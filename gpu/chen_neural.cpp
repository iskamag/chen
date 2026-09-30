#include "chen_neural.h"
#include <torch/script.h>
#include <torch/version.h>
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/hip/HIPStream.h>
#include <hip/hip_runtime.h>
#include <cstdio>
#include <sstream>
#include <string>

extern "C" int32_t chen_neural_decode(const ChenRulePosition *,uint64_t,
    float *,hipStream_t);
struct ChenNeuralContext {
    torch::jit::Module module;
    at::Tensor features;
    int32_t device;
    uint64_t max_batch;
};
namespace {
int32_t fail(char *error,uint64_t capacity,const char *message) {
    if(error && capacity) std::snprintf(error,size_t(capacity),"%s",message);
    return -1;
}
void check(hipError_t code) {
    TORCH_CHECK(code==hipSuccess,hipGetErrorString(code));
}
// All exits retire GPU uses before locally-owned tensors or context can die.
struct Completion {
    ~Completion() { (void)hipDeviceSynchronize(); }
};
}
extern "C" uint32_t chen_neural_abi_version() { return 1; }
extern "C" const char *chen_neural_torch_version() { return TORCH_VERSION; }
extern "C" uint64_t chen_neural_runtime_identity(uint32_t selector) {
    switch(selector) {
        case 0:return reinterpret_cast<uint64_t>(&hipPointerGetAttributes);
        case 1:return reinterpret_cast<uint64_t>(&hipMemGetAddressRange);
        case 2:return reinterpret_cast<uint64_t>(static_cast<hipError_t (*)(void **,size_t)>(&hipMalloc));
        case 3:return reinterpret_cast<uint64_t>(&hipFree);
        case 4:return reinterpret_cast<uint64_t>(&hipStreamSynchronize);
        default:return 0;
    }
}
extern "C" ChenNeuralContext *chen_neural_create(const void *archive,
    uint64_t archive_bytes,const char *expected_manifest,uint64_t manifest_bytes,
    int32_t device,uint64_t max_batch,char *error,uint64_t error_capacity) {
    if(error && error_capacity) *error='\0';
    try {
        TORCH_CHECK(archive && archive_bytes && archive_bytes<=64*1024*1024,
            "invalid bounded inference archive");
        TORCH_CHECK(expected_manifest && manifest_bytes && manifest_bytes<=65536,
            "invalid inference manifest");
        TORCH_CHECK(max_batch && max_batch<=4096,"invalid neural batch capacity");
        check(hipSetDevice(device));
        c10::InferenceMode inference;
        // Construct owners before the fence, so its destructor runs first.
        auto ctx=std::make_unique<ChenNeuralContext>();
        Completion completion;
        torch::jit::ExtraFilesMap extra{{"chen_manifest.json",""}};
        std::istringstream input(std::string(static_cast<const char *>(archive),size_t(archive_bytes)));
        ctx->module=torch::jit::load(input,c10::Device(c10::kCUDA,device),extra);
        TORCH_CHECK(extra.at("chen_manifest.json")==
            std::string(expected_manifest,size_t(manifest_bytes)),
            "embedded inference manifest mismatch");
        ctx->device=device;ctx->max_batch=max_batch;
        ctx->module.eval();
        ctx->features=at::empty({int64_t(max_batch),18,8,8},
            at::TensorOptions().dtype(at::kFloat).device(c10::Device(c10::kCUDA,device)));
        check(hipDeviceSynchronize());
        return ctx.release();
    } catch(const std::exception &exc) {
        fail(error,error_capacity,exc.what());return nullptr;
    } catch(...) {
        fail(error,error_capacity,"unknown neural creation failure");return nullptr;
    }
}
extern "C" void chen_neural_destroy(ChenNeuralContext *ctx) {
    if(!ctx) return;
    (void)hipSetDevice(ctx->device);
    (void)hipDeviceSynchronize();
    delete ctx;
}
extern "C" int32_t chen_neural_evaluate(void *owner,int32_t device,
    const ChenRulePosition *positions,uint64_t count,int32_t *scores,
    char *error,uint64_t capacity) {
    if(error && capacity) *error='\0';
    auto ctx=static_cast<ChenNeuralContext *>(owner);
    if(!ctx || ctx->device!=device || count>ctx->max_batch ||
        (count && (!positions || !scores))) return fail(error,capacity,"invalid neural callback arguments");
    if(!count) return 0;
    try {
        check(hipSetDevice(device));
        c10::InferenceMode inference;
        // Completion must outlive all enqueued uses of local output storage.
        at::Tensor output;
        Completion completion;
        const auto stream=c10::hip::getCurrentHIPStream(device).stream();
        auto features=ctx->features.narrow(0,0,int64_t(count));
        check(static_cast<hipError_t>(chen_neural_decode(positions,count,
            features.data_ptr<float>(),stream)));
        output=ctx->module.forward({features}).toTensor();
        TORCH_CHECK(output.scalar_type()==at::kInt && output.dim()==1 &&
            output.size(0)==int64_t(count) && output.is_contiguous() &&
            output.device()==features.device(),"neural output shape/dtype/device mismatch");
        // Output validation reads one bounded scalar. JIT may also read small
        // scalars internally; boards and score payloads remain on device.
        TORCH_CHECK(((output>=-29000)&(output<=29000)).all().item<bool>(),
            "neural static score outside range");
        check(hipMemcpyAsync(scores,output.data_ptr<int32_t>(),count*sizeof(int32_t),
            hipMemcpyDeviceToDevice,stream));
        check(hipDeviceSynchronize());
        return 0;
    } catch(const std::exception &exc) {
        return fail(error,capacity,exc.what());
    } catch(...) {
        return fail(error,capacity,"unknown neural evaluation failure");
    }
}
