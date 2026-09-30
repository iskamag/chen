#include "chen_neural.h"
#include <torch/script.h>
#include <torch/version.h>
#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/hip/HIPStream.h>
#include <c10/hip/HIPGuard.h>
#include <c10/hip/HIPCachingAllocator.h>
#include <ATen/hip/HIPGraph.h>
#include <hip/hip_runtime.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdlib>
#include <cstdio>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

extern "C" int32_t chen_neural_decode(const ChenRulePosition *,uint64_t,
    float *,hipStream_t);
namespace {
std::atomic<bool> graph_runtime_poisoned{false};
std::recursive_mutex plugin_mutex;
namespace Allocator=c10::hip::HIPCachingAllocator;
struct InjectedFailure:std::runtime_error {using std::runtime_error::runtime_error;};
struct AdmissionFailure:std::runtime_error {using std::runtime_error::runtime_error;};
[[noreturn]] void retirement_fault(const char *stage) noexcept {
    std::fprintf(stderr,"chen fatal graph retirement fault: %s; exit70 preserves live storage until process teardown\n",stage);
    std::fflush(stderr);std::_Exit(70);
}
void retire_check(hipError_t status,const char *stage) noexcept {
    if(status!=hipSuccess) retirement_fault(stage);
}
struct GraphEntry {
    int32_t device;
    uint64_t count, pool_bytes=0, tick=0;
    c10::hip::HIPStream stream;
    c10::MempoolId_t pool;
    hipGraph_t graph=nullptr;
    hipGraphExec_t executable=nullptr;
    unsigned long long capture_id=0;
    bool pool_owned=false, pool_active=false;
    bool test_retirement_failure=false;
    at::Tensor output, valid;
    GraphEntry(int32_t d,uint64_t n):device(d),count(n),
        stream(c10::hip::getStreamFromPool(false,d)),pool(at::cuda::graph_pool_handle()) {}
    // Never release potentially live storage. Retirement failures fail-stop
    // before tensor/pool reclamation; actual capture faults poison all plugin use.
    ~GraphEntry() {
        if(test_retirement_failure) retirement_fault("injected unproven completion fence");
        retire_check(hipSetDevice(device),"setdevice");
        hipStreamCaptureStatus status=hipStreamCaptureStatusNone;
        retire_check(hipStreamIsCapturing(stream.stream(),&status),"capture status");
        if(status!=hipStreamCaptureStatusNone) {
            hipGraph_t aborted=nullptr;
            (void)hipStreamEndCapture(stream.stream(),&aborted);
            if(aborted) retire_check(hipGraphDestroy(aborted),"aborted graph destroy");
            retire_check(hipStreamIsCapturing(stream.stream(),&status),"retired capture status");
            if(status!=hipStreamCaptureStatusNone) retirement_fault("capture still active");
        }
        if(pool_active) {
            try { Allocator::endAllocateToPool(device,pool);pool_active=false; }
            catch(...) { retirement_fault("allocator end"); }
        }
        retire_check(hipDeviceSynchronize(),"completion fence");
        if(executable) retire_check(hipGraphExecDestroy(executable),"executable destroy");
        if(graph) retire_check(hipGraphDestroy(graph),"graph destroy");
        // HIP graph destruction can defer release until a subsequent sync.
        retire_check(hipDeviceSynchronize(),"destroy completion fence");
        output=at::Tensor();valid=at::Tensor();
        if(pool_owned && !pool_active) {
            try { Allocator::releasePool(device,pool);Allocator::emptyCache(pool); }
            catch(...) { retirement_fault("allocator release/trim"); }
        }
    }
};
}
struct ChenNeuralContext {
    torch::jit::Module module;
    at::Tensor features, positions;
    int32_t device;
    uint64_t max_batch;
    ChenNeuralExecutionConfig execution{1,0,0,0,0};
    ChenNeuralExecutionStats stats{1,0,0,0,0,0,0,0,0,0,0,0,0};
    uint64_t tick=0;
    std::vector<std::unique_ptr<GraphEntry>> graphs;
    uint32_t test_failure=0;
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
    ~Completion() { retire_check(hipDeviceSynchronize(),"local owner completion fence"); }
};
void validate_output(const at::Tensor &output,const at::Tensor &features,uint64_t count) {
    TORCH_CHECK(output.scalar_type()==at::kInt && output.dim()==1 &&
        output.size(0)==int64_t(count) && output.is_contiguous() &&
        output.device()==features.device(),"neural output shape/dtype/device mismatch");
}
void injected(ChenNeuralContext *ctx,uint32_t stage) {
    if(ctx->test_failure==stage) {
        ctx->test_failure=0;
        throw InjectedFailure("injected graph failure stage "+std::to_string(stage));
    }
}
void evict(ChenNeuralContext *ctx) {
    const auto oldest=std::min_element(ctx->graphs.begin(),ctx->graphs.end(),
        [](const auto &a,const auto &b){return a->tick<b->tick;});
    TORCH_CHECK(oldest!=ctx->graphs.end(),"graph retained budget exhausted");
    check(hipDeviceSynchronize());
    ctx->stats.retained_pool_bytes-=(*oldest)->pool_bytes;
    ctx->graphs.erase(oldest);++ctx->stats.evictions;
    ctx->stats.cached_shapes=ctx->graphs.size();
    TORCH_CHECK(!graph_runtime_poisoned.load(),"graph cleanup fault; process restart required");
}
GraphEntry *capture(ChenNeuralContext *ctx,uint64_t count) {
    auto entry=std::make_unique<GraphEntry>(ctx->device,count);
    entry->test_retirement_failure=ctx->test_failure==5;
    const auto started=std::chrono::steady_clock::now();
    // Owned input/features are reused only under exclusive context ownership.
    // All upstream staging uses have completed before this side stream starts.
    check(hipDeviceSynchronize());
    c10::hip::HIPStreamGuard stream_guard(entry->stream);
    const auto features=ctx->features.narrow(0,0,int64_t(count));
    for(unsigned i=0;i<5;++i) {
        check(static_cast<hipError_t>(chen_neural_decode(
            reinterpret_cast<const ChenRulePosition *>(ctx->positions.data_ptr<uint8_t>()),
            count,features.data_ptr<float>(),entry->stream.stream())));
        auto output=ctx->module.forward({features}).toTensor();
        validate_output(output,features,count);
        TORCH_CHECK(((output>=-29000)&(output<=29000)).all().item<bool>(),
            "neural static score outside range");
    }
    check(hipDeviceSynchronize());
    try {
    // Version-pinned native allocator protocol: one unique pool from Torch's
    // graph_pool_handle, one begin/end and one release per admitted capture.
    // Restrict graph mode to native allocator and deterministic generated IR.
    try {
        Allocator::beginAllocateToPool(ctx->device,entry->pool,
            [owner=entry.get()](hipStream_t candidate){
                hipStreamCaptureStatus status=hipStreamCaptureStatusNone;
                unsigned long long id=0;
                check(hipStreamGetCaptureInfo(candidate,&status,&id));
                return status==hipStreamCaptureStatusActive && id==owner->capture_id;
            });
        entry->pool_owned=true;entry->pool_active=true;
    } catch(...) {
        // beginAllocateToPool is not documented as transactional on exception.
        graph_runtime_poisoned.store(true);throw;
    }
    injected(ctx,1);
    check(hipStreamBeginCapture(entry->stream.stream(),hipStreamCaptureModeGlobal));
    hipStreamCaptureStatus capture_status=hipStreamCaptureStatusNone;
    check(hipStreamGetCaptureInfo(entry->stream.stream(),&capture_status,&entry->capture_id));
    TORCH_CHECK(capture_status==hipStreamCaptureStatusActive,"graph capture did not become active");
    injected(ctx,2);
    injected(ctx,5);
    check(static_cast<hipError_t>(chen_neural_decode(
        reinterpret_cast<const ChenRulePosition *>(ctx->positions.data_ptr<uint8_t>()),
        count,features.data_ptr<float>(),entry->stream.stream())));
    entry->output=ctx->module.forward({features}).toTensor();
    validate_output(entry->output,features,count);
    entry->valid=((entry->output>=-29000)&(entry->output<=29000)).all();
    check(hipStreamEndCapture(entry->stream.stream(),&entry->graph));
    Allocator::endAllocateToPool(ctx->device,entry->pool);entry->pool_active=false;
    injected(ctx,3);
    TORCH_CHECK(entry->graph,"empty neural graph capture");
    check(hipGraphInstantiate(&entry->executable,entry->graph,nullptr,nullptr,0));
    check(hipGraphDestroy(entry->graph));entry->graph=nullptr;
    check(hipDeviceSynchronize());
    for(const auto &segment:Allocator::snapshot(entry->pool).segments) {
        if(segment.device==ctx->device && segment.owner_private_pool_id==entry->pool)
            entry->pool_bytes+=segment.total_size;
    }
    ctx->stats.peak_observed_pool_bytes=std::max(ctx->stats.peak_observed_pool_bytes,
        ctx->stats.retained_pool_bytes+entry->pool_bytes);
    ++ctx->stats.captures;
    ctx->stats.capture_wall_ms+=std::chrono::duration<double,std::milli>(
        std::chrono::steady_clock::now()-started).count();
    if(entry->pool_bytes>ctx->execution.retained_budget_bytes)
        throw AdmissionFailure("graph exceeds retained pool budget");
    while(ctx->graphs.size()>=ctx->execution.max_cached_shapes ||
        ctx->stats.retained_pool_bytes>ctx->execution.retained_budget_bytes-entry->pool_bytes)
        evict(ctx);
    injected(ctx,4);
    entry->tick=++ctx->tick;
    auto *result=entry.get();
    ctx->graphs.push_back(std::move(entry));
    ctx->stats.retained_pool_bytes+=result->pool_bytes;
    ctx->stats.cached_shapes=ctx->graphs.size();
    return result;
    } catch(const InjectedFailure &) {throw;}
    catch(const AdmissionFailure &) {throw;}
    catch(...) {graph_runtime_poisoned.store(true);throw;}
}
void reject_random_ops(const std::shared_ptr<torch::jit::Graph> &graph) {
    // Raw HIP capture has no Torch RNG replay protocol. This check supplements
    // the trusted generated deterministic StaticEvaluator admission contract;
    // it is not a validator for arbitrary executable TorchScript archives.
    for(const auto *node:graph->nodes()) {
        const std::string kind=node->kind().toQualString();
        TORCH_CHECK(kind.find("aten::rand")==std::string::npos &&
            kind.find("aten::normal")==std::string::npos &&
            kind.find("aten::bernoulli")==std::string::npos &&
            kind.find("aten::dropout")==std::string::npos &&
            kind.find("aten::multinomial")==std::string::npos,
            "graph execution rejects RNG operators");
        TORCH_CHECK(node->blocks().empty(),"graph execution rejects dynamic control flow");
    }
}
}
extern "C" uint32_t chen_neural_abi_version() { return 1; }
extern "C" uint32_t chen_neural_execution_abi_version() { return 1; }
extern "C" int32_t chen_neural_set_execution(ChenNeuralContext *ctx,
    const ChenNeuralExecutionConfig *config,char *error,uint64_t capacity) {
    std::lock_guard<std::recursive_mutex> lock(plugin_mutex);
    if(error && capacity) *error='\0';
    if(!ctx || !config || config->abi_version!=1 || config->reserved || config->mode>1 ||
        ctx->stats.first_used || (config->mode==0 && (config->max_cached_shapes || config->retained_budget_bytes)) ||
        (config->mode==1 && (!config->max_cached_shapes || config->max_cached_shapes>16 ||
            !config->retained_budget_bytes || config->retained_budget_bytes>uint64_t(4)*1024*1024*1024)))
        return fail(error,capacity,"invalid or immutable neural execution config");
    try {
        TORCH_CHECK(!graph_runtime_poisoned.load(),"graph capture fault; process restart required");
        check(hipSetDevice(ctx->device));
        c10::InferenceMode inference;
        at::Tensor positions;
        Completion completion;
        if(config->mode) {
            TORCH_CHECK(!graph_runtime_poisoned.load(),"graph cleanup fault; process restart required");
            TORCH_CHECK(Allocator::name()=="native" && Allocator::isEnabled(),
                "graph execution requires native caching allocator");
            const auto allocator_config=Allocator::snapshot().config_metadata;
            TORCH_CHECK(!allocator_config.expandable_segments &&
                !allocator_config.graph_capture_record_stream_reuse,
                "graph execution requires fixed segments and default capture reuse policy");
            reject_random_ops(ctx->module.get_method("forward").graph());
            positions=at::empty({int64_t(ctx->max_batch*sizeof(ChenRulePosition))},
                at::TensorOptions().dtype(at::kByte).device(c10::Device(c10::kCUDA,ctx->device)));
        }
        check(hipDeviceSynchronize());
        ctx->positions=std::move(positions);ctx->execution=*config;
        ctx->stats.mode=config->mode;ctx->stats.max_cached_shapes=config->max_cached_shapes;
        ctx->stats.staging_logical_bytes=config->mode?ctx->max_batch*sizeof(ChenRulePosition):0;
        return 0;
    } catch(const std::exception &exc) { return fail(error,capacity,exc.what()); }
    catch(...) { return fail(error,capacity,"unknown execution configuration failure"); }
}
extern "C" int32_t chen_neural_execution_stats(const ChenNeuralContext *ctx,
    ChenNeuralExecutionStats *stats,char *error,uint64_t capacity) {
    std::lock_guard<std::recursive_mutex> lock(plugin_mutex);
    if(error && capacity) *error='\0';
    if(!ctx || !stats) return fail(error,capacity,"invalid execution stats arguments");
    *stats=ctx->stats;return 0;
}
extern "C" int32_t chen_neural_test_inject_failure(ChenNeuralContext *ctx,uint32_t stage) {
    std::lock_guard<std::recursive_mutex> lock(plugin_mutex);
#ifdef CHEN_NEURAL_TESTING
    if(!ctx || ctx->execution.mode!=1 || stage>5) return -1;
    ctx->test_failure=stage;return 0;
#else
    (void)ctx;(void)stage;return -1;
#endif
}
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
    std::lock_guard<std::recursive_mutex> lock(plugin_mutex);
    if(error && error_capacity) *error='\0';
    try {
        TORCH_CHECK(!graph_runtime_poisoned.load(),"graph capture fault; process restart required");
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
    std::lock_guard<std::recursive_mutex> lock(plugin_mutex);
    if(!ctx) return;
    retire_check(hipSetDevice(ctx->device),"context destroy setdevice");
    retire_check(hipDeviceSynchronize(),"context destroy completion fence");
    ctx->graphs.clear();
    delete ctx;
}
extern "C" int32_t chen_neural_evaluate(void *owner,int32_t device,
    const ChenRulePosition *positions,uint64_t count,int32_t *scores,
    char *error,uint64_t capacity) {
    std::lock_guard<std::recursive_mutex> lock(plugin_mutex);
    if(error && capacity) *error='\0';
    auto ctx=static_cast<ChenNeuralContext *>(owner);
    if(!ctx || ctx->device!=device || count>ctx->max_batch ||
        (count && (!positions || !scores))) return fail(error,capacity,"invalid neural callback arguments");
    if(graph_runtime_poisoned.load()) return fail(error,capacity,"graph capture fault; process restart required");
    if(!count) return 0;
    try {
        check(hipSetDevice(device));
        c10::InferenceMode inference;
        // Completion must outlive all enqueued uses of local output storage.
        at::Tensor output;
        Completion completion;
        ctx->stats.first_used=1;
        const auto stream=c10::hip::getCurrentHIPStream(device).stream();
        auto features=ctx->features.narrow(0,0,int64_t(count));
        if(ctx->execution.mode) {
            TORCH_CHECK(!graph_runtime_poisoned.load(),"graph cleanup fault; process restart required");
            check(hipMemcpyAsync(ctx->positions.data_ptr<uint8_t>(),positions,
                count*sizeof(ChenRulePosition),hipMemcpyDeviceToDevice,stream));
            GraphEntry *entry=nullptr;
            for(auto &candidate:ctx->graphs) if(candidate->count==count) {entry=candidate.get();break;}
            if(entry) { ++ctx->stats.hits;entry->tick=++ctx->tick; }
            else { ++ctx->stats.misses;entry=capture(ctx,count); }
            check(hipGraphLaunch(entry->executable,stream));
            TORCH_CHECK(entry->valid.item<bool>(),"neural static score outside range");
            check(hipMemcpyAsync(scores,entry->output.data_ptr<int32_t>(),count*sizeof(int32_t),
                hipMemcpyDeviceToDevice,stream));
            check(hipDeviceSynchronize());
            return 0;
        }
        check(static_cast<hipError_t>(chen_neural_decode(positions,count,
            features.data_ptr<float>(),stream)));
        output=ctx->module.forward({features}).toTensor();
        validate_output(output,features,count);
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
