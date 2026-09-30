// Device-only synthetic callback fixture, not a learned inference model.
#include "chen_external.h"
#include <hip/hip_runtime.h>
#include <cstdio>
struct ProbeOwner { uint64_t calls;uint32_t mode,fail_at;int32_t score;uint32_t reserved; };
__global__ void fill_scores(int32_t *scores,uint64_t count,uint32_t mode,int32_t value) {
    const uint64_t i=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(i<count && !(mode==1 && (i&1))) scores[i]=(mode==2 && !i)?29001:value;
}
extern "C" int32_t chen_external_probe_callback(void *opaque,int32_t device,
    const ChenRulePosition *,uint64_t count,int32_t *scores,char *error,uint64_t cap) {
    auto *owner=static_cast<ProbeOwner *>(opaque);++owner->calls;
    hipError_t status=hipSetDevice(device);hipStream_t stream=nullptr;
    if(status==hipSuccess) status=hipStreamCreateWithFlags(&stream,hipStreamNonBlocking);
    if(status==hipSuccess && count) {
        hipLaunchKernelGGL(fill_scores,dim3(uint32_t((count+63)/64)),dim3(64),0,stream,
            scores,count,owner->mode,owner->score);
        status=hipGetLastError();
    }
    const hipError_t drain=stream?hipStreamSynchronize(stream):hipSuccess;
    if(status==hipSuccess) status=drain;
    if(stream) (void)hipStreamDestroy(stream);
    if(status!=hipSuccess) { if(error&&cap) std::snprintf(error,size_t(cap),"%s",hipGetErrorString(status));return -1; }
    if(owner->mode==3 && owner->calls==owner->fail_at) {
        if(error&&cap) std::snprintf(error,size_t(cap),"intentional synthetic callback failure");return -1;
    }
    return 0;
}
