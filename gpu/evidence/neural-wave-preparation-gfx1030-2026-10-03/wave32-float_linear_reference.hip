// Isolated experimental full-wave mapping; physical wave 32.
#define CHEN_WAVE 32
// Controlled FP32 / packed FP16 comparisons for chen_int8_linear. Inputs and
// weights are prepared before timing; both paths accumulate and output FP32.
#include <hip/hip_runtime.h>
#include <cstdint>
using Half2 = __attribute__((vector_size(4))) __fp16;
union Pair { uint32_t bits; Half2 halves; };
namespace {

template<bool Half>
__global__ void affine(const void* a,const void* w,const float* bias,float* y,
                       int m,int n,int k) {
    const uint64_t wave=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/CHEN_WAVE;
    const int tiles=(n+(CHEN_WAVE-1))/CHEN_WAVE,lane=threadIdx.x&(CHEN_WAVE-1);
    const int row=__builtin_amdgcn_readfirstlane(int(wave/tiles));
    if(row>=m) return;
    const int column=int(wave%tiles)*CHEN_WAVE+lane,npad=tiles*CHEN_WAVE;
    float sums[4]={0,0,0,0};
    const int steps=Half?k/2:k;
    for(int start=0;start<steps;start+=4) {
        #pragma unroll
        for(int j=0;j<4;++j) if(start+j<steps) {
            const int p=start+j;
            if(Half) {
                Pair av{static_cast<const uint32_t*>(a)[uint64_t(row)*steps+p]};
                Pair wv{static_cast<const uint32_t*>(w)[uint64_t(p)*npad+column]};
                sums[j]=__builtin_amdgcn_fdot2(av.halves,wv.halves,sums[j],false);
            } else {
                const float av=static_cast<const float*>(a)[uint64_t(row)*k+p];
                const float wv=static_cast<const float*>(w)[uint64_t(p)*npad+column];
                sums[j]=__builtin_fmaf(av,wv,sums[j]);
            }
        }
    }
    if(column<n) y[uint64_t(row)*n+column]=((sums[0]+sums[1])+(sums[2]+sums[3]))+(bias?bias[column]:0);
}
}
extern "C" int32_t chen_float_linear(const void* a,const void* w,const float* bias,
    float* y,uint64_t m,uint64_t n,uint64_t k,uint64_t half,hipStream_t stream) {
    if(m>1048576||!n||n>4096||!k||k>4096||(k&3)||half>1) return int32_t(hipErrorInvalidValue);
    if(!m) return int32_t(hipSuccess);
    if(!a||!w||!y) return int32_t(hipErrorInvalidValue);
    const dim3 grid((m*((n+(CHEN_WAVE-1))/CHEN_WAVE)*CHEN_WAVE+127)/128),block(128);
    if(half) hipLaunchKernelGGL((affine<true>),grid,block,0,stream,a,w,bias,y,int(m),int(n),int(k));
    else hipLaunchKernelGGL((affine<false>),grid,block,0,stream,a,w,bias,y,int(m),int(n),int(k));
    return int32_t(hipGetLastError());
}

// Experimental admission only; initializes the HIP runtime when called.
extern "C" int32_t chen_wave_resources(uint64_t kind,uint64_t block,uint64_t* out) {
    if(!out || !block || block>1024) return int32_t(hipErrorInvalidValue);
    const void* function=nullptr;
    switch(kind) { case 0: function = reinterpret_cast<const void*>(affine<false>); break;
case 1: function = reinterpret_cast<const void*>(affine<true>); break; default: return int32_t(hipErrorInvalidValue); }
    hipFuncAttributes attr{};
    hipError_t rc=hipFuncGetAttributes(&attr,function);
    if(rc!=hipSuccess) return int32_t(rc);
    int active=0;
    rc=hipOccupancyMaxActiveBlocksPerMultiprocessor(&active,function,int(block),0);
    if(rc!=hipSuccess) return int32_t(rc);
    out[0]=attr.numRegs; out[1]=attr.sharedSizeBytes; out[2]=attr.localSizeBytes;
    out[3]=attr.maxThreadsPerBlock; out[4]=active; out[5]=CHEN_WAVE;
    return int32_t(hipSuccess);
}
