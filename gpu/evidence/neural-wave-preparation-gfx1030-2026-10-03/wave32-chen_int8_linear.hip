// Isolated experimental full-wave mapping; physical wave 32.
#define CHEN_WAVE 32
// gfx1030 mixed INT8 linear operator. No allocations, synchronization, ownership
// transfers, or hidden default-stream work. Python owns tensors and stream.
#include <hip/hip_runtime.h>
#include <cstdint>
#include <cfloat>

namespace {

__device__ __forceinline__ int chen_wave_sum(int value) {
    #pragma unroll
    for(int step=16;step>0;step/=2) value+=__shfl_down(value,step,32);
    if(CHEN_WAVE==64) value=__builtin_amdgcn_readlane(value,0)+__builtin_amdgcn_readlane(value,32);
    return value;
}
__device__ __forceinline__ float chen_wave_max(float value) {
    #pragma unroll
    for(int step=16;step>0;step/=2) value=fmaxf(value,__shfl_xor(value,step,32));
    if(CHEN_WAVE==64) {
        union Bits { float f; int i; } bits{value},lower{},upper{};
        lower.i=__builtin_amdgcn_readlane(bits.i,0);
        upper.i=__builtin_amdgcn_readlane(bits.i,32);
        value=fmaxf(lower.f,upper.f);
    }
    return value;
}

__device__ __forceinline__ int dot4(int a,int b,int acc) {
    return __builtin_amdgcn_sdot4(a,b,acc,false);
}
template<bool Packed>
__global__ void linear(const int32_t* __restrict__ a,const int32_t* __restrict__ w,
                      const float* sa,const float* sw,const float* bias,
                      const float* residual,float* y,int m,int n,int chunks) {
    const uint64_t wave=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/CHEN_WAVE;
    const int lane=threadIdx.x&(CHEN_WAVE-1),tiles=(n+(CHEN_WAVE-1))/CHEN_WAVE;
    const int row=__builtin_amdgcn_readfirstlane(int(wave/tiles));
    if(row>=m) return; // Full wave returns, never a partial-wave broadcast.
    const int column=int(wave%tiles)*CHEN_WAVE+lane,npad=tiles*CHEN_WAVE;
    int s0=0,s1=0,s2=0,s3=0;
    int k=0;
    // Independent accumulators hide the instruction dependency chain. Packed
    // weights coalesce the wave's output-channel reads at each K/4 step.
    for(;k+3<chunks;k+=4) {
        int av0=a[uint64_t(row)*chunks+k],av1=a[uint64_t(row)*chunks+k+1];
        int av2=a[uint64_t(row)*chunks+k+2],av3=a[uint64_t(row)*chunks+k+3];
        int w0=0,w1=0,w2=0,w3=0;
        if(Packed) {
            w0=w[uint64_t(k)*npad+column]; w1=w[uint64_t(k+1)*npad+column];
            w2=w[uint64_t(k+2)*npad+column]; w3=w[uint64_t(k+3)*npad+column];
        } else if(column<n) {
            w0=w[uint64_t(column)*chunks+k];w1=w[uint64_t(column)*chunks+k+1];
            w2=w[uint64_t(column)*chunks+k+2];w3=w[uint64_t(column)*chunks+k+3];
        }
        s0=dot4(av0,w0,s0);s1=dot4(av1,w1,s1);s2=dot4(av2,w2,s2);s3=dot4(av3,w3,s3);
    }
    for(;k<chunks;++k) {
        const int av=a[uint64_t(row)*chunks+k];
        const int wv=Packed?w[uint64_t(k)*npad+column]:(column<n?w[uint64_t(column)*chunks+k]:0);
        s0=dot4(av,wv,s0);
    }
    if(column<n) {
        const uint64_t index=uint64_t(row)*n+column;
        float value=float((s0+s1)+(s2+s3))*(sa[row]*sw[column]);
        if(bias) value+=bias[column];
        if(residual) value+=residual[index];
        y[index]=value;
    }
}

// The scalar readout is a separate mapping: lanes partition K rather than
// wasting (CHEN_WAVE-1) lanes per output. All lanes participate in the integer reduction.
template<bool Packed>
__global__ void scalar(const int32_t* a,const int32_t* w,const float* sa,
                       const float* sw,const float* bias,const float* residual,
                       float* y,int m,int chunks) {
    const uint64_t row=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/CHEN_WAVE;
    const int lane=threadIdx.x&(CHEN_WAVE-1);
    if(row>=uint64_t(m)) return;
    int sum=0;
    for(int k=lane;k<chunks;k+=CHEN_WAVE) sum=dot4(a[row*chunks+k],w[Packed?uint64_t(k)*CHEN_WAVE:k],sum);
    sum=chen_wave_sum(sum);
    if(!lane) {
        float value=float(sum)*(sa[row]*sw[0]);
        if(bias) value+=bias[0];
        if(residual) value+=residual[row];
        y[row]=value;
    }
}
__global__ void pack_weights(const int32_t* w,int32_t* packed,int n,int chunks,int npad) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=uint64_t(chunks)*npad) return;
    const int k=int(index/npad),column=int(index%npad);
    packed[index]=column<n?w[uint64_t(column)*chunks+k]:0;
}
__device__ __forceinline__ int quant_word(const float* a,float scale,int available=4) {
    uint32_t word=0;
    #pragma unroll
    for(int byte=0;byte<4;++byte) {
        const int q=byte<available?int(nearbyintf(fminf(127.f,fmaxf(-127.f,a[byte]/scale)))):0;
        word|=(uint32_t(q)&255)<<(byte*8);
    }
    return int(word);
}
// Dynamic row quantization + dot + dequantization in one dispatch. This saves
// absmax/scale/round/cast/pad tensor launches for the narrow semantic layers.
// The padded API provides zero K padding; the rows API reads tails safely
// without allocating a padded FP32 tensor. Residual and bias remain FP32.
template<bool Scalar>
__global__ void dynamic_linear(const float* a,const int32_t* w,const float* sw,
    const float* bias,const float* residual,float* y,int m,int n,int k,int source_k) {
    const uint64_t wave=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/CHEN_WAVE;
    const int tiles=(n+(CHEN_WAVE-1))/CHEN_WAVE,lane=threadIdx.x&(CHEN_WAVE-1);
    const int row=__builtin_amdgcn_readfirstlane(int(wave/tiles));
    if(row>=m) return;
    const int column=int(wave%tiles)*CHEN_WAVE+lane,npad=tiles*CHEN_WAVE,chunks=k/4;
    const float* inputs=a+uint64_t(row)*source_k;
    float maximum=0;
    for(int i=lane;i<source_k;i+=CHEN_WAVE) maximum=fmaxf(maximum,fabsf(inputs[i]));
    maximum=chen_wave_max(maximum);
    // Torch's division by a CPU scalar first rounds its reciprocal to FP32.
    // Keep that ordering explicit: true division changes halfway byte rounding.
    const float scale=maximum==0?1.f:fmaxf(maximum*(1.f/127.f),FLT_MIN);
    int sums[4]={0,0,0,0};
    if(Scalar) {
        for(int chunk=lane;chunk<chunks;chunk+=CHEN_WAVE)
            sums[0]=dot4(quant_word(inputs+4*chunk,scale,source_k-4*chunk),w[uint64_t(chunk)*npad],sums[0]);
    } else for(int start=0;start<chunks;start+=4) {
        #pragma unroll
        for(int j=0;j<4;++j) if(start+j<chunks) {
            const int chunk=start+j;
            const int av=__builtin_amdgcn_readfirstlane(quant_word(inputs+4*chunk,scale,source_k-4*chunk));
            sums[j]=dot4(av,w[uint64_t(chunk)*npad+column],sums[j]);
        }
    }
    int sum=(sums[0]+sums[1])+(sums[2]+sums[3]);
    if(Scalar) {
        sum=chen_wave_sum(sum);
    }
    if((Scalar?!lane:column<n)) {
        const int c=Scalar?0:column;
        const uint64_t index=uint64_t(row)*n+c;
        float value=float(sum)*(scale*sw[c]);
        if(bias) value+=bias[c];
        if(residual) value+=residual[index];
        y[index]=value;
    }
}
bool aligned(const void* p) { return p && !(reinterpret_cast<uintptr_t>(p)&3); }
bool dimensions(uint64_t m,uint64_t n,uint64_t k) {
    return m<=1048576 && n>0 && n<=4096 && k>0 && k<=4096 && !(k&3);
}
template<bool Packed>
int32_t dispatch(const int8_t* a,const void* w,const float* sa,const float* sw,
                 const float* bias,const float* residual,float* y,
                 uint64_t m,uint64_t n,uint64_t k,hipStream_t stream) {
    if(!dimensions(m,n,k)) return int32_t(hipErrorInvalidValue);
    if(!m) return int32_t(hipSuccess);
    if(!aligned(a)||!aligned(w)||!aligned(sa)||!aligned(sw)||!aligned(y)||
       (bias&&!aligned(bias))||(residual&&!aligned(residual))) return int32_t(hipErrorInvalidValue);
    dim3 block(128);
    const auto* words=reinterpret_cast<const int32_t*>(a);
    const auto* weights=reinterpret_cast<const int32_t*>(w);
    if(n==1) {
        hipLaunchKernelGGL((scalar<Packed>),dim3((m*CHEN_WAVE+127)/128),block,0,stream,
                          words,weights,sa,sw,bias,residual,y,int(m),int(k/4));
    } else {
        hipLaunchKernelGGL((linear<Packed>),dim3((m*((n+(CHEN_WAVE-1))/CHEN_WAVE)*CHEN_WAVE+127)/128),block,0,stream,
                          words,weights,sa,sw,bias,residual,y,int(m),int(n),int(k/4));
    }
    return int32_t(hipGetLastError());
}
}
extern "C" int32_t chen_int8_linear(const int8_t* a,const int8_t* w,const float* sa,
    const float* sw,const float* bias,float* y,uint64_t m,uint64_t n,uint64_t k,hipStream_t stream) {
    return dispatch<false>(a,w,sa,sw,bias,nullptr,y,m,n,k,stream);
}
extern "C" int32_t chen_int8_linear_packed(const int8_t* a,const int32_t* w,const float* sa,
    const float* sw,const float* bias,float* y,uint64_t m,uint64_t n,uint64_t k,hipStream_t stream) {
    return dispatch<true>(a,w,sa,sw,bias,nullptr,y,m,n,k,stream);
}
extern "C" int32_t chen_int8_linear_packed_residual(const int8_t* a,const int32_t* w,const float* sa,
    const float* sw,const float* bias,const float* residual,float* y,uint64_t m,uint64_t n,uint64_t k,hipStream_t stream) {
    return dispatch<true>(a,w,sa,sw,bias,residual,y,m,n,k,stream);
}
extern "C" int32_t chen_int8_pack_weights(const int8_t* w,int32_t* packed,
    uint64_t n,uint64_t k,hipStream_t stream) {
    if(!dimensions(1,n,k)||!aligned(w)||!aligned(packed)) return int32_t(hipErrorInvalidValue);
    const int npad=int((n+(CHEN_WAVE-1))&~uint64_t((CHEN_WAVE-1)));
    hipLaunchKernelGGL(pack_weights,dim3(((k/4)*npad+255)/256),dim3(256),0,stream,
                      reinterpret_cast<const int32_t*>(w),packed,int(n),int(k/4),npad);
    return int32_t(hipGetLastError());
}
extern "C" int32_t chen_int8_linear_dynamic_packed(const float* a,const int32_t* w,
    const float* sw,const float* bias,const float* residual,float* y,
    uint64_t m,uint64_t n,uint64_t k,hipStream_t stream) {
    if(!dimensions(m,n,k)) return int32_t(hipErrorInvalidValue);
    if(!m) return int32_t(hipSuccess);
    if(!aligned(a)||!aligned(w)||!aligned(sw)||!aligned(y)||
       (bias&&!aligned(bias))||(residual&&!aligned(residual))) return int32_t(hipErrorInvalidValue);
    const dim3 grid((m*((n+(CHEN_WAVE-1))/CHEN_WAVE)*CHEN_WAVE+127)/128),block(128);
    if(n==1) hipLaunchKernelGGL((dynamic_linear<true>),grid,block,0,stream,a,w,sw,bias,residual,y,int(m),int(n),int(k),int(k));
    else hipLaunchKernelGGL((dynamic_linear<false>),grid,block,0,stream,a,w,sw,bias,residual,y,int(m),int(n),int(k),int(k));
    return int32_t(hipGetLastError());
}
extern "C" int32_t chen_int8_linear_dynamic_packed_rows(const float* a,const int32_t* w,
    const float* sw,const float* bias,const float* residual,float* y,
    uint64_t m,uint64_t n,uint64_t source_k,uint64_t k,hipStream_t stream) {
    if(!dimensions(m,n,k)||!source_k||k!=((source_k+3)&~uint64_t(3))) return int32_t(hipErrorInvalidValue);
    if(!m) return int32_t(hipSuccess);
    if(!aligned(a)||!aligned(w)||!aligned(sw)||!aligned(y)||
       (bias&&!aligned(bias))||(residual&&!aligned(residual))) return int32_t(hipErrorInvalidValue);
    const dim3 grid((m*((n+(CHEN_WAVE-1))/CHEN_WAVE)*CHEN_WAVE+127)/128),block(128);
    if(n==1) hipLaunchKernelGGL((dynamic_linear<true>),grid,block,0,stream,a,w,sw,bias,residual,y,int(m),int(n),int(k),int(source_k));
    else hipLaunchKernelGGL((dynamic_linear<false>),grid,block,0,stream,a,w,sw,bias,residual,y,int(m),int(n),int(k),int(source_k));
    return int32_t(hipGetLastError());
}

// Experimental admission only; initializes the HIP runtime when called.
extern "C" int32_t chen_wave_resources(uint64_t kind,uint64_t block,uint64_t* out) {
    if(!out || !block || block>1024) return int32_t(hipErrorInvalidValue);
    const void* function=nullptr;
    switch(kind) { case 0: function = reinterpret_cast<const void*>(linear<true>); break;
case 1: function = reinterpret_cast<const void*>(dynamic_linear<false>); break;
case 2: function = reinterpret_cast<const void*>(scalar<true>); break;
case 3: function = reinterpret_cast<const void*>(dynamic_linear<true>); break; default: return int32_t(hipErrorInvalidValue); }
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
