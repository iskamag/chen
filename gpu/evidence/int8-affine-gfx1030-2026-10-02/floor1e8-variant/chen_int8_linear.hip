// gfx1030 mixed INT8 linear operator. No allocations, synchronization, ownership
// transfers, or hidden default-stream work. Python owns tensors and stream.
#include <hip/hip_runtime.h>
#include <cstdint>

namespace {
__device__ __forceinline__ int dot4(int a,int b,int acc) {
    return __builtin_amdgcn_sdot4(a,b,acc,false);
}
template<bool Packed>
__global__ void linear(const int32_t* __restrict__ a,const int32_t* __restrict__ w,
                      const float* sa,const float* sw,const float* bias,
                      const float* residual,float* y,int m,int n,int chunks) {
    const uint64_t wave=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const int lane=threadIdx.x&31,tiles=(n+31)/32;
    const int row=__builtin_amdgcn_readfirstlane(int(wave/tiles));
    if(row>=m) return; // Full wave returns, never a partial-wave broadcast.
    const int column=int(wave%tiles)*32+lane,npad=tiles*32;
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
// wasting 31 lanes per output. All lanes participate in the integer reduction.
template<bool Packed>
__global__ void scalar(const int32_t* a,const int32_t* w,const float* sa,
                       const float* sw,const float* bias,const float* residual,
                       float* y,int m,int chunks) {
    const uint64_t row=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const int lane=threadIdx.x&31;
    if(row>=uint64_t(m)) return;
    int sum=0;
    for(int k=lane;k<chunks;k+=32) sum=dot4(a[row*chunks+k],w[Packed?uint64_t(k)*32:k],sum);
    #pragma unroll
    for(int step=16;step>0;step/=2) sum+=__shfl_down(sum,step,32);
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
__device__ __forceinline__ int quant_word(const float* a,float scale) {
    uint32_t word=0;
    #pragma unroll
    for(int byte=0;byte<4;++byte) {
        const int q=int(nearbyintf(fminf(127.f,fmaxf(-127.f,a[byte]/scale))));
        word|=(uint32_t(q)&255)<<(byte*8);
    }
    return int(word);
}
// Dynamic row quantization + dot + dequantization in one dispatch. This saves
// absmax/scale/round/cast/pad tensor launches for the narrow semantic layers.
// The caller provides zero K padding. Residual and bias remain FP32.
template<bool Scalar>
__global__ void dynamic_linear(const float* a,const int32_t* w,const float* sw,
    const float* bias,const float* residual,float* y,int m,int n,int k) {
    const uint64_t wave=(uint64_t(blockIdx.x)*blockDim.x+threadIdx.x)/32;
    const int tiles=(n+31)/32,lane=threadIdx.x&31;
    const int row=__builtin_amdgcn_readfirstlane(int(wave/tiles));
    if(row>=m) return;
    const int column=int(wave%tiles)*32+lane,npad=tiles*32,chunks=k/4;
    const float* inputs=a+uint64_t(row)*k;
    float maximum=0;
    for(int i=lane;i<k;i+=32) maximum=fmaxf(maximum,fabsf(inputs[i]));
    #pragma unroll
    for(int step=16;step>0;step/=2) maximum=fmaxf(maximum,__shfl_xor(maximum,step,32));
    // Torch's division by a CPU scalar first rounds its reciprocal to FP32.
    // Keep that ordering explicit: true division changes halfway byte rounding.
    const float scale=fmaxf(maximum,1e-8f)*(1.f/127.f);
    int sums[4]={0,0,0,0};
    if(Scalar) {
        for(int chunk=lane;chunk<chunks;chunk+=32)
            sums[0]=dot4(quant_word(inputs+4*chunk,scale),w[uint64_t(chunk)*npad],sums[0]);
    } else for(int start=0;start<chunks;start+=4) {
        #pragma unroll
        for(int j=0;j<4;++j) if(start+j<chunks) {
            const int chunk=start+j;
            const int av=__builtin_amdgcn_readfirstlane(quant_word(inputs+4*chunk,scale));
            sums[j]=dot4(av,w[uint64_t(chunk)*npad+column],sums[j]);
        }
    }
    int sum=(sums[0]+sums[1])+(sums[2]+sums[3]);
    if(Scalar) {
        #pragma unroll
        for(int step=16;step>0;step/=2) sum+=__shfl_down(sum,step,32);
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
        hipLaunchKernelGGL((scalar<Packed>),dim3((m*32+127)/128),block,0,stream,
                          words,weights,sa,sw,bias,residual,y,int(m),int(k/4));
    } else {
        hipLaunchKernelGGL((linear<Packed>),dim3((m*((n+31)/32)*32+127)/128),block,0,stream,
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
    const int npad=int((n+31)&~uint64_t(31));
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
    const dim3 grid((m*((n+31)/32)*32+127)/128),block(128);
    if(n==1) hipLaunchKernelGGL((dynamic_linear<true>),grid,block,0,stream,a,w,sw,bias,residual,y,int(m),int(n),int(k));
    else hipLaunchKernelGGL((dynamic_linear<false>),grid,block,0,stream,a,w,sw,bias,residual,y,int(m),int(n),int(k));
    return int32_t(hipGetLastError());
}
