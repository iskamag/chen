// Isolated RDNA2 experiment, deliberately independent of the production ABI.
#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
struct Input { uint64_t bits; uint32_t square, reserved; };
static_assert(sizeof(Input)==16);
#define HIP_OK(x) do { hipError_t e=(x); if(e!=hipSuccess) { fprintf(stderr,"%s: %s\n",#x,hipGetErrorString(e)); exit(2); } } while(0)
__device__ __forceinline__ uint64_t transpose_swar(uint64_t x) {
    uint64_t t=(x^(x>>7))&0x00aa00aa00aa00aaull; x^=t^(t<<7);
    t=(x^(x>>14))&0x0000cccc0000ccccull; x^=t^(t<<14);
    t=(x^(x>>28))&0x00000000f0f0f0f0ull; return x^t^(t<<28);
}
__device__ __forceinline__ uint64_t transpose_loop(uint64_t x) {
    uint64_t y=0;
    #pragma unroll
    for(int s=0;s<64;++s) y|=((x>>s)&1ull)<<((s%8)*8+s/8);
    return y;
}
__device__ __forceinline__ uint32_t pop_swar(uint64_t x) {
    x-=(x>>1)&0x5555555555555555ull;
    x=(x&0x3333333333333333ull)+((x>>2)&0x3333333333333333ull);
    x=(x+(x>>4))&0x0f0f0f0f0f0f0f0full;
    x+=x>>8; x+=x>>16; x+=x>>32; return uint32_t(x)&127;
}
template<int Shift, bool Left> __device__ __forceinline__ uint64_t fill(uint64_t g,uint64_t p) {
    if(Left) { g|=p&(g<<Shift); p&=p<<Shift; g|=p&(g<<(2*Shift)); p&=p<<(2*Shift); g|=p&(g<<(4*Shift)); return g<<Shift; }
    else { g|=p&(g>>Shift); p&=p>>Shift; g|=p&(g>>(2*Shift)); p&=p>>(2*Shift); g|=p&(g>>(4*Shift)); return g>>Shift; }
}
__device__ __forceinline__ uint64_t rook_fill(uint64_t x,int s) {
    uint64_t g=1ull<<s, empty=~x;
    return (fill<1,true>(g,empty&0xfefefefefefefefeull)&0xfefefefefefefefeull)
         | (fill<1,false>(g,empty&0x7f7f7f7f7f7f7f7full)&0x7f7f7f7f7f7f7f7full)
         | fill<8,true>(g,empty) | fill<8,false>(g,empty);
}
__device__ __forceinline__ uint64_t rook_walk(uint64_t x,int s) {
    uint64_t a=0;
    for(int d=0;d<4;++d) {
        int df=d==0?1:d==1?-1:0, dr=d==2?1:d==3?-1:0;
        int f=s%8+df,r=s/8+dr;
        while(f>=0&&f<8&&r>=0&&r<8) { uint64_t b=1ull<<(r*8+f); a|=b; if(x&b) break; f+=df;r+=dr; }
    }
    return a;
}
// All variants use this recurrence; operation result is XOR-reduced into output.
// Every iteration changes input, preventing immutable-input hoisting.
__device__ __forceinline__ uint64_t next_bits(uint64_t x,int r) {
    return ((x<<9)|(x>>55))^(0x9e3779b97f4a7c15ull+uint32_t(r));
}
template<int Mode> __global__ void independent(const Input* in,uint64_t* out,int n,int rounds) {
    int i=blockIdx.x*blockDim.x+threadIdx.x; if(i>=n) return;
    uint64_t x=in[i].bits,y=0; int s=in[i].square;
    for(int r=0;r<rounds;++r) {
        uint64_t v;
        if constexpr(Mode==0) v=transpose_loop(x);
        if constexpr(Mode==1) v=transpose_swar(x);
        if constexpr(Mode==2) v=__popcll(x);
        if constexpr(Mode==3) v=__popc(uint32_t(x))+__popc(uint32_t(x>>32));
        if constexpr(Mode==4) v=pop_swar(x);
        if constexpr(Mode==5) v=rook_walk(x,s);
        if constexpr(Mode==6) v=rook_fill(x,s);
        y^=v; x=next_bits(x,r); s=(s+13)&63;
    }
    out[i]=y;
}
// One wave32 per board. Each lane owns two destination squares. Masks use
// cached between-square tables, independent of occupancy and origin square.
__global__ void cooperative(const Input* in,uint64_t* out,const uint64_t* between,int n,int rounds) {
    int i=(blockIdx.x*blockDim.x+threadIdx.x)/32,lane=threadIdx.x&31;
    if(i>=n) return; // Entire wave participates or returns.
    uint64_t raw=in[i].bits;
    // The intrinsic returns signed int even for an unsigned argument. Preserve
    // each 32-bit pattern before widening, otherwise low bit31 sign-extends.
    uint64_t x=uint64_t(uint32_t(__builtin_amdgcn_readfirstlane(uint32_t(raw)))) |
               (uint64_t(uint32_t(__builtin_amdgcn_readfirstlane(uint32_t(raw>>32))))<<32);
    int s=__builtin_amdgcn_readfirstlane(in[i].square); uint64_t y=0;
    for(int r=0;r<rounds;++r) {
        int t0=lane,t1=lane+32;
        bool a0=t0!=s&&(t0/8==s/8||t0%8==s%8)&&!(between[s*64+t0]&x);
        bool a1=t1!=s&&(t1/8==s/8||t1%8==s%8)&&!(between[s*64+t1]&x);
        uint32_t lo=__builtin_amdgcn_ballot_w32(a0),hi=__builtin_amdgcn_ballot_w32(a1);
        y^=uint64_t(lo)|(uint64_t(hi)<<32); x=next_bits(x,r); s=(s+13)&63;
    }
    if(lane==0) out[i]=y;
}
static void launch(int mode,const Input* in,uint64_t* out,const uint64_t* between,int n,int rounds) {
    dim3 block(128),grid((n+127)/128);
    switch(mode) {
    case 0: hipLaunchKernelGGL(independent<0>,grid,block,0,0,in,out,n,rounds); break;
    case 1: hipLaunchKernelGGL(independent<1>,grid,block,0,0,in,out,n,rounds); break;
    case 2: hipLaunchKernelGGL(independent<2>,grid,block,0,0,in,out,n,rounds); break;
    case 3: hipLaunchKernelGGL(independent<3>,grid,block,0,0,in,out,n,rounds); break;
    case 4: hipLaunchKernelGGL(independent<4>,grid,block,0,0,in,out,n,rounds); break;
    case 5: hipLaunchKernelGGL(independent<5>,grid,block,0,0,in,out,n,rounds); break;
    case 6: hipLaunchKernelGGL(independent<6>,grid,block,0,0,in,out,n,rounds); break;
    case 7: grid.x=(n*32+127)/128; hipLaunchKernelGGL(cooperative,grid,block,0,0,in,out,between,n,rounds); break;
    default: exit(3);
    }
    HIP_OK(hipGetLastError());
}
int main(int argc,char** argv) {
    if(argc!=6) { fprintf(stderr,"usage: microbench INPUT OUTPUT_PREFIX N ROUNDS REPEATS\n"); return 1; }
    int n=atoi(argv[3]),rounds=atoi(argv[4]),repeats=atoi(argv[5]);
    if(n<1||n>1048576||rounds<1||rounds>4096||repeats<1||repeats>101) return 1;
    std::vector<Input> inputs(n); FILE* f=fopen(argv[1],"rb"); if(!f) return 1;
    bool ok=fread(inputs.data(),sizeof(Input),n,f)==size_t(n)&&fgetc(f)==EOF; fclose(f); if(!ok) return 1;
    for(auto& x:inputs) if(x.square>63||x.reserved) return 1;
    hipDeviceProp_t p; HIP_OK(hipGetDeviceProperties(&p,0));
    if(strncmp(p.gcnArchName,"gfx1030",7)||p.warpSize!=32) return 4;
    printf("{\"device\":\"%s\",\"arch\":\"%s\",\"warp_size\":%d,\"cu\":%d,\"n\":%d,\"rounds\":%d,\"warmups\":3,\"samples\":[",p.name,p.gcnArchName,p.warpSize,p.multiProcessorCount,n,rounds);
    Input* di; uint64_t *dout,*db; HIP_OK(hipMalloc(&di,n*sizeof(Input))); HIP_OK(hipMalloc(&dout,n*8)); HIP_OK(hipMalloc(&db,4096*8));
    std::vector<uint64_t> b(4096,0),outputs(n);
    for(int s=0;s<64;++s) for(int t=0;t<64;++t) {
        if(s==t||!(s/8==t/8||s%8==t%8)) continue;
        int delta=s/8==t/8?(t>s?1:-1):(t>s?8:-8);
        for(int q=s+delta;q!=t;q+=delta) b[s*64+t]|=1ull<<q;
    }
    HIP_OK(hipMemcpy(di,inputs.data(),n*sizeof(Input),hipMemcpyHostToDevice)); HIP_OK(hipMemcpy(db,b.data(),b.size()*8,hipMemcpyHostToDevice));
    hipEvent_t start,stop; HIP_OK(hipEventCreate(&start)); HIP_OK(hipEventCreate(&stop));
    std::vector<std::vector<float>> times(8);
    for(int m=0;m<8;++m) for(int w=0;w<3;++w) launch(m,di,dout,db,n,rounds);
    HIP_OK(hipDeviceSynchronize());
    // Reverse order on alternating samples rather than benchmarking modes in long blocks.
    for(int rep=0;rep<repeats;++rep) for(int k=0;k<8;++k) {
        int m=rep%2?7-k:k; HIP_OK(hipEventRecord(start)); launch(m,di,dout,db,n,rounds); HIP_OK(hipEventRecord(stop)); HIP_OK(hipEventSynchronize(stop));
        float ms; HIP_OK(hipEventElapsedTime(&ms,start,stop)); times[m].push_back(ms);
    }
    for(int m=0;m<8;++m) {
        launch(m,di,dout,db,n,rounds); HIP_OK(hipMemcpy(outputs.data(),dout,n*8,hipMemcpyDeviceToHost));
        char name[4096]; if(snprintf(name,sizeof(name),"%s.%d.bin",argv[2],m)>=int(sizeof(name))) return 1;
        f=fopen(name,"wb"); if(!f) return 1; if(fwrite(outputs.data(),8,n,f)!=size_t(n)||fclose(f)) return 1;
        printf("%s{\"mode\":%d,\"milliseconds\":[",m?",":"",m);
        for(int rep=0;rep<repeats;++rep) printf("%s%.9g",rep?",":"",times[m][rep]); printf("]}");
    }
    printf("]}\n"); HIP_OK(hipEventDestroy(start)); HIP_OK(hipEventDestroy(stop)); HIP_OK(hipFree(di)); HIP_OK(hipFree(dout)); HIP_OK(hipFree(db));
}
