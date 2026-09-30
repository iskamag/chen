#include "chen_rules.h"
#include <hip/hip_runtime.h>

namespace {
__global__ void decode(const ChenRulePosition *positions, uint64_t count,
    float *features) {
    const uint64_t index=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(index>=count*18*64) return;
    const uint64_t row=index/(18*64);
    const unsigned plane=(index/64)%18, square=index%64;
    const ChenRulePosition p=positions[row];
    const unsigned raw=square^(p.side?56:0);
    float value=0;
    if(plane<12) {
        const unsigned color=plane<6?p.side:1-p.side;
        value=float((p.boards[color*6+plane%6]>>raw)&1);
    } else if(plane<16) {
        const unsigned right=(plane-12)^(p.side?2:0);
        value=float((p.castling>>right)&1);
    } else if(plane==16) {
        value=float(p.ep_square>=0 && unsigned(p.ep_square)==raw);
    } else {
        // Python's canonical encoder divides in double before float32 storage.
        value=float(p.halfmove>=100?1.0:double(p.halfmove)/100.0);
    }
    features[index]=value;
}
}
extern "C" int32_t chen_neural_decode(const ChenRulePosition *positions,
    uint64_t count,float *features,hipStream_t stream) {
    if(count) hipLaunchKernelGGL(decode,dim3((count*18*64+255)/256),dim3(256),
        0,stream,positions,count,features);
    return int32_t(hipGetLastError());
}
