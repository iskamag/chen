#pragma once
#include <cstdint>

#ifndef PEEL_WAVE_SIZE
#define PEEL_WAVE_SIZE 32
#endif
#if PEEL_WAVE_SIZE != 32 && PEEL_WAVE_SIZE != 64
#error "PEEL_WAVE_SIZE must be 32 or 64"
#endif

// Use explicit AMD ballots: HIP's runtime warpSize stays 32 on RDNA even
// when the experimental compiler switch forces wave64 code generation.
__device__ __forceinline__ uint64_t peel_ballot(bool predicate)
{
#if PEEL_WAVE_SIZE == 64
    return __builtin_amdgcn_ballot_w64(predicate);
#else
    return __builtin_amdgcn_ballot_w32(predicate);
#endif
}

struct PeelResult { uint64_t active[4], checksum; };

// The active mask is uniform within a wave: ballots return SGPR values.
// Preserve that property instead of rematerializing the bitmap in VGPRs.
__device__ __forceinline__ void peel_scalar_reset(uint32_t (&a)[8])
{
    #pragma unroll
    for (int w = 0; w < 8; ++w) a[w] = ~uint32_t(0);
    // Opaque scalar identity emits no instructions and keeps graph work
    // inside every benchmark repetition instead of permitting hoisting.
    asm volatile("" : "+s"(a[0]), "+s"(a[1]), "+s"(a[2]), "+s"(a[3]),
                      "+s"(a[4]), "+s"(a[5]), "+s"(a[6]), "+s"(a[7]));
}

// Starting from all ones, F(A) is a descending chain: F(top)<=top and
// monotonicity gives F(A_p)<=A_p inductively. Thus A & F(A) == F(A), so
// the old active mask need not be retained after computing all ballots.
// This shortcut is NOT valid for an arbitrary initial active mask.
template<int Passes, bool Capture = false>
__global__ void __launch_bounds__(PEEL_WAVE_SIZE)
live_set_scalar(const uint64_t * __restrict__ refs, PeelResult *out, int runs)
{
    constexpr int nodes = 256 / PEEL_WAVE_SIZE;
    uint32_t incoming[nodes][8];
    int lane = threadIdx.x;
    #pragma unroll
    for (int k = 0; k < nodes; ++k)
        #pragma unroll
        for (int w = 0; w < 4; ++w) {
            uint64_t row = refs[(lane + k * PEEL_WAVE_SIZE) * 4 + w];
            incoming[k][2*w] = uint32_t(row);
            incoming[k][2*w+1] = uint32_t(row >> 32);
        }
    uint32_t active[8];
    uint64_t checksum = 0;
    for (int run = 0; run < runs; ++run) {
        peel_scalar_reset(active);
        #pragma unroll
        for (int pass = 0; pass < Passes; ++pass) {
            uint32_t hit[nodes] = {};
            // Interleave independent dot products to hide the latency of
            // v_and_or chains; avoid a 64-bit comparison just to test !=0.
            #pragma unroll
            for (int w = 0; w < 8; ++w)
                #pragma unroll
                for (int k = 0; k < nodes; ++k)
                    hit[k] |= incoming[k][w] & active[w];
            uint32_t next[8];
            #pragma unroll
            for (int k = 0; k < nodes; ++k) {
                uint64_t mask = peel_ballot(hit[k] != 0);
#if PEEL_WAVE_SIZE == 64
                next[2*k] = uint32_t(mask); next[2*k+1] = uint32_t(mask >> 32);
#else
                next[k] = uint32_t(mask);
#endif
            }
            #pragma unroll
            for (int w = 0; w < 8; ++w) {
                active[w] = next[w];
            }
        }
        // Uniform scalar checksum avoids lane-0 divergence every repetition.
        uint32_t lo = active[0] ^ active[2] ^ active[4] ^ active[6] ^ uint32_t(run);
        uint32_t hi = active[1] ^ active[3] ^ active[5] ^ active[7];
        checksum ^= uint64_t(lo) | (uint64_t(hi) << 32);
        if (Capture && lane == 0) {
            #pragma unroll
            for (int w = 0; w < 4; ++w)
                out[run].active[w] = uint64_t(active[2*w]) | (uint64_t(active[2*w+1]) << 32);
            out[run].checksum = checksum;
        }
    }
    if (!Capture && lane == 0) {
        #pragma unroll
        for (int w = 0; w < 4; ++w)
            out->active[w] = uint64_t(active[2*w]) | (uint64_t(active[2*w+1]) << 32);
        out->checksum = checksum;
    }
}
