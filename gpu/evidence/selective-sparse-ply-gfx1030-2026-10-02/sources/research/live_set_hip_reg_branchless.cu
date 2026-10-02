/* live_set_hip_reg_branchless.cu
 * Explicit register assignment to force 32 VGPRs.
 * The masked passes (1..PASSES-1) use a branchless 0/-1 mask + v_and_or_b32
 * (always-issue LDS reads), instead of the divergent exec-masked branches
 * the compiler emits for `if (live & (1<<i))` in live_set_hip_reg.cu.
 */
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <ctime>

#include <hip/hip_runtime.h>

#define NUM_VECTORS 256
#define WORDS       4
#define PASSES      5
#define SET_BITS    8

#define HIP_CHECK(call) do { \
    hipError_t err = call; \
    if (err != hipSuccess) { \
        fprintf(stderr, "HIP error %s:%d: %s (%d)\n", \
                __FILE__, __LINE__, hipGetErrorString(err), err); \
        exit(1); \
    } \
} while (0)

static void build_graph(uint64_t *vectors)
{
    ::memset(vectors, 0, sizeof(uint64_t) * NUM_VECTORS * WORDS);
    srand(time(NULL));
    for (int i = 0; i < NUM_VECTORS; i++) {
        int range = i + 1; if (range > 256) range = 256;
        int max = SET_BITS < range ? SET_BITS : range;
        int set = 0;
        while (set < max) {
            int bit = rand() % range, w = bit >> 6, b = bit & 63;
            uint64_t mask = 1ULL << b;
            if (!(vectors[i * WORDS + w] & mask)) {
                vectors[i * WORDS + w] |= mask;
                set++;
            }
        }
    }
    int root = NUM_VECTORS - 1, rw = root >> 6, rb = root & 63;
    uint64_t rmask = 1ULL << rb;
    for (int i = 0; i < NUM_VECTORS; i++)
        vectors[i * WORDS + rw] |= rmask;
}

__global__ void __launch_bounds__(32, 16)
live_set_v_reg(
    const uint64_t * __restrict__ graph,
    uint64_t *results,
    int num_runs)
{
    __shared__ uint32_t lds[256 * 8];

    int lane = threadIdx.x;
    for (int i = lane; i < 256 * 8; i += 32)
        lds[i] = ((const uint32_t *)graph)[i];

    uint64_t checksum = 0;

    /* Explicit register variables to force compiler allocation */
    register uint32_t a0l asm("v4") = ~0u;
    register uint32_t a0h asm("v5") = ~0u;
    register uint32_t a1l asm("v6") = ~0u;
    register uint32_t a1h asm("v7") = ~0u;
    register uint32_t a2l asm("v8") = ~0u;
    register uint32_t a2h asm("v9") = ~0u;
    register uint32_t a3l asm("v10") = ~0u;
    register uint32_t a3h asm("v11") = ~0u;

    for (int run = 0; run < num_runs; run++) {
        register uint32_t r0l asm("v12") = 0;
        register uint32_t r0h asm("v13") = 0;
        register uint32_t r1l asm("v14") = 0;
        register uint32_t r1h asm("v15") = 0;
        register uint32_t r2l asm("v16") = 0;
        register uint32_t r2h asm("v17") = 0;
        register uint32_t r3l asm("v18") = 0;
        register uint32_t r3h asm("v19") = 0;

        #pragma unroll
        for (int i = 0; i < 8; i++) {
            int off = ((lane << 3) + i) << 3;
            r0l |= lds[off+0]; r0h |= lds[off+1];
            r1l |= lds[off+2]; r1h |= lds[off+3];
            r2l |= lds[off+4]; r2h |= lds[off+5];
            r3l |= lds[off+6]; r3h |= lds[off+7];
        }

        /* DPP reduction */
        asm volatile(
            "v_or_b32_dpp %0, %0, %0 row_shr:8 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %1, %1, %1 row_shr:8 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %2, %2, %2 row_shr:8 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %3, %3, %3 row_shr:8 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %4, %4, %4 row_shr:8 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %5, %5, %5 row_shr:8 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %6, %6, %6 row_shr:8 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %7, %7, %7 row_shr:8 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %0, %0, %0 row_shr:4 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %1, %1, %1 row_shr:4 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %2, %2, %2 row_shr:4 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %3, %3, %3 row_shr:4 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %4, %4, %4 row_shr:4 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %5, %5, %5 row_shr:4 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %6, %6, %6 row_shr:4 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %7, %7, %7 row_shr:4 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %0, %0, %0 row_shr:2 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %1, %1, %1 row_shr:2 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %2, %2, %2 row_shr:2 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %3, %3, %3 row_shr:2 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %4, %4, %4 row_shr:2 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %5, %5, %5 row_shr:2 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %6, %6, %6 row_shr:2 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %7, %7, %7 row_shr:2 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %0, %0, %0 row_shr:1 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %1, %1, %1 row_shr:1 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %2, %2, %2 row_shr:1 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %3, %3, %3 row_shr:1 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %4, %4, %4 row_shr:1 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %5, %5, %5 row_shr:1 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %6, %6, %6 row_shr:1 row_mask:0xf bank_mask:0xf\n"
            "v_or_b32_dpp %7, %7, %7 row_shr:1 row_mask:0xf bank_mask:0xf\n"
            : "+v"(r0l), "+v"(r0h), "+v"(r1l), "+v"(r1h),
              "+v"(r2l), "+v"(r2h), "+v"(r3l), "+v"(r3h)
            :
            :);

        int word = lane >> 3;
        uint32_t peer = (lane == 15) ? 31*4 : (lane == 31) ? 15*4 : 0;
        uint32_t src  = (lane < 16) ? 15*4 : 31*4;
        uint32_t t;

        if (lane == 15 || lane == 31) {
            t = __builtin_amdgcn_ds_bpermute(peer, r0l); r0l |= t;
            t = __builtin_amdgcn_ds_bpermute(peer, r0h); r0h |= t;
            t = __builtin_amdgcn_ds_bpermute(peer, r1l); r1l |= t;
            t = __builtin_amdgcn_ds_bpermute(peer, r1h); r1h |= t;
            t = __builtin_amdgcn_ds_bpermute(peer, r2l); r2l |= t;
            t = __builtin_amdgcn_ds_bpermute(peer, r2h); r2h |= t;
            t = __builtin_amdgcn_ds_bpermute(peer, r3l); r3l |= t;
            t = __builtin_amdgcn_ds_bpermute(peer, r3h); r3h |= t;
        }
        t = __builtin_amdgcn_ds_bpermute(src, r0l); r0l |= t;
        t = __builtin_amdgcn_ds_bpermute(src, r0h); r0h |= t;
        t = __builtin_amdgcn_ds_bpermute(src, r1l); r1l |= t;
        t = __builtin_amdgcn_ds_bpermute(src, r1h); r1h |= t;
        t = __builtin_amdgcn_ds_bpermute(src, r2l); r2l |= t;
        t = __builtin_amdgcn_ds_bpermute(src, r2h); r2h |= t;
        t = __builtin_amdgcn_ds_bpermute(src, r3l); r3l |= t;
        t = __builtin_amdgcn_ds_bpermute(src, r3h); r3h |= t;

        a0l = r0l; a0h = r0h; a1l = r1l; a1h = r1h;
        a2l = r2l; a2h = r2h; a3l = r3l; a3h = r3h;

        for (int p = 1; p < PASSES; p++) {
            r0l = 0; r0h = 0; r1l = 0; r1h = 0;
            r2l = 0; r2h = 0; r3l = 0; r3h = 0;

            uint32_t mlo = (word == 0) ? a0l : (word == 1) ? a1l :
                           (word == 2) ? a2l : a3l;
            uint32_t mhi = (word == 0) ? a0h : (word == 1) ? a1h :
                           (word == 2) ? a2h : a3h;
            uint64_t mask = ((uint64_t)mhi << 32) | mlo;
            uint64_t live = mask >> ((lane << 3) & 63);

            #pragma unroll
            for (int i = 0; i < 8; i++) {
                /* branchless: always issue the LDS reads, conditionally OR via
                   a 0/-1 mask.  Lowers to ds_read + v_and_or_b32, mirroring the
                   hand-written ASM kernel (no exec-masked divergent branch). */
                uint32_t m = (uint32_t)(0u - ((live >> i) & 1ULL));
                int off = ((lane << 3) + i) << 3;
                r0l |= m & lds[off+0]; r0h |= m & lds[off+1];
                r1l |= m & lds[off+2]; r1h |= m & lds[off+3];
                r2l |= m & lds[off+4]; r2h |= m & lds[off+5];
                r3l |= m & lds[off+6]; r3h |= m & lds[off+7];
            }

            asm volatile(
                "v_or_b32_dpp %0, %0, %0 row_shr:8 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %1, %1, %1 row_shr:8 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %2, %2, %2 row_shr:8 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %3, %3, %3 row_shr:8 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %4, %4, %4 row_shr:8 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %5, %5, %5 row_shr:8 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %6, %6, %6 row_shr:8 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %7, %7, %7 row_shr:8 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %0, %0, %0 row_shr:4 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %1, %1, %1 row_shr:4 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %2, %2, %2 row_shr:4 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %3, %3, %3 row_shr:4 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %4, %4, %4 row_shr:4 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %5, %5, %5 row_shr:4 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %6, %6, %6 row_shr:4 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %7, %7, %7 row_shr:4 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %0, %0, %0 row_shr:2 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %1, %1, %1 row_shr:2 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %2, %2, %2 row_shr:2 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %3, %3, %3 row_shr:2 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %4, %4, %4 row_shr:2 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %5, %5, %5 row_shr:2 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %6, %6, %6 row_shr:2 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %7, %7, %7 row_shr:2 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %0, %0, %0 row_shr:1 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %1, %1, %1 row_shr:1 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %2, %2, %2 row_shr:1 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %3, %3, %3 row_shr:1 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %4, %4, %4 row_shr:1 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %5, %5, %5 row_shr:1 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %6, %6, %6 row_shr:1 row_mask:0xf bank_mask:0xf\n"
                "v_or_b32_dpp %7, %7, %7 row_shr:1 row_mask:0xf bank_mask:0xf\n"
                : "+v"(r0l), "+v"(r0h), "+v"(r1l), "+v"(r1h),
                  "+v"(r2l), "+v"(r2h), "+v"(r3l), "+v"(r3h)
                :
                :);

            if (lane == 15 || lane == 31) {
                t = __builtin_amdgcn_ds_bpermute(peer, r0l); r0l |= t;
                t = __builtin_amdgcn_ds_bpermute(peer, r0h); r0h |= t;
                t = __builtin_amdgcn_ds_bpermute(peer, r1l); r1l |= t;
                t = __builtin_amdgcn_ds_bpermute(peer, r1h); r1h |= t;
                t = __builtin_amdgcn_ds_bpermute(peer, r2l); r2l |= t;
                t = __builtin_amdgcn_ds_bpermute(peer, r2h); r2h |= t;
                t = __builtin_amdgcn_ds_bpermute(peer, r3l); r3l |= t;
                t = __builtin_amdgcn_ds_bpermute(peer, r3h); r3h |= t;
            }
            t = __builtin_amdgcn_ds_bpermute(src, r0l); r0l |= t;
            t = __builtin_amdgcn_ds_bpermute(src, r0h); r0h |= t;
            t = __builtin_amdgcn_ds_bpermute(src, r1l); r1l |= t;
            t = __builtin_amdgcn_ds_bpermute(src, r1h); r1h |= t;
            t = __builtin_amdgcn_ds_bpermute(src, r2l); r2l |= t;
            t = __builtin_amdgcn_ds_bpermute(src, r2h); r2h |= t;
            t = __builtin_amdgcn_ds_bpermute(src, r3l); r3l |= t;
            t = __builtin_amdgcn_ds_bpermute(src, r3h); r3h |= t;

            a0l &= r0l; a0h &= r0h; a1l &= r1l; a1h &= r1h;
            a2l &= r2l; a2h &= r2h; a3l &= r3l; a3h &= r3h;
        }

        if (lane == 0)
            checksum ^= (((uint64_t)a0h<<32)|a0l) ^ (((uint64_t)a1h<<32)|a1l) ^
                        (((uint64_t)a2h<<32)|a2l) ^ (((uint64_t)a3h<<32)|a3l) ^ (uint64_t)run;
    }

    if (lane == 0)
        results[0] = checksum;
}

int main(int argc, char **argv)
{
    hipDeviceProp_t prop;
    HIP_CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s, CU: %d, Wavefront: %d\n\n", prop.name, prop.multiProcessorCount, prop.warpSize);

    uint64_t h_vectors[NUM_VECTORS * WORDS];
    build_graph(h_vectors);

    uint64_t *d_data, *d_out;
    HIP_CHECK(hipMalloc(&d_data, NUM_VECTORS * WORDS * sizeof(uint64_t)));
    HIP_CHECK(hipMalloc(&d_out, sizeof(uint64_t)));
    HIP_CHECK(hipMemcpy(d_data, h_vectors, NUM_VECTORS * WORDS * sizeof(uint64_t), hipMemcpyHostToDevice));

    printf("=== explicit register assignment (branchless masked passes) ===\n");
    printf("CPU AVX2 baseline: ~448 ns/run (89.6 ns/pass)\n\n");
    printf("%-12s %12s %12s %12s\n", "runs", "time(ms)", "ns/run", "ns/pass");

    for (int n = 1000; n <= 128000; n *= 4) {
        hipEvent_t start, stop;
        HIP_CHECK(hipEventCreate(&start));
        HIP_CHECK(hipEventCreate(&stop));
        HIP_CHECK(hipEventRecord(start));

        live_set_v_reg<<<1, 32>>>(d_data, d_out, n);

        HIP_CHECK(hipEventRecord(stop));
        HIP_CHECK(hipEventSynchronize(stop));

        float ms;
        HIP_CHECK(hipEventElapsedTime(&ms, start, stop));

        double ns_per = (ms * 1e6) / n;
        printf("%-12d %12.3f %12.1f %12.1f\n", n, ms, ns_per, ns_per / PASSES);

        HIP_CHECK(hipEventDestroy(start));
        HIP_CHECK(hipEventDestroy(stop));
    }

    HIP_CHECK(hipFree(d_data));
    HIP_CHECK(hipFree(d_out));
    return 0;
}
