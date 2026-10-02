#include "chen_gpu.h"
#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>

struct ChenGpuContext {
    int device = 0;
    uint64_t capacity = 0;
    ChenPosition *positions = nullptr, *staging_positions = nullptr;
    int32_t *scores = nullptr, *staging_scores = nullptr;
    hipStream_t stream = nullptr;
    hipEvent_t events[4] = {};
    ChenDeviceInfo info = {};
};

static int32_t fail(char *error, uint64_t capacity, const char *message) {
    if (error && capacity) std::snprintf(error, size_t(capacity), "%s", message);
    return -1;
}
static int32_t hip_fail(char *error, uint64_t capacity, hipError_t code) {
    return fail(error, capacity, hipGetErrorString(code));
}
static void clear_error(char *error, uint64_t capacity) {
    if (error && capacity) *error = '\0';
}

__device__ static int evaluate_position(const ChenPosition &p, uint32_t evaluator) {
    constexpr int values[6] = {100, 320, 330, 500, 900, 0};
    constexpr int phases[6] = {0, 1, 1, 2, 4, 0};
    int mg = 0, eg = 0, phase = 0;
    for (int color = 0; color < 2; ++color) {
        const int sign = color ? -1 : 1;
        for (int piece = 0; piece < 6; ++piece) {
            uint64_t bb = p.pieces[color * 6 + piece];
            const int count = __popcll(bb);
            if (evaluator == 0) { mg += sign * values[piece] * count; continue; }
            phase += phases[piece] * count;
            while (bb) {
                const int sq = __ffsll(static_cast<unsigned long long>(bb)) - 1;
                bb &= bb - 1;
                const int file = sq % 8, rank = color ? 7 - sq / 8 : sq / 8;
                const int fx = 2 * file - 7, ry = 2 * rank - 7;
                const int center = 14 - (fx < 0 ? -fx : fx) - (ry < 0 ? -ry : ry);
                int m = 0, e = 0;
                switch (piece) {
                    case 0: m = rank * 8 + center * 2; e = rank * 14 + center; break;
                    case 1: m = center * 6; e = center * 4; break;
                    case 2: m = center * 3 + rank * 2; e = center * 3; break;
                    case 3: m = rank * 2 + (rank == 6 ? 18 : 0); e = center; break;
                    case 4: m = center; e = center * 2; break;
                    case 5: m = -center * 5 - rank * 4 +
                                    ((rank == 0 && (file == 2 || file == 6)) ? 35 : 0);
                            e = center * 6; break;
                }
                mg += sign * (values[piece] + m);
                eg += sign * (values[piece] + e);
            }
        }
        if (evaluator == 1 && __popcll(p.pieces[color * 6 + 2]) >= 2) {
            mg += sign * 25; eg += sign * 40;
        }
    }
    if (phase > 24) phase = 24;
    const int score = evaluator == 0 ? mg : (mg * phase + eg * (24 - phase)) / 24;
    return p.side ? -score : score;
}

__global__ void evaluate_kernel(const ChenPosition *positions, int32_t *scores,
                                uint64_t count, uint32_t evaluator) {
    const uint64_t index = uint64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (index < count) scores[index] = evaluate_position(positions[index], evaluator);
}

extern "C" uint32_t chen_gpu_abi_version() { return 1; }
extern "C" void chen_gpu_destroy(ChenGpuContext *ctx) {
    if (!ctx) return;
    (void)hipSetDevice(ctx->device);
    if (ctx->stream) (void)hipStreamSynchronize(ctx->stream);
    for (auto event : ctx->events) if (event) (void)hipEventDestroy(event);
    if (ctx->positions) (void)hipFree(ctx->positions);
    if (ctx->scores) (void)hipFree(ctx->scores);
    if (ctx->staging_positions) (void)hipHostFree(ctx->staging_positions);
    if (ctx->staging_scores) (void)hipHostFree(ctx->staging_scores);
    if (ctx->stream) (void)hipStreamDestroy(ctx->stream);
    delete ctx;
}

extern "C" ChenGpuContext *chen_gpu_create(int32_t device, uint64_t capacity,
                                           char *error, uint64_t error_capacity) {
    clear_error(error, error_capacity);
    if (std::getenv("HSA_OVERRIDE_GFX_VERSION")) {
        fail(error, error_capacity, "HSA_OVERRIDE_GFX_VERSION is forbidden: verify the actual GPU architecture");
        return nullptr;
    }
    if (!capacity || capacity > std::numeric_limits<size_t>::max() / sizeof(ChenPosition) ||
        capacity > uint64_t(std::numeric_limits<uint32_t>::max()) * 256) {
        fail(error, error_capacity, "invalid capacity"); return nullptr;
    }
    auto *ctx = new (std::nothrow) ChenGpuContext;
    if (!ctx) { fail(error, error_capacity, "context allocation failed"); return nullptr; }
    ctx->device = device; ctx->capacity = capacity;
    hipDeviceProp_t prop{};
    hipError_t code = hipSetDevice(device);
    if (code == hipSuccess) code = hipGetDeviceProperties(&prop, device);
    if (code != hipSuccess) {
        hip_fail(error, error_capacity, code); chen_gpu_destroy(ctx); return nullptr;
    }
    if (std::strncmp(prop.gcnArchName, "gfx1030", 7) != 0 ||
        (prop.gcnArchName[7] != '\0' && prop.gcnArchName[7] != ':')) {
        fail(error, error_capacity, "this build requires an actual gfx1030 device");
        chen_gpu_destroy(ctx); return nullptr;
    }
    std::snprintf(ctx->info.name, sizeof(ctx->info.name), "%s", prop.name);
    std::snprintf(ctx->info.architecture, sizeof(ctx->info.architecture), "%s", prop.gcnArchName);
    ctx->info.total_memory = prop.totalGlobalMem;
    ctx->info.compute_units = prop.multiProcessorCount;
    ctx->info.wave_size = prop.warpSize;
    ctx->info.async_engines = prop.asyncEngineCount;
    int runtime = 0;
    code = hipRuntimeGetVersion(&runtime); ctx->info.runtime_version = uint32_t(runtime);
    if (code == hipSuccess) code = hipStreamCreateWithFlags(&ctx->stream, hipStreamNonBlocking);
    for (int i = 0; i < 4 && code == hipSuccess; ++i) code = hipEventCreate(&ctx->events[i]);
    if (code == hipSuccess) code = hipMalloc(&ctx->positions, capacity * sizeof(ChenPosition));
    if (code == hipSuccess) code = hipMalloc(&ctx->scores, capacity * sizeof(int32_t));
    if (code == hipSuccess) code = hipHostMalloc(&ctx->staging_positions, capacity * sizeof(ChenPosition));
    if (code == hipSuccess) code = hipHostMalloc(&ctx->staging_scores, capacity * sizeof(int32_t));
    if (code != hipSuccess) {
        hip_fail(error, error_capacity, code); chen_gpu_destroy(ctx); return nullptr;
    }
    return ctx;
}

extern "C" int32_t chen_gpu_device_info(ChenGpuContext *ctx, ChenDeviceInfo *out,
                                        char *error, uint64_t error_capacity) {
    clear_error(error, error_capacity);
    if (!ctx || !out) return fail(error, error_capacity, "null context/device info");
    const hipError_t code = hipSetDevice(ctx->device);
    if (code != hipSuccess) return hip_fail(error, error_capacity, code);
    *out = ctx->info; return 0;
}

extern "C" int32_t chen_gpu_evaluate(ChenGpuContext *ctx, const ChenPosition *positions,
                                     int32_t *scores, uint64_t count, uint32_t evaluator,
                                     ChenTimings *timings, char *error, uint64_t error_capacity) {
    clear_error(error, error_capacity);
    if (!ctx || !timings) return fail(error, error_capacity, "null context/timings");
    *timings = {0., 0., 0., count, ctx->capacity};
    if (count > ctx->capacity) return fail(error, error_capacity, "batch exceeds context capacity");
    if (evaluator > 1) return fail(error, error_capacity, "unknown evaluator");
    if (!count) return 0;
    if (!positions || !scores) return fail(error, error_capacity, "null position/score buffer");
    for (uint64_t i = 0; i < count; ++i) {
        if (positions[i].side > 1 || positions[i].reserved)
            return fail(error, error_capacity, "invalid position side/reserved field");
    }
    std::memcpy(ctx->staging_positions, positions, count * sizeof(ChenPosition));
    hipError_t code = hipSetDevice(ctx->device);
    if (code != hipSuccess) return hip_fail(error, error_capacity, code);
    if (code == hipSuccess) code = hipEventRecord(ctx->events[0], ctx->stream);
    if (code == hipSuccess) code = hipMemcpyAsync(ctx->positions, ctx->staging_positions,
        count * sizeof(ChenPosition), hipMemcpyHostToDevice, ctx->stream);
    if (code == hipSuccess) code = hipEventRecord(ctx->events[1], ctx->stream);
    if (code == hipSuccess) {
        hipLaunchKernelGGL(evaluate_kernel, dim3(uint32_t((count + 255) / 256)), dim3(256),
                          0, ctx->stream, ctx->positions, ctx->scores, count, evaluator);
        code = hipGetLastError();
    }
    if (code == hipSuccess) code = hipEventRecord(ctx->events[2], ctx->stream);
    if (code == hipSuccess) code = hipMemcpyAsync(ctx->staging_scores, ctx->scores,
        count * sizeof(int32_t), hipMemcpyDeviceToHost, ctx->stream);
    if (code == hipSuccess) code = hipEventRecord(ctx->events[3], ctx->stream);
    // Even error paths drain the stream before caller buffers/context are reused.
    const hipError_t sync = hipStreamSynchronize(ctx->stream);
    if (code == hipSuccess) code = sync;
    float times[3]{};
    for (int i = 0; i < 3 && code == hipSuccess; ++i)
        code = hipEventElapsedTime(&times[i], ctx->events[i], ctx->events[i + 1]);
    if (code != hipSuccess) return hip_fail(error, error_capacity, code);
    std::memcpy(scores, ctx->staging_scores, count * sizeof(int32_t));
    timings->h2d_ms = times[0]; timings->kernel_ms = times[1]; timings->d2h_ms = times[2];
    return 0;
}
