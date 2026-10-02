#ifndef CHEN_GPU_H
#define CHEN_GPU_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 1: all buffers are caller-owned; context calls must be serialized.
 * Pieces: white P/N/B/R/Q/K, black P/N/B/R/Q/K. A1 is bit zero.
 * side: 0 white, 1 black; reserved must be zero. Scores: side-to-move cp.
 * No chess history is represented: this is evaluation data, not search state. */
typedef struct ChenPosition {
    uint64_t pieces[12];
    uint32_t side;
    uint32_t reserved;
} ChenPosition;
typedef struct ChenDeviceInfo {
    char name[128];
    char architecture[64];
    uint64_t total_memory;
    uint32_t compute_units; /* HIP multiProcessorCount: 30 WGPs here, 60 physical CUs. */
    uint32_t wave_size;
    uint32_t async_engines;
    uint32_t runtime_version;
} ChenDeviceInfo;
typedef struct ChenTimings {
    double h2d_ms;
    double kernel_ms;
    double d2h_ms;
    uint64_t count;
    uint64_t capacity;
} ChenTimings;
typedef struct ChenGpuContext ChenGpuContext;
uint32_t chen_gpu_abi_version(void);
ChenGpuContext *chen_gpu_create(int32_t device, uint64_t capacity,
                                char *error, uint64_t error_capacity);
int32_t chen_gpu_device_info(ChenGpuContext *, ChenDeviceInfo *, char *, uint64_t);
/* evaluator = 0 material, 1 tapered; 0 return means success.
 * Synchronous completion, persistent device and pinned host buffers.
 * HIP-event transfer timings exclude staging memcpy/host validation.
 * count above capacity fails; no hidden reallocations or CPU fallback. */
int32_t chen_gpu_evaluate(ChenGpuContext *, const ChenPosition *, int32_t *,
                         uint64_t count, uint32_t evaluator, ChenTimings *,
                         char *error, uint64_t error_capacity);
void chen_gpu_destroy(ChenGpuContext *);
#ifdef __cplusplus
}
static_assert(sizeof(ChenPosition) == 104, "position ABI");
static_assert(sizeof(ChenDeviceInfo) == 216, "device ABI");
static_assert(sizeof(ChenTimings) == 40, "timings ABI");
#endif
#endif
