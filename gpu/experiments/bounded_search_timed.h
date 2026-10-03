#ifndef CHEN_EXPERIMENT_BOUNDED_SEARCH_TIMED_H
#define CHEN_EXPERIMENT_BOUNDED_SEARCH_TIMED_H
#include "bounded_search.h"
#ifdef __cplusplus
extern "C" {
#endif
#define CHEN_BOUNDED_TIMED_ABI 2
#define CHEN_BOUNDED_ABORT_DEADLINE 6
#define CHEN_BOUNDED_ABORT_CANCELLED 7
#define CHEN_BOUNDED_TIMED_MAX_NS 5000000000ULL
/* Separate ABI: embedded search job/result retain their exact ABI1 semantics.
 * elapsed_ns is device elapsed-work allowance, 1..5e9, NOT an absolute host clock.
 * Device wall_clock64 rate is admitted using hipDeviceAttributeWallClockRate.
 * Deadline and cancellation are polled every32 DFS loop transitions and before
 * root publication; one legal-generation/evaluation operation is indivisible.
 * Deadline does not cover host copies/stream fences; the Rust scheduler includes
 * them in its wall budget and passes only remaining allowance to each launch. */
typedef struct ChenBoundedTimedJob {
    ChenBoundedJob search;
    uint64_t elapsed_ns;
    uint32_t abi_version,struct_size;
} ChenBoundedTimedJob;
typedef struct ChenBoundedTimedResult {
    ChenBoundedResult search;
    uint64_t elapsed_ticks;
    uint32_t legal_root_count;
    ChenRuleMove fallback;
    uint32_t abi_version,struct_size;
} ChenBoundedTimedResult;
typedef struct ChenBoundedTimedTelemetry {
    double h2d_ms,kernel_ms,d2h_ms;
    uint64_t wall_clock_khz,scratch_bytes,history_capacity;
} ChenBoundedTimedTelemetry;
typedef struct ChenBoundedTimedContext ChenBoundedTimedContext;
uint32_t chen_bounded_timed_abi_version(void);
ChenBoundedTimedContext *chen_bounded_timed_create(int32_t device,uint32_t capacity,
    char *error,uint64_t error_capacity);
/* Only cancel is concurrently callable, from a host monitor. It performs a
 * system-visible atomic store into explicitly coherent mapped host memory.
 * clear_cancel must be called by the serialized owner before spawning monitor.
 * run never clears cancellation itself. No host callbacks execute onGPU. */
void chen_bounded_timed_cancel(ChenBoundedTimedContext *);
void chen_bounded_timed_clear_cancel(ChenBoundedTimedContext *);
int32_t chen_bounded_timed_history(ChenBoundedTimedContext *,const uint64_t *,uint64_t,
    double *h2d_ms,char *,uint64_t);
int32_t chen_bounded_timed_run(ChenBoundedTimedContext *,const ChenBoundedTimedJob *,uint32_t,
    ChenBoundedTimedResult *,ChenBoundedTimedTelemetry *,char *,uint64_t);
/* Returns-1 after uncertain retirement; resources are retained and the caller
 * must retire its process. All configuration/transport failures leave output
 * buffers untouched. Typed device aborts publish counters/fallback but score,
 * PV and completed_depth stay zero. Only full-window Exact is a root decision.
 * Caller retains previous completed root. Fallback is canonical legal move0
 * only if root legal generation completed; it is never an optimality claim. */
int32_t chen_bounded_timed_destroy(ChenBoundedTimedContext *);
#ifdef __cplusplus
}
static_assert(sizeof(ChenBoundedTimedJob)==208,"timed job ABI");
static_assert(sizeof(ChenBoundedTimedResult)==328,"timed result ABI");
static_assert(sizeof(ChenBoundedTimedTelemetry)==48,"timed telemetry ABI");
#endif
#endif
