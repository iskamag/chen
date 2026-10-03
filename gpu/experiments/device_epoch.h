#ifndef CHEN_DEVICE_EPOCH_H
#define CHEN_DEVICE_EPOCH_H
#include "chen_selective.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef struct ChenDeviceEpochContext ChenDeviceEpochContext;
typedef struct ChenEpochTrace {
    ChenSelectiveStats stats;
    ChenRulesResult result;
    ChenRuleMove pv[CHEN_SELECTIVE_MAX_PLY];
} ChenEpochTrace;
typedef struct ChenEpochReport {
    uint32_t abi_version,struct_size,attempts,mode;
    uint32_t advance_bound,error,resident_blocks_per_multiprocessor,grid_blocks;
    double setup_ms;
    ChenEpochTrace trace[4];
} ChenEpochReport;
/* Separate research ABI: mode0 gated graph;1 one block;2 cooperative grid;
 * mode3 ordinary queued launches of exactly the mode0 graph kernels.
 * Advances2/4. Host checks cancellation/deadline BETWEEN epoch calls.
 * No external/Torch evaluator callback runs inside these classical epochs.
 * Errors are sticky/context-invalidating. Output report stays unchanged on
 * failure; caller retains its last completed host result. */
ChenDeviceEpochContext *chen_epoch_create(int32_t device,uint64_t capacity,
    uint32_t mode,uint32_t block_size,char *error,uint64_t error_capacity);
void chen_epoch_destroy(ChenDeviceEpochContext *);
int32_t chen_epoch_reset(ChenDeviceEpochContext *,const ChenRulePosition *,
    const uint64_t *history,uint64_t history_count,const ChenSelectiveConfig *,
    char *error,uint64_t error_capacity);
int32_t chen_epoch_advance(ChenDeviceEpochContext *,uint64_t absolute_node_limit,
    uint32_t evaluator,uint32_t advances,ChenEpochReport *,char *error,uint64_t error_capacity);
#ifdef __cplusplus
}
static_assert(sizeof(ChenEpochTrace)==360,"epoch trace ABI");
static_assert(sizeof(ChenEpochReport)==1480,"epoch report ABI");
#endif
#endif
