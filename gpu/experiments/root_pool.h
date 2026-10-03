#ifndef CHEN_ROOT_POOL_H
#define CHEN_ROOT_POOL_H
#include "bounded_search_timed.h"
#ifdef __cplusplus
extern "C" {
#endif
#define CHEN_ROOT_POOL_ABI 3
#define CHEN_ROOT_POOL_PENDING 0
#define CHEN_ROOT_POOL_PAUSED 1
#define CHEN_ROOT_POOL_COMPLETE 2
#define CHEN_ROOT_POOL_FAILED 3
/* Root provenance is fixed by reset: state, prior history, evaluator, q policy,
 * epoch. Each begin changes ONLY main depth. No cross-depth bound reuse.
 * One scalar DFS per independent root-child workgroup; concurrent workgroups,
 * persistent device stacks and a shared completed-Exact incumbent.
 * First epoch gives previous completed PV (canonical first at depth1) one
 * bounded quantum to seed an incumbent; following epochs run all children.
 * Paused is resumable unfinished work, NOT a bound or an Aborted score packet.
 * Challenger window uses incumbent-1, preserving earliest canonical ties.
 * Prior exact root publication survives begin, pause, cancellation and failure.
 * CPU performs no leaf evaluation or legal generation. Calls serialized except
 * mapped coherent cancel. Failed retirement poisons context; retire process.
 */
typedef struct ChenRootPoolChild {
    ChenBoundedResult bound; /* CHILD score frame, root_ply=1 */
    uint32_t status,canonical_index,stack_pointer,reserved;
    int32_t original_alpha,original_beta;
    ChenRuleMove action;
    uint32_t padding;
} ChenRootPoolChild;
typedef struct ChenRootPoolSnapshot {
    ChenBoundedResult published; /* last completed EXACT, parent score frame */
    uint64_t root_hash,root_epoch,nodes,qnodes,iteration_nodes,elapsed_ticks;
    uint32_t abi_version,struct_size,requested_depth,root_count;
    uint32_t completed_children,exact_children,upper_children,paused_children;
    uint32_t abort_reason,iteration_complete,terminal,reserved;
    ChenRuleMove fallback;
    uint32_t padding;
} ChenRootPoolSnapshot;
typedef struct ChenRootPoolContext ChenRootPoolContext;
uint32_t chen_root_pool_abi_version(void);
ChenRootPoolContext *chen_root_pool_create(int32_t device,char *error,uint64_t cap);
int32_t chen_root_pool_destroy(ChenRootPoolContext *ctx);
void chen_root_pool_cancel(ChenRootPoolContext *ctx);
/* reset job requires depth/root_ply/history_offset=0, full window, reserved=0;
 * node budget is TOTAL across depths; history <=65535 previous hashes only.
 * Mandatory root classification is disclosed separately and not node budget. */
int32_t chen_root_pool_reset(ChenRootPoolContext *ctx,const ChenBoundedJob *job,
    const uint64_t *history,uint64_t count,uint64_t root_epoch,
    ChenRootPoolSnapshot *snapshot,char *error,uint64_t cap);
int32_t chen_root_pool_begin(ChenRootPoolContext *ctx,uint32_t depth,
    ChenRootPoolSnapshot *snapshot,char *error,uint64_t cap);
/* elapsed_ns 1..50ms is per-epoch device elapsed cap, checked every8 DFS
 * transitions against a common epoch clock. Host wall deadline checked between
 * epochs. Epoch retirement/prologue/epilogue overhead remains outside this cap.
 * optional children requires capacity256; configuration failure leaves outputs
 * unchanged. Whole children/stack inspection is diagnostic, not required UCI. */
int32_t chen_root_pool_advance(ChenRootPoolContext *ctx,uint64_t elapsed_ns,
    ChenRootPoolSnapshot *snapshot,ChenRootPoolChild *children,uint32_t capacity,
    ChenBoundedTimedTelemetry *telemetry,char *error,uint64_t cap);
#ifdef __cplusplus
}
static_assert(sizeof(ChenRootPoolChild)==336,"pool child ABI");
static_assert(sizeof(ChenRootPoolSnapshot)==408,"pool snapshot ABI");
#endif
#endif
