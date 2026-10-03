#ifndef CHEN_EXPERIMENT_BOUNDED_SEARCH_H
#define CHEN_EXPERIMENT_BOUNDED_SEARCH_H
#include "../chen_rules.h"
#ifdef __cplusplus
extern "C" {
#endif
#define CHEN_BOUNDED_ABI 1
#define CHEN_BOUNDED_MAX_JOBS 64
#define CHEN_BOUNDED_STACK 64
#define CHEN_BOUNDED_MAX_DEPTH 16
#define CHEN_BOUNDED_MAX_QPLY 32
#define CHEN_BOUNDED_MAX_NODES 1000000
#define CHEN_BOUNDED_EXACT 0
#define CHEN_BOUNDED_LOWER 1
#define CHEN_BOUNDED_UPPER 2
#define CHEN_BOUNDED_ABORTED 3
#define CHEN_BOUNDED_ABORT_NONE 0
#define CHEN_BOUNDED_ABORT_NODES 1
#define CHEN_BOUNDED_ABORT_QPLY 2
#define CHEN_BOUNDED_ABORT_STACK 3
#define CHEN_BOUNDED_ABORT_POSITION 4
#define CHEN_BOUNDED_ABORT_RULES 5
typedef struct ChenBoundedJob {
    ChenRulePosition root;
    uint64_t node_budget, history_offset, history_count;
    uint32_t abi_version, struct_size, depth, max_qply;
    uint32_t evaluator, quiescence, root_ply, reserved;
    int32_t alpha, beta;
} ChenBoundedJob;
typedef struct ChenBoundedResult {
    uint32_t abi_version, struct_size, kind, abort_reason;
    int32_t score;
    uint32_t completed_depth, pv_length, reserved;
    uint64_t nodes, qnodes;
    ChenRuleMove pv[CHEN_BOUNDED_STACK];
} ChenBoundedResult;
/* Independent jobs, one nonrecursive device DFS worker per job. History is
 * previous positions only; local stack index0 supplies the current job root.
 * root_ply changes mate-distance scores, not history indexing.
 * Window -32000<=alpha<beta<=32000. Evaluator0 material/1 tapered.
 * quiescence0 is a static-horizon control; quiescence1 searches tactical moves
 * and every evasion. An unresolved continuation at max_qply aborts unless a
 * sound cutoff has already completed the requested window. No TT/PVS/selective pruning.
 * Canonical legal-generation order only; no capture/history ordering yet.
 * completed_depth is requested main depth and excludes q continuation.
 * Lower/Upper PV is a legal search witness, not a certified optimal line.
 * Exact/Lower/Upper refer only to the completed requested finite search policy,
 * not true game values or neural confidence. Aborted score/PV/depth are zero.
 * Caller owns last-completed iteration publication; this API has none.
 * Invalid host configuration or HIP failure returns-1 without modifying output.
 * All calls are synchronous, caller serialized. On unproven HIP retirement
 * resources are intentionally retained and the caller must retire its process.
 * No deadline/cancel interface yet: node/q/stack bounds and an external process
 * watchdog are the only limits. Nodes count entered states including qsearch.
 */
int32_t chen_bounded_search_run(int32_t device,const ChenBoundedJob *jobs,uint32_t count,
    const uint64_t *history,uint64_t history_count,ChenBoundedResult *results,
    char *error,uint64_t error_capacity);
#ifdef __cplusplus
}
static_assert(sizeof(ChenBoundedJob)==192,"bounded job ABI");
static_assert(sizeof(ChenBoundedResult)==304,"bounded result ABI");
#endif
#endif
