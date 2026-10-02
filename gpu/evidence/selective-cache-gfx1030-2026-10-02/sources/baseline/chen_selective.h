#ifndef CHEN_SELECTIVE_H
#define CHEN_SELECTIVE_H
#include "chen_external.h"
#ifdef __cplusplus
extern "C" {
#endif

#define CHEN_SELECTIVE_MAX_PLY 64u
#define CHEN_SELECTIVE_MAX_BATCH 256u
#define CHEN_SELECTIVE_MAX_CANDIDATES 1024u
#define CHEN_SELECTIVE_POLICY_BEST_OPEN 0u

#define CHEN_SELECTIVE_ROOT_PUBLISHED 1u
#define CHEN_SELECTIVE_ROOT_TERMINAL 2u
#define CHEN_SELECTIVE_HORIZON_COMPLETE 4u
#define CHEN_SELECTIVE_MATE_PROOF 8u

#define CHEN_SELECTIVE_PROGRESS 0u
#define CHEN_SELECTIVE_CLOSED 1u
#define CHEN_SELECTIVE_NODE_BUDGET 2u
#define CHEN_SELECTIVE_ARENA_CAPACITY 3u
#define CHEN_SELECTIVE_CANDIDATE_SCAN 4u
#define CHEN_SELECTIVE_FALLBACK 5u

typedef struct ChenSelectiveConfig {
    uint32_t abi_version, struct_size;
    uint32_t max_ply, policy;
    uint32_t batch_leaves, candidate_limit;
    uint64_t reserved;
} ChenSelectiveConfig;

typedef struct ChenSelectiveStats {
    uint32_t abi_version, struct_size;
    uint64_t arena_positions, classified_positions, evaluated_positions;
    uint64_t completed_steps, expanded_leaves, open_frontier_leaves;
    uint32_t last_selected, last_added;
    double kernel_ms;
    uint32_t minimum_frontier_depth, maximum_frontier_depth, pv_depth, root_flags;
    uint32_t stop_reason, reserved;
} ChenSelectiveStats;

typedef struct ChenSelectiveContext ChenSelectiveContext;
uint32_t chen_selective_abi_version(void);
ChenSelectiveContext *chen_selective_create(int32_t device, uint64_t max_positions,
    char *error, uint64_t error_capacity);
void chen_selective_destroy(ChenSelectiveContext *);

/* An independent arena: no inner rules context or device pointer escapes.
 * Serialized, synchronous calls may move exclusively between host threads;
 * every device call sets its device. Full-width rules ABI1 is unchanged.
 * Config is ABI1/sizeof32, max_ply1..64, policy0, batch_leaves1..256,
 * candidate_limit batch_leaves..1024, reserved0. Root/history use rules ABI1:
 * previous-history excludes current root and contains at most65535 hashes.
 * Reset uploads/validates only: no classified-node admission or evaluation.
 * Successful reset returns FALLBACK with PUBLISHED unset. */
int32_t chen_selective_reset(ChenSelectiveContext *, const ChenRulePosition *root,
    const uint64_t *history, uint64_t history_count, const ChenSelectiveConfig *,
    ChenSelectiveStats *, char *error, uint64_t error_capacity);

/* absolute_node_limit caps cumulative newly classified nodes, including root.
 * Evaluator0 material/1 tapered requires external=null; evaluator2 requires a
 * checked External80 descriptor. Its callback/owner/runtime/max_batch identity
 * is locked until reset; descriptor storage is borrowed for this call only.
 * Select distinct unexpanded leaves by descending through best open children,
 * max -child.score, tie first legal child. Reserve each examined candidate and
 * all ancestors for this step, including candidates rejected for admission.
 * Admit only COMPLETE sibling sets within node limit and arena capacity, in
 * selection order. New children form one contiguous block, evaluated using
 * existing exact rules/terminal or external evaluation. All ancestor scores and
 * closure metadata are backed up in device reverse-index order.
 * Terminal and max-ply leaves stay in minimax, but are never eligible again.
 * A nonterminal root is published only after its entire legal set is scored.
 * No board/frontier/score-array host download is needed by these operations.
 *
 * Return0 includes explicit clean budget/capacity/candidate-scan/closure stops;
 * no partial sibling commit or node-cap overshoot. Such stops preserve the last
 * completed root/PV. Reserved candidates are cleared at each advance. Already
 * admitted evaluation/backup finishes even at the exact node boundary. Host
 * cancellation is checked BETWEEN calls, not inside a running GPU/callback.
 * Negative reset/advance errors invalidate only this selective context: finish
 * refuses publication until reset succeeds. Actual HIP/native runtime faults
 * may require process recreation; resetting does not promise fault recovery.
 * Caller retains its last published host snapshot on runtime failure. */
int32_t chen_selective_advance(ChenSelectiveContext *, uint64_t absolute_node_limit,
    uint32_t evaluator, const ChenExternalEvaluator *external,
    ChenSelectiveStats *, char *error, uint64_t error_capacity);

/* PUBLISHED root/PV only; otherwise returns an error, never stale publication.
 * Insufficient PV capacity errors instead of truncating. All frontier depth
 * statistics describe ACTUAL retained terminal/horizon/open leaf plies.
 * They do not assert an exact fixed-horizon score for a mixed-depth tree.
 * Only HORIZON_COMPLETE promises equality to full-width target max_ply minimax
 * with the same evaluator/rules. MATE_PROOF establishes a forced outcome under
 * current-position draw policy; its finite-tree mate distance is not promised
 * globally shortest while other branches remain unresolved.
 * kernel_ms is cumulative rules/selection/backup event spans, including host
 * dispatch gaps and external callback spans; not pure GPU active time.
 * Only scalar metadata and this final legal PV cross the host boundary. */
int32_t chen_selective_finish(ChenSelectiveContext *, ChenRulesResult *,
    ChenRuleMove *pv, uint64_t pv_capacity, ChenSelectiveStats *,
    char *error, uint64_t error_capacity);

#ifdef __cplusplus
}
static_assert(sizeof(ChenSelectiveConfig)==32, "selective config ABI");
static_assert(sizeof(ChenSelectiveStats)==96, "selective stats ABI");
#endif
#endif
