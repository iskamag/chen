#ifndef CHEN_EXTERNAL_H
#define CHEN_EXTERNAL_H
#include "chen_rules.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Independently versioned, synchronous leaf-evaluator protocol. Every pointer
 * passed to a callback is borrowed for that invocation only. It must not escape.
 * Positions are count contiguous 128-byte device RulePosition packets, scores
 * count writable int32 device elements. These are private arena allocations:
 * callbacks may read positions and write scores, never modify positions.
 * The rules stream is complete before invocation. The callback completes ALL
 * its GPU work before returning, including failure, and catches C++ exceptions.
 * Terminal positions may be included, but their output scores are ignored.
 * Successful outputs must be in [-29000,29000], side-to-move centipawns. Rules
 * own terminal adjudication/mate scores. No fallback is implied on failure. */
typedef int32_t (*ChenExternalEvaluate)(void *owner, int32_t device,
    const ChenRulePosition *positions, uint64_t count, int32_t *scores,
    char *error, uint64_t error_capacity);
typedef struct ChenExternalEvaluator {
    uint32_t abi_version, struct_size;
    void *owner;
    ChenExternalEvaluate evaluate;
    /* Bound HIP tokens in the same order as chen_replay_runtime_identity.
     * Compare before any context callback; mismatches are errors. */
    uint64_t runtime_identity[5];
    uint64_t max_batch;
    uint64_t reserved;
} ChenExternalEvaluator;
uint32_t chen_external_abi_version(void);
/* No static evaluator outputs committed until every callback chunk and score
 * preflight succeeds. Classification can commit exact terminal scores/counts.
 * First invocation fixes evaluator identity (callback and owner) until reset;
 * mixing classical/external evaluators fails. Repeated scoring is idempotent.
 * The descriptor and owner remain live for this synchronous call only. Caller
 * serializes context use and guarantees trusted callback behavior/ownership.
 * Invalid phases, runtime mismatch and partial callback output fail explicitly.
 * Failed iterations must be discarded. No board/score payload host download. */
int32_t chen_rules_evaluate_external(ChenRulesContext *, uint64_t start,
    uint64_t count, const ChenExternalEvaluator *, ChenRulesStats *,
    char *, uint64_t);
#ifdef __cplusplus
}
static_assert(sizeof(ChenExternalEvaluator)==80, "external evaluator ABI");
#endif
#endif
