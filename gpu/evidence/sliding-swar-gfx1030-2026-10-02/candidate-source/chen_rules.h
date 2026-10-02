#ifndef CHEN_RULES_H
#define CHEN_RULES_H
#include "chen_gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
#define CHEN_RULES_MAX_MOVES 256
#define CHEN_RULES_CAPACITY -2
typedef struct ChenRulePosition {
    uint64_t boards[12];
    uint32_t side;
    uint32_t castling;
    int32_t ep_square;
    uint32_t halfmove;
    uint32_t fullmove;
    uint32_t reserved;
    uint64_t hash;
} ChenRulePosition;
typedef struct ChenRuleMove { uint8_t from, to, promotion, reserved; } ChenRuleMove;
/* Promotion: 0 none, 1 knight, 2 bishop, 3 rook, 4 queen; A1=0.
 * Castle bits WK/WQ/BK/BQ=1/2/4/8; EP=-1 absent. Hash exactly src/core.rs.
 * Positions must satisfy Rust's local FEN invariants, halfmove<=65535,
 * fullmove>=1, no reserved data, and a matching recomputed hash. */
typedef struct ChenRulesStats {
    /* Counts and kernel/event spans accumulate since reset. Event spans include
     * dispatch/enqueue gaps; they exclude host staging and debug/PV downloads. */
    uint64_t arena_positions;
    uint64_t classified_positions;
    uint64_t evaluated_positions;
    uint64_t next_start;
    uint64_t next_count;
    double kernel_ms;
} ChenRulesStats;
typedef struct ChenRulesResult { int32_t score; uint32_t pv_length; } ChenRulesResult;
typedef struct ChenRulesContext ChenRulesContext;
uint32_t chen_rules_abi_version(void);
ChenRulesContext *chen_rules_create(int32_t device, uint64_t max_positions,
                                    char *error, uint64_t error_capacity);
void chen_rules_destroy(ChenRulesContext *);
/* All APIs synchronous and serialized per context; every device call sets its
 * device. Returns 0 success, -2 bounded arena/move overflow, -1 other failure.
 * On failure outputs must be discarded. Generic legal/make invalidate the
 * resident arena; reset before using tree APIs after a generic call. */
int32_t chen_rules_legal(ChenRulesContext *, const ChenRulePosition *, uint64_t count,
    ChenRuleMove *moves /* count*256 */, uint32_t *counts, uint32_t *flags,
    ChenTimings *, char *, uint64_t);
/* Legal flags bit0=in-check, bit1=insufficient material. */
int32_t chen_rules_make(ChenRulesContext *, const ChenRulePosition *,
    const ChenRuleMove *, uint64_t count, ChenRulePosition *out,
    ChenTimings *, char *, uint64_t);
/* history contains previous positions only, <=65535 hashes. */
int32_t chen_rules_reset(ChenRulesContext *, const ChenRulePosition *root,
    const uint64_t *history, uint64_t history_count, char *, uint64_t);
/* classify executes exact legal generation/terminal adjudication without
 * allocating children; adjudicate=0 pure perft, =1 chess search draws.
 * The first classify/expand/evaluate locks mode until reset; mixed modes fail.
 * Repeated classification/evaluation is idempotent for scores/count telemetry.
 * classify/evaluate on expanded nodes fail. */
int32_t chen_rules_classify(ChenRulesContext *, uint64_t start, uint64_t count,
    uint32_t adjudicate, ChenRulesStats *, char *, uint64_t);
/* expand first classifies then assigns ranges/creates children entirely onGPU.
 * Arena capacity includes every retained tree level; overflow does not commit
 * children. Expanding a statically evaluated nonterminal node is invalid.
 * next_start/count are in stats. No device pointers escape ABI1. */
int32_t chen_rules_expand(ChenRulesContext *, uint64_t start, uint64_t count,
    uint32_t adjudicate, ChenRulesStats *, char *, uint64_t);
/* evaluate classifies/adjudicates leaves before evaluating nonterminals;
 * evaluator0 material/1 tapered, locked on first use until reset; clamps +/-29000. Must not evaluate expanded
 * nodes. It can finish already admitted leaves at a host node budget boundary. */
int32_t chen_rules_evaluate(ChenRulesContext *, uint64_t start, uint64_t count,
    uint32_t evaluator, ChenRulesStats *, char *, uint64_t);
/* reduce parents at one level after their child levels have complete scores.
 * A read-only preflight rejects mixed plies, overlapping children, incomplete
 * child scores or invalid phases before any reduction scores are written. */
int32_t chen_rules_reduce(ChenRulesContext *, uint64_t start, uint64_t count,
    ChenRulesStats *, char *, uint64_t);
/* root result/PV only; insufficient PV capacity errors instead of truncating. */
int32_t chen_rules_finish(ChenRulesContext *, ChenRulesResult *, ChenRuleMove *pv,
    uint64_t pv_capacity, ChenRulesStats *, char *, uint64_t);
/* Debug download never needed for normal tree expansion/evaluation/reduction. */
int32_t chen_rules_download(ChenRulesContext *, uint64_t start, uint64_t count,
    ChenRulePosition *out, char *, uint64_t);
/* Pure legal perft; counters/draw claims never prune legal branches.
 * depth0=1; arena overflow explicit. This invalidates any prior resident tree. */
int32_t chen_rules_perft(ChenRulesContext *, const ChenRulePosition *, uint32_t depth,
    uint64_t *nodes, ChenRulesStats *, char *, uint64_t);
#ifdef __cplusplus
}
static_assert(sizeof(ChenRulePosition)==128, "rules position ABI");
static_assert(sizeof(ChenRuleMove)==4, "rules move ABI");
static_assert(sizeof(ChenRulesStats)==48, "rules stats ABI");
static_assert(sizeof(ChenRulesResult)==8, "rules result ABI");
#endif
#endif
