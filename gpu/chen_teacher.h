#ifndef CHEN_TEACHER_H
#define CHEN_TEACHER_H
#include "chen_replay.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Independent teacher ABI1. Caller-owned tightly packed GPU tensors:
 * scores int32[N], best_actions int64[N], parent_ply int32[N]. */
typedef struct ChenTeacherSink {
    uint32_t abi_version, struct_size;
    int32_t device;
    uint32_t reserved;
    uint64_t capacity, scores, best_actions, parent_ply;
} ChenTeacherSink;
typedef struct ChenTeacherExport {
    uint64_t rows, root_epoch;
    double kernel_ms;
    /* -1 no evaluator locked (terminal-only scoring);0 Material;1 Tapered. */
    int32_t evaluator;
    /* 1=scored resident finite-tree classical minimax. No uniform depth claim. */
    uint32_t label_kind;
} ChenTeacherExport;
uint32_t chen_teacher_abi_version(void);
/* Same existing child-edge row selection, epoch, runtime-identity, pointer and
 * serialized ownership rules as replay export. Compare all5 runtime tokens
 * before context creation. Caller synchronizes Torch work before writable use,
 * retains owners until synchronous return, and discards output on any error.
 * Source parents must be expanded/nonterminal and completely backed up: scored,
 * all children scored, best is the first maximum of -child.score, parent score
 * equals that maximum. Preflight all selected edges before writing output.
 * Exact parent side-to-move scores: static CP clamped +/-29000, mate terminal
 * -30000+absolute-root-ply, propagated by negation. parent_ply makes the mate
 * frame explicit. best_actions use replay's current-mover canonical encoding;
 * sibling edge rows share the same parent labels. No normalization/outcome claim.
 * Board/node/legal/count state preserved; scratch/timing telemetry may change.
 * No board/label/support D2H, only bounded scalar status reads. No allocation.
 * Returns0 success,-2 capacity,-1 validation/runtime. Runtime faults can leave
 * partial output; recreate context after genuine HIP faults. Empty count with a
 * valid search arena/schema is a no-op. Uniform horizon/evaluator construction
 * provenance is the producer's responsibility, not inferred from arbitrary trees.
 */
int32_t chen_rules_export_teacher(ChenRulesContext *, uint64_t child_start,
    uint64_t count, uint64_t expected_epoch, const ChenTeacherSink *,
    ChenTeacherExport *, char *, uint64_t);
#ifdef __cplusplus
}
static_assert(sizeof(ChenTeacherSink)==48, "teacher sink ABI");
static_assert(sizeof(ChenTeacherExport)==32, "teacher result ABI");
#endif
#endif
