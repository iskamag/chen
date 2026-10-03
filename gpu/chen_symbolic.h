#ifndef CHEN_SYMBOLIC_H
#define CHEN_SYMBOLIC_H
#include "chen_rules.h"
#ifdef __cplusplus
extern "C" {
#endif
#define CHEN_SYMBOLIC_CHANNELS 12
/* Independent current-state snapshot ABI. Caller-owned contiguous GPU outputs:
 * features float32[N,18,8,8], channels float32[N,64,12], history int64[N,4],
 * position_hash/node_id/root_epoch int64[N] preserving uint64 bit patterns.
 * Channel order matches neural.symbolic_residual.CHANNELS; current-mover
 * orientation flips black ranks XOR56 and swaps colors. No score/teacher input.
 * History: halfmove, repetition count INCLUDING current state, retained
 * reversible window length INCLUDING current state, absolute search-root ply.
 * Window is min(halfmove+1, retained ancestors+root-history+current).
 * Phase=min(sum(N+B+2R+4Q over both colors),24)/24.
 * Counts clip8/div8; history scales100/3/256/256 then clip1; pins/checkers binary.
 * Pseudoattacks include own occupied blockers, and ignore pin legality. */
typedef struct ChenSymbolicSink {
    uint32_t abi_version,struct_size;
    int32_t device;uint32_t reserved;
    uint64_t capacity,features,channels,history,position_hash,node_id,root_epoch;
} ChenSymbolicSink;
typedef struct ChenSymbolicExport { uint64_t rows,root_epoch;double kernel_ms; } ChenSymbolicExport;
uint32_t chen_symbolic_abi_version(void);
/* Root and contiguous child IDs are accepted in a valid reset/search arena,
 * including expanded or terminal nodes. Perft mode is rejected. Export changes
 * no node/position/search counters; private scratch/timing telemetry may change.
 * Root epoch must match, without wrapping.
 * Same ownership/runtime/serialization/stream completion contract as replay
 * export. Caller completes Torch producers before entry, keeps all allocations
 * alive/exclusive through return and discards outputs on failure. All advertised
 * capacity intervals are allocation-bounded, disjoint and non-aliasing to arena.
 * Separate read-only preflight completes before output writes. Runtime faults
 * can leave partial output; no result metadata is published on failure.
 * No board download, dense attack matrix or per-call device allocation. */
int32_t chen_rules_export_symbolic(ChenRulesContext *,uint64_t start,uint64_t count,
    uint64_t expected_epoch,const ChenSymbolicSink *,ChenSymbolicExport *,char *,uint64_t);
/* Independent topology ABI1; existing symbolic ABI1/layout is unchanged.
 * Caller-owned uint64 outputs: source_attacks[N,64], pinned[N,2], checkers[N],
 * pinner_by_square[N,64] int64 and position_hash/node_id/root_epoch[N] uint64.
 * Torch int64 storage preserves uint64 bit patterns.
 * Source index AND target bits use current-mover rank orientation (black XOR56).
 * Empty sources are zero. Both colors' sources are present; ownership comes from
 * the observed board, not a separate color dimension. Slider masks include the
 * first occupied blocker, including friendly pieces; pinned pieces retain their
 * pseudoattacks. Pawn masks exclude pushes/EP and king masks exclude castling.
 * pinned columns0/1 are mover/opponent piece bits. checkers contains opponent
 * sources checking the current mover's king. These are not legal-move masks.
 * pinner_by_square is -1 for empty/unpinned sources; otherwise it is the canonical
 * pinner square. It covers both colors and never changes the pseudoattack mask.
 * Hash remains the absolute simulator hash; node/epoch are arena identity.
 * Same preflight, ownership, zero-copy-of-board, failure and synchronous stream
 * contract as chen_rules_export_symbolic. No dense matrix/device allocation.
 * All seven output intervals advertise full capacity and require8-byte alignment.
 * The result metadata uses the unchanged ChenSymbolicExport layout. */
typedef struct ChenTopologySink {
    uint32_t abi_version,struct_size;
    int32_t device;uint32_t reserved;
    uint64_t capacity,source_attacks,pinned,checkers,pinner_by_square,position_hash,node_id,root_epoch;
} ChenTopologySink;
uint32_t chen_topology_abi_version(void);
int32_t chen_rules_export_topology(ChenRulesContext *,uint64_t start,uint64_t count,
    uint64_t expected_epoch,const ChenTopologySink *,ChenSymbolicExport *,char *,uint64_t);
#ifdef __cplusplus
}
static_assert(sizeof(ChenSymbolicSink)==72,"symbolic sink ABI");
static_assert(sizeof(ChenSymbolicExport)==24,"symbolic export ABI");
static_assert(sizeof(ChenTopologySink)==80,"topology sink ABI");
#endif
#endif
