#ifndef CHEN_REPLAY_H
#define CHEN_REPLAY_H
#include "chen_rules.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Independently versioned export ABI. Addresses are caller-owned device buffers,
 * never borrowed arena pointers. Rows are tightly packed with these exact types:
 * pieces/next_pieces int64[N,12], state/next_state int32[N,5],
 * legal_actions int16[N,K], legal_counts int32[N], actions int64[N],
 * material_proxy float32[N], terminal uint8[N], identities int64[N].
 * State columns: side, castling, raw EP(-1 absent), halfmove, fullmove.
 * Unsigned boards/clocks/hash/IDs preserve their bit patterns in signed tensors.
 * Canonical actions ((from*64+to)*5+promotion), with black squares XOR56.
 * Active legal IDs sorted unique, padding -1, no truncation. */
typedef struct ChenReplaySink {
    uint32_t abi_version, struct_size;
    int32_t device;
    uint32_t legal_slots;
    uint64_t capacity;
    uint64_t pieces, state, next_pieces, next_state;
    uint64_t legal_actions, legal_counts, actions, material_proxy, terminal;
    uint64_t parent_hash, child_hash, parent_node, child_node, root_epoch;
} ChenReplaySink;
typedef struct ChenReplayExport {
    uint64_t rows, root_epoch;
    /* HIP event span for validation+export dispatches, including enqueue gaps.
     * Excludes pointer inspection and result/status downloads. */
    double kernel_ms;
} ChenReplayExport;
uint32_t chen_replay_abi_version(void);
/* Address tokens of bound HIP functions, not arena storage. Selectors:
 * 0 PointerGetAttributes,1 MemGetAddressRange,2 Malloc,3 Free,4 StreamSynchronize.
 * Invalid selector returns0. Caller compares all five to the producing runtime
 * before context creation; foreign instances cannot prove allocation bounds. */
uint64_t chen_replay_runtime_identity(uint32_t selector);
/* Host metadata only; serialized with every context call. A fresh context has
 * epoch0; epoch alone does not prove an available search arena. */
int32_t chen_replay_epoch(ChenRulesContext *, uint64_t *epoch, char *, uint64_t);
/* Select contiguous resident child IDs, ordered increasing; count0 is a no-op.
 * Nonempty selections exclude root. Each source parent must be expanded,
 * nonterminal, in a search-mode arena. Children may already be expanded.
 * No board packets are downloaded; board/node/legal/count state is preserved.
 * Private scratch and timing telemetry may change. Exact successor terminal follows
 * legal mate/stalemate first, then draw counters/material/retained repetition.
 * Proxy=tanh(current-mover material centipawns/800), no game-result claim.
 * Epoch increases per successful reset and generic legal/make invalidation,
 * without wrapping, and is scoped to this context. expected_epoch must match.
 * Caller synchronizes its GPU producers/consumers before this call, exclusively
 * owns writable buffers until return, retains their allocation owners, and
 * serializes all operations on the context. The private HIP stream drains
 * before return (including device errors). This API does not synchronize Torch.
 * HIP checks full advertised capacity allocation ranges, device/type/alignment,
 * and disjoint byte intervals; caller proves tensor logical shape/type/size.
 * Returns0 success,-2 capacity,-1 validation/runtime. Output rows [0,count)
 * valid on success only; discard output on any failure. Preflight capacity and
 * selection checks complete before output writes; runtime faults may leave
 * partial output. No pointers escape and no per-call device allocations. */
int32_t chen_rules_export_transitions(ChenRulesContext *, uint64_t child_start,
    uint64_t count, uint64_t expected_epoch, const ChenReplaySink *,
    ChenReplayExport *, char *, uint64_t);
#ifdef __cplusplus
}
static_assert(sizeof(ChenReplaySink)==136, "replay sink ABI");
static_assert(sizeof(ChenReplayExport)==24, "replay result ABI");
#endif
#endif
