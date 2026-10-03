#ifndef CHEN_TACTICAL_H
#define CHEN_TACTICAL_H
#include "chen_symbolic.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Exact chen_tactical_exact_v1 observation, parent-mover frame.
 * P=32,K=256,E=edge_slots>=4992. Caller owns whole disjoint device allocations.
 * Tensor dimensions/dtypes in field order below:
 * f32[N,P,16],f32[N,64,12],f32[N,K,12],bool[N,P],i64[N,P],bool[N,K],i64[N,K],
 * i64[N,E,2],i64[N,E],bool[N,E],f32[N,E,4],bool[N],f32[N],f32[N,K],
 * i64[N],i64[N],i64[N],i64[N,4],i32[N],i32[N,3].
 * Terminal:0 ongoing,1 mate,2 stalemate,3 halfmove100,4 threefold,5 insufficient.
 * Counts:pieces,legal actions,active edges. Terminal draw may retain legal support.
 * Hash is absolute; node/epoch bind source. History fourth counter is arena ply.
 * Geometry computed once per state; exact action effects never use teacher/PV.
 * No board download, dense adjacency or per-call device allocation. Synchronous
 * private-stream completion and status readbacks match symbolic ABI1 contract.
 * Preflight failure preserves outputs. Kernel/runtime failure may partially write;
 * discard all outputs; result metadata is published only on complete success.
 * More than32 occupied squares reject. Unsupported move overflow is a device
 * error, never truncated success. Output padding is initialized and masked.
 */
typedef struct ChenTacticalSink {
    uint32_t abi_version,struct_size;
    int32_t device;uint32_t reserved;
    uint64_t capacity;
    uint32_t legal_slots,edge_slots;
    uint64_t pieces,squares,actions,piece_mask,piece_squares,action_mask,action_ids;
    uint64_t edge_index,edge_kind,edge_mask,edge_features,edge_overflow;
    uint64_t value_prior_cp,action_prior_cp,position_hash,node_id,root_epoch,history,terminal,slot_counts;
} ChenTacticalSink;
uint32_t chen_tactical_abi_version(void);
int32_t chen_rules_export_tactical(ChenRulesContext *,uint64_t,uint64_t,uint64_t,
    const ChenTacticalSink *,ChenSymbolicExport *,char *,uint64_t);
/* Independently versioned bucket API2; same tensor layout/feature schema as ABI1,
 * sink.abi_version=2, 1<=E<=65536. Returns 0 only for a COMPLETE graph.
 * -2 means edge bucket capacity: discard ALL tensor outputs, then retry with a
 * larger disjoint bucket. Other schema/move capacity failures are -1, not retryable.
 * Completion metadata is untouched on any failure; tensor payloads may be partial.
 * 0/-1/-2 guarantee no in-flight writes by this call when it returns. -3 means
 * retirement could not be established: retain allocations, stop using context,
 * and retire the owning process. Never free/retry an unretired packet.
 * All ownership, alias, exact-terminal/history and complete-action conditions
 * of ABI1 continue to apply. No payload/count download selects the bucket. */
#define CHEN_TACTICAL_BUCKET_CAPACITY -2
#define CHEN_TACTICAL_BUCKET_UNRETIRED -3
uint32_t chen_tactical_bucket_abi_version(void);
int32_t chen_rules_export_tactical_bucket(ChenRulesContext *,uint64_t,uint64_t,uint64_t,
    const ChenTacticalSink *,ChenSymbolicExport *,char *,uint64_t);
#ifdef __cplusplus
}
static_assert(sizeof(ChenTacticalSink)==192,"tactical sink ABI");
#endif
#endif
