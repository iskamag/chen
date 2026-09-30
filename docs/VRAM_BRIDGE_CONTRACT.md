# Resident rules arena to Torch replay

This contract defines the first direct ingress from Chen's resident HIP rules
arena into [compact replay](COMPACT_REPLAY.md). Torch owns bounded output
tensors; a synchronous HIP export operation gathers legal parent/child records
into those tensors. Exact states, legal support and simulator identity remain
on the device. There is no borrowed arena pointer and no DLPack dependency in
this first implementation. Training through this seam initially exercises
mechanical losses and explicitly named toy targets; it establishes no strength
or calibrated outcome claim.

The baseline inspected on 2026-09-30 is `gpu/chen_rules.h` ABI 1, SHA-256
`1ece98e87769197389362e903976123195efa10fd13c7aeafe1ec55137320391`,
and `neural/compact.py`, SHA-256
`a52ba420dca331edf999e539cc37ea5b9208630ed0f9d9587fe24501b2c944d3`.
The frozen export header is `gpu/chen_replay.h`, SHA-256
`4522e2d08f1cc34684877464f16a1fe5175fb3b61b77abb51cf70a8eceb8ded6`.
Its corresponding implementation and binding have separate verification work.
The requirements below are acceptance criteria, not results of a bridge
hardware run. No GPU operation was performed to write this document.

## Runtime identity before context creation

The Torch bridge must bind the producer to **the HIP runtime that owns Torch's
allocations and streams**. On this installation Torch bundles HIP 7.0 while the
system compiler/runtime installation is 7.2. A diagnostic initially loaded the
system HIP before Torch's bundled runtime; allocation-extent queries through
the foreign instance then produced misleading bounds. This is an interop
initialization failure, not evidence that a larger tensor or weakened bounds
check would be correct.

Import Torch, identify its actually loaded HIP library, retain that handle and
promote it with `RTLD_GLOBAL` **before** loading the engine with `CDLL`. Use the
actual loaded object, not an unqualified `libamdhip64.so` search that can select
the system installation. The first binding pins this installed wheel's
`torch/lib/libamdhip64.so`, rejects a missing file and checks all five bound
functions; it does not claim to discover every possible Torch distribution's
runtime layout. Linux documents that global symbols participate in
later library resolution and that an already loaded local object can be
promoted. [Linux dlopen manual](https://man7.org/linux/man-pages/man3/dlopen.3.html).
Promotion is not proof by itself: prior loading or symbol interposition can
still bind functions differently. Do not attempt to repair a mismatched live
process by unloading/rebinding a runtime with outstanding allocations; fail
initialization and restart with the correct load order.

Before creating the C context, compare all five nonzero producer-bound function
addresses returned by `chen_replay_runtime_identity(selector)` with the same
functions resolved from Torch's actual HIP runtime:

| Selector | HIP API |
| --- | --- |
| 0 | `hipPointerGetAttributes` |
| 1 | `hipMemGetAddressRange` |
| 2 | `hipMalloc` |
| 3 | `hipFree` |
| 4 | `hipStreamSynchronize` |

An invalid selector returns zero. These are process-local identity tokens, not
allocation pointers or stable serialized IDs. Every mismatch is an error before
context/output allocation; checking only pointer attributes is insufficient.
Capture `dladdr` paths, runtime file SHA-256, compiler version, Torch/HIP versions
and the identity result. Addresses alone are not comparable across processes.
The guard verifies this five-API binding; hardware pointer/range and stream
tests are still required for the whole seam.

Standalone Rust can use the system HIP runtime when that runtime owns its arena
and every operation stays there. The Torch bridge has a different binding
prerequisite. Compiler 7.2 can produce a library executed against bundled HIP
7.0, but compatibility is scoped to the actual compiled artifact and hardware
tests; matching compiler/runtime version numbers is neither required by this
contract nor sufficient evidence of interop correctness. Keep standalone Rust
and Torch-bridge evidence separate. A preloaded copy tracer must also preserve
this load order rather than importing system HIP as an early dependency.

The reviewed native report `gpu/evidence/replay-gfx1030-2026-09-30.json`
records all five `dladdr` results at the actual Torch runtime, SHA-256
`c13a873d6444aa9c32d846feaa888de2108a5ed3c15f8a1b7708c5742a894e81`.
Its tested producer library SHA-256 is
`333e3427c23c9de726387e62cc85f28bf20f8af27d8d8abc48882e586c101444`;
the compiler is HIP 7.2.53211 and Torch's HIP build is 7.0.51831. The finite
native oracle covers 136 transitions across seven fixtures. Its nine marked
export calls each issued two 16-byte summary D2H copies through `hipMemcpy`
or `hipMemcpyAsync`, with no H2D/D2D calls through those APIs. A separate
intentional 256-byte D2H payload control was detected. This is a scoped API
diagnostic, not a complete trace of runtime-internal traffic or all transfers
through optimizer training. It verifies the prerequisite on the captured
artifact without extending it to arbitrary runtimes or strength.

## Ownership and synchronous publication

The rules context exclusively owns its arena, raw moves, node links, legal
counts and retained root-history hashes. The Python adapter exclusively owns
contiguous Torch output tensors, their shape/dtype metadata and all references
needed to keep their storage alive during the C call. Calls are serialized
per context and per output bundle; the adapter holds a lock across validation,
synchronization and export. Another thread must not submit work using or
overwriting the same bundle during that interval.

Publication follows this order:

1. Allocate a fixed-capacity output bundle on the context's actual HIP device.
   Check every tensor's dtype, shape, contiguity, logical byte capacity and
   storage ownership. Keep strong references to every tensor and the loaded
   library/context throughout the call.
2. Call `torch.cuda.synchronize(device)` before handing writable pointers to
   HIP. This drains preceding Torch readers and writers, including replay
   copies that still read a bundle being reused. Torch uses the `cuda` API
   spelling on ROCm. This conservative device barrier also covers freshly
   allocated buffers that the caching allocator could have reused.
3. Pass only the output pointers and bounded control metadata to the exporter.
   HIP selects records and translates states/actions on its private stream.
   No CPU list of states, moves or labels is constructed. The simulator context
   cannot be reset or destroyed concurrently with that operation.
4. HIP drains the private stream before returning, on success and on failure.
   A successful return makes every published output available to subsequent
   Torch operations on any stream. On failure the adapter discards all output
   contents; it never admits a partially written bundle to replay.
5. Construct the auxiliary target/history/game tensors on Torch, then drain
   Torch before returning the complete export bundle. These writes need ordering
   too; HIP completion covers only the fields produced by HIP.
6. Use `ExportedTransitions.append_to` for a completed handoff: it drains prior
   Torch work, calls `CompactReplay.append`, then drains the replay copies before
   returning. `index_copy_` gives the ring its own device storage; the source
   owners can retire immediately after this handoff, even across Torch streams.
   Sampling produces independent gathered tensors, which survive ring overwrite.
   Direct `CompactReplay.append` retains its separate caller-ordered-stream
   contract and does not provide these bridge barriers.

The initial design trades overlap for a small, reviewable lifetime protocol.
Synchronizing a stream waits for computation; it does not itself download board
state. Cross-stream ordering is required even when two libraries use the same
GPU. AMD documents explicit stream synchronization and event dependencies in
the [HIP 7.0 asynchronous execution guide](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/how-to/hip_runtime_api/asynchronous.html).

Neither C nor the Python adapter frees the other's allocations. C retains no
output pointer after returning. Destroying/resetting the simulator after a
successful export cannot change the Torch snapshot. Destruction of the Python
context closes the C context before unloading its library. External raw pointer
access remains a same-process trusted-caller interface; pointer validation does
not make arbitrary addresses or concurrent frees safe.

Recoverable validation/capacity failures allow another valid export after the
failed output is discarded. A genuine HIP device/runtime fault does not certify
that the context remains usable; close/recreate it before resuming production.
Completion fences must also run on exceptions after asynchronous Torch work
has been submitted. Use `finally` around auxiliary publication and replay
admission so a partial operation cannot release source owners while another
stream still reads them. A failure of the completion fence itself is a runtime
fault, not a successful handoff.

## C boundary and bounded selection

The export extension has an independent ABI/version check; it preserves the
layout and behavior of rules ABI 1. The agreed draft is:

```c
uint32_t chen_replay_abi_version(void); /* export ABI 1 */
uint64_t chen_replay_runtime_identity(uint32_t selector);
int32_t chen_replay_epoch(ChenRulesContext *, uint64_t *epoch,
                         char *error, uint64_t error_capacity);
typedef struct ChenReplaySink {
    uint32_t abi_version, struct_size;
    int32_t device;
    uint32_t legal_slots;
    uint64_t capacity;
    uint64_t pieces, state, next_pieces, next_state;
    uint64_t legal_actions, legal_counts, actions, material_proxy, terminal;
    uint64_t parent_hash, child_hash, parent_node, child_node, root_epoch;
} ChenReplaySink; /* 136 bytes */
typedef struct ChenReplayExport {
    uint64_t rows, root_epoch;
    double kernel_ms;
} ChenReplayExport; /* 24 bytes */
int32_t chen_rules_export_transitions(
    ChenRulesContext *, uint64_t child_start, uint64_t count,
    uint64_t expected_epoch,
    const ChenReplaySink *, ChenReplayExport *,
    char *error, uint64_t error_capacity);
```

The uint64 address fields carry same-process device addresses, not exported
arena addresses. `capacity` is rows; the fixed schema computes each field's
required bytes using the full advertised capacity, not just the returned count.
Validate ABI version, exact struct size, legal_slots in 1..256,
context device and every pointer before launching. Counts, spans and byte
multiplications use checked arithmetic. Overflow and capacity exhaustion are
errors, never truncation. A selected parent's legal count exceeding the configured
legal bound fails before writes. C count=0 is a no-op only after validating the
search arena and sink header/device/schema: zero capacity permits zero pointers;
nonzero capacity still requires valid full-capacity buffer intervals. The first
Python adapter admits only nonempty export calls. Host-visible return count,
epoch, status and timing
are control metadata. The final compiled header is authoritative and must be
captured by hash with the corresponding binding.

Select an existing contiguous child ID span. Each exported row identifies an
actual existing
parent/child edge, with the child's recorded move legal in the parent's support.
No arbitrary board pair may be presented as a transition. Current parents must
be classified, expanded nonterminal nodes in adjudicated search mode, with
nonempty legal support. The exporter computes child terminal adjudication
locally using retained root history and ancestry; it need not mutate/classify
the arena child first; already expanded children are valid. A root without a
parent, out-of-range span, unavailable arena or mixed search/perft mode is an
explicit error. An epoch query returns host context metadata without downloading
state. The Python selector retains `(context_uuid, epoch, child_start, count)`
and checks context identity before export; C checks expected epoch against live
epoch before preflight. This rejects stale spans even if their integer IDs alias
valid new nodes, and cross-context equal epochs are rejected by Python identity.
Export preflight finishes before output writes; later runtime failure still
invalidates all outputs.

The C side independently checks nonnull pointers, native element alignment,
the actual device-memory type and device ordinal, sufficient allocation range,
and disjoint writable byte ranges. Reject host, pinned host, managed,
wrong-device and unrecognized pointers; reject overlap among outputs and with
the simulator's allocations. For a pointer inside an allocation, require
`offset <= allocation_bytes` and
`needed_bytes <= allocation_bytes - offset`. A caller's capacity integer alone
is insufficient. HIP's pointer attributes and allocation-range queries provide
these runtime checks. [HIP 7.0 memory API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/doxygen/html/group___memory.html).

Torch can suballocate a tensor inside a larger allocator block. HIP allocation
range therefore does not prove the tensor's logical storage capacity. The Python
adapter separately validates that capacity and keeps the owning storage alive.
The first binding freshly allocates ordinary dense tensors on each export and
requires zero storage offset and logical storage bytes equal to tensor bytes.
Views and custom allocators require separate acceptance tests. Reusable bundles
are a future optimization governed by the same pre-write synchronization rule.
Runtime/compiler/library compatibility and same-device ordinal mapping are
checked at initialization; a Torch device name alone is not that proof.

## Exact record layout and action translation

All arrays have leading dimension `B`; only the returned active prefix is
published. The initial legal bound is 256, matching the rules ABI. The table
describes compact schema 1; exact simulator identity is a separate sidecar.

| Field | Torch dtype and shape | Meaning |
| --- | --- | --- |
| `pieces`, `next_pieces` | int64 `[B,12]` | Absolute white PNBRQK, then black PNBRQK bit patterns |
| `state`, `next_state` | int32 `[B,5]` | Side, absolute castling bits, raw EP, halfmove, fullmove |
| `legal_actions` | int16 `[B,256]` | Sorted unique current-parent canonical actions; -1 padding |
| `legal_counts` | int32 `[B]` | Active prefix length, 1 through 256 |
| `actions`, `target_actions` | int64 `[B]` | Executed action and separately declared target, each in current support |
| `history_handles`, `next_history_handles` | int64 `[B]` | External history-provenance IDs; -1 means absent |
| `material_proxy` | float32 `[B]` | Declared toy scalar in [-1,1] |
| `terminal` | bool `[B]` | Successor terminal under the declared simulator draw policy |
| `game_ids` | int64 `[B]` | Stable source trajectory identity |

`ChenRulePosition` is 128 bytes and additionally contains its simulator hash
and reserved-zero field. The export kernel transposes current and successor
packets into the SoA arrays. Bit 63 must survive uint64-to-int64 transport as a
bit pattern, not a numeric saturation or floating-point conversion. State side
is 0/1; castling WK/WQ/BK/BQ bits are 1/2/4/8; absent raw EP is -1. Halfmove is
0..65535. Fullmove is 1..4294967295, transported through signed int32 and decoded
with `int64(column) & 0xffffffff`. The raw clocks stay exact even though the
network feature clips halfmove at 100.

Squares are absolute A1=0. Canonicalize both endpoints of a move using the
**current parent's** side: xor 56 for black, unchanged for white. Promotion
codes are 0 none, 1 knight, 2 bishop, 3 rook, 4 queen. The index is
`((canonical_from * 64 + canonical_to) * 5 + promotion)`, in 0..20479.
Generate all legal indices, sort them and pad inactive slots with -1. Moving
into successor orientation before encoding the executed action is an error.
Promotion variants must remain distinct; castling and EP use the ordinary king
or pawn endpoints, with their special effects preserved in successor state.

Device decoding must exactly reproduce `encode_fen`: own PNBRQK then enemy
PNBRQK, rank flip for black, own K/Q then opponent K/Q castling planes, raw EP
plane and clipped halfmove/100 plane. Raw EP includes ineffective and pinned
targets. Repetition hashing uses effective legal EP and has a different meaning.
The current compact decoder uses double division then float32 rounding for
clock parity. Reduced feature precision is a separate experiment; exact rule
storage never follows that precision change. Fullmove and history are omitted
from the 18-plane model input, while remaining part of simulator provenance.

The sink exports `parent_hash`, `child_hash`, `parent_node`, `child_node` and
`root_epoch` as int64 `[B]` sidecars. Hashes and epoch use uint64 bit-pattern
transport. The epoch advances after a successful reset and when generic calls
invalidate the arena; it must not silently wrap. Together with a host context
UUID, root-history provenance and schema
versions it scopes every node ID. An arena ID is not a global history handle.
Keep sidecar identity aligned with ring admissions, ideally using the ring's
monotonic `sequence_ids` and explicit ownership when rows are overwritten.
CompactReplay schema 1 itself does not store these sidecars; retaining them is
the adapter's separate responsibility.

The sink deliberately omits `target_actions`, `history_handles`,
`next_history_handles` and `game_ids`. The adapter constructs these fields on
the device from declared labels/provenance, then supplies exactly the compact
schema keys to admission. Sidecars are not silently added as replay fields.

## History, terminal labels and target provenance

Board/rule packets are insufficient to reconstruct repetition context. The
simulator owns the root's preceding hashes and each node's ancestry. Record
their origin and ordering before arena reset; copying a node ID into replay
does not preserve that ancestry. For later replay adjudication, an external
history owner must retain a bounded immutable root checkpoint/move-prefix or
equivalent history object until every row/sample reference has retired. Exact
history payload, lifetime, byte budget and collision-checking witness are
separate requirements from the neural feature schema. History-free mechanics
records use -1 handles and explicitly say history was not retained.

The inspected HIP `classify_kernel` treats no-legal-move positions as mate or
stalemate; in adjudicate=1 it also terminates on insufficient material,
halfmove>=100 or a current third hash repetition. It does **not** inspect a
future move to determine whether a claim could be made on the following ply.
Consequently, bridge terminal parity uses this current-position simulator
predicate, not Python's unrestricted `claim_draw=True` predicate. Declare the
policy in the manifest, including the initial hash-history collision assumption.
An empty supplied history does not certify that a played game's history was
empty. Terminal current positions belong to a separate inference API; compact
transition schema 1 requires a legal executed move.

The exporter produces exact observations and move identities, not training
truth. A separate target adapter states the algorithm and provenance for each
label. For a mechanical smoke, `target_actions = actions` is executed-action
imitation. It is not a best-move label or the existing corpus's tactical toy
teacher. A GPU material adapter can produce the documented current-mover
`tanh(material/8)` proxy; record piece weights and units. The first exporter
actually calls material evaluator 0 with centipawn weights 100/320/330/500/900/0
and its +/-29000 clamp, then divides by 800 before `tanhf`. Neither target is an
outcome distribution. If a resident minimax score/PV becomes a teacher, retain evaluator
ID, search depth, node budget, perspective, seed and model/source hash.

Actual game-outcome WDL, visits-based policies, delayed targets or missing labels
need a new versioned target interface with explicit validity/type and perspective.
Never encode an absent label as draw/zero or optimal-policy mass. Delayed writes
must target a surviving `(game_id, sequence_id, epoch)` and cannot mutate a
published snapshot without an ordered update protocol. Initial bridge training
and reports use mechanical/proxy terminology throughout.

## What a zero-host-copy result must establish

The first milestone is **zero host transfer of state payload during export and
replay sampling**. Payload includes board/rule arrays, hashes, legal support,
move/action arrays and per-row targets/identity. HIP gathers into Torch buffers
and replay copies those buffers into owned device storage: these are device
operations and must still be counted in bandwidth and latency reports.

Control transfers are reported separately. Rules ABI 1 already downloads
summary/error/count telemetry, and `CompactReplay.append` currently converts
validation reductions to Python booleans. Those scalar reads synchronize and
transfer device data. Therefore the current path cannot claim zero **all** D2H
bytes or asynchronous ingestion. Root/history CPU upload is a declared setup
cost. Repeated CPU root upload is not GPU self-play. A later strict no-scalar-read
admission path needs validated producer ownership and failure-atomic device
admission, rather than silently skipping validation.

Capture a bounded region from resident export through replay sample/decode,
without debug downloads. Use HIP/ROCm API and memory-copy tracing to classify
every H2D, D2H, D2D and default-direction operation by pointer type and purpose.
Retain exact bytes and call sites, including scalar validation and summary
reads. Instrument all state download/export fallbacks. A TorchDispatchMode audit
also rejects `.cpu()`, `.item()`, scalar extraction and CPU tensor conversions
in the sampled hot path, but cannot see arbitrary native HIP copies; it is
supporting evidence rather than the complete proof. Pinned/mapped host payload
is forbidden even if a kernel accesses it without a memcpy call.

Adversarially add a board `.cpu()` copy, an `.item()` reduction and a native
device-to-host fallback; the corresponding audits must fail. Perform oracle
downloads in a separately marked verification region after completion. Report
both regions, setup, barriers, export gather, admission validation/copy, decode,
model/optimizer time and peak bytes. An unprofiled run may report source-level
absence of state-download calls, not a traced exact-zero-byte measurement.

## Acceptance tests

| Concern | Required positive and negative evidence |
| --- | --- |
| Runtime binding | All five API identities match Torch's actual HIP runtime before context creation; deliberate foreign-runtime token mismatch rejects. Record actual bound library paths/hashes and separate standalone-system-runtime evidence |
| Exact payload | Independent CPU oracle compares every current/successor bitboard, raw EP, rights, clocks, support, executed action and preserved hash; both movers, all promotions, castling, EP, pinned EP, capture, check evasion and terminal successors |
| Boundary representation | Bit 63, signed high fullmove values, halfmove 99/100/65535, maximum support, action indices 0/20479, padding and overflow; never truncate |
| Draw semantics | Repetition with retained root history and ancestry, insufficient material, fifty-move boundary, mate/stalemate; prospective-claim fixture confirms the documented policy difference |
| Span/phase | Root without parent, out-of-range span, unexpanded or terminal parent, unavailable arena, perft mode and capacity failure reject without replay mutation; stale expected epoch and cross-context equal-epoch selectors reject, and unclassified/expanded legal successors export correctly |
| Pointer safety | Host/pinned/managed/wrong-device/null/misaligned/undersized/interior-overrun/overlapping buffers and noncontiguous views reject before launch; correct Torch suballocations pass |
| Lifetime/order | Write/read on a nondefault Torch stream before bundle reuse, immediate simulator reset after export, repeated ring wrap and garbage collection; exported/gathered snapshots stay unchanged |
| Failure | Inject preflight/kernel failures and a Torch exception after queued source reads; completion fences run on exceptions, discard output, ring size/cursor/sequence IDs unchanged for preflight failures; runtime faults require recreation |
| Training parity | Identical sample identities and decoded tensors produce the same losses/gradients as an oracle-packed record stream; toy labels remain named and scoped |
| Copy audit | Native trace plus Torch audit reports zero state-payload host bytes within the declared region, with known scalar-control bytes listed separately; deliberate violations are detected |

Tests must run on the actual gfx1030 with the installed Torch/HIP runtime and
compiled library hashes captured together. CPU-only tests establish schema and
error handling but cannot certify pointer interoperability or stream ordering.
Measure several repeated bounded workloads before making throughput claims;
successful ingress alone says nothing about action dynamics or chess strength.

## Optional later asynchronous/DLPack seam

DLPack is useful only if a later experiment needs shared externally owned
snapshots and overlap. It introduces ownership obligations absent from the
first Torch-owned sink. A future pool snapshot must be immutable, bounded and
independent of the mutable arena. Publication records a producer-ready event;
the consumer stream waits before reads. After replay copies, a consumer-done
event fences pool reuse. Reclaim only after all descriptor references retire
and the last consumer fence completes. Capsule destruction alone does not
prove queued GPU reads are finished. HIP documents event record/wait ordering
in its [asynchronous guide](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/how-to/hip_runtime_api/asynchronous.html).

Use an object implementing `__dlpack_device__` and the full keyword signature
`__dlpack__(stream=None, max_version=None, dl_device=None, copy=None)`. Return
ROCm device type `kDLROCM`, native signed-int64/int32/float32/bool descriptors,
explicit element strides and correctly owned shape/stride arrays. The installed
PyTorch 2.10 protocol requests version 1.0 and forwards the current stream;
pin that behavior rather than assuming latest DLPack APIs. Import on the same
device with `copy=False`; pointer equality confirms descriptor sharing, while
earlier snapshot gathering is still a device copy. [PyTorch 2.10 implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/torch/utils/dlpack.py),
[DLPack C API](https://dmlc.github.io/dlpack/latest/c_api.html).

ROCm stream 0 is its default stream; CUDA's special integers 1 and 2 do not
apply. Reject unsupported stream modes explicitly, including no-synchronization
requests when readiness cannot be guaranteed. Cross-device/copy-required export
with `copy=False` raises `BufferError`. Do not raise `TypeError` for real
validation failures: consumers may interpret that as an unsupported keyword
and retry with reduced arguments. [Array API DLPack protocol](https://data-apis.org/array-api/latest/API_specification/generated/array_api.array.__dlpack__.html).

Each capsule is consumed once and has one managed-descriptor deleter; create a
fresh descriptor/reference for another import. Descriptor ownership transfers
to the consumer, while the producer retains allocation ownership through its
reference-counted pool lease. The lease retains the HIP context/runtime/library
needed to release memory even after the original simulator object closes.
Unconsumed capsules release their descriptor reference too. [DLPack Python interchange specification](https://dmlc.github.io/dlpack/latest/python_spec.html).

Treat imported tensors as read-only by application contract: Torch shares
storage and permits in-place operations, and the DLPack read-only flag does not
universally enforce immutability. `record_stream` documents Torch allocator
reuse ordering, but a HIP-owned pool needs its own fences and lease protocol;
recording a stream is not a substitute for that external ownership agreement.
[PyTorch from_dlpack](https://docs.pytorch.org/docs/2.10/generated/torch.from_dlpack.html),
[PyTorch record_stream](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.record_stream.html).

Before adopting this alternative, add double-consume/deleter tests, failed import,
early Python destruction, consumer side-stream reads, pool exhaustion and arena
reset stress. The synchronous Torch-owned contract remains the comparison
baseline so lifetime complexity and actual overlap gains can be measured.
