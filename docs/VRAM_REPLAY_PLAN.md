# Resident replay scaling on gfx1030

The next priority is to connect exact state production to compact resident replay
and measure the entire learning/search pipeline. `DeviceReplay.sample()` already
gathers minibatches on the device. Its producer is still CPU python-chess, and
the Rust HIP evaluator currently accepts host slices and returns host scores.
Neither interface yet produces exact legal experience directly into VRAM.
This document proposes the next experiment; its capacities and throughput gains
are not measured results.

## Current allocations and a first correction

Schema 1 allocates **29,733 bytes per record** in the ring: 9,216 bytes for two
float32 18×8×8 features, 20,480 bytes for the current bool legal mask, and 37 bytes
for actions, proxy label, terminal flag, game id and sequence id. These are tensor
payload bytes; allocator overhead, alignment, temporary buffers and provenance
objects are separate.

`train.run_comparison` currently retains the full device corpus after copying it
into the ring. The corpus costs 29,725 bytes per record, without ring sequence
ids. Thus this smoke pipeline retains **59,458 bytes per generated record**,
before minibatches, models or activations: 100,000 records would occupy about
5.54 GiB, versus 2.77 GiB for the ring alone. Warm-up slices also keep their
parent corpus allocation alive. Before interpreting a capacity benchmark,
either produce directly into replay or obtain warm-up rows from replay and
release all corpus tensor references after ingress. Keep CPU provenance and
fingerprints without retaining a second full device dataset.

The card has 16 GiB total physical VRAM, shared with other allocations. Derive
replay capacity from observed free memory after reserving model/optimizer,
training activations, search arenas, resident legal-generation queues and a
safety margin. Do not size replay from total memory alone. Measure driver free
memory as well as Torch allocated/reserved peaks: Rust HIP allocations and
external tensor owners may lie outside Torch allocator accounting.

For float32 Adam-style training, parameters, gradients and two moment tensors
normally contribute roughly 16 bytes per learned parameter before temporary
optimizer storage. Mixed precision may add master weights and different moment
types; inspect actual storage instead of assuming it halves that total.
Model geometry buffers, allocator caches and graph capture also consume memory.

A 64×64 float32 latent uses 16,384 bytes. Fifty thousand such search latents
would use about 0.76 GiB before edges, exact boards or masks. A 512-row float32
20,480-action policy uses 40 MiB per dense logits tensor; masking and backward
can retain additional tensors. A compact replay therefore helps much less if
each sampled row immediately expands an enormous policy or search frontier.

## Representation candidates

All candidates preserve the exact action vocabulary: `((from*64+to)*5+promo)`,
with promotion none, N, B, R, Q. No candidate may truncate the legal set or
silently drop an underpromotion.

| Variant | Per-record payload before history/extra strength labels | Capacity in an illustrative 8 GiB replay budget | Main tradeoff |
| --- | ---: | ---: | --- |
| Current float32 features + bool mask | 29,733 B | 288,902 | Simple gather; dominates storage and read bandwidth |
| Same features + packed mask | 11,813 B | 727,159 | Cheap first compression; mask decode still materializes dense policy support |
| Compact state pair + packed mask | 2,829 B | 3,036,385 | Device feature decoding; legal vocabulary stays fixed |
| Compact state pair + 256-slot sparse action list | 789 B | 10,887,116 | Small fixed gather; requires explicit overflow handling |
| Compact state pair + sparse arena at mean 32 actions | 341 B | 25,190,424 | Lowest illustrative payload; offsets/lifetime/fragmentation complexity |

These are arithmetic comparisons, not target capacities. Compact rows below
include two 108-byte exact-state payloads, the existing 37 bytes of transition
metadata, two 8-byte history references, and legal support storage. The sparse
examples allocate an 8-byte offset/count descriptor plus two bytes per action.
They exclude the history arena itself, allocation padding, generation metadata,
teacher visit distributions and game outcomes. Mean 32 is an experimental input,
not an observed corpus statistic. A real policy target with float32 weight per
legal action adds four bytes per action; the mean-32 example then adds 128 B.

Packed masks store 20,480 bits as 640 uint32 words: exactly 2,560 bytes. A device
decoder can materialize a bool minibatch mask, or the policy loss can inspect
packed support directly. Compare both; unpacking without changing the full
dense decoder may save replay storage while leaving training throughput nearly
unchanged. Prove every action's bit, including word boundaries and promotions.

Sparse support stores each canonical legal action in int16 (20,479 fits), with
count/offset and a defined ordering. Convert to int64 only for operations that
require it. A legal-only decoder can evaluate endpoint pair factors and
promotion terms directly for listed actions, avoiding the 20,480-wide logits.
The current model's factorized heads make this plausible, but the path must
produce the same legal logits and normalization as dense decoding. Policy
target visits must share the action ordering and cannot be confused with slot
indices. Terminal rows have count zero and bypass policy sampling.

The 256-slot choice is an initial storage capacity, **not** a theorem about the
maximum legal moves in every admitted state. The producer checks the actual
count. Overflow must use a measured secondary arena or reject the whole record
with an explicit counter; clipping a move list is invalid. Fixed-stride and CSR
arenas should be compared with the same exact corpus and legal-count histogram.
Arena offsets alone are unsafe after reuse: carry generations and pin their
allocations until all sampled readers complete.

## Compact exact states and feature decoding

Proposed state payload, stored as structure-of-arrays:

| Field | Payload |
| --- | ---: |
| White P/N/B/R/Q/K then Black P/N/B/R/Q/K bitboards | 96 B |
| Absolute side, castling rights, raw EP square, effective EP square | 4 B |
| Exact halfmove clock and fullmove number, uint32 each | 8 B |

Total is 108 B per state before history reference and allocator alignment. This
uses **twelve 64-bit bitboards**, not twelve total bits. Standard chess only is
the initial rules domain. An ABI AoS layout may require padding; assert its
actual size rather than assuming the SoA payload calculation is its C size.

The existing `ChenPosition` evaluator ABI is 104 bytes and intentionally omits
castling, EP, clocks and history. It cannot become exact replay state by treating
reserved padding as rule fields. Introduce an explicitly versioned resident
state/experience ABI and keep evaluation-data and exact-state contracts clear.

Decode schema-1 features on the consumer stream after gathering compact rows:

1. Expand the twelve bitboards into one-hot piece planes.
2. For Black, flip ranks with square xor 56, swap absolute piece colors into
   own/enemy order, and reorder castling rights as own K/Q then enemy K/Q.
   A rank flip preserves files; bit reversal would incorrectly flip files too.
3. Plane 16 uses the **raw FEN EP target** or no target, matching current replay
   and diamond supervision. Keep effective legal EP separately for position
   identity, move generation and repetition. Determining effective EP requires
   actual king-safe legality, not merely an adjacent pawn.
4. Plane 17 is `min(exact_halfmove/100, 1)` in the requested float dtype. Retain
   the unclipped exact clock for adjudication. Fullmove is retained for protocol
   reconstruction/provenance even though schema 1 does not encode it.
5. Decode successors using the successor's side. The stored transition action
   uses the current state's side; each rollout action uses that ply's side.

Reject out-of-range packed fields and overlapping piece bitboards before
publishing a valid record. If source clocks exceed the uint32 ABI range, reject
or move to a versioned wider format; saturating exact rule state is invalid.
Mixed-precision feature decoding is a separate accuracy experiment. Compare
float32 outputs exactly against the CPU encoder, including Black orientation,
EP, rights and clocks, then set explicit tolerances for reduced precision.

Repetition is not reconstructible from board plus clocks. Store a reference to
an immutable game-history prefix or a generation-protected resident history
arena. Retain actual prior position identities and the moves/checkpoints needed
for exact verification. A count for the current position alone cannot answer
all possible next-move repetition claims. Any pruning at an irreversible
transition must be justified by the exact rules and tested against complete
history. Hash matches need an exact-state comparison path; a finite hash alone
is not a mathematical equality proof.

The neural history extension needs its own schema version. It may encode
repetition counts or a bounded history sequence, while exact adjudication still
uses full permitted context. Two quiet diamond endpoints share board/rule
features and safe *current* draw flags but retain different future repetition
maps; do not merge their exact history handles because their latents are close.

## Producer, replay and consumer ownership

The initial producer can be CPU Rust exact chess rules writing compact shards
through one explicit ingress transfer. This establishes format/oracle parity
without pretending the generator is already resident. The target producer is
exact GPU legal move generation plus self-play/search, publishing experience
into the same resident interface. Compare them as separate producer modes.

Proposed capability boundaries are `ExperienceProducer`, `ReplayStorage`,
`FeatureDecoder` and `LegalPolicyDecoder`. Keep the current host
`BatchEvaluator` for its measured CPU/GPU evaluation experiments. A future
resident capability returns owned device leases and completion events rather
than accepting host position slices and promising residency indirectly.

An initial same-process protocol can use these operations:

| Operation | Contract |
| --- | --- |
| `reserve(count)` | Return writable row range, generation and bounded device buffers; never silently reallocate |
| `publish(write_lease, ready_event, metadata)` | Publish only fully written/validated rows; metadata includes schema, rules/producer generation and label provenance |
| `sample(count, rng_state, consumer_stream)` | Return owned gathered compact minibatch or a pinned read lease with its generations |
| `decode(read_lease, feature_workspace, consumer_stream)` | Expand features/support entirely on device after the producer event |
| `release(read_lease, completed_event)` | Permit overwrite only after all consumer kernels have finished reading |

Begin with one producer and one ordered stream, matching the current ring's
supported ownership. For two streams, enqueue explicit event dependencies;
[HIP streams do not automatically order separate streams](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/how-to/hip_runtime_api/asynchronous.html).
Generation checks alone cannot prevent torn reads if a writer overwrites during
a gather. Pin rows until the gather completes or use separately published
immutable slots with retries on version mismatch. Publication/overwrite states
should be visible to the implementation, not inferred from Python object life.

[Torch DLPack import shares the producer allocation](https://docs.pytorch.org/docs/2.10/generated/torch.from_dlpack.html),
so a same-process bridge can avoid a host data round trip. Its owner/deleter
must outlive imported tensors, report the correct ROCm device and dtype/shape,
and honor stream synchronization. Share once under the supported protocol;
do not fabricate a borrowed tensor from an integer pointer with no allocation
lifetime. Verify the installed Torch/HIP versions and actual pointer ownership
experimentally before adopting this bridge. A device-to-device gather/copy may
remain the simplest safe minibatch snapshot and should be measured honestly.

Capacity exhaustion returns a defined choice: overwrite eligible oldest rows,
backpressure the producer, or reject the whole new record with a counter.
Never overwrite a pinned row, dangling history/list arena, or in-flight
inference buffer. Sequence ids stay monotonic across clear/reuse; epoch rollover
is explicit before integer overflow. A completed game's outcome may arrive
after early rows were evicted. Apply delayed labels only when both game id and
row generation still match, and count missed updates; otherwise the result can
corrupt a replacement game's targets.

## Oracle and concurrency gates

Before performance measurement, replay a frozen set of legal positions and
trajectories through CPU Rust, independent python-chess and the proposed device
producer/decoder. Include captures, checks, pinned EP, both castles, lost
castling rights, all promotions/underpromotions, double pushes with effective
and ineffective EP, clocks near claim/automatic-draw boundaries, repeated
histories, terminal roots and color/rank symmetry. Compare exact action sets,
canonical action round trips, successor state and termination context, not just
the chosen move or policy sum.

Exercise every mask bit boundary and sparse overflow path. Use small artificial
capacities to force record/list/history wraparound; check sequence generations,
delayed-result updates, clear behavior, rejection before publication, and
sampled-batch survival after overwrite. A test that merely sees `device=cuda`
does not prove residency: use dispatch audits and a HIP trace to distinguish
H2D/D2H data movement from device gathers and small telemetry reads. Stress
cross-stream consumers with deliberate delay before releasing leases. Reject
torn or stale batches rather than quietly training on them.

## Falsifiable end-to-end experiment

Freeze one legal corpus, its source/position/action hashes, the same uniform
index schedule, model initialization, optimizer, feature precision and label
source. First compare the dense ring against packed masks with unchanged
models. Then compare compact-state decoding with packed support, and finally
legal-only sparse policy decoding. Hold the usable replay byte budget fixed
for a capacity comparison and the record count fixed for a throughput comparison.
Measure batch sizes 1, 16, 128 and 512, plus the actual search queue histogram.

Run both bounded CPU ingress and an independently verified resident producer.
Include legal generation, label generation/search, publication, sample/gather,
feature/support decode, forward/backward and optimizer step in wall time. Keep
completed games per hour and training positions per second separate: a fast
learner with a starved or strategically weak producer is not a strong engine.
For streaming ingestion, report producer/consumer rates, queue occupancy,
backpressure, retained data age and evictions. Keep telemetry transfers small,
named and outside the claimed resident data path.

After warm-up and optimizer-state allocation, record replicated wall times,
P50/P95 batch latency, phase device events, actual H2D/D2H bytes, peak driver
and Torch memory, decoded workspace size, fragmentation, and search latency
while training runs. Archive the source hashes, hardware/runtime/compiler
versions and trace. Validate identical logits/losses/gradients for the lossless
representations before comparing speed. Reduced precision receives a separate
numerical and paired-game gate.

Predeclare these proposed acceptance tests: packed masks reduce fixed payload
as calculated without semantic changes; compact states reduce stored board
payload without encoder disagreement; sparse decoding reproduces legal logits,
normalization and targets without overflow loss; the chosen end-to-end variant
retains at least 95% of baseline training throughput at fixed record count while
increasing usable capacity, or improves throughput with a declared memory cost.
Reject a “win” based only on fewer host copies when decode/search contention
reduces overall completed-game/learning throughput. Thresholds are experiment
design choices to freeze before execution, not existing measured achievements.

Promotion to engine training also requires independent held-out task results
and paired offline match evidence from [NEURAL_TRAINING.md](NEURAL_TRAINING.md).
Residency, storage capacity and chess strength remain separate measured claims.
