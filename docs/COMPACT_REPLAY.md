# Compact resident replay, version 1

`neural.compact.CompactReplay` keeps exact current/successor bitboards, rule
metadata and sparse legal actions in device tensors. Sampling gathers owned
snapshots and decodes schema-1 features and a dense legal mask on the device;
`neural.train.loss_terms` and `train_step` consume the resulting batch directly.
CPU ingestion is explicit and happens once. This implementation has no direct
HIP producer bridge yet.

## Storage and producer contract

All fields have a leading capacity dimension. The default legal bound is 256.

| Fields per record | Representation | Bytes |
| --- | --- | ---: |
| `pieces`, `next_pieces` | Twelve signed int64 bit patterns each | 192 |
| `state`, `next_state` | Five int32 columns each | 40 |
| `history_handles`, `next_history_handles` | Opaque int64 IDs | 16 |
| `legal_actions`, `legal_counts` | 256 int16 indices and int32 count | 516 |
| `actions`, `target_actions`, `game_ids` | int64 labels/identity | 24 |
| `material_proxy`, `terminal` | float32, bool | 5 |
| Ring-owned `sequence_ids` | int64 monotonic admission identity | 8 |
| **Total** | | **801** |

Bitboards use absolute white PNBRQK then black PNBRQK order, A1=0. Signed storage
preserves all 64 bits, including bit 63. State columns are side (0 white/1 black),
castling (WK/WQ/BK/BQ bits 1/2/4/8), raw EP square (-1 absent), halfmove and
fullmove. Clock columns transport uint32 bit patterns: halfmove is bounded to
0..65535, matching Rust's u16 counter; fullmove supports 1..4294967295. Interpret
the columns with `unsigned_clock`, never a signed fullmove comparison. Exact
unclipped clocks remain resident separately from the clipped neural feature.

These values align with the rules arena's 128-byte `ChenRulePosition` packet.
That packet additionally carries reserved-zero and exact simulator hash fields;
the replay uses separate current/next SoA tensors and does not store the hash.
A future bridge must explicitly transpose the packet into these tensors and
canonicalize legal raw moves into schema-1 action indices. It must prove
device/context compatibility, allocation lifetime and producer-to-consumer
stream ordering before exposing borrowed memory.

The action index is `((canonical_from*64 + canonical_to)*5 + promotion)`, with
promotion 0 none, 1 knight, 2 bishop, 3 rook, 4 queen. Active sparse entries must
be sorted, unique indices in 0..20479, followed by -1 padding. Counts exceeding
the configured bound fail; actions are never truncated. Admission requires at
least one current legal action and both executed and target actions in that
support. Terminal successors are allowed. `decode_legal_mask` also handles
zero-count terminal supports as all false, although terminal current rows are
not trainable transition records in this schema.

The label names and meanings match the existing synthetic corpus: `terminal`
describes the successor; `material_proxy` is a rule-derived toy material target,
not a game outcome or calibrated WDL probability. The target policy uses the
same documented tactical toy heuristic. `game_ids` identify source trajectories;
sequence IDs identify ring admissions, including repeated source records.

## Exact decoding and ownership

For black to move, the decoder flips ranks with square xor 56, swaps own/enemy
piece planes, and orders castling rights own K/Q then enemy K/Q. Files remain
unchanged. The EP plane preserves the **raw** FEN target, including an ineffective
or pinned EP opportunity. Effective EP used in repetition hashing has a different
meaning and must not replace this field. Halfmove clipping reproduces Python's
double-precision division before float32 rounding, then converts to an optional
float16/bfloat16 output; this avoids reciprocal-rounding differences on GPU.

Admission validates tensor keys, shape, dtype, device, disjoint occupancy, one
king per color, metadata bounds, EP rank, sparse support, label membership and
finite proxy targets. These checks synchronize at ingestion. They do not prove
reachability, all Rust local FEN invariants, legal move generation, successor
correctness or terminal adjudication. The exact simulator/validated producer
owns those proofs; generated data tests compare them with python-chess.

Append validates every supplied record before changing the ring. Oversized
valid writes retain the newest capacity records and still advance the sequence
counter by the entire supplied length. Capacity is fixed; a configured byte
budget fails before storage allocation. Sequence int64 exhaustion fails before
mutation and requires a new explicit epoch. `clear` retains allocations and
does not reuse sequence IDs. `sample_compact` performs independent gathers, so
later overwrite or clear cannot mutate a sampled batch. It is an owned snapshot,
not a borrowed arena lease.

The ring is single-owner and its operations require an ordered stream. There is
no concurrent producer/consumer protocol, external stream event handshake, or
delayed-label update API. History handles (-1 absent, otherwise opaque IDs) are
preserved through gathering, but the caller owns their lifetime and provenance.
The 18-plane model input does not include repetition history; these handles do
not make the neural input Markov-complete.

## Measured gfx1030 comparison

The final [machine-readable report](../neural/evidence/compact-gfx1030-2026-09-30.json)
and [console log](../neural/evidence/compact-gfx1030-2026-09-30.log) record source
hashes, provenance, seed and schedule. The
[initial report](../neural/evidence/compact-gfx1030-initial-2026-09-30.json) is
preserved separately. Runtime was Torch 2.10.0+rocm7.0 / Torch HIP 7.0.51831 on
actual gfx1030; this is distinct from the separately installed HIP compiler.

The benchmark uses 192 legal CPU-oracle transition records, seed 20260930,
max_plies 48, tiled into 65,536 resident rows. Tiling exercises capacity mechanics,
not corpus diversity. Only one capacity-sized device replay exists at a time;
the device ingress staging copy remains bounded to the 192-row source and is
deleted before residency measurement. CPU generation/conversion and actual
source H2D plus ring-fill time are reported separately. Models are relation,
width 16, depth 1, AdamW lr 0.0003, batch 128, 16 measured steps. Sampling uses
100 measured repeats after warmup. Both modes reset model/optimizer and sampler
seeds identically; initial optimizer allocation remains inside measured training.

| Final repeat | Dense | Compact |
| --- | ---: | ---: |
| Replay payload per row | 29,733 B | 801 B |
| Replay payload, 65,536 rows | 1,948,581,888 B | 52,494,336 B |
| Baseline Torch allocated / reserved | 0 / 0 B | 67,108,864 / 67,108,864 B |
| Resident Torch allocated | 1,948,581,888 B | 119,603,200 B |
| Resident minus baseline allocated | 1,948,581,888 B | 52,494,336 B |
| Resident Torch reserved | 1,973,420,032 B | 127,926,272 B |
| Peak Torch allocated during measured training | 2,123,039,232 B | 227,475,968 B |
| Torch reserved after training | 2,132,803,584 B | 241,172,480 B |
| After cleanup allocated / reserved | 67,108,864 / 67,108,864 B | 67,108,864 / 67,108,864 B |
| Source ingress and ring fill | 0.519 s | 0.622 s |
| Sample + decode per batch | 99.0 us | 922.5 us |
| Short measured training | 9,518 positions/s | 9,134 positions/s |

The payload reduction is 37.1x. The surviving 64 MiB allocation belongs to
process/runtime state after dense training, rather than replay storage. Its exact
allocator owner has not been identified. Driver free/total snapshots are also
recorded; they include unrelated processes and driver accounting and are not
interpreted as replay bytes. Default cache cleanup does not remove all persistent
runtime allocations.

Dense and compact have identical sampled-sequence hashes and identical first
and final loss dictionaries in both runs. Sampling/decode is about 9.3x slower
for compact in the final repeat, while end-to-end short training is 96.0% of
dense throughput; the initial run was 99.3%. These are two short smoke timings,
not statistical steady-state estimates or evidence of strength. Dense legal
masks and all 20,480 policy logits are still materialized per sampled minibatch.
Compilation, fusion and a sparse policy/loss interface remain experiments.

## Reproduction and tests

Use a Torch/ROCm environment with the optional python-chess development oracle
installed (the captured run used chess 1.11.2). Torch spells ROCm devices `cuda`.

```sh
CHEN_TEST_DEVICE=cpu python -m unittest discover -s neural/tests -p test_compact.py -v
CHEN_TEST_DEVICE=cuda python -m unittest discover -s neural/tests -p test_compact.py -v
python -m neural.compact --device cuda --capacity 65536 --source-samples 192 \
  --batch-size 128 --repeats 100 --steps 16 --seed 20260930 \
  --output results/neural/compact-gfx1030.json
```

All 12 compact tests passed on CPU and gfx1030; captured
[CPU](../neural/evidence/compact-cpu-tests-2026-09-30.log) and
[GPU](../neural/evidence/compact-gpu-tests-2026-09-30.log) logs are retained.
They exercise wraparound/oversized writes, failure atomicity, clock uint32
bit patterns, immutable gathers, clear/ID lifecycle, sparse action boundaries,
raw EP/castling/promotion/color symmetry, generated exact successor and legal
support against the oracle, float32/16/bfloat16 decoding, and all three existing
models' scores/losses/gradients against dense replay. TorchDispatchMode rejects
scalar host extraction and device-to-CPU copy in the sampled decode path.
The final source additionally passed a CPU rerun and the actual GPU benchmark
after adding reporting-only baseline/cleanup fields.

The next producer experiment should feed distinct legal simulator transitions
directly from an event-owned HIP device arena, retain exact repetition-history
provenance, and compare whole producer-to-optimizer throughput and peak memory
over several capacities. Admission validation and stream synchronization belong
in that timing. Require byte-exact decoded features/support/successors plus
identical sampled losses before optimizing them. Report kernel/decode, staging,
producer, allocation and training time separately, and measure repeated runs;
a host-copy microbenchmark alone cannot establish an end-to-end gain.
