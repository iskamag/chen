# Chen GPU research: gfx1030

Status: executable HIP batched evaluation, a CPU-rules frontier search integration,
and a research plan for device-resident search/training. Full GPU chess rules,
resident self-play, trained inference, and the requested 2500 Lichess target are
unfinished. Evaluation throughput is one component of engine strength.

## Actual machine and build

On 2026-09-30 the existing `cpp-rocm` container executed Chen kernels compiled
with `/usr/bin/hipcc -O3 -std=c++17 --offload-arch=gfx1030`. The compiler identifies
itself as HIP 7.2.53211 / AMD clang 22 / ROCm 7.2.4. No architecture override is
present; the library rejects `HSA_OVERRIDE_GFX_VERSION`.

| Query | Observed result |
|---|---|
| HIP device name / architecture | AMD Radeon RX 6800 / gfx1030 |
| HIP total memory | 17,163,091,968 bytes |
| HIP wave size | 32 |
| HIP multiprocessors | 30 |
| rocprofv3 physical CUs / SIMDs | 60 / 120 |
| Physical PCI vendor / device | 0x1002 / 0x73bf |
| HIP async engines | 8 |
| rocprofv3 / runtime version | 1.1.0 / 70253211 |

HIP's reported multiprocessor count is 30 work-group processors on this RDNA2
device; the profiler's agent table reports 60 physical CUs. Preserve the raw
counts and their query names in benchmark metadata. RDNA's native wave size is
32; AMD warns that forcing its experimental wave64 option is unsupported by the
HIP runtime. Kernel launch sizes are multiples of 32.
[AMD hardware features](https://rocm.docs.amd.com/projects/HIP/en/develop/reference/hardware_features.html)

`gpu/evidence/gfx1030-agent-info-2026-09-30.csv` records physical agent discovery;
`gpu/evidence/gfx1030-kernel-trace-2026-09-30.csv` records real device dispatches.
The trace contains Chen's `evaluate_kernel` with 24 VGPRs, zero scratch, and zero
LDS allocation. The compiler retained one kernel with an evaluator selector.

## Reproduce the baseline

```sh
scripts/gpu_probe.sh --max-batch 65536 --repetitions 9
```

The script copies only `gpu/` into a unique `/tmp/chen-gpu.*` directory inside the
running container, builds, executes, then removes that temporary directory. Set
`CHEN_ROCM_CONTAINER` to select a different existing container. It does not
modify `/work` or the user's mounted AI files. No Python ML framework is required
for this probe.

For a persistent local build in a ROCm environment:

```sh
gpu/build.sh /absolute/path/libchen_gpu.so
python3 gpu/probe.py /absolute/path/libchen_gpu.so
```

The C ABI in `gpu/chen_gpu.h` is version 1. A context preallocates positions,
scores, pinned host staging, a nonblocking stream, and four events. Every
evaluation finishes before returning. Calls on one context must be serialized;
an exclusively owned context can move to another host thread because evaluation
and destruction select its device. Keep the dynamic library loaded until after
context destruction.

The 104-byte packet contains 12 bitboards, side to move, and a reserved word.
The order is white pawn/knight/bishop/rook/queen/king, then black in the same
order; A1 is bit 0. Scores are integer centipawns from the side to move.
`Material` and `Tapered` match `src/eval.rs`, including C++/Rust division toward
zero. Capacity overflow, invalid side/reserved fields, and unknown evaluator
selectors return errors. Empty batches succeed. There is no automatic CPU
fallback, reallocation on the hot path, or architecture spoofing.

This packet describes evaluation inputs. GPU move generation must use a richer
state with castling, en passant, halfmove clock, and repetition history.
Aliasing an evaluation packet with complete chess state would silently corrupt
rule semantics.

## Measured scope and implications

The deterministic standalone oracle checked 4096 disjoint, varied bitboard
packets against each of the two integer evaluators. Five boundary cases passed.
These packets cover arithmetic and geometry, including unusual material and
phase saturation; they are deliberately broader than legal chess positions.
The Rust integration must separately compare legal positions and exercise
search/cancellation. The oracle alone does not validate search or chess rules.

The saved JSONL sweep uses a warmup and nine repetitions per batch size. Each
reported statistic is the median of that metric. `wall_ms` measures the ctypes
call, host validation/staging copies, transfers, launch, event extraction, and
completion. It excludes caller position packing, model training, and search.
The event spans are H2D interval, kernel interval, and D2H interval; enqueue
gaps and launch overhead can be inside those spans. They should not be read as
isolated shader instruction time. Dispatch timestamps from a profiler answer a
different question and perturb execution. The separate profile run is not used
for the throughput table.

| Tapered batch | Call wall time | H2D span | Kernel span | D2H span | Positions/s |
|---|---:|---:|---:|---:|---:|
| 1 | 44.52 µs | 9.20 µs | 10.16 µs | 8.88 µs | 22,462 |
| 128 | 52.76 µs | 9.32 µs | 18.20 µs | 8.76 µs | 2.43M |
| 2048 | 70.86 µs | 18.88 µs | 19.04 µs | 9.00 µs | 28.9M |
| 8192 | 124.20 µs | 46.64 µs | 19.84 µs | 11.44 µs | 66.0M |
| 32768 | 326.62 µs | 145.04 µs | 29.00 µs | 15.04 µs | 100.3M |
| 65536 | 736.54 µs | 279.00 µs | 83.40 µs | 19.48 µs | 89.0M |

These are initial machine-local observations, not a stable speed record or a
CPU/GPU speedup claim. The largest batch loses throughput in this run. Resolve
variance with repeated independent sweeps, hot/cold device state, and profiling
before optimizing from it. Store all samples in later benchmark revisions and
report tail latency as well as medians. Raw initial measurements are in
`gpu/evidence/gfx1030-eval-2026-09-30.jsonl`.

The immediate design conclusion is concrete: calling the GPU once per leaf in
the existing depth-first alpha-beta search would pay roughly 45 µs per call
before doing meaningful search work. The batch interface and a search scheduler
that produces many useful leaves are essential. Cheap handcrafted evaluation
is primarily a correctness and transport oracle; it is too small to predict
neural arithmetic throughput.

## Literature and architecture candidates

**CPU principal variation plus queued GPU siblings.** Alpha-beta benefits from
ordering and early bounds; letting GPU work race ahead can multiply useless
nodes. Strnad and Guid's GPU alpha-beta study uses Reversi and finds that small
boards lack enough parallel work, while larger boards benefit. It supports
measuring parallel slack, not assuming that chess inherits a speedup.
[Parallel Alpha-Beta Algorithm on the GPU](https://hrcak.srce.hr/77340)

Our experiment: search the first child to a useful bound, enqueue bounded sibling
subtrees, and report both effective root work and speculative work discarded
after cutoffs. Sweep queue depth and subtree horizon. Keep exact minimax tests
for shallow trees. This hybrid is useful if low-latency tactical paths coexist
with enough parallel sibling work; it can remain CPU-heavy.

**Batched PUCT with inference caching.** Cazenave separates cached network
inferences from path-specific search statistics, and uses Virtual Mean during
pending descents to discourage duplicate work while preserving an edge's mean.
His Go/MobileNet experiments demonstrate that useful batch occupancy and
selection quality both matter; larger batches change the search's information
schedule. The reported RTX2080Ti/TensorFlow throughput is not a gfx1030 result.
[Batch Monte Carlo Tree Search](https://arxiv.org/html/2104.04278v1)

Our experiment: tag leaf requests with a ticket, search generation, model hash,
and full rule-state key; deduplicate pending inferences while preserving each
path's backup. Compare Virtual Mean and loss at equal wall time and equal useful
evaluations. Measure duplicate selection, pending time, cache reuse, tactical
accuracy, and match score. A cache keyed only by board hash is insufficient when
clock/history features change the network input.

**Prior-budgeted breadth-first search.** Frankston and Howard's 2026 RMCTS
allocates descendant budget using priors, batches each depth, and computes a
locally regularized posterior policy bottom-up. It trades adaptive expansion for
larger batches. Their speed/quality experiments cover Connect-4, Dots-and-Boxes,
and Othello with TensorRT on RTX3080. There is no chess or AMD measurement in
that evidence.
[Accelerating Monte-Carlo Tree Search with Optimized Posterior Policies](https://arxiv.org/html/2601.01301v2)

Our experiment: implement a faithful reference next to frontier minimax, then
add a separate scheduler using uncertainty and tactical volatility to refill a
bounded frontier. Equalize learned model, root budgets, and training data before
claiming a scheduling gain. Incorrect priors can omit a forcing reply; reserve
an explicit verification budget for forcing lines and compare with a reference
search. The uncertainty scheduler is Chen's proposed hypothesis, not a result
proven by RMCTS.

**Device-resident frontier and many games.** Move generation, move application,
terminal detection, evaluation, and backup can run as bulk kernels over a
preallocated arena. A single game often has narrow early frontiers; hundreds of
games give independent roots without distorting each game's move choice to fill
a batch. This is the most direct route to producing replay data in VRAM. The
first implementation should use bounded bulk stages; a persistent queue kernel
comes after rules parity and overflow semantics are trustworthy.

## Module boundaries for experiments

Keep chess semantics in one tested Rust reference, with GPU kernels checked
against it. The chosen modules are:

| Module | Owns | Experimental replacements |
|---|---|---|
| Rule state | board, legality, history, terminals | GPU bulk rules using the same semantics |
| Search scheduler | expansion, tickets, budgets, backup | frontier minimax, PUCT, posterior frontier, hybrid bounds |
| Batch evaluator | values/policies/uncertainty and timing | CPU classical, HIP classical, neural inference |
| GPU transport/arena | streams, capacity, buffers, lifetimes | AoS/SoA, graphs, resident replay, persistent queue |
| Training/replay | example ownership, model versions, sampling | learned latents, planning auxiliaries, curriculum |
| Evaluation harness | frozen conditions and adjudication | matches, tactics, calibration, throughput |

A evaluator replacement must not implicitly change the search, data split, or
time control. A scheduler replacement must use the same model and legal action
space for a fair comparison. Evaluation and training have independent arenas;
model replacement publishes a new immutable version after outstanding tickets
for the old version are resolved or discarded.

### Proposed resident arena contract

This is a next-stage design, not ABI-1 functionality. Use a separate search
packet, initially 128 bytes: twelve bitboards, a 64-bit hash, a 64-bit history
handle, and four 32-bit fields for side/castling/en-passant/halfmove clock.
History handles reference a bounded parent-linked history arena; hash equality
is checked against the relevant state when necessary. Fullmove numbering is
host notation metadata. Draw semantics must match the harness's declared rules.

Allocate node, edge, history, frontier, score/policy, and replay arrays up front.
The producer first counts each node's legal children; a prefix sum assigns
disjoint edge ranges; expansion writes only within committed ranges. Overflow
returns an explicit partial status and the last fully completed iteration;
never truncate legal moves silently. Cancellation discards an unfinished
generation. Node IDs pair arena index with generation to reject stale tickets.

Device-resident replay uses fixed-size board/metadata records plus variable
policy ranges, priority, game ID, model version, and terminal label. Producer
publication occurs after payload writes become visible; the trainer reads only
published records. A ring may overwrite only released slots. If training falls
behind, the producer reports dropped records or applies backpressure. Index
sampling and decoding stay on-device; exporting PGN, checkpoints, diagnostics,
or archival datasets is an explicit operation. Randomly creating bitboards in
VRAM is an arithmetic workload, not self-play training data.

The proposed replay budget starts at 4 GiB and is measured against actual
weights, gradients, optimizer state, inference scratch, and tree growth. One
million 128-byte position records alone use about 122 MiB; policy targets and
history can dominate. Size the arena from measured allocations rather than the
nominal 16 GiB total.

### Transfers and execution

Keep the current synchronous API as the correctness oracle. A future asynchronous
API exposes `submit -> ticket`, `poll`, `wait`, and cancellation/generation
retirement with explicit buffer lifetimes. AMD documents stream ordering,
cross-stream event dependencies, and pinned memory for efficient asynchronous
copies. Overlap must be verified in traces; having multiple streams does not
prove concurrency. Do not let a host buffer be reused before its transfer is
complete.
[AMD asynchronous execution](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/how-to/hip_runtime_api/asynchronous.html)

First benchmark two or three pinned staging slots with compute/transfer streams;
then try HIP graphs for repeated fixed-shape batches. Device-resident frontier
stages remove repeated PCIe transfers. A struct-of-arrays arena can coalesce
same-field accesses, but a one-time transpose and small-batch latency must be
counted. Query model operators on the real RDNA2 GPU before choosing a network
that depends on accelerator-specific instructions.

## Experiment gates

1. **Rules parity:** sorted legal moves, applied state, hash/history, perft, and
   terminal outcomes against Rust and an independent chess implementation;
   explicitly cover castling through check, en passant discovered check, all
   promotions, pinned pieces, clocks, repetition, and arena overflow.
2. **Batch parity:** legal trajectories and wide arithmetic packets;
   CPU/HIP score equality; empty, partial-wave, capacity, error, context thread
   transfer, cancellation, and generation retirement cases.
3. **Search parity:** exact shallow minimax and terminal/mate-distance tests;
   stopped searches return a legal move from a completed iteration. Scheduling
   experiments report their changed information assumptions.
4. **GPU evidence:** architecture, compile target, dispatch trace, memory use,
   per-stage event spans, packing time, wall latency, useful batch occupancy,
   speculative/duplicate work, and clocks/power state. Run experiments serially
   on this one GPU when collecting performance numbers.
5. **Learning evidence:** frozen disjoint game splits, labels and provenance,
   training on-device, saved model/optimizer seeds, held-out policy/value loss,
   latent reconstruction/transition metrics, and strength ablations at equal
   inference time. A declining synthetic loss is not playing-strength evidence.
6. **Strength evidence:** paired openings, both colors, fixed time/hardware,
   legal move/termination checks, complete PGNs, pinned opponents, and confidence
   intervals. A local relative rating requires an external calibration before
   making a 2500 Lichess claim.

## Profiling recovery observed here

The installed `rocprofv3 --version` works, but the first actual trace failed
because `libdw.so.1` is absent from the container. We fetched Ubuntu Noble's
`libdw1t64_0.190-1.1ubuntu0.1_amd64.deb`, extracted it into
`/tmp/chen-gpu/profiler-deps`, and supplied that directory through
`LD_LIBRARY_PATH`. No package was installed and no unrelated mount was changed.
[Ubuntu Noble package metadata](https://packages.ubuntu.com/noble/libdw1t64)
The successful command was:

```sh
LD_LIBRARY_PATH=/tmp/chen-gpu/profiler-deps/usr/lib/x86_64-linux-gnu:/opt/rocm/lib \
  /opt/rocm/bin/rocprofv3 --kernel-trace --output-format csv \
  --output-directory /tmp/chen-gpu/profile -- \
  python3 /tmp/chen-gpu/probe.py /tmp/chen-gpu/build/libchen_gpu.so \
  --max-batch 8192 --repetitions 1
```

Record profiler version and support when requesting deeper counters. Installed
tools and documentation can differ; trace-only RDNA support must not be
mistaken for the full Instinct counter feature set.
[AMD rocprofv3 7.2.1 usage](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/docs-7.2.1/how-to/using-rocprofv3.html)
[AMD thread-trace architecture support](https://rocm.docs.amd.com/projects/rocprofiler-sdk/en/develop/how-to/using-thread-trace.html)
