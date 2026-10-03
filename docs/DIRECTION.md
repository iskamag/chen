# Chen architecture and near-term direction

Chen has two cooperating worktrees: `chen` on `master` owns the runnable Rust
engine, gfx1030 runtime and evaluation harness; `../chen-neural` on
`research/neural` owns model, latent-transition and training experiments. The
target is a GPU engine with useful learned representations, adjustable play and
eventual 2500 Lichess performance. A working UCI engine exists; a rating above
750 at one second per move now has a useful provisional reference: [13 wins,
3 draws against Stockfish configured 1320](harness/one-second-reference-2026-10-02.md),
giving an uncalibrated configured-opponent proxy of 1714. This remains separate
from an actual Lichess rating; larger reference-pool checks are still needed.
The subsequent [GPU selective reference](harness/gpu-one-second-reference-2026-10-03.md)
scored 4 wins, 5 draws and 7 losses, proxy1254. The CPU result does not establish
GPU strength. The [algorithmic redesign](research/ALGORITHMIC_REDESIGN.md) now
prioritizes parallel retained alpha-beta/quiescence after the now-runnable
bounded GPU reference, a scalable transformer and one recurrent tactical neural
processor. Hardware throughput and playing
strength remain
separate comparisons; further selector/dispatch microtuning is deprioritized.

This file defines architectural direction. [STATUS](research/STATUS.md) owns the
latest verified results and open gates; [PROGRAM](research/PROGRAM.md) owns the
overall research requirements. Historical evidence directories are snapshots,
not alternative implementations to import or edit.

## Ownership and interfaces

| Boundary | Authority | Extensions belong here |
| --- | --- | --- |
| Exact chess state and legal moves | `src/core.rs`, `src/rules.rs`; CPU reference and `RulesBackend` | Faster attacks or GPU transitions, checked against exact state/hash/legal-action oracles |
| Scheduling, budgets, cancellation and terminal decisions | `SearchBackend`; `src/search.rs`, `frontier.rs`, `resident.rs`, `selective.rs` | Named search policies with explicit work, completed horizon and failure semantics |
| Position scores | `Evaluator`, `BatchEvaluator`; `src/eval.rs`, `backend.rs` | Classical, fully observed neural or incremental scoring, always with a declared score frame |
| Device storage and dispatch | `src/hip.rs`, `device.rs`, `gpu/` | Versioned HIP ABI, bounded arenas, stream ownership and measured kernels |
| Deployment admission and inference | `src/neural.rs`; neural worktree exporter/plugin | Fingerprinted model artifacts, exact feature contracts, same-model numerical and lifetime checks |
| Data and neural experiments | `experiments/semantic_corpus/`; `../chen-neural/neural/` | Immutable split packs, resident sampling, independent encoders, frozen-target transitions and controlled studies |
| Protocol and resource levels | `src/main.rs`, `strength.rs` | UCI, worker lifetime and `StrengthPolicy`; level numbers remain resource policies until calibrated |
| Performance and strength evidence | `scripts/bench.py`, `scripts/match.py`, pinned Fastchess integration | Reproducible raw timings, paired legal games and separately scoped statistical conclusions |

The CPU rules oracle stays available when a kernel is optimized. Search owns
full history and terminal adjudication; schema-1 neural features omit some of
that state. A latent representation never authorizes a legal move or overrides
mate/draw handling. Scores from native classical search and normalized Stockfish
teacher queries have different contracts: conversion and ordinary-score masks
must remain explicit.

Resident full-width and selective search share host/device scheduling contracts,
but answer different depth questions. Selective depth is not a completed
uniform horizon. `hip-frontier` transfers CPU-generated leaves; `hip-resident`
and `hip-selective` execute rules and tree operations on the device. Preserve
these names and scopes in every benchmark. A CPU fallback must be explicit.

## Full GPU execution, symbolic residuals and INT8

Full GPU execution remains the deployment target: exact rules, symbolic feature
production, latent updates, neural scoring, frontier decisions and backup retain
their state on the device. CPU implementations remain references for comparison.
The new [PVS/aspiration controls](CLASSICAL_SEARCH.md) stay independently
selectable; their completed16-game direct one-second diagnostic favors PVS
6W/6D/4L, with broader play still needed before default adoption.
CPU/proof controls, bounded reference and parallel GPU candidates were developed
in isolated temporary worktrees. Their reviewed milestones are integrated on
main; clean temporary worktrees are removed at the requested wrap-up, retaining
their branches. Model/data work remains in the neural worktree. See the
[handoff](HANDOFF.md) before resuming experiments.
Measure host scheduling and publication costs explicitly while moving suitable
control work into the GPU. A CPU result does not replace this target.

The [native tactical observation](GPU_TACTICAL.md) now exposes a versioned, typed
residual-input boundary to the neural engine. The separately named
[complete edge-bucket producer](TACTICAL_BUCKETS.md) now measures the full
export/copy/captured-inference path; ABI1 remains the conservative control.
Derive channels from the current exact state and legal actions:
occupancy, attacks, pins, check constraints, material/phase, clocks/history and
move events. Keep parent/child identity, player orientation, validity masks,
stream dependencies and allocation ownership explicit. Learned projections may
inject these channels into neural residual streams. Teacher scores and future
search/PV information cannot enter a branch prediction as observed state.

Quantized inference is a separate model/backend variant. Start with optimized
INT8 packing, accumulation and fused scale/residual operations supported by the
actual gfx1030 ISA; inspect the compiled code and compare against FP32/FP16 on
real resident batches. Fit any static activation ranges using TRAIN only; dynamic
per-row scaling is inference arithmetic whose overhead must also be measured.
Freeze the quantization policy, fitted scales and weights before numerical,
action-ranking and playing-strength comparisons.
Report calibration/packing overhead, dispatch/transfer time and end-to-end
latency; integer arithmetic alone does not demonstrate an optimized backend.

Timing is the primary optimization criterion. Memory limits protect ownership and
bounded execution, but they are configurable engineering bounds, not the target
metric. Permit larger justified artifacts or resident allocations when measured
latency improves; avoid duplicated checkpoint tensors and needless transfers.

The [completed precision/capture controls](../../chen-neural/docs/PRECISION_CAPTURE.md)
use captured FP32 as the next integration reference. Captured INT8 wins the
tested batch-one anchors, loses at batches32/256, and nearly ties complete-root
transitions. Retain INT8 as a named small-batch option rather than applying one
precision globally. FP16 remains a useful fidelity control. Measure the complete
producer/inference/search path before deployment; these TRAIN diagnostics do
not change the selected-NONE quality gate.

The theory track uses Lean to prove finite-model invariants and execution-policy
equivalence: stable argmax under monotone reservations, dependency-ordered backup,
packed field separation and exact symbolic transition facts. Keep assumptions and
refinement boundaries explicit. A machine-checked model theorem needs a stated
connection to the executed kernel; it is not a claim of globally optimal chess.
Investigate latency/query lower bounds under declared oracle and hardware models
before calling a strategy optimal. Lean 4.27.0 is available locally.

## Parallel research tracks

**GPU search and profiling.** Profile the actual one-second UCI path and its
host/device phases before choosing the next optimization. Separate launch and
synchronization cost, transfers, legal expansion, allocation/compaction,
evaluation and backup; also report end-to-end latency, time-control overshoot,
nodes, complete sibling coverage and achieved horizons. Use frozen baseline and
candidate binaries, alternating repeated runs, identical positions and both
equal-work and equal-time comparisons. Kernel throughput alone cannot select a
search policy. The existing selective fusion experiment addresses repeated
fences; its adoption depends on parity and measured end-to-end benefit.
The [gfx1030 SWAR study](RDNA_BITBOARDS.md) reads AMD RDNA2 instruction
definitions and the neighboring bitmatrices kernels, verifies emitted runtime
ISA, and independently checks every primitive output. Keep branchless sliding
fills as a measured candidate; the integrated selective-search result improves
less than 0.2%, so the default remains unchanged. Preserve independent board lanes
and ordinary popcount intrinsics. The next search work changes tactical
continuation and useful work allocation; selection-path tuning resumes only
for a demonstrated policy or a measured integration bottleneck.

**2FFS paper implementation.** Implement a separate CPU reference for
[Two-Fidelity Best-Action Identification for Stochastic Minimax Tree,
arXiv 2606.01708v2](https://arxiv.org/abs/2606.01708v2), then compare batched GPU
execution of the same policy. Keep stochastic-tree reproduction and a chess
adapter distinct. The paper's fixed-confidence claim needs its stated slow
oracle and fast-bias assumptions; deterministic neural scores, finite-node
Stockfish labels and reporting-quantization intervals do not supply those
assumptions. A chess adaptation must report an uncertified recommendation unless
it independently establishes them. Add a dedicated policy contract rather than
overload `SelectiveTree` with unrelated confidence semantics. Compare CPU/GPU
sampling and decision traces before interpreting speed or strength.

**Transformers and latent architectures.** Keep transformer candidates beside
the existing square/relation/plan/threat encoders in the neural worktree. Record
the original paper and precise local hypothesis before adding a candidate.
Compare identical admitted data, legal action support, score frames, seeds,
readout and selection rules; report parameter counts, retained VRAM, training
time and inference/search latency rather than calling equal width equal compute.
Include fully observed transformer scoring and classical baselines. Latent
transition models must separately demonstrate decision quality and multistep
refresh agreement before imagined search is deployed. A successful reconstruction
loss or a larger model alone does not establish a strength gain.
The recurrent tactical operator in the
[redesign](research/ALGORITHMIC_REDESIGN.md#one-new-processor-with-two-substantive-extensions)
uses actual piece/square/legal-action nodes and supplied exact typed edges. Establish
exact-child and recurrent-action controls, fresh data, convergence and bounded
hyperparameter tuning before extending it to persistent object memory or
search-aware adaptive refresh. The [scalable transformer](../../chen-neural/docs/SCALING_STUDY.md)
is also a serious candidate: compare roughly 1M through 270M descriptors and
width/depth alternatives rather than assume the tiny processor is preferable.
The recurrent processor's separate restricted-exchange head has its first two
DEV HPO fits in neural `861b3d3`; recurrence/typing comparisons and transformer
learning are still pending. Those labels describe an artificial recapture game,
not strategic chess value. No architecture winner or neural engine is selected.
Offline training may substantially exceed
compute-efficient schedules when extra training improves inference-budget
quality; short update counts are learning-curve controls, not size verdicts.

**Resident corpus and learning.** The main-worktree corpus package is a shared
data boundary, not a second model library. Keep complete legal-root groups,
exact state/history and teacher-query provenance together; only feature tensors
enter the learned observation path. A packed CPU-origin upload is distinct from
native GPU generation. Native export currently requires complete independent
base allocations; aligned corpus views must not bypass that ownership check.
Training, frozen-checkpoint validation and selected-checkpoint test access remain
separate. Do not reopen spent families to tune the next architecture.

## Order of evidence and integration

1. The device-owned alpha-beta/quiescence reference and retained root-child pool
   are implemented. Keep their admitted window/abort controls; fix useful work
   allocation before adding precision changes. The child-local resource retry
   candidate still cannot bootstrap an Exact incumbent on Kiwipete. Next test
   deterministic capture/promotion/PV ordering as a no-pruning permutation,
   preserving canonical root identities and charging all work. Compare completed
   decisions and paired equal-time play before promoting a new policy or adding TT/PVS.
2. Implement the recurrent tactical processor and shared exact topology ingress,
   and scale the fully observed transformer through explicit model/data/compute
   studies. Model cache capacity does not define the preferred size.
   Compare exact-child evaluation to parent/action processing. Keep terminal,
   history, state identity and score-frame authority in the symbolic engine.
3. Admit fresh root families and a consistent teacher protocol. Report learning
   and data-scaling curves; predeclare a bounded development sweep and freeze
   choices before fresh evaluation. Existing negative results are not tuned
   architecture verdicts.
4. Test a frozen neural winner in a fixed search at equal time before persistent
   memory or adaptive refresh. Keep 2FFS and modern transformers as named
   controls with their additional assumptions and budgets visible.
5. Merge only reviewed contracts and measured decision/strength improvements.
   Profile newly useful paths end to end before precision or kernel tuning.
   Calibrate resource levels separately and establish the final Lichess pool
   and rating criterion before claiming the target.

Every result binds source, binary/library/model hashes, compiler/runtime/device,
configuration, data or opening hashes and raw observations. Tests establish
their actual scope: rule parity, numerical inference, protocol reliability,
performance and playing strength are distinct gates.

## Coherence rules for contributors

The [causal profiling policy](CAUSAL_PROFILING.md#avoiding-redundant-work)
governs new experiments: use a bounded intervention with an end-to-end progress
metric, reuse reviewed source-bound results, and repeat broad verification only
for a specific changed boundary or unresolved concern.

- Give each parallel task explicit file ownership; coordinate shared Git staging
  and commits. Keep each commit reviewable with its validation and evidence scope.
- Add a new implementation behind an existing seam when its semantics fit;
  introduce a new seam when inputs, guarantees or ownership actually differ.
- Keep live source out of evidence archives and imported archive code out of
  runtime modules. Archive duplication preserves past runs; it is not a reason
  to maintain duplicate active feature decoders or training loops.
- Route new readers here and to `STATUS`. `docs/EXPERIMENTS.md` retains the
  original proposed optimization protocol; its historical “future” sections do
  not supersede implemented GPU, neural or harness contracts.
- Update capability claims when the underlying implementation or verification
  changes. Keep unverified work and rejected experiments visible without
  presenting them as production behavior.
