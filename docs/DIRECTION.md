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
ISA, and independently checks every primitive output. Prefer branchless sliding
fills as the next rules candidate; preserve independent board lanes and ordinary
popcount intrinsics. Primitive wins require full rules parity and integrated
register, kernel and end-to-end measurements before adoption.

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

**Resident corpus and learning.** The main-worktree corpus package is a shared
data boundary, not a second model library. Keep complete legal-root groups,
exact state/history and teacher-query provenance together; only feature tensors
enter the learned observation path. A packed CPU-origin upload is distinct from
native GPU generation. Native export currently requires complete independent
base allocations; aligned corpus views must not bypass that ownership check.
Training, frozen-checkpoint validation and selected-checkpoint test access remain
separate. Do not reopen spent families to tune the next architecture.

## Order of evidence and integration

1. Freeze a runnable release and run the one-second-per-move reference match
   with pinned opponents, options, openings and artifact hashes. Independently
   replay games and report failures and uncertainty. An offline estimate needs
   a declared reference calibration; it is not automatically Lichess Elo.
2. Capture heavy CPU/GPU profiles for that same engine configuration and
   representative opening, tactical and endgame positions. Use profiles to
   choose a bounded optimization, then verify its oracle and repeated timing.
3. Finish resident corpus hardware admission and a predeclared learning study.
   Keep negative models and the selected-`NONE` outcome. Promote a candidate
   only after joint value/action gates, reliable inference and equal-time play.
4. Reproduce 2FFS and add transformer comparisons as named research variants.
   Keep their additional assumptions and budgets visible instead of silently
   changing the baseline engine.
5. Merge only reviewed contracts and measured improvements into deployment.
   Calibrate resource levels separately and establish the final Lichess pool
   and rating criterion before claiming the target.

Every result binds source, binary/library/model hashes, compiler/runtime/device,
configuration, data or opening hashes and raw observations. Tests establish
their actual scope: rule parity, numerical inference, protocol reliability,
performance and playing strength are distinct gates.

## Coherence rules for contributors

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
