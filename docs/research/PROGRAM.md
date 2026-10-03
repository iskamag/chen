# GPU and latent chess research program

Objective: a modular gfx1030 GPU chess engine whose neural representations and
search architectures can be compared under controlled compute budgets, ultimately
validated at 2500 Lichess rating in a declared time-control pool. Most
development evaluations remain offline. Adjustable play strength is also a target.

## Teams and branches

- Main worktree `chen`, branch `master`: GPU researcher plus integration/harness lead.
- Separate worktree `../chen-neural`, branch `research/neural`: neural architecture
  researcher and neural data/training/methods researcher, with disjoint file ownership.
- Baseline commit `d3d6e5e` freezes the original Rust rules/search oracle.

Both teams verify primary literature and record hypotheses before choosing a
production architecture. Neural designs are independent experiments; global
novelty is unproven. Branch isolation preserves a runnable engine while models,
features, and losses evolve.

## Hardware evidence

2026-09-30: running Docker container `cpp-rocm` (image `pufferlib-rocm:4.0`)
has HIP compiler, ROCm 7.0, and PyTorch 2.10.0+rocm7.0. Live PyTorch reports
`torch.cuda.is_available() == True`, arch `gfx1030`, 30 HIP multiprocessors (WGPs), and
16,368 MiB memory. The profiler identifies an RX 6800 with 60 physical CUs
and 120 SIMDs. HIP compiler/runtime is 7.2.4, distinct from PyTorch's 7.0 build. Container task files use isolated `/tmp/chen-*` directories;
its unrelated `/work` bind mount is untouched.

## Working contracts

Rules remain the CPU correctness oracle. Batched evaluation accepts ordered
positions and returns scores in exactly that order, from each position's side
to move. CPU and HIP implementations share the same interface. Search algorithms
own scheduling, cancellation, draw history, and budgets; evaluation does not
adjudicate mate. The first GPU evaluator scores classical features and is an
execution/throughput oracle. Neural inference is a distinct subsequent backend.

Alpha-beta's serial evaluation path is insufficient evidence for GPU suitability.
A full-width batched frontier search supplies an explicit, bounded comparison
and batching substrate. GPU-resident frontier expansion, irregular search
scheduling, device-side move generation, and neural-guided planning require
separate experiments, oracles, and end-to-end wall-clock evidence.

Neural research investigates legal-move-conditioned latent transitions and
relational board representations, with full-observation refresh and multi-step
consistency as validation targets. Replay is resident on the GPU for sampling
and training; initial upload is recorded. Generated rule-derived toy data tests
mechanics, not strength. Real labels, dataset licensing, holdout game/position
separation, and cross-domain validation remain production gates.

## Execution and theory priorities

Full GPU execution, including resident symbolic-to-neural residual inputs,
remains the target. Optimized INT8 inference is an explicit research/deployment
variant with TRAIN-only calibration and measured numerical, policy and latency
comparisons. Timing takes priority over minimizing memory; resource ceilings may
be adjusted when a concrete allocation and latency comparison justify them.

A dedicated theory track uses the installed Lean toolchain for search-policy and
symbolic-state invariants, plus assumption-explicit bounds on query/latency costs.
Keep proven finite-model equivalence separate from kernel refinement evidence
and claims about optimal chess decisions. [Architecture direction](../DIRECTION.md)
defines the residual, quantization and proof ownership boundaries.

Assigned execution ownership (all three agents stopped at the service usage
limit as of 2026-10-03; root is integrating directly): the coherence agent owns native symbolic production,
INT8 whole-model fidelity/timing and architecture contracts; the RDNA agent owns
device epoch scheduling and hardware/resource evidence; the fusion agent owns
selection experiments and Lean/theoretical strategy analysis. The integration
lead reviews source and evidence, maintains the transformer comparison arm,
and commits verified milestones. GPU experiments take exclusive sequential
hardware reservations while source review, CPU fixtures and proofs proceed in
parallel. Frozen comparison arms remain available independently of defaults.

## Gates

1. Core state/move/hash oracles; GPU batch parity and error/lifetime tests.
2. Device identity/target checks; transfer, kernel and end-to-end latency measured
   separately across batch sizes. No silent CPU substitution.
3. Neural shape/dtype/masking/gradient tests; repeatable optimization; full-refresh
   versus transition representation agreement and latent ablations.
4. UCI compliance and bounded paired legal games before strength testing.
5. Predeclared offline reference pool, clocks, opening pairs, concurrency and
   statistical settings; relative-patch SPRT is separate from absolute rating.
   Lichess uses Glicko-2 and distinct pools: offline engine Elo does not certify
   2500 Lichess. The final target requires a real rating in the selected pool
   with reported uncertainty, or an explicitly scoped calibration estimate.
6. Adjustable levels initially cap search resources; neural skill conditioning
   and policy temperature require independent strength/behavior calibration.

Rating sources: [Lichess rating systems](https://lichess.org/page/rating-systems)
and [Mark Glickman's Glicko-2 specification](https://www.glicko.net/glicko/glicko2.pdf).

The goal stays active until the requested GPU/neural integration and rating gate
are actually demonstrated. Research notes, compilation, synthetic training, or
short smoke games do not meet that gate.

Latest checkpoint: [STATUS.md](STATUS.md).
