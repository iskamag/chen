# Research checkpoint — 2026-09-30

The original scope remains active: gfx1030 GPU execution, modular comparable
search, independent neural architecture/latent research, resident training data,
offline strength evaluation, adjustable play and eventual 2500 Lichess rating.
This checkpoint records implemented foundations and open gates.

## Verified foundations

- Baseline Rust engine frozen at `d3d6e5e`.
- Traits: `Evaluator`, `BatchEvaluator`, `SearchBackend`, `StrengthPolicy`,
  `RulesBackend`, `ResidentTree`.
- Real gfx1030 HIP Material/Tapered kernels with persistent buffers, versioned
  ABI, runtime device checks, strict compilation, explicit errors and dispatch
  traces. Physical RX 6800: wave32, 60 CUs / 30 HIP WGPs, about 16 GiB VRAM.
- CPU-rules batched frontier minimax selects either CPU or HIP evaluation.
  All six depth-3 suite positions match CPU-frontier by depth, score, best move,
  total nodes and PV. Alpha-beta without quiescence agrees on root scores/depths.
- Portable build: 37 Rust tests pass. HIP-feature host build: 39 pass, with
  six actual-hardware tests run separately and passing in `cpp-rocm`. Strict
  Clippy, formatting, Python compilation and shell syntax checks pass.
- Hardware tests compare ordered batches of legal positions at sizes
  0/1/7/64/257/4096 for both evaluators, exercise capacity/length errors and
  exclusive thread transfer, and compare GPU search including exact node cap.
- Full-rule 128-byte state and GPU legal generation, incremental transitions,
  draw adjudication, leaf evaluation and minimax reduction now run in a bounded
  resident arena. `cpu-resident` and `hip-resident` share one scheduler; all six
  depth-3 suite results agree by completed depth, score, move, nodes and PV.
  Independent Python rules compare 256 roots and 7090 legal transitions; the
  Rust hardware suite additionally checks 527 roots, ordered legal lists and
  every successor. GPU start-position depth-4 perft is 197281 leaves.
  [Rules contract and evidence](../GPU_RULES.md) specify phase/capacity limits.
- Seven harness tests pass. Four ordinary-opening smoke games finish 30 legal
  plies each without technical failure, remaining explicitly inconclusive.
  Two-game mate fixtures finish for CPU and real GPU UCI implementations;
  these verify protocol/adjudication rather than strength.
- Match artifacts can preserve actual copied Rust/HIP binaries and source,
  compiler, image ID and SHA-256 manifest. Wrapper-only hashes are labeled.
- `Level` 1..20 caps resources monotonically; no rating labels are assigned.
- Separate neural worktree `../chen-neural`, branch `research/neural`:
  `697fee7` establishes square/relation/plan models, latent transitions,
  resident replay, controlled legal-data training, and saved smoke checkpoints.
  Additional audited checkpoint validation and legal-diamond experiments are
  retained on that branch; its earlier combined GPU suite passed 29 tests.
  `e262feb` adds compact replay and observed-action transport experiments;
  its compact/dynamics suite passes 48 actual-GPU tests without skips.
  Compact payload is 801 B/record versus 29733 B dense (37.1x smaller), with
  byte-exact decoded inputs/losses/gradients. Sampling is slower; two short
  training runs reach 96–99.3% of dense throughput. Those measurements used
  CPU-made ingress. The later [HIP–Torch bridge](../GPU_REPLAY_EXPORT.md) exports
  resident edges into Torch-owned VRAM, admits compact records and trains all
  three model families with toy targets. Its final combined GPU suite passes
  62 tests without skips; all six Rust hardware tests also pass after the export
  extension. Runtime identity, stale selections, allocation bounds, cross-stream
  handoff and failure-path completion are tested. Scoped copy diagnostics see
  only two 16-byte status copies during each successful C export call.

## Research findings

The standalone tapered kernel reaches roughly 100 million positions/second at
batch 32768 in one measured sweep. This excludes position packing, rules,
search and network inference. Small-batch launch latency is substantial; a
serial alpha-beta GPU call per leaf is not the intended architecture.

The legal-diamond experiment supplies a falsifiable chess-specific hypothesis:
independent legal move orders with equal exact endpoints should agree in latent
space. Its first controlled comparison finds negligible additional benefit
from crosspath consistency. Learned action separation is only 2.4–7.2% of the
corresponding exact re-encoding separation. Reconstruction is weak. This is a
negative architectural result to retain, not playing-strength evidence.

Observed move transport raises learned action separation from about 5% to 34%
in a controlled four-model smoke. Action-delta error and successor semantics
remain worse than the previous transition control; this is mixed/negative
evidence. Exact piece transport is simulator-owned and kept separate from
learned predictions. The neural branch retains full metrics, logs and hashes.

Independent review found and fixed an exact-node-limit frontier bug and unsafe
acceptance of malformed deterministic checkpoint buffers. It also drove actual
artifact capture and canonical held-out split checks.

## Next work

1. Optimize the executed GPU rules/resident arena against its frozen oracles:
   attack generation, parallel prefix assignment, compact edges and level reuse.
2. Compare full-width frontier, faithful regularized batched search and a
   separately declared adaptive scheduler at equal wall-clock budgets.
3. Extend the executed HIP-to-compact-replay bridge to learned batched inference
   and sustained data generation, retaining its stream/lifetime/ownership contract.
4. Improve action-sensitive dynamics with exact transition reconstruction and
   counterfactual tactical probes; test multistep held-out drift before latent
   search. Integrate a trained evaluator only after these gates.
5. Select licensed real training data and game/position holdouts, train with
   meaningful outcome/search targets, and measure equal-budget paired play.
6. Establish offline reference pools and statistical acceptance. Calibrate
   adjustable levels separately. The 2500 Lichess claim requires evidence in
   the chosen Lichess pool; no account deployment or rating result exists yet.

No requirement has been dropped. Selective resident search and GPU self-play,
production trained inference, long
matches/SPRT and the final rating target remain incomplete.
