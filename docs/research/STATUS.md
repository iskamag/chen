# Research checkpoint — 2026-09-30

The original scope remains active: gfx1030 GPU execution, modular comparable
search, independent neural architecture/latent research, resident training data,
offline strength evaluation, adjustable play and eventual 2500 Lichess rating.
This checkpoint records implemented foundations and open gates.

## Verified foundations

- Baseline Rust engine frozen at `d3d6e5e`.
- Traits: `Evaluator`, `BatchEvaluator`, `SearchBackend`, `StrengthPolicy`.
- Real gfx1030 HIP Material/Tapered kernels with persistent buffers, versioned
  ABI, runtime device checks, strict compilation, explicit errors and dispatch
  traces. Physical RX 6800: wave32, 60 CUs / 30 HIP WGPs, about 16 GiB VRAM.
- CPU-rules batched frontier minimax selects either CPU or HIP evaluation.
  All six depth-3 suite positions match CPU-frontier by depth, score, best move,
  total nodes and PV. Alpha-beta without quiescence agrees on root scores/depths.
- Portable build: 31 Rust tests pass. HIP-feature host build: 33 pass, with
  two actual-hardware tests run separately and passing in `cpp-rocm`. Strict
  Clippy, formatting, Python compilation and shell syntax checks pass.
- Hardware tests compare ordered batches of legal positions at sizes
  0/1/7/64/257/4096 for both evaluators, exercise capacity/length errors and
  exclusive thread transfer, and compare GPU search including exact node cap.
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
  retained on that branch. Latest combined GPU suite passes 29 tests.

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

Independent review found and fixed an exact-node-limit frontier bug and unsafe
acceptance of malformed deterministic checkpoint buffers. It also drove actual
artifact capture and canonical held-out split checks.

## Next work

1. Freeze a richer full-rule device state and test GPU move/transition kernels
   against the Rust perft/hash oracle. Resident expansion/reduction remains open.
2. Compare full-width frontier, faithful regularized batched search and a
   separately declared adaptive scheduler at equal wall-clock budgets.
3. Add compact resident replay/feature decoding and an ordered HIP-to-neural
   inference boundary with explicit stream/lifetime/ownership semantics.
4. Improve action-sensitive dynamics with exact transition reconstruction and
   counterfactual tactical probes; test multistep held-out drift before latent
   search. Integrate a trained evaluator only after these gates.
5. Select licensed real training data and game/position holdouts, train with
   meaningful outcome/search targets, and measure equal-budget paired play.
6. Establish offline reference pools and statistical acceptance. Calibrate
   adjustable levels separately. The 2500 Lichess claim requires evidence in
   the chosen Lichess pool; no account deployment or rating result exists yet.

No requirement has been dropped. Device-resident full search/self-play,
direct HIP–Torch experience ingestion, production trained inference, long
matches/SPRT and the final rating target remain incomplete.
