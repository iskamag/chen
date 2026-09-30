# Research checkpoint — 2026-10-01

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
- Portable build: 37 Rust tests pass. HIP-feature host build: 43 pass, with
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
- Twenty harness tests pass. Four ordinary-opening smoke games finish 30 legal
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

- The [search-teacher export](../GPU_TEACHER.md) now supplies exact backed-up
  classical minimax labels. Its independent Rust oracle checks 140 parents and
  1358 edge rows, including mate-distance frames. A resident training run uses
  3872 train and 1048 heldout unique parents from disjoint source-game families,
  with zero canonical overlap. All three trained models trail material on
  heldout score/action metrics; artifacts and failures are retained.
- The [external evaluator contract](../GPU_EXTERNAL_EVAL.md) supports interchangeable
  synchronous GPU inference callbacks. Partial callback failures commit no static
  scores. [Native same-model inference](../GPU_NEURAL.md) preserves the trained
  square/relation/plan encoders and scalar heads; their 255 trained numerical
  rows and 510 architecture-fixture rows match eager/scripted GPU scores exactly.
  Schema features match bit-for-bit. Rust `Eval=neural` selects the loaded model,
  and bench records the archive SHA256. Learned latent planning is still open.
  The hardened combined neural suite passes all 76 actual-GPU tests without
  skips; all six Rust hardware tests pass. Both UCI option orders work after
  startup prebinding of the producing HIP runtime. Independent Python minimax
  agrees on all 18 trained-model/position cases and 11982 iterative leaf
  evaluations. [Integration evidence](../../gpu/evidence/neural-integration-2026-09-30.json)
  separates current verification from earlier source snapshots.

## Research findings

The latest checkpoint adds optional exact-count native HIP graph inference,
Rust execution-policy admission/telemetry, and process-lifetime single-provider
ownership. [Native timings](../../GPU_INFERENCE_PERF.md) measure 1.75/2.04/2.00x
warm batch-one callback gains for square/relation/plan, including owned input
staging and unchanged guards. Batch 256 gains fall to 1.05–1.06x. Cold small-shape
capture costs 450–1242 ms, so this remains an opt-in experiment. All 27 native
model/count cases, borrowed-pointer changes, LRU/admission checks and tested
failure policies pass. The [Rust integration record](../../gpu/evidence/neural-graph-integration-2026-10-01.json)
checks 18 cases per mode against independent minimax with identical outcomes.
Current host suites pass 37/43 tests, six Rust hardware checks pass, and the
combined neural suite passes 88 CPU/GPU tests in ROCm without skips.
Both modes pass option ordering and resource/stop checks. Graph mode completes
40 legal game plies; final eager play again times out on the first neural move.
The raw failure is preserved and long-play reliability remains open.
The [follow-up investigation](../GPU_NEURAL_RELIABILITY.md) reproduces the stall
and captures native waits in BLAS/HIP completion. Handle priming fails its
reliability criterion and is rejected. A single original-plugin run with direct
dispatch disabled passes; repeated controls are required before adopting it.
The [repeated dispatch study](../GPU_DISPATCH_STUDY.md) ended partially after an
external container stop. Eight actual timeout stacks survive across four trial
directories, under both requested settings. Full replacement stdout/stderr and
ready-time environment records were lost from host temporary storage; they are
not counted as replayable numerical verification. No dispatch workaround is
adopted. The same container was restarted for infrastructure recovery after
archiving the interrupted state; the partial study was not resumed.

Neural commit `090f2a4` retains a
[predeclared 27-run value study](../../../chen-neural/docs/VALUE_STUDY.md): three
encoders, three readouts, three seeds, 2048 updates each on the same resident
corpus/sample streams. All learned runs trail material's 227.1 cp heldout MAE.
Residual heads reach 232.0/246.9/241.4 cp mean MAE; none improves aggregate
selected-child regret across seeds. Improved sibling concordance does not
produce stronger top decisions. Low-score corrections and a training/heldout
gap suggest specific regularization/ranking experiments on a fresh split;
saturation was not observed. These adapters remain outside production inference.

Neural commit `ce63edd` retains the subsequent
[strategic transport study](../../../chen-neural/docs/STRATEGIC_STUDY.md), which
completes 30 resident training runs over three encoders, five action/anchor
controls and two seeds. Prediction receives the root and legal action only;
exact piece transport, material and pseudoattack covariates are declared priors.
Fresh whole-lineage splits exclude previously consumed inputs. Native depth-3
Tapered supervision retains all legal actions for 184 training, 45 validation
and 44 test roots, omitting incomplete/terminal/mate-band groups explicitly.
No model passes the predeclared joint validation MAE and root-regret gate against
both material and static Tapered. Selection is sealed before all 30 frozen models
are scored once on test. Test baselines are 162.4/156.8 cp MAE and 118.0/101.8 cp
regret respectively; individual ranking gains do not override failed selection.
Plan action-blind predictions show micro-centipawn same-material gaps on the GPU;
their raw-float ranking differences are not evidence of learned action awareness.
A CPU-only frozen-checkpoint diagnostic preserves exact material ties and does
not reproduce the GPU effect; the specific numerical cause remains unverified.
The original five-run attempt stopped at a scalar-parameter checksum defect;
its results remain archived. The bookkeeping-only repair passes 18 actual-GPU
tests, and repeated initial controls have bit-exact parameters/sample streams.
Independent cold artifact admission verifies all 30 checkpoint hashes, source
pins and the observed selection/test journal ordering. These research predictors
are separate from deployed inference and establish no match or rating gain.

Main commit `c51e78c` pins official offline tools and records a
[64-game reference match](../harness/reference-2026-10-01.md) against Stockfish 19
at requested 2000 nodes per move: 61 losses, 3 draws, 0 wins, 32 complete color-swapped
pairs, 4015 independently replayed legal plies. The original bootstrap output is
retained historically and excluded from inference because the authored openings
are correlated. New intervals require an explicit independence assertion.
This diagnostic does not calibrate Elo or complete the Fastchess/SPRT gate.
The [pinned Fastchess integration](../harness/FASTCHESS.md) now freezes paired
experiments, delegates SPRT to the pinned release, independently checks PGN body/
header results, normal rule endpoints, pair identities, native configuration and
scoreboard, and treats the finite game cap as inconclusive. Twenty harness tests
and actual node/depth/clock/failure/interrupt fixtures pass. These authored
protocol fixtures establish result handling; real strength matches and actual
statistical boundary acceptance remain unverified. Container matches run the
whole harness beside local engine executables; host remote-wrapper retirement
is outside this integration's ownership contract.

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

The finite-depth search-teacher comparison is a negative result: the best neural
model has 308 cp heldout MAE and 49.6% action agreement; material has 191 cp
MAE and a greedy material policy has 54.7% agreement. Shared heads and sample
streams control initialization/data, but architectures are not parameter-matched.
This single-seed, short run has no rating interpretation.

Independent review found and fixed an exact-node-limit frontier bug and unsafe
acceptance of malformed deterministic checkpoint buffers. It also drove actual
artifact capture and canonical held-out split checks.

## Next work

1. Optimize the executed GPU rules/resident arena against its frozen oracles:
   attack generation, parallel prefix assignment, compact edges and level reuse.
2. Compare full-width frontier, faithful regularized batched search and a
   separately declared adaptive scheduler at equal wall-clock budgets.
3. Optimize the executed trained batched inference and sustained resident data
   generation, retaining its stream/lifetime/ownership contract. Measure graph
   shape reuse and cold costs before cache admission changes; investigate the
   repeated eager-game timeout with thread diagnostics.
4. Improve action-sensitive dynamics with exact transition reconstruction and
   counterfactual tactical probes; test multistep held-out drift before latent
   search. Fully observed static inference is already an experimental option;
   imagined latent planning requires these additional gates.
5. Select licensed real training data and game/position holdouts, train with
   stronger outcome/search targets, and measure equal-budget paired play.
6. Establish offline reference pools and statistical acceptance. Calibrate
   adjustable levels separately. The 2500 Lichess claim requires evidence in
   the chosen Lichess pool; no account deployment or rating result exists yet.

No requirement has been dropped. Selective resident search and GPU self-play,
strong trained evaluation, sustained inference optimization, reliable long
matches/SPRT and the final rating target remain incomplete.
