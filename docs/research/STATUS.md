# Research checkpoint — 2026-10-02

The original scope remains active: gfx1030 GPU execution, modular comparable
search, independent neural architecture/latent research, resident training data,
offline strength evaluation, adjustable play and eventual 2500 Lichess rating.
This checkpoint records implemented foundations and open gates.

## Verified foundations

- Baseline Rust engine frozen at `d3d6e5e`.
- Traits: `Evaluator`, `BatchEvaluator`, `SearchBackend`, `StrengthPolicy`,
  `RulesBackend`, `ResidentTree`, `SelectiveTree`.
- Real gfx1030 HIP Material/Tapered kernels with persistent buffers, versioned
  ABI, runtime device checks, strict compilation, explicit errors and dispatch
  traces. Physical RX 6800: wave32, 60 CUs / 30 HIP WGPs, about 16 GiB VRAM.
- CPU-rules batched frontier minimax selects either CPU or HIP evaluation.
  All six depth-3 suite positions match CPU-frontier by depth, score, best move,
  total nodes and PV. Alpha-beta without quiescence agrees on root scores/depths.
- Portable build: 44 Rust tests pass. HIP-feature host build: 50 pass, with
  eight classical actual-hardware tests run separately and passing in `cpp-rocm`.
  One neural selective hardware test passes for each of three pinned models.
  Strict
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
- Twenty-three harness tests pass. Four ordinary-opening smoke games finish 30 legal
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
At that integration checkpoint, host suites passed 37/43 tests, six Rust hardware checks passed, and the
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
The subsequent [parent-context control](../GPU_PARENT_CONTEXT.md) completes
exactly six declared fresh consumers in CPU/HIP, HIP/CPU, CPU/HIP order.
All pass 18 same-model oracle comparisons and two 20-ply games: 108 comparisons,
71,892 iterative evaluated leaves and 240 postopening legal plies in total.
Independent cold replay admits all PVs/games, 30 observed dispatch-1/eager child
environments and recorded retirement; host and copied remote journals match
byte-for-byte. Neither the Torch-free nor live-idle-HIP parent reproduces the
stall. These finite controls establish no cause, failure rate or production fix;
warming/history and oracle work remain confounded. Production policy is unchanged.

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
scoreboard, and treats the finite game cap as inconclusive. Twenty-three harness tests
and actual node/depth/clock/failure/interrupt/effective-EP fixtures pass. These authored
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

The [selective resident foundation](../GPU_SELECTIVE_SEARCH.md) now executes
on CPU and gfx1030 with complete legal sibling admission, exact cumulative
node caps, explicit closure and separately reported maximum frontier depth.
All per-step classical parity checks pass, as do nine closed-horizon neural
comparisons and failure/reset checks across three pinned eager models. At a
2,000-node cap and maximum ply four, all five nonmate suite positions reach
some ply-four leaves, while full-width search completes depth one to three.
This is mixed-depth coverage, not a uniform depth or strength gain. The first
three-repeat diagnostic is slower on GPU: median sum of six searches is
159.46 ms versus 18.86 ms for CPU selective search. Repeated phase fences and
scalar readbacks are the next classical fusion target; the verified baseline
and raw records remain frozen.

Neural branch `2a08ebb` adds an untrained eight-role supervised bottleneck,
an independent typed threat encoder, a frozen shared readout and cached blind
sibling ties. Ten new CPU tests pass. Stockfish 19 labels are converted to a
pinned approximate reference-material frame with reporting intervals; parent
MultiPV and standalone position queries retain distinct provenance and search
contexts. No new model training or heldout performance result is claimed yet.

The first semantic input set admits 256/64/64 game families from one pinned
Lichess game shard. An [independent CPU replay](evidence/semantic-admission-replay-2026-10-01/README.md)
reparses all 384 selected game chunks, 3,072 observation histories and 14,056
legal root children. It confirms balanced observation movers and zero canonical
input overlap across splits or with 751,183 previously used inputs. Two altered
history/clock controls are rejected after their outer fingerprints are updated.
The producer is staged in `results/neural-staging`. All 23 CPU tests pass,
including parent/worker cancellation, inherited signal masks, orphan reaping and
failed-engine quarantine; an independent reviewer reran the same suite. The
[frozen producer archive](evidence/semantic-producer-2026-10-01/semantic-study-label-sealed-2026-10-01.json)
records the separately sealed one-attempt label declaration. Real CPU Stockfish
production completed all 3,456 queries in 119.96 seconds, with both workers
exiting zero. The frozen whole-family gate retained only 6/0/1 train/validation/test
families. Its empty validation split prevents training/model-selection admission.
Structural review finds that latest standalone packets bearing bound flags
dominate rejection, rather than shallow reported search depth. The [closed
result](evidence/semantic-teacher-2026-10-01/README.md) and full
mechanical audit verify every record/hash, command, packet, action, score-frame
conversion and accepted-family list. All raw records and transcripts are
archived; no heldout score-quality metrics have been inspected.
This is a negative dataset-admission result, with no relaxed
gate, replacements or retry. All 17,128 planned canonical inputs are marked for
exclusion from future fresh studies. No new model training result is claimed.

A separately declared [ordinary-packet study](ORDINARY_SCORE_PACKETS.md) uses
seed 20261008 and fresh 256/64/64 families. Independent admission replay
confirms zero overlap against all 768,311 earlier inputs and across splits.
The [frozen preparation](evidence/semantic-packet-v2-preparation-2026-10-01/README.md)
includes 16 source modules, 31 passing CPU tests, independent review and the
separate root seal. Its one bounded CPU Stockfish attempt completed all 3,456
queries in 134.96 seconds, with both workers exiting zero. Highest eligible
ordinary complete packets retain their emission identity and reported depth;
final choices are separate. Mate/tablebase/decisive evidence and protocol
defects still reject queries, and rejected families are never replaced.
The whole-family gate admits 233/62/55 train/validation/test families:
1,864/496/440 standalone rows and 8,477/2,106/2,118 complete-root action rows.
The [closed teacher archive](evidence/semantic-teacher-packet-v2-2026-10-01/README.md)
preserves every raw record, the full query plan, independent audit and a ledger
of all 17,084 planned inputs, including rejected families, for future exclusion.
Independent mechanical audit passed every transcript, frozen source, query,
history, canonical action, emission selection and Decimal score-frame check.
This provides labels for a later corpus/learning declaration, with no model
training, heldout score-quality analysis, playing-strength result or rating.
The [resident corpus implementation](SEMANTIC_RESIDENT_CORPUS.md) now preserves
immutable exact state/history, complete root groups, teacher query frames and
terminal masks in three separately sealed split packs. Its independent CPU
audit checks all 15,501 state snapshots and all 350 complete legal expansions
(12,701 children) against the actual Rust rules oracle; every table byte,
history, PV and score interval matches independent reconstruction. There are
2,800 standalone observations and 550,040 previous-history hashes in 7,232,128
payload bytes. Forty-four CPU tests pass. The [closed evidence](evidence/semantic-corpus-2026-10-02/manifest.json)
includes payloads, sources, seals, compiled oracle closure and reports.

Actual TRAIN gfx1030 [resident mechanics](../../gpu/evidence/semantic-resident-gfx1030-2026-10-02/attempt2/runtime-report.json)
pass all 233 roots, 8,477 legal child edges and 1,864 observations. Features,
intervals and histories match CPU, gathered batches own their storage and
prepared batches allocate no new host index tensors. The loader makes one
instrumented packed upload call for 8,006,592 bytes including resident schedule
and history-gather indices; physical DMA counts are unmeasured. This is CPU-origin
corpus ingestion, separate from native GPU generation. No model training,
heldout quality metrics or learning-selection admission is claimed.

The classical selective fusion now passes [actual gfx1030 parity and timing](../../gpu/evidence/selective-fusion-gfx1030-2026-10-02/report.json).
Both libraries pass per-step portable-oracle checks; all 36 benchmark rows agree
on scores, PVs, admissions and counters. Across six alternating cold-process
runs, the median sum of six search times falls from 164.763101 to 144.956140 ms,
a 12.02% reduction for that workload. Equal-time playing strength and sustained
throughput remain unmeasured. The exact preserved baseline artifact and fresh
fused artifact are archived; two fresh baseline builds differ in hash despite
identical source bytes, with cause unestablished.

Docker/Git access was restored on October 2. Commit `3976f26` archives the audited
teacher studies and spent-input ledgers. [Architecture direction](../DIRECTION.md)
now defines ownership, 2FFS reproduction, transformer comparisons, heavy
profiling and strength gates. The [closed one-second classical diagnostic](../harness/one-second-reference-2026-10-02.md)
finishes 16 games with 13 wins, 3 draws, no losses against Stockfish19 configured
`UCI_Elo=1320`. Independent replay checks 1,085 legal plies and per-move timing;
Chen's maximum observed wall time is 1004.750 ms. Its configured-opponent proxy
is 1714, with no confidence interval or Lichess calibration. This supplies a
provisional strength estimate above the user's 750 expectation rather than
certifying an absolute rating. Commits `aae0977` and `6fb7130` deliver the fusion
and resident corpus; `7e5398d` adds per-move harness telemetry. The 2FFS CPU
implementation passes 12 tests and 600 independent fixtures against the paper's
declared oracle contracts. Its exposed-node bookkeeping optimization preserves
exact traces across 10 paired runs and improves the larger synthetic median from
1330.677 to 742.852 ms. [2FFS](../TWO_FIDELITY.md) remains a CPU research controller
with a batched fast-oracle seam; no GPU execution or chess guarantee is claimed.

[Heavy profiling](../PROFILING.md) covers 21 bounded UCI workloads, 935 legal PVs and
1,792 search-worker samples. GPU serial backup consumes 77.59–79.29% of dispatch
time; selection 17.37–18.89%; child rules/evaluation 2.42–2.81%. CPU move application
and ray attacks dominate. This motivates a sparse-ancestor backup experiment
and RDNA2/SWAR kernel work informed by official ISA guidance and `../bitmatrices`.
Commit `7243a74` adopts classical sparse ancestor backup after 72 equal-work rows,
1,336 native per-step snapshots and maximum depth/batch guard checks pass.
The large-arena six-position median drops from 1593.306 to 818.045 ms (48.66%).
Classical completion now processes unique admitted ancestors by descending ply;
external neural evaluation retains the reverse-index reference. Selection now
consumes roughly 71–75% of kernel duration. Commit `4e5782a` then adopts the
classical packed best-open-child cache: 72 fixed-work rows retain identical
scores, PVs and counters, and the large-arena median drops from 817.527 to
405.193 ms (50.44%). An independent root replay verifies all 47 payloads and
246 raw members and reconstructs the summary byte for byte. Tiny mate positions
show observed 1.50–2.82% slowdowns. This is a latency result, not a strength claim;
the external neural selector retains its uncached reference.

Commit `eff0e6b` caches completed 2FFS scale certificates until evidence refresh.
All ten positive-action traces remain byte identical, with 600 independent
fixtures and 12 tests passing. Fresh synthetic baseline/candidate medians improve
6.013→2.667 ms and 766.957→287.009 ms; these are CPU research measurements.

Neural commit `041e68d` adds paired semantic study mechanics and the separate
transformer proposal, with 10 passing fixture tests. Neural commit `e50660d` adds the independently reviewed outer runner for 8 anchors
and 32 transition models, with 10 passing orchestration tests. Review fixes bind
phase/suite corpus identities and reject over-budget completed work. The first root-sealed actual gfx1030 study completed 3,072 anchor and 6,144
transition updates, then failed closed before validation because its combined
checkpoint was 137,594,521 bytes, above the 128 MiB gate. Neural commit `0f5140e`
archives all 40 finite individual states, exact runtime and terminal logs.
No heldout quality metrics or model promotion occurred. Neural commit `7d0a199`
replaces the aggregate checkpoint with a hash-bound manifest of all 40 individual
states. Thirteen orchestration fixtures and actual checkpoint admission pass.
A separately root-sealed fresh corrected attempt completed all 40 phases and
reached validation in 142.158 seconds. Neural commit `68b7176` archives its exact
closure, individual states, full validation metrics and independent mechanical
replay. Root checks all 45 payloads and 109 tar members and independently replays
the strict paired-seed selection arithmetic. The unchanged joint gate selects
NONE; TEST remains closed and no model is promoted. Some models improve scalar
error and regret, but optimal-action agreement does not consistently meet the
joint gate across both seeds. The original attempt remains closed and is not
resumed. The [native baseline witness](evidence/semantic-baselines-2026-10-02/build-binding.json)
joins actual Rust Material/Tapered scores to all 15,501 exact corpus hashes and
classifies the sole terminal state as a draw using the history-aware oracle.
It contains no heldout quality metrics.

The [RDNA/SWAR study](../RDNA_BITBOARDS.md), committed as `af6fda2`, independently
checks 401,920 outputs. Kogge–Stone rook fills improve isolated primitive medians
by 2.35–4.98×; cooperative wave32-per-board loses despite lower VGPR usage.
Runtime ISA shows identical two-instruction vector counts for the 64-bit popcount
intrinsic and explicit DWORD halves. The integrated GPU sliding-fill candidate
passes 2,439,488 independent host checks and exact fixed-work comparisons, but
changes whole-search latency by less than 0.2%; the default GPU walk remains.
The separate [CPU SWAR study](../CPU_SLIDING_ATTACKS.md) improves the six-position
200,000-node search median from 2018.104 to 1729.688 ms (14.3%). Commit `08f79e0`
adopts that measured CPU candidate and retains the ray reference as the
`ray-attacks` build feature; both release configurations pass all 56 tests.
Perft and equal-time effects vary by position, with regressions retained.

Full GPU execution remains the target, including simulator-owned symbolic
channels injected into neural residual streams. Separate mixed INT8 and symbolic
role-injection candidates now have arithmetic/causality fixtures. Commit
`6846c50` adds the [actual gfx1030 INT8 affine backend](../RDNA_INT8_AFFINE.md):
16 matrix shapes and 674,337 outputs per checked path pass independent integer
oracles, including zero/tiny rows, ties, residuals and unpadded input tails.
Fused dynamic quantization improves small affine calls but loses to wider routes
at larger batches. The actual loaded Torch HIP provider is recorded separately
from the native compiler. Whole-model fidelity/latency and simulator integration
remain pending; no neural deployment or strength result follows. Timing
is primary; memory bounds are engineering controls rather than the optimization
objective. Commit `b1505ef` adds [22 Lean certificates and a latency strategy
comparison](FINITE_MODEL_STRATEGIES.md), including finite selection obstruction,
cache stability, backup equivalence and INT8 signed32 bounds. Root independently
compiles all 22 and reproduces their actual axioms audit; HIP correspondence,
full neural fidelity and global strategy optimality are not established.

Commit `4df4129` retains an isolated best-open GPU selection seed and its
[independently reconstructed evidence](evidence/best-open-seed-root-review-2026-10-02.json).
All 1,336 native steps preserve parity. Twelve balanced timing pairs measure
4.47% lower aggregate search time at 50,000 nodes, with a small mate-case
regression and uncontrolled shared cache/power effects. Private metadata grows
by 8 bytes per node. This is a frozen comparison arm; production still uses the
committed packed cache.

Neural commit `d8f886b` adds a separate
[multihead semantic transformer control](../../../chen-neural/docs/SEMANTIC_TRANSFORMER_CONTROL.md)
with eight role tokens, exact empty-key exclusion, preserved EP context, and
zero-initial current-state symbolic residual projections. Seven CPU mechanics
tests pass after independent review caught and fixed an extreme-logit masking
bug and incompatible-batch reshape. Auxiliary symbolic-to-role attention pools
all symbolic rows; spatial reasoning enters through the square residual path.
Training, native GPU integration, quantization fidelity and strength for this
new architecture remain open. Frozen study modules and deployment loaders
remain unchanged.

The [device-owned epoch experiment](../DEVICE_EPOCH_EXPERIMENT.md) compares
gated graphs, a single resident workgroup and an occupancy-admitted cooperative
grid. Its full-GPU classical phase controller is under hardware validation;
final independent archive admission and equal-work score/PV timing checks are
pending. The current Torch callback still requires an explicit host boundary;
native symbolic and INT8 graph phases are the next integration work. The
whole-model TRAIN-only INT8 diagnostic is separately bounded and measures
same-checkpoint numerical error and latency without retraining or promotion.

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
