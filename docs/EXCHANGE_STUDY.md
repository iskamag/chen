# Restricted-exchange learning study

`neural/exchange_model.py` gives the existing tactical processor a separate task head. It predicts the correction to the initial capture/promotion material delta in the artificial legal destination-recapture game defined in [EXCHANGE_TARGETS.md](EXCHANGE_TARGETS.md). P=100, N=320, B=330, R=500 and Q=900 are physical material units. This is **not strategic CP**, a chess value evaluator, an Elo estimate, or a trained deployment candidate.

The input is the complete current parent legal-action graph from [TACTICAL_INPUTS.md](TACTICAL_INPUTS.md): pieces, squares, every legal action, all 14 exact typed relations, history and causal physical material priors. Action edits and immediate material changes are calculable from the current position and legal action. No exchange return, dependency depth, future trace, child teacher score or future PV enters the observation. Point losses apply only to fully closed capture/promotion slots; quiet actions remain present in the processor. Missing check-evasion, terminal and resource-bounded targets supply no guessed material return.

## Model and controls

The three input projections and recurrent tactical cell are shared across iterations. A single linear readout consumes the action latent plus pooled piece/square/action context. Its zero initialization reproduces the immediate-material baseline exactly; processor gradients become active after the readout starts learning. There are no strategic value heads or unused semantic decoders.

The default width256 model has 1,067,521 stored parameters, including 1,055,488 processor parameters. Width is configurable and an optional caller parameter budget rejects allocation before building the model. Recurrence1/2/4/8 changes computation, not parameter count. Typed/untyped controls retain the same topology/features/weights; the untyped route zeros the learned relation embedding. It does **not** erase exact piece identities, occupancy, action effects or geometry, and therefore is a relation-type ablation rather than removal of all chess knowledge. The unused relation embedding remains counted in stored parameters for this control.

`destination_continuation` is a same-head exact-child reference seam: encode the independently observed complete immediate-child graph, query its occupied destination square in the child mover frame, and negate its predicted continuation to recover parent correction. CPU admission rejects empty/out-of-range queries. Parent-child replay and history identity remain the external adapter's responsibility. This seam has a mechanics fixture; **it is not a trained arm in the current declaration**. No deeper future trace is supplied as an input.

## Fresh data and label yield

The source-bound evidence lives in `neural/evidence/exchange-learning-inputs-2026-10-03`. The admission first fixed256 TRAIN,64 DEV and64 closed TEST reachable whole-game families, then generated exchange labels only for TRAIN/DEV. Its397 attempts took39.92s onCPU3, with12 empty-family and1 spent-feature rejection. The known-spent ledger binds20 sources covering838,236 legacy input keys and1,042 family IDs, including the777,264-key previous corpus exclusion snapshot, semantic/value/strategic admissions, packet-v2 spent inputs, both reachable mechanics pilots, prior exchange traces, native topology/tactical source fixtures and their complete one-ply support. These are explicit known sources, not a claim of an omniscient history registry.

Every planned parent and every complete legal one-ply child feature key is protected against that ledger and against other families/splits. Source move identities and whole-family IDs are also protected. The legacy key covers mover-oriented pieces, castling rights, raw EP and clipped halfmove clock; full replay/history is bound separately, and equivalent board features with different histories are conservatively excluded. Deeper exchange trace states are supervision-only for this parent/exact-one-child design. Future designs observing deeper states need a broader admission closure.

| Split | Families | Admitted roots | Initial task actions | Closed exact | Nonzero correction | Unsupported check escape | Typed terminal |
|---|---:|---:|---:|---:|---:|---:|---:|
| TRAIN |256|1,443|3,430|2,783|1,117|599|48|
| DEV |64|378|795|651|244|139|5|
| TEST |64|362|not labeled|not labeled|not labeled|not labeled|not labeled|

Closed dependency-round counts are TRAIN1:1501,2:897,3:251,4:113,5:17,6:2,7:2; DEV1:378,2:199,3:47,4:23,6:4. Thus higher-dependency strata remain small; their estimates must carry counts rather than imply broad tactical coverage. Roots without a point target stay in the raw admission/missing-yield report and contribute no loss. The training packs contain910 TRAIN and229 DEV task-bearing roots. Complete parent graphs are separate tensor dictionaries from masked corrections and diagnostic strata. TEST labeling/packing/loading is rejected before artifact access. Prior Stockfish pilot scores are not used here.

The restricted-label generator uses the unchanged16-ply/4096-node-per-action semantics, with500,000 entered-state and180s safety ceilings per split. Actual TRAIN/DEV generation took4.60s/1.18s and6,619/1,562 entered states, with no resource-aborted target. The source used at admission/label time is retained byte-exact as `admission-label-source.py`; current additional pack checks are separately source-pinned.

## Convergence and bounded DEV tuning

`neural/exchange_study.py` declares eight arms: recurrence1/2/4/8 crossed with typed/untyped. For each arm, seed6101 screens learning rates0.001/0.0003 and weight decays0/0.0001. All four controls must reach the declared DEV plateau before selecting a pair. Selection minimizes DEV macro-family mean absolute correction error, with deterministic LR/weight-decay ties. The chosen seed6101 fit is reused; seeds6102/6103 supply two additional paired repetitions. This gives32 screening fits plus16 additional fits, not repeated training of an already finished checkpoint.

Training uses active-action mean SmoothL1 on corrections normalized by800, AdamW, batch8, FP32 throughout and gradient clipping1. DEV's primary metric averages action errors within a root, roots within a family, then families. Reports also include micro-action MAE, nonzero-correction family MAE, motif/dependency strata and their sample counts. The immediate-material control predicts correction zero. Its error measures this restricted task, not a complete chess ranking.

A fit runs at least20 epochs. An improvement must exceed max(0.5 material units,0.2% of its tracked best metric). Eight evaluations without such improvement cause a0.3 LR reduction; after two reductions, a third plateau terminates the fit. Epoch-by-epoch training loss and DEV strata are retained, with the best actual DEV checkpoint. This establishes a **declared DEV plateau**, not a theorem of optimizer convergence or sufficient synthetic-data coverage. More data, longer schedules and alternative regularization can still change results.200 epochs,1800s per fit or21600s aggregate active time are safety ceilings: a stopped fit is explicitly unconverged and cannot satisfy HPO admission. An incomplete experiment supplies no architecture winner.

## Resource and execution boundary

Offline CPU collation pads only to a batch's active K/E capacities, buckets by edge work and includes collation/upload in synchronized wall accounting. For the actual TRAIN pack at batch8: maximum E468,K68; a width256 edge-expanded FP32 tensor is3.83MB. One structural forward processor iteration over an epoch is approximately124.35G MACs. Eight saved edge tensors across eight iterations would occupy approximately245MB, an illustrative structural calculation rather than a measured autograd peak. Parameters/gradients/two Adam arrays require approximately17.08MB. Kernel launch/reduction/autograd overhead, allocator liveness and actual throughput still need execution; these figures do not predict latency or cache residency.

The declaration pins the local relative-import source closure, TRAIN/DEV packs and admission reports. Runtime ingress requires actual Torch2.10.0+rocm7.0/gfx1030 without overrides or CPU fallback, HIP runtime70051831 and the previously observed Torch HIP-provider hash. It does not load the native INT8 library or alter the deployment producer.

Execution is chunked into at most two fits and3900s per group. A source-bound journal records active jobs, immutable retired report/checkpoint hashes, aggregate active wall time and the next fit. It resumes **between completed fits only**; interrupted active jobs reject automatic restart. Every group finishes all GPU fences and emits a terminal record before another hardware owner may enter. An external owner must also impose a process-group watchdog (inner3900s, outer3960s maximum), retain partial failures and verify process retirement. Waiting between groups is not counted as training time. HPO training timing cannot be attributed to isolated hardware when another GPU job overlaps.

CPU preparation, seven mechanics fixtures plus a chunk-schedule fixture, and actual label collection are complete. **No study training, TEST evaluation or learned-quality claim has occurred.** After frozen convergence runs, a separately declared evaluation should compare restricted-task accuracy against compute/inference cost; strategic complete-root decisions and same-time chess search require their own fresh evidence.

```sh
# CPU admission/label/pack commands are separate; TEST has no label command.
python -m neural.exchange_study declare --directory PATH --output declaration.json --width 256
# On an independently admitted exclusive ROCm window, under an owned watchdog:
python -m neural.exchange_study run --declaration declaration.json --sha256 PIN --output RUN --max-fits 2
```
