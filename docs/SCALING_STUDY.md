# Scaling the existing transformer candidate

This is a study design and allocation-free configuration/reporting milestone.
No large model has been allocated, trained, profiled or strength-tested. The
old tiny negative experiments do not measure transformer scaling. The tactical
operator's 18,850-parameter default is a mechanics fixture, not a size target.

## Candidate and primary evidence

Keep one existing `TransformerSemantics` family: 64 square tokens, pre-LN
multihead residual blocks, per-head geometry/optional exact rule bias, eight
role queries, current-state symbolic residual injection and material-residual
value readout. Width, independent depth and head count can now grow without
the old 256/8/16 ceilings. Old configurations keep the same arithmetic and
checkpoint tensor schema. The eight-role readout remains an explicit bottleneck;
its adequacy must be tested, not assumed from model size.

Ruoss et al.'s **Grandmaster-Level Chess Without Search**, published as
[Amortized Planning with Large-Scale Transformers: A Case Study on Chess](https://arxiv.org/abs/2402.04494),
studies supervised state-value, action-value and behavioral-cloning targets,
with models up to270M parameters and ChessBench's10M games/~15B annotated data
points. Its reported2895 Lichess blitz rating is against humans and belongs to
that model/protocol. It is evidence that serious transformer scale can work in
chess, not a strength prediction for Chen. The
[author implementation](https://github.com/google-deepmind/searchless_chess)
offers9M/136M/270M checkpoints and a decoder-only board/action-value model;
this encoder with semantic roles is not a reproduction or a compatible loader.
A source-bound board/action-value reference is a useful future comparison;
weights/data download, porting and execution are outside this milestone.

[Chessformer](https://arxiv.org/abs/2605.19091) supports a square-token geometry
direction but uses dynamic geometric attention bias and a policy design that
Chen has not reproduced. Neither these papers nor our typed symbolic inputs
establish novelty for larger Chen transformers.

## Allocation-free descriptors

`TransformerConfig.estimate(batch_size=..., value_only=...)` and
`scaling_profiles()` construct no modules or tensors. Parameters are calculated
exactly from the declared layers. Reports separate stored, value-path and
forward-active parameters, auxiliary decoder parameters, FP32 parameter bytes,
dense matmul MACs and attention logit elements. MAC counts exclude topology
production, material, pointwise operations, dispatch and synchronization;
summed logit elements are not peak memory. Small CPU execution fixtures compare
predicted counts with actual parameter counts and intercepted matrix shapes.

The provisional bands below target **value-path parameters**, not stored
training weights. They are descriptors, not chosen winners or launches.
Geometry is enabled, rule bias/injection absent, expansion4, B1/value-only.

| Nominal band | Width / heads / depth | Stored parameters | Value-path parameters | Forward matmul MACs |
| --- | --- | ---: | ---: | ---: |
| ~1M | 128 /2 /5 | 2,077,891 | 1,087,293 | 71,066,624 |
| ~10M | 320 /5 /8 | 12,885,175 | 10,410,801 | 666,667,520 |
| ~50M | 640 /10 /10 | 56,300,895 | 51,353,561 | 3,261,445,120 |
| ~100M | 768 /12 /14 | 108,200,567 | 102,264,049 | 6,521,247,744 |
| ~270M | 1280 /20 /13 | 274,033,055 | 264,139,801 | 16,740,935,680 |

One MAC is one multiply-accumulate, not two FLOPs. Static geometry bias is
computed once and broadcast; batched rule bias scales with B. Reports preserve
that distinction. Increasing heads at fixed width changes logit workspace and
optional bias parameters without necessarily changing dense matmul work.
Masking piece keys still does not compact tokens. Existing symbolic topology
construction costs are deliberately excluded, not claimed free.

For allocation, callers can declare `max_parameters` on the constructor.
It checks the estimate BEFORE buffers or weights allocate; `None` is uncapped.
This is a caller resource policy, not an architectural size limit. The tactical
config likewise supports allocation-free `estimate()` and caller budgets,
without an arbitrary width256 ceiling. No inference speed follows from fitting
weights in VRAM/cache. Activations, optimizer state, precision and workspace
must be measured or separately accounted. Timing is the deployment priority.

## Separate experimental axes

- **Model N:** record total/body/auxiliary/value-path counts, width, independent
  depth, heads, head dimension and expansion. A parameter curve follows a pinned
  width-depth schedule; it does not isolate parameter count from shape.
- **Data D:** unique independently sampled root/game families and deduplicated
  state+history identities, plus states, complete legal action groups and
  labelled children. Replaying an epoch increases processed examples, not D.
- **Training C:** processed roots/actions, optimizer steps, forward/backward
  matrix work, teacher-generation work, wall time and energy when available.
  Declared forward MACs alone are not training compute. Do not silently turn
  `6NT` language-model heuristics into measured chess work.
- **Inference I:** exact producer, topology, value-only encoder, complete-root
  scoring, precision conversion, transfers, capture setup, steady calls and
  synchronized wall time. The search contract includes all these costs.
- **Computation shape:** compare wider/shallow and narrower/deep models at
  approximately matched N and matched measured work; do not fit them as one
  unconstrained size curve. Tactical recurrence1/2/4/8 shares weights and changes
  execution depth; it is distinct from independently parameterized transformer
  layers. Do not silently add recurrence to this transformer's checkpoint.

NNUE/event updates, tactical operators and the transformer remain separate
mechanisms within the same quality/compute comparison. The larger transformer
is a serious candidate, not merely a control for an assumed graph winner.

## Data, convergence and bounded HPO

The main deployment comparison is **converged, possibly heavily overtrained
models at matched inference budgets**. Offline training can use successive
new synthetic data increments and substantially more examples than a
training-compute-optimal allocation would recommend. Here "overtrained" means
spending extra offline compute/data on a fixed deployable model, not fitting
the final TEST set or repeatedly memorizing one small corpus.

The [synthetic data program](research/SYNTHETIC_DATA_PROGRAM.md)
defines which legal/history families, exact topology/action targets, bounded
exchange/quiescence traces, strategic complete-root rankings and counterexamples
to generate. More positions help only when their diversity, label quality and
task coverage are useful. Exact rule supervision and strategic teacher values
are different targets; unlimited exact rule examples do not supply an unlimited
strong value oracle. Declare the teacher budget/quality and sufficient coverage
for each data increment, rather than assuming quantity removes label error.

Use fresh family-separated TRAIN/development/selection/TEST splits with a
hashed exclusion ledger. Old validation is development evidence. Group related
positions, games and full legal sibling sets; preserve history distinctions.
Keep one teacher version, finite-budget protocol and score/WDL perspective;
record confidence intervals/bounds and label coverage. Do not mix standalone
child and root MultiPV horizons as interchangeable exact values. All terminal
history/legality decisions stay exact. Teacher labels or future PV/search
results cannot enter current-state symbolic features or priors.

For each scale, save train/development curves versus processed examples and
measured compute, several checkpoints, gradient/numerical diagnostics and
per-stratum root decisions. Equal short update counts are learning-curve
controls, not architecture-selection gates. Fixed 384 updates are not a
convergence criterion, and a capped unconverged run is labelled undertrained.

Tune LR/warmup/decay and regularization per size on development data; a recipe
that converges at 1M parameters need not do so at 270M. Declare plateau criteria
for complete-root regret/ranking and tactical, quiet and endgame strata, using
fixed development families and uncertainty-aware tolerances. A suggested
protocol requires three scheduled evaluations with changes below declared
tolerances after an LR reduction, then a new TRAIN family increment and a
continued-training check. Tolerances, evaluation spacing, data increments and
work ceilings must be pinned before a run. Save every checkpoint and outcome.
Training-loss convergence alone or a quiet-position plateau is insufficient
when tactical errors continue changing. If teacher quality or fresh data raises
the plateau, record that change as a new protocol arm rather than hiding it.

A proposed bounded pilot is 24 optimizer trials: three declared size anchors,
four LR/warmup/decay recipes and two paired seeds, at pinned pilot budgets.
The actual anchors, trial budgets and recipes require a separate executable
declaration. Record failed and truncated trials; do not retain only winners.
The pilot narrows development tuning; it does not pick the deployment winner.
After per-size convergence tuning, freeze the schedule or its declared
size-dependent rule and evaluate the five-scale ladder with at least three
paired seeds. New offline data/training increments are supported through fresh
declared budgets. This bounds jobs and preserves provenance without restricting
the program to short runs. A separate data ladder uses at least five nested
TRAIN family subsets and records both unique data and processed examples.

Run a **separate compute-constrained study** if we want the best loss under an
offline training budget. Its iso-compute checkpoints, truncation and fitted
frontier answer a resource-allocation question. They do not select the model
for a deployment where offline training is plentiful and online latency is
limited. The deployment winner comes from converged/overtrained checkpoints,
complete-root quality and same-time search, with all inference costs included.

Primary objectives should align with complete-root decisions: calibrated value
and complete-group ranking loss, with interval/tie handling. Report regret,
agreement, severe blunders and tactical/quiet/endgame strata. Keep auxiliary
reconstruction weights, material prior shrinkage and symbolic injection as
explicit bounded ablations. They are not confounded with N in the primary
curve. No single-seed best checkpoint or small-sample accuracy improvement
qualifies as a winner.

## Offline learning and online computation

Fit weights, calibration, optimizer schedules and data curricula offline.
Freeze and hash the resulting checkpoint before selection/final evaluation.
During online search, use inference-only execution with no backpropagation,
optimizer or weight update. Cached latents, exact event updates and causal
current-search residuals are computation/state, not online weight learning;
their validity still requires state/history/perspective/epoch ownership.
Any future online-learning experiment is a separately declared protocol,
not an implicit part of playing-strength results. The current scaling work
implements neither training nor online adaptation.

## Local scaling fits and playing-strength gates

[The root-owned fitter](../scripts/analyze_scaling.py) accepts
`chen_scaling_curve_v1` development data. Required top-level fields are `axis`,
`metric`, predeclared nonnegative `declared_floor`, its `floor_provenance`,
`controls` and `observations`. Controls pin split, teacher, training, metric and
axis-design SHA256s, architecture family, precision, development split and a
nonempty `held_fixed` mapping. Each observation binds distinct run/checkpoint,
integer seed, positive x, metric value strictly above floor and identical
controls. The source declaration and run archive must substantiate those pins;
the fitter validates metadata, not the truth of a submitted metric.

Supported primary axes are `parameters`, `unique_train_roots` and
`training_macs`; supported errors are value MAE, root regret and ranking cross
entropy. It requires at least five levels across16x and three paired seed
curves. Declare whether N means stored or value-path parameters in axis design.
Parameter curves hold the data and declared per-size convergence/tuning policy
fixed; data curves hold model and exposure/convergence rule fixed; compute
curves hold model/data fixed or explicitly declare a compute-constrained
frontier with its selection protocol. N, D and C
are often coupled, so a single diagonal sweep cannot identify three exponents.

The fit is `error-floor = amplitude * x^(-exponent)` in log space. Report the
range, residuals, per-seed slopes and seed-curve bootstrap interval. That
interval describes training seed variation conditional on fixed evaluation
families; family-sampling uncertainty requires a separate analysis. Do not
choose the floor after seeing the points. Reject or report poor curvature/fits,
and test prospective predictions on new declared scales. The current four
recurrence settings form a discrete ablation, not enough levels for this fit.

[Jones](https://arxiv.org/abs/2104.03113) studies train/test-compute tradeoffs in
AlphaZero/Hex, not a chess-transformer exponent.
[Hoffmann et al.](https://arxiv.org/abs/2203.15556) demonstrate model/data
allocation tradeoffs for language models; their compute-optimal exponents
must not be imported into chess. A Chen fit establishes a local empirical
relationship under its teacher, data, objective and training protocol. It
does not establish a universal optimal-FLOP law, algorithmic reasoning, a
270M optimum or
2500 Elo. There are currently no admissible real Chen scaling curves.

Freeze checkpoints and selection before final evaluation. First compare the
complete legal-action policy against exact-child re-encoding, material/Tapered
and appropriate stronger references on fresh families, with paired uncertainty
estimates. Then compare same-time resident search at1s/move with fixed search
controls/openings/opponents and include every producer/inference cost. Also
report a fixed-work search control to diagnose evaluator versus throughput.
Fitting an offline error exponent does not substitute for these matches; the
fitter deliberately rejects Elo as a power-law error metric. Optimize kernels
after identifying useful quality, while preserving full GPU execution and
causal symbolic residual streams.
