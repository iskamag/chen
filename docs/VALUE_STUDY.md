# Controlled resident value-head study

This experiment isolates value learning on the existing square, relation and
plan encoders. Its criterion is heldout ordinary centipawn error and observed
child ordering against a deterministic material baseline. Finite-depth
classical search labels are supervision; they do not supply game outcomes,
calibrated WDL or playing-strength evidence. **The captured 27-run result is
negative:** every learned model has worse heldout CP MAE than material, and no
architecture/head improves mean selected-child regret across its three seeds.

The [predeclaration](../neural/evidence/value-study-predeclared-2026-10-01.json)
fixes the workload before the main run: 128 training and 32 heldout source-game
families, corpus seed 20261001, three initialization/sample seeds
20261001/20261002/20261003, all three encoders and all three heads, 2,048 steps
per combination, batch 128, width 32, depth 1. This is 27 runs. Pilot results
established runtime and mechanics only. The workload does not expand in response
to heldout metrics.

## Head and target contracts

All adapters consume the same schema-1 observations and the same mean-pooled
64-square encoder output. `neural/value_heads.py` owns their implementation and
restricted checkpoint contract; the original models, search teachers and
deployment-version-1 loader are unchanged.

| Head | Differentiable value prediction | Initial prediction |
| --- | --- | --- |
| `three_logit` | 800 × atanh(clamp(softmax(logits)[2] − softmax(logits)[0], ±(1−1e−6))) | Original seeded three-logit head |
| `direct_cp` | 800 × scalar linear readout | 0 cp |
| `material_residual` | Exact material cp + 800 × scalar linear readout | Exact material cp |

Material counts own-minus-opponent pieces with values 100/320/330/500/900/0,
clamped to ±29000. It ignores castling, EP and clocks. New scalar readouts start
at zero without consuming random numbers. That produces explicit 0/material
initial baselines; their encoder gradients activate once the readout has learned
nonzero weights. For a fixed architecture/seed, every adapter starts from an
identical `teacher.comparable_model` encoder. Across architectures, common input,
position, norm and original heads share initialization, while architecture
blocks and total parameter counts differ. Head initialization and parameter
counts remain declared differences rather than hidden matching claims.

Every head receives **the same normalized-CP SmoothL1 criterion**:
`SmoothL1(value_cp / 800, teacher_cp / 800)`, averaged over records with
`abs(teacher_cp) < 29000`. Mate-band records have no value gradient. No policy,
board reconstruction or dynamics objective is active. AdamW uses lr 0.0003,
the adapter's active value parameters and gradient clipping at norm 1.

Training uses unrounded differentiable CP. The bounded utilities are separate
diagnostics: `tanh(cp/800)` for new heads and the existing softmax difference for
the three-logit control. Atanh clamping can flatten the existing head at its
extremes, and near saturation it amplifies inverse-utility gradients. Reports
include utility saturation, raw-score outliers and preclip gradient norms rather
than assuming the three heads have identical conditioning. Static deployment
rounding/clamping is separate from training and from the float ordering metric.

## Resident corpus, split and comparisons

The unchanged [search teacher](SEARCH_TEACHER.md) produces full breadth-first
depth-3 trees, Tapered static leaves, and complete bottom-up reductions. Exported
parents are at root ply one, giving a two-ply remaining horizon. Sibling rows
are deduplicated to one parent vote. CPU ingress uploads roots and their retained
history only; exact observations and labels flow from the GPU simulator to the
owned compact replay.

`value_study.gather_values` gathers current bitboards/rule metadata and score/
lineage labels, then decodes features on the device. It never materializes dense
legal-policy masks or next-observation features for scalar training, and does
not allocate another persistent dense corpus. Each seed has one resident index
schedule reused by every architecture/head, with replacement. Training tensors
stay in VRAM. Scalar simulator validation and AdamW control are allowed; tests
forbid payload CPU conversion through loss/backward/update and host scalar reads
in the strict gather/loss/backward path.

Game families are entirely separated. The existing canonical collision check
covers roots, observed parents and stored successors, including color/rank
equivalence, raw EP, rights and clipped clocks. Uniform legal walks share early
start-position ancestry outside the trained observations. Their distribution is
artificial; this split establishes lineage and feature separation, not human-
game representativeness. Teacher repetition adjudication uses retained history,
which remains absent from the encoder's feature schema.

Three seeds vary initialization and sampling on a **fixed corpus**, so their
heldout samples are correlated. Each run reports before-training, after-training
and training-set metrics. All 27 checkpoints are retained. There is no stopping
or tuning based on the heldout set. The previous teacher run also used policy
loss and a different corpus; changes versus that run cannot be attributed solely
to the new value heads.

## Score, ordering and uncertainty diagnostics

Ordinary score metrics use unrounded float CP: MAE, RMSE, signed bias, p95
absolute error and maximum predicted magnitude. Compressed utility RMSE is
reported separately. Teacher-magnitude bins isolate low scores, 200–800 cp,
800–2000 cp and 2000–29000 cp. Material-imbalance bins distinguish positions
with `abs(material_cp) <= 100` from larger imbalances. These bins expose which
errors remain after adding the anchor, rather than allowing an aggregate win to
hide large tactical misses.

Ordering groups observed first-ply parents by `teacher_root_id`. Their scores
are from each child's mover, so **lower child CP is better for the root mover**.
It measures mean regret of the selected child relative to the minimum teacher
score, optimal-child fraction accepting all teacher ties, and concordance on all
unordered teacher-nontied sibling pairs. Prediction ties get half concordance
credit; top choice uses first resident-record ties. Only roots with at least two
ordinary observed children enter selection statistics.

This is an **unrounded float-CP ordering diagnostic over ordinary nonterminal
children**. Terminal and mate-band root actions are absent; it cannot claim exact
full-root move agreement. Future nearest-even integer deployment may add ties,
and no neural search or engine game is measured here. The identical material
predictions provide the comparison on exactly this subset.

Per-family MAE summaries cross the device boundary only after evaluation. Paired
bootstrap intervals resample entire source-game families with equal family
weight and compare network minus material MAE. They describe corpus uncertainty
for one initialization seed. Seed variation is reported separately; a family
interval is not evidence of independent new heldout games for each seed.

## Captured result and explanation

The corpus contains 7,861 training and 1,935 heldout unique parents, with 37/36
mate-band records excluded from value metrics. The split checked 243,099 and
63,494 canonical positions with zero overlap. One candidate family was rejected
for terminal/repeated history; none collided canonically. Every seeded run used
exactly the same 2,048×128 resident sample schedule as its other eight controls.

The table reports heldout mean ± sample SD across the three initialization/
sampling seeds. Material MAE is **227.1 cp** and mean selected-child regret is
**158.3 cp** on the identical ordinary nonterminal subset.

| Encoder | Head | Heldout CP MAE, mean ± seed SD | Mean selected-child regret cp |
| --- | --- | ---: | ---: |
| Square | Three-logit | 311.2 ± 13.1 | 231.4 |
| Square | Direct CP | 296.8 ± 6.9 | 234.2 |
| Square | Material residual | 232.0 ± 1.0 | 192.4 |
| Relation | Three-logit | 317.1 ± 59.0 | 270.2 |
| Relation | Direct CP | 326.6 ± 28.9 | 322.2 |
| Relation | Material residual | 246.9 ± 14.3 | 164.6 |
| Plan | Three-logit | 309.9 ± 46.5 | 250.3 |
| Plan | Direct CP | 310.6 ± 18.4 | 265.6 |
| Plan | Material residual | 241.4 ± 13.5 | 167.9 |

The anchored initial prediction equals material exactly, so these runs test
whether learned corrections add value. They did not improve aggregate heldout
MAE. Residual heads do improve pair concordance to 0.628–0.637 from material's
0.571 over 28,168 teacher-nontied pairs. That is not sufficient for top choice:
material selects an optimal observed child in 40.6% of roots; the three-seed
residual fractions are 33.3% / 36.5% / 39.6%. One plan seed improves selection
regret (135.7 cp), but its other seeds do not, and its score MAE remains worse.
This exception is retained without selecting it as a deployment winner.

The bins identify a concrete harmful correction pattern. For the 794 ordinary
positions with `abs(teacher_cp) < 200`, material MAE is 86.2 cp; residual-head
seed means are 139.7 / 150.7 / 143.1 cp. For the 207 positions in the 800–2000 cp
bin, they improve material's 446.1 cp to 393.7 / 382.2 / 352.7 cp. Improvements
on larger scores do not compensate for worse corrections on common low scores.
All three residual heads are also worse than material in the aggregate
`abs(material_cp) <= 100` and `>100` subsets. The failure is not confined to an
inability to represent materially imbalanced boards.

No ordinary teacher or heldout prediction reaches `abs(utility) >= .99`; the
largest heldout raw prediction across all runs is approximately 1401 cp. Thus
utility saturation is **not an observed explanation** for this run. Switching
to a direct scalar with the shared CP criterion also fails to remove the gap.
Relation/plan residual training MAE averages 152.6/146.5 cp, versus heldout
246.9/241.4 cp, while training material MAE is 216.9 cp. This supports a
generalization problem alongside objective/selection misalignment, rather than
a claim that the trained encoders learned a stronger search evaluator. Family
bootstrap intervals are broad; none of the nine residual runs shows a paired
family MAE improvement whose interval excludes zero. One relation run's interval
instead establishes a positive network-minus-material difference on this corpus.

These are observations, not a diagnosis of a unique cause. The next
predeclared experiment should introduce a separate validation-family set for
residual shrinkage/regularization, keep a fresh test set untouched, and compare
root listwise ranking against the shared score objective. Its acceptance gate
should require both ordinary score improvement and selected-child regret
improvement. A static Tapered baseline/anchor would also distinguish learning
positional evaluation from learning the search correction beyond that evaluator.
More diverse families and explicit tactical features can test the generalization
hypothesis; more steps on this same heldout-selected setting cannot establish it.

## Artifacts and reproduction

The new `chen_value_adapter_checkpoint` kind preserves adapter configuration,
deterministic buffers, exact weights, teacher construction and study provenance.
It has a restricted CPU-first loader in `value_heads.py`; existing teacher and
deployment-version-1 artifact kinds do not silently accept it. Checkpoints live
under ignored `results/neural/value-study/`. Writing weights and post-training
payload fingerprints are explicit cold CPU audit boundaries; those bytes are
never used as training ingress.

```sh
PYTHONPATH=/path/to/python-chess python -m unittest neural.tests.test_value_study -v
CHEN_VALUE_STUDY_TEST_GPU=1 PYTHONPATH=/path/to/python-chess \
  python -m unittest neural.tests.test_value_study -v
python -m neural.value_study --library /path/to/teacher-only/libchen_gpu.so \
  --producer-manifest /path/to/teacher-producer-evidence.json \
  --train-families 128 --heldout-families 32 --corpus-seed 20261001 \
  --seeds 20261001 20261002 20261003 --steps 2048 --batch-size 128 \
  --width 32 --depth 1 --tree-depth 3 --evaluator 1 --arena-capacity 250000 \
  --output results/neural/value-study-gfx1030.json
```

The [full report](../neural/evidence/value-study-gfx1030-2026-10-01.json),
[run log](../neural/evidence/value-study-gfx1030-2026-10-01.log),
[GPU test log](../neural/evidence/value-study-tests-gfx1030-2026-10-01.log),
[CPU test log](../neural/evidence/value-study-tests-cpu-2026-10-01.log) and
[artifact/source manifest](../neural/evidence/value-study-artifacts-2026-10-01.json)
retain all controls, full root histories/construction, bins, family uncertainty,
stream/data fingerprints and actual checkpoint hashes. All six study tests
passed on actual GPU; the final CPU run passes five and skips one hardware test.
The current restricted loader separately validates all 27 actual trained
checkpoints on CPU. The exact adapter source used during training is archived;
the final adapter adds float32/input and exact-integer metadata admission guards
without changing forward mathematics or stored state.

The captured environment is actual gfx1030 under Torch 2.10.0+rocm7.0 and the
immutable teacher-only producer library. Training wall times include sampling,
loss/update and lazy optimizer allocations, but exclude initial/heldout
evaluation and checkpoints. Parent CPU-only match work may run concurrently;
these are research-job costs rather than an isolated throughput benchmark.
All training loops together took 227.94 s. Exact active value-parameter counts
are 7075/7009 (square), 11370/11304 (relation), 15850/15784 (plan), where the
first value is the three-logit control and the second is either scalar head.

The next inference/strength step requires an explicit adapter deployment
contract, score/decoder parity and offline paired games. The 2500 Lichess
objective remains open. Latent planning remains a separate research question;
this experiment trains an evaluation readout without rollout objectives.
