# Learned latent depth: fixed input, tied weights, no search

Additional learned recurrence improved held-out resource-feasibility accuracy
over a single update, but **more latent depth did not reliably produce deeper
reasoning**. Two four-round-trained models reached 98.82% and 98.38% random
held-out accuracy at round four, then fell to 92.87% and 55.74% at round 32.
Training over variable depths made extrapolation more stable in one seed, but
both variable-depth models stayed at 50% on the degree-matched Hall pairs at
every measured depth. This is a useful control and a negative result for
dependable structural extrapolation, not a proposed chess engine.

## What changes with depth

The input is a bipartite resource-to-target incidence graph. Every target is
demanded; each resource can satisfy at most one target. All edges and counts
are available before the first update. No new input appears with additional
rounds, and there are no runtime oracle calls, branch decisions, child boards,
or alternative assignment trajectories.

Each target/resource starts from `[1, degree/10, 1]`, projected to a typed
24-dimensional state. At round k, each node receives an adjacency-masked sum
of projected opposite-type states from round k−1, divided by ten. Separate
resource and target GRU cells synchronously update the states from that
message, their previous state, and the unchanged initial embedding. The same
weights are reused at every round. A graph readout pools sums and maxima and
also receives target count, resource count, and edge count. It emits a
feasibility probability.

The model has **16,897 parameters** at every tested depth. More rounds buy
additional state updates and longer possible dependency propagation, not
additional parameters or longer input. They can also merely repeat a learned
nonlinearity, change confidence, or destabilize a state. The experiment must
distinguish those possibilities from successful collective reasoning.

This small typed recurrent network is an experimental learned control. It is
not claimed to be a novel JPEG architecture. Labels are exact offline matching
answers supplied by the shared corpus; `maximum_matching`, labels, family,
and split never enter the network. IDs seed only the alignment-corruption
control, not the learned forward pass.

## Frozen training and corpus

Protocol `e929588` was committed before execution; source `497bd7a` was
committed before training. There were four fits: seeds 17 and 29 under either
fixed four-round training or a uniformly sampled one-to-eight-round depth per
minibatch. Each fit used 600 AdamW steps, batch 128, width 24, learning rate
0.002, and the preregistered BCE/gradient-clipping settings. The fixed regime
uses four recurrent rounds per training example; variable-depth training has
a different, approximately 4.5-round average training compute budget.

Checkpoint selection used only DEV balanced accuracy at the regime's maximum
training depth, with DEV logloss breaking ties. All four checkpoint selections
were frozen before any EVAL/STRUCTURE predictions were computed. Selected
steps were 600, 600, 600, and 400 respectively. Depth curves below are reported
without selecting a winner or tuning on held-out results.

The shared corpus contains 1,760 TRAIN and 440 DEV random graphs with three
through six resources/targets; EVAL contains 880 such small graphs and 480
larger graphs at sizes seven, eight, and ten. Random splits are balanced by
label and exclude duplicate raw graphs, but are **not isomorphism-held-out**.
The separate STRUCTURE set has 240 graphs: 16 Gram-pair examples, 64
degree-matched Hall-pair examples, 39 cycles, 35 resource-deficit cycles,
44 forced chains, and 42 chains with a missing final resource. Structural
examples include relabelled versions; those repetitions are not independent
structural discoveries. Their class counts are 123 feasible and 117 infeasible.

CPU affinity was changed from the protocol's CPU 1 to CPU 2 at the parent's
request to avoid contention; PyTorch intra/inter-op threads were one. All
training and evaluation completed in 43.675 seconds without a GPU. No
additional fits or evaluation-based changes followed the result.

## Same frozen weights at different depths

Accuracy in percent on all 1,360 random EVAL graphs; balanced accuracy is
identical here because the classes are balanced:

| Training / seed | 1 | 2 | 4 | 8 | 16 | 32 rounds |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 4 / 17 | 90.37 | 97.50 | 98.82 | 94.56 | 92.65 | 92.87 |
| Fixed 4 / 29 | 77.57 | 97.79 | 98.38 | 89.34 | 61.69 | 55.74 |
| Variable 1–8 / 17 | 95.74 | 96.25 | 98.46 | 98.90 | 96.40 | 90.81 |
| Variable 1–8 / 29 | 95.96 | 97.65 | 98.16 | 98.16 | 98.16 | 98.38 |

The larger EVAL subset also improves at trained depths. At rounds one, four,
and 32, its accuracy is respectively:

| Training / seed | 1 | 4 | 32 rounds |
|---|---:|---:|---:|
| Fixed 4 / 17 | 94.58 | 98.75 | 96.25 |
| Fixed 4 / 29 | 86.25 | 97.71 | 65.21 |
| Variable 1–8 / 17 | 94.17 | 98.33 | 93.33 |
| Variable 1–8 / 29 | 95.00 | 98.33 | 98.33 |

This is size transfer on random graphs, not proof of long dependency handling.
The structural results are more discriminating. Accuracy on all 240 STRUCTURE
graphs is:

| Training / seed | 1 | 2 | 4 | 8 | 16 | 32 rounds |
|---|---:|---:|---:|---:|---:|---:|
| Fixed 4 / 17 | 61.67 | 82.08 | 86.67 | 80.00 | 83.33 | 66.67 |
| Fixed 4 / 29 | 51.67 | 86.67 | 86.67 | 70.00 | 48.75 | 48.75 |
| Variable 1–8 / 17 | 72.08 | 80.00 | 86.67 | 86.67 | 86.67 | 83.33 |
| Variable 1–8 / 29 | 72.08 | 83.33 | 83.33 | 83.33 | 83.33 | 83.33 |

These structural aggregate accuracies are slightly affected by class imbalance;
all family metrics, FP/FN, and balanced accuracies are in the committed JSON.
For example, 48.75% above means predicting every structural graph infeasible,
with balanced accuracy exactly 50%.

The decisive **64 Hall-pair graphs** preserve every target degree and every
resource degree, have target degree two, and have no isolated resources.
Both variable-depth seeds and fixed seed 29 score exactly 50% at every depth.
Fixed seed 17 scores `50, 50, 50, 87.5, 75, 50%` at the six measured depths.
That transient improvement is real, but it does not replicate across seeds
and disappears at greater depth. It cannot be promoted to a reliable latent
matching procedure.

Post-hoc descriptive size slices of the saved logits show that fixed seed 17's
round-eight Hall gain covers sizes four, six, and eight, while size ten stays
at 50%. Variable seed 17 improves forced-chain size-eight and size-ten accuracy
from 0% at one/two rounds to 100% at four rounds; size ten falls back to 0%
at round 32. Variable seed 29 never gets the size-ten forced chains right.
These slices involved no further fitting or selection. All forced-chain
examples in this corpus are feasible, so getting that family right alone
does not establish a complete compatibility solver. Missing-resource chains
also admit simpler count/degree cues.

## Controls: recurrence and destructive coupling interventions

The controls use the **same selected weights**, not separately trained models.
Their four-round random EVAL accuracies are:

| Training / seed | Full feedback | Reset state each round | Zero coupling | Independently scrambled alignment |
|---|---:|---:|---:|---:|
| Fixed 4 / 17 | 98.82 | 90.37 | 50.00 | 91.84 |
| Fixed 4 / 29 | 98.38 | 77.57 | 50.00 | 91.84 |
| Variable 1–8 / 17 | 98.46 | 95.74 | 50.00 | 92.43 |
| Variable 1–8 / 29 | 98.16 | 95.96 | 50.00 | 91.62 |

Reset executes the requested number of updates but starts every one from the
initial states. Its logits are bit-identical at all six depths, exactly equal
to one feedback round. Repeated compute without feedback provides no gain in
this control. This is not a separately trained, optimized shallow alternative.

Zero coupling removes edge messages while retaining degree/global inputs and
local recurrence. Its failure shows that these trained weights use coupling;
it does not show that a separately trained degree-only classifier must fail
on the random distribution. Independently permuting resource identities per
target corrupts message alignment while leaving original degree/global inputs
and labels fixed. In particular the retained resource degrees generally
contradict the altered adjacency, and original labels need not describe the
rewired graph. This is an inconsistent-input, distribution-shifting lesion,
not an isolated causal test of alignment or accuracy on rewired graphs.
Coherent joint node relabelling, in contrast, preserves logits to within
`4.77e-7` on the preregistered numerical check.

## Stability, confidence, and limits

Round count is not a monotonic confidence or accuracy dial. Fixed seed 29
makes 378 additional classification flips between rounds eight and 16 on
random EVAL; its Brier score worsens from 0.0106 at round four to 0.3727 at
round 32, with 602 false negatives. Variable seed 29 is much more stable:
no random-EVAL classification changes from four to eight or eight to 16,
then three from 16 to 32; its Brier improves slightly from 0.0126 at four to
0.0120 at 32. Variable seed 17 instead worsens from Brier 0.0092 at eight to
0.0917 at 32. Calibration/error and probability drift are reported for every
family and depth. We did not record hidden-state distances, so performance
collapse is evidence of unstable extrapolation, not a diagnosed oversmoothing
mechanism or a proven oscillatory attractor.

Inference is a deterministic recurrent circuit: no runtime matching labels or
ordinary search are hidden in the output. The measured feedback prefix for
2,040 non-training graphs, batched 256, is about 0.075 seconds through four
rounds and 0.529 through 32. These are cumulative timing probes including
snapshot readouts, not engine move latency or optimized scalar inference.

The evidence supports useful learned interaction updates under a bounded
training regime. It does not establish systematic algorithmic depth
generalization. A next falsifiable test would train with explicit identity-
sensitive compatibility obligations and adversarial degree-matched structures,
then seal new long-range families. Simply adding recurrence to the same
aggregate message network is insufficient on the present result. Any new
operator must be compared at matched training compute and on the difficult
paired structures, rather than selected for its random accuracy alone.

## Artifacts and reproduction

`experiments/codec_lab/latent_depth/learned/results/` contains four selected
checkpoints, all per-query logits, checkpoint-selection logs, and complete
family/depth/control metrics: about 1.6 MiB. Every fit passed checkpoint
save/load with exactly zero logit difference on eight DEV graphs. These are
the actual selected weight SHA-256 values:

```text
fixed4 seed17       2bf02e11a0c454d130162db7e1648b4f092a415e6b467d0455efa4a642b6e89a
fixed4 seed29       9944a1fc03cb50660319b7bf6464644ab484f6daaa4570ee71080f4a60b15a7d
variable1to8 seed17 0663d65925498b2f795a471a8ddd8835756680513b25888deef6989b38c2f12c
variable1to8 seed29 0f43b426c60a79dc24ed94647179afce967dc773a5490b1a79f529986d58cbc0
```

Protocol SHA-256 is
`9ee1f5601a8470c31ee7d8050c511395969e27c7747c13ae6d4eeda22f02a6ad`;
source SHA-256 is
`2d8c07a004d817573ae2fc7edfe9505ef89f38197bc40cdd973387344b9078d0`;
shared corpus SHA-256 is
`441a37e200d9460403e95b9318a9d4751790c13d4813e6f51cb3dcf5568311ab`.
The environment was PyTorch 2.14.1+cpu and NumPy 2.5.4.

From the worktree/repository root, repeat all four frozen training configurations:

```sh
/tmp/chen-jpeg2-cpu-venv/bin/python experiments/codec_lab/latent_depth/learned/run.py \
  --corpus /home/iskam/src/vibe/chen/experiments/codec_lab/latent_depth/corpus.json \
  --out /tmp/chen-latent-learned-rerun --cpu 2
```

For inference-only replay, first copy the committed results directory to a
temporary directory, then pass that copy as `--out` with `--replay`. Replay
checks corpus/weight hashes and recomputes logits/metrics; it rewrites its
output directory, so use a copy to preserve the recorded first-run timings.
