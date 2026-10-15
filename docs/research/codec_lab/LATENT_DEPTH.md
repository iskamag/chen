# Latent depth without a move tree

Extra computation inside a fixed state can resolve additional interactions.
It is not automatically useful: local messages can stall, and learned
recurrence can become less accurate with additional rounds.

Three agents tested different update laws on the same frozen interaction
corpus. Every input edge is present at round zero. A round does not reveal
another target, choose a move, make a successor board, or expand a branch.

    h0 = encode(input)
    h[k+1] = update(h[k], fixed_input)
    answer[k] = readout(h[k])

The update and latent dimensions stay fixed within a run. These are resource
interaction probes, not measurements of chess plies or chess playing strength.
Algebraic interaction order, message dependency radius, and learned recurrent
inference are different notions of depth; their round counts are not equal-cost
units and should not be ranked against one another.

## Shared task and controls

Every target needs a distinct adjacent resource. The corpus has 3,800 graphs:
1,760 TRAIN, 440 DEV, 1,360 random EVAL, and 240 structured cases. Random EVAL
contains 880 graphs at training sizes 3–6 and 480 larger graphs at sizes 7,8,10.
It is label-balanced by construction. All maximum-matching labels were checked
by both Hall deficiency and independently implemented subset dynamic programming
before the lanes ran. Neither oracle is part of an inference update.

Structural cases include forced chains, cycles, shared-resource deficits,
degree-preserving positive/negative pairs, and the previous Gram-equal pair.
The degree pairs preserve every target degree and resource degree while changing
whether a complete assignment exists. Raw graphs are distinct across splits;
this is not an isomorphism holdout, and coherent permutation replicas are not
independent evidence.

Protocols, source, corpus, seeds, and learned training budget were frozen before
measurement. Models use TRAIN and DEV only for fitting/checkpoint selection;
all four learned checkpoints freeze before EVAL/STRUCTURE predictions.

## 1. Repeated exact algebraic composition

The latent state contains simultaneous target/resource subset relations.
Initially it has all single-edge terms. Each round closes it under disjoint
composition, excluding repeated target or resource use. This raises interaction
order without revealing new input or storing individual assignment paths.

On a declared bounded subset of 210 cases:

| Rounds | Correct feasible | Correct impossible | Unknown | Incorrect certificates |
| ---: | ---: | ---: | ---: | ---: |
| 0 | 0 | 0 | 210 | 0 |
| 1 | 5 | 3 | 202 | 0 |
| 2 | 70 | 25 | 115 | 0 |
| 3 | 112 | 70 | 28 | 0 |
| 4 | 112 | 98 | 0 | 0 |

All maximum-cardinality facts were already complete at round three. Round four
only observes saturation to issue conservative negative certificates. Frozen
state and additive reinjection remain unknown throughout; removing resource
exclusion produces 77 false feasible claims.

This is exact circuit depth with explicit combinatorial state. Its tensor has
2^(targets+resources) Boolean coordinates, and its composition indices grow
exponentially too. Avoiding a branch tree does not remove that cost. It is not
evidence of a scalable learned latent solver.

[Algebra details](LATENT_ALGEBRA.md)

## 2. Synchronous constraint messages

Targets send exactly-one support constraints; resources send at-most-one
constraints. Every update reads only the previous round. No assignment is
chosen, no branch is made, and an unresolved fixed point stays UNKNOWN.

Across all 1,600 EVAL plus structural graphs:

| Rounds | Correct feasible | Correct impossible | Unknown |
| ---: | ---: | ---: | ---: |
| 1 | 0 | 521 | 1,079 |
| 2 | 0 | 588 | 1,012 |
| 4 | 20 | 645 | 935 |
| 8 | 78 | 684 | 838 |
| 16 | 105 | 695 | 800 |
| 19 | 113 | 695 | 792 |
| 32 | 113 | 695 | 792 |

There are zero incorrect trusted certificates. Supported/required-edge audits
also pass on all 803 feasible roots.

An n-target forced chain needs exactly 2n−1 rounds: one round forces a target's
only choice, the next excludes competing resource use, and the consequence
reaches the next target in the following round. Ten-target chains need 19
rounds. Repeating the original one-pass calculation without feedback never
achieves this propagation.

Cycles and the degree-preserving Hall pairs reach an immediate unresolved fixed
point. More iterations of this update law cannot resolve their collective
constraint. The operator lacks that inference, regardless of its allowed depth.

[Message details](LATENT_MESSAGES.md)

## 3. Learned tied recurrence

Four small typed recurrent probes have 16,897 parameters each. Two seeds train
at four rounds; two train with sampled depths one through eight. Within each
frozen checkpoint, evaluation changes only the number of recurrent updates.
The states are target/resource vectors with adjacency-masked messages and shared
GRU updates, not a proposed final JPEG architecture.

Mean accuracy over the two seeds, on 1,360 random EVAL graphs:

| Latent rounds | Trained at 4 | Trained at 1–8 |
| ---: | ---: | ---: |
| 1 | 83.97% | 95.85% |
| 2 | 97.65% | 96.95% |
| 4 | 98.60% | 98.31% |
| 8 | 91.95% | 98.53% |
| 16 | 77.17% | 97.28% |
| 32 | 74.30% | 94.60% |

These are descriptive curves, not a held-out choice of the best depth. One
four-round-trained model falls from 98.38% at four rounds to 55.74% at 32.
Resetting state each round reproduces the one-round answer exactly, despite
the repeated computation.

The structural test is less encouraging: both variable-depth models score
50% on all 64 degree-preserving Hall-pair cases at every tested depth. One
fixed-depth seed transiently reaches 87.5% at eight rounds; the other remains
at 50%, and both are at 50% at 32 rounds. Random accuracy gains do not establish
reliable higher-order compatibility reasoning or depth extrapolation.

Zero-message and shuffled-coupling controls are destructive interventions using
the same weights, not separately trained baselines. The shuffled input also
retains original resource-degree features and labels, making it inconsistent
with the altered adjacency; it does not isolate alignment alone. Training
regimes differ in training compute, sampling trajectory, and checkpoint
selection depth, so this is not a causal comparison of those regimes.

[Learned probe details](LATENT_LEARNED.md)

## What this means for the architecture

Latent recurrence can carry consequences farther and compose higher-order
relations while the input stays fixed. Depth is useful when the update creates
new relevant interactions. Repeating an operator that has already stalled adds
nothing; repeating a learned update outside its trained range can make the
answer worse. The choice of interaction law and the stability of its repeated
application therefore need to be evaluated together.

The parent independently reran exact algebra, full message inference/audits,
and every learned checkpoint's depth/control predictions. Five message
schedule/certificate tests pass. Learned logits, metrics and transitions match
the frozen outputs exactly. These checks establish the reported small-task
behavior, not chess move quality. [Reproduction and audit](../../../experiments/codec_lab/latent_depth/README.md).
