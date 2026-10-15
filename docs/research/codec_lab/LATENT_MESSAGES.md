# Latent depth as synchronous constraint interaction

Extra rounds can transmit consequences across a longer chain without exploring
child positions. This experiment demonstrates that effect on fixed resource
graphs, and also shows its limit: cycles and Hall-deficient cores can reach an
uninformative fixed point immediately. More depth does not repair an operator
that has no further deductions to make.

This is a discrete, hand-designed recurrent state, not a trained latent chess
model. There is no chess-strength result. All input incidences and constraints
are available at round zero. No branch, tentative assignment, rollout, child
world, or runtime matching solver appears in inference.

## Frozen experiment

Protocol commit `b130622`; measured source commit `1627eb4`, both before runs.
The common corpus is supplied by the parent, SHA256
`441a37e200d9460403e95b9318a9d4751790c13d4813e6f51cb3dcf5568311ab`.
Only its 1,360 EVAL and 240 structural rows are read by this lane. There is no
training or parameter selection. The EVAL partition has 680 feasible and 680
infeasible graphs; structure has 123 feasible and 117 infeasible graphs.

Each target requires exactly one resource, and a resource can serve at most one
target. Allowed target-resource incidences are fixed bitmasks. Every allowed edge
has a Boolean variable whose domain initially permits both zero and one.

The recurrent state retains which values remain possible for each edge. A target
factor forces its only remaining candidate, or excludes its other candidates
when one choice is already forced. A resource factor with a forced claimant
excludes competing claims. Local zero-candidate, incompatible-domain, or
multiple-forced-choice contradictions are recorded.

In compact form, with domain intersection understood componentwise,

    D[t+1] = D[t] ∩ target_support(D[t]) ∩ capacity_support(D[t]).

Every factor reads **only the previous state**. Updated rows are never consumed
by capacity factors in the same round, and updated capacities cannot propagate
through subsequent targets in that round. Counts evaluate these local exactly-one
and at-most-one rules directly; no assignment combinations are enumerated.

Every retained assignment is preserved by each local support rule. Thus induction
over rounds gives sound deductions. `FEASIBLE` requires a complete injective
selection of forced edges, checked directly. `INFEASIBLE` requires a local
contradiction from these valid deductions. Otherwise the answer is `UNKNOWN`,
which asserts neither feasibility nor infeasibility.

An independent offline augmenting-matching implementation audits corpus labels
and, on every feasible root, checks the support and necessity of individual
edges. Those audit-only forced/deleted-edge queries are never imported by
`solver.py` or fed back into its state.

## A real dependency-radius effect

Consider three targets with initial candidates:

    A → {a}
    B → {a,b}
    C → {b,c}

All of these facts are present before round one. The messages unfold as follows:

| Round | New information | Certified complete result? |
|---:|---|---|
| 0 | Every allowed edge permits zero or one | UNKNOWN |
| 1 | Target A requires `A=a` | UNKNOWN |
| 2 | Resource a excludes `B=a` | UNKNOWN |
| 3 | Target B requires `B=b` | UNKNOWN |
| 4 | Resource b excludes `C=b` | UNKNOWN |
| 5 | Target C requires `C=c` | FEASIBLE |
| 6 | No change | FEASIBLE, fixed point |

The interaction alternates: a target's obligation becomes a capacity exclusion,
which becomes another target's obligation. A later message depends on the earlier
message, not merely on evaluating the same independent features again.

All 44 permuted forced-chain fixtures follow the predicted rule: an n-target
chain first certifies at **2n−1 rounds**, then reaches a fixed point at 2n.

| Targets | Fixtures | First certificate round | First fixed-point round |
|---:|---:|---:|---:|
| 2 | 4 | 3 | 4 |
| 3 | 8 | 5 | 6 |
| 4 | 8 | 7 | 8 |
| 6 | 8 | 11 | 12 |
| 8 | 8 | 15 | 16 |
| 10 | 8 | **19** | 20 |

The round-19 probe matters: the ten-target chain is unresolved at round 16 and
resolved at 19; reporting only 16 and 32 would obscure its actual propagation
distance. Missing-final-resource chain variants propagate a contradiction from
both ends and certify at round n, for every tested n=2,3,4,6,8,10.

This is not numerical settling. Domains are discrete monotone support sets and
the trace identifies the exact new edge value at each step. Permuting target
order does not change the schedule; an in-place traversal would fail that test.

## Whole-corpus depth curves and controls

| Synchronous rounds | FEASIBLE certificates | INFEASIBLE certificates | UNKNOWN | False certificates |
|---:|---:|---:|---:|---:|
| 1 | 0 | 521 | 1,079 | 0 |
| 2 | 0 | 588 | 1,012 | 0 |
| 4 | 20 | 645 | 935 | 0 |
| 8 | 78 | 684 | 838 | 0 |
| 16 | 105 | 695 | 800 | 0 |
| 19 | 113 | 695 | 792 | 0 |
| 32 | 113 | 695 | 792 | 0 |

These are certified-coverage counts, not accuracy with UNKNOWN guessed as a
class. Many early negative cases have empty target neighborhoods. The clean
forced-chain schedule is stronger evidence of propagation radius than the
aggregate improvement alone.

On EVAL alone, certificates grow from 521 at depth one to 722 at depth 16
(69 feasible, 653 infeasible); 638 remain UNKNOWN. On the held-out structural
families, depth one certifies none, while depth 19 certifies all 44 positive and
42 negative chain fixtures. The remaining 154 structural fixtures stay UNKNOWN.

Three original-graph controls freeze at the round-one state forever:

- **Input-only one pass** caches the one-pass result.
- **Repeated original input without feedback** executes the original pass again
  at every depth, but never consumes its previous deductions.
- **Target-only messages** retain recurrence while removing capacity messages.

Each still certifies only the initial 521 negative cases, with no complete
positive witness and no additional exclusions. Performing more copies of the
same computation does not extend its dependency radius; recurrence requires the
right cross-factor information.

A fourth control independently relabels resource identities within each row.
This preserves target degrees but deliberately changes which targets compete for
the same resource. A single consistent global relabeling would merely be an
isomorphism and would not ablate coupling. By depth 16 the rowwise-shuffled
control produces 32 false-positive and 52 false-negative classifications against
the original graphs. These are **untrusted original-graph answers**, not valid
certificates; their edge-support audit is marked not applicable. The experiment
shows why preserving the actual shared identity matters, not that recurrence
alone guarantees correctness.

On all 803 feasible original roots, the independent audit finds zero removed
edges that participate in a feasible matching, and zero wrongly forced edges.
This holds for every original-graph control as well. Final monotone domains
contain all earlier deductions, so auditing the final domains also covers all
earlier prunings. All original-graph status claims agree with exact labels.

## Where more depth cannot help this operator

A feasible cycle such as `{a,b}, {b,c}, {c,a}` has no initially forced edge. Every
local factor supports all its edge values, so the state is unchanged at the
first round. It is correctly UNKNOWN even though complete assignments exist.
The solver does not choose one merely to obtain a result.

The infeasible graph `{a,b}, {a,b}, {a,b}` also has no singleton. Each individual
target/capacity factor remains locally consistent, while three simultaneous
obligations have only two shared resources. This Hall deficit requires an
aggregate interaction that the chosen local support state does not express.

All tested degree-preserving Hall-core pairs remain at the initial state:
32 infeasible graphs and their 32 feasible partners. They use every resource and
all targets have degree two; an isolated-resource shortcut cannot distinguish
them. The 39 feasible cycles, 35 resource-deficit cycles, and 16 Gram-equal
paired fixtures also remain UNKNOWN at a round-one fixed point. Extra rounds
16→19→32 produce no additional deduction on those 154 fixtures.

Thus depth and operator expressivity are separate. Coupled local support can
carry a real chain interaction farther. At its fixed point, repeating it cannot
invent a missing set-level capacity constraint. A richer state/operator would be
a new hypothesis, not evidence that unlimited repetitions of this one solve the
remaining graphs.

## Exploratory compute and reproducibility

Python 3.14.8 standard library, CPU4 affinity, five repetitions, first 512 EVAL
roots plus all 240 structural roots. Setup, corpus JSON, rewiring, exact labels,
and offline support audits are outside timings. Literal requested rounds,
domain-state materialization and a shape check are inside. Contradiction states
are absorbing; no unbounded convergence loop is hidden in a round.

| Mode | Depth 1, µs/root | Depth 8 | Depth 19 | Depth 32 |
|---|---:|---:|---:|---:|
| Coupled synchronous | 4.73 | 31.16 | 70.20 | 115.44 |
| Input-only cached pass | 4.75 | 4.95 | 5.31 | 5.75 |
| Repeated original, no feedback | 4.71 | 32.91 | 77.06 | 129.30 |
| Target-only, no capacity | 2.63 | 14.70 | 33.32 | 55.28 |

The no-feedback control spends comparable repeated compute without gaining new
facts. These are Python laboratory costs, not a kernel optimization or chess
engine performance claim. No compression or representation-size objective is
used.

```sh
python3 experiments/codec_lab/latent_depth/messages/run.py \
  --corpus /absolute/path/to/experiments/codec_lab/latent_depth/corpus.json \
  --out /tmp/latent-messages-rerun \
  --verify experiments/codec_lab/latent_depth/messages/results/curves.json
python3 -m unittest discover \
  -s experiments/codec_lab/latent_depth/messages -p test_solver.py
```

The output directory must be new. Five schedule/boundary tests pass. Frozen
per-family/per-size/depth curves, first-certificate/fixed-point histograms,
selected round-by-round factor messages, timings and source hashes are retained
under `experiments/codec_lab/latent_depth/messages/results/`. Two runs are compared
on deterministic curves; timing variation is kept separate.
