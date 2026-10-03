# Algorithmic redesign — 2026-10-03

The immediate question is useful chess work per decision. The current GPU
selective engine's one-second reference scored 4W/5D/7L against configured
Stockfish 1320, an uncalibrated proxy 1254. It used classical Tapered evaluation,
not a neural candidate. Kernel parity and throughput do not establish a good
search policy. Separate CPU and GPU reference matches do not isolate a calibrated
rating difference. [Match evidence](../harness/gpu-one-second-reference-2026-10-03.md).

This document supersedes the near-term optimization order in older research
notes. Frozen experiments remain comparison arms. No new search or neural
candidate described below has established a playing-strength gain.

## Findings that change the implementation order

| Mechanism | CPU alpha-beta | Production GPU selective |
| --- | --- | --- |
| Leaf continuation | Quiescence: captures/promotions, all check evasions | Immediate static score, including nonterminal checked positions |
| Bounds | Fail-soft alpha-beta, typed TT bounds | Mixed-depth backed-up estimates, no alpha/beta windows |
| Work allocation | Ordered depth-first search with cutoffs | Greedy best-open refinement; a batch uses one score snapshot |
| Completion | Iterative deepening retains last completed iteration | Growing mixed-depth tree; maximum frontier depth is not uniform coverage |
| Reuse | History-sensitive direct-mapped TT | No transposition table; UCI Hash does not create one here |

Sources: [CPU search](../../src/search.rs),
[GPU selective classification and allocation](../../gpu/chen_selective.inc).
The GPU tree admits complete legal siblings and has coherent terminal/mate
propagation. Its ordinary mixed-depth estimates are not bounds on an unresolved
deeper horizon. These differences are plausible causes of wasted search, not a
causal explanation of every lost game.

[CPW Search Progression](https://chessprogramming.org/Search_Progression) identifies
quiescence, TT and PVS as foundational search work before extensive pruning and
tuning. [Quiescence](https://chessprogramming.org/Quiescence_Search) addresses
tactical horizon errors and disallows stand-pat while checked. We are missing
these foundations in the production GPU policy. Korf and Chickering's original
[best-first minimax study](https://www.maxchickering.com/publications/aij96.pdf)
also motivates minimum coverage before refinement when static estimates can
underestimate alternatives; its random-tree/Othello results are not chess Elo
predictions.

The CPU reference has another tradeoff: its TT identity folds ordered reversible
history. This protects draw context but loses ordinary board transpositions
across different move orders. Investigate a board-only ordering cache or a
carefully defined compatible-context score cache. Removing history from score
identity without a replacement correctness argument is not the next change.

## Search candidate: device-owned bounded negamax

Implement a separate experimental GPU worker, initially without TT, PVS, LMR or
null move. Reuse exact HIP legal generation, transitions and classical evaluation.
Use an explicit device stack and branch-local history, with ordered moves and
alpha-beta windows. Tactical continuation includes captures/promotions and all
legal evasions in check. Give stack exhaustion, node exhaustion and cancellation
explicit failure semantics rather than silently static-scoring unfinished work.

The result contract is:

```
state identity + evaluator identity + horizon/search-semantics identity
requested window + score + Exact/Lower/Upper/Aborted
completed depth + best move/PV + charged work + stop reason
```

An aborted job supplies no reusable score bound. A completed bound is relative
to the declared horizon, terminal/draw rules and quiescence semantics, not the
game-theoretic value of chess. A bounded quiescence horizon must be declared;
resource exhaustion cannot masquerade as completing that horizon. A prototype
may return only the first PV move; it must say so rather than invent a full PV.

First compare independent root-position jobs and root-split jobs with full
windows. Then introduce device-owned iteration/publication: retain the previous
completed root result while the next iteration runs. This keeps the full-GPU
target; CPU work is protocol, limits and oracle comparison. A one-thread-per-job
reference establishes semantics, not the final GPU scheduling architecture.

After reference parity, the consequential parallel experiment is a dependency
work pool: establish the ordered first child's bound before releasing later
siblings, use zero windows with correct re-search, and cancel work rendered
irrelevant by completed bounds. Add a device TT with depth/horizon, bound, mate
normalization, evaluator and draw-context rules. Neural values can order jobs;
they cannot certify bounds. See [PVS](https://chessprogramming.org/Principal_Variation_Search)
and [Stockfish's actual search](https://github.com/official-stockfish/Stockfish/blob/sf_19/src/search.cpp).
Stale windows, divergent workers and speculative sibling work must be measured.

### Controls and decision metrics

Keep Tapered fixed while comparing:

1. Current selective-static policy.
2. Selective policy with tactical leaf continuation.
3. Bounded alpha-beta worker, followed by separately admitted PVS/TT variants.

Compare completed fixed-horizon decisions to an unpruned oracle with identical
draw, leaf and quiescence rules. Exercise narrow-window inequalities and aborted
jobs separately. Exact-horizon equality does not automatically cover conventional
deeper-entry TT reuse.

For equal-time decisions, record per-root-action open refinement coverage,
first-opponent-reply coverage, unresolved tactical leaves, score reversals,
stale/cancelled work, TT reuse, completion horizon and stop/cap reasons. Measure
time to find or refute a move and paired playing strength. Current counters do
not quantify the fraction of tactically wasted work. CPU visits and GPU newly
classified positions are different units; equal node counts alone are inadequate.

## Neural audit: what exists and what failed

All current learned encoders begin with schema1 mover-oriented
`[B,18,8,8]` observations. Learned features omit repetition; the simulator owns
history and terminal decisions.

| Implemented family | Actual computation | Quality status |
| --- | --- | --- |
| Square | 64 square tokens, residual MLP with global mean context | Trained small studies; no selected quality winner |
| Relation | Single-head dense 64x64 attention, static geometry/pseudoattack bias | Same negative studies; not a modern multihead transformer |
| Plan | Relation body plus eight learned queries and square/query attention | Same negative studies |
| Threat | Four typed attack/support message operators over dense square tensors | Latest study; masking empties does not make sparse objects |
| Semantic transitions | Frozen eight-role anchor; legal action and derived child diagram predict child roles | Action-conditioned/blind/shuffled controls; selection NONE |
| Multihead transformer + symbolic projection | Pre-LN attention and learned current-state residual injections | Untrained mechanics prototypes |

Attention and learned query bottlenecks borrow from
[Transformer attention](https://arxiv.org/abs/1706.03762) and
[Perceiver](https://proceedings.mlr.press/v139/jaegle21a.html).
Typed relation messages are established [graph-network](https://arxiv.org/abs/1806.01261)
techniques, with chess graph prior art in [AlphaGateau](https://arxiv.org/abs/2410.23753).
The multihead control is not a [Chessformer](https://arxiv.org/abs/2605.19091)
reproduction. Supervised latent transitions are related to action-conditioned
dynamics in [MuZero](https://arxiv.org/abs/1911.08265); our code does not implement
MuZero's planning/self-play algorithm. Local exact-diagram routing and role
composition are hypotheses, not validated novelty claims.

Completed studies:

- [Value](../../../chen-neural/docs/VALUE_STUDY.md): 27 runs, three encoders,
  three heads, three seeds. No aggregate MAE or chosen-child regret winner over
  material; width 32/depth 1, 2048 updates.
- [Strategic](../../../chen-neural/docs/STRATEGIC_STUDY.md): 30 runs,
  conditioning/anchor controls, two seeds. Joint gate selected NULL;
  width 16/depth 1, 1024 updates.
- [Semantic corpus](../../../chen-neural/docs/SEMANTIC_CORPUS_STUDY.md): eight
  anchors and 32 transitions; width 32/depth 1, 384 anchor/192 transition updates.
  Selection NONE; TEST closed. Some scalar/ranking improvements miss the joint
  two-seed gate. Transport and conditioned rankings coincide despite latent
  differences.

Head types, encoder families, conditioning/blinding/shuffling and small refresh
controls are actual ablations. Broad hyperparameter tuning has not happened.
Learning rate, auxiliary weights, role count, recurrence/depth, width, teacher
budget and ranking weights were mostly fixed. No convergence/data-scaling curve,
matched inference-time architecture comparison or neural strength match exists.
Exact-child re-encoding was a target, not an admitted competing policy control.

Latest supervision has 233 TRAIN/62 validation root families. Finite-node teacher
queries have differing standalone/parent-MultiPV contexts. At width 32, the board
decoder is 213824 parameters, around 79–83% of active anchor parameters; all training
heads account for around 93–96%. Inference skips these heads. The concern is
untested allocation of learning objectives, not decoder cost at deployment.

## One new processor, with two substantive extensions

### A. Recurrent tactical operator network — implement first

Represent actual piece objects (at most 32), 64 squares and every legal root action
as nodes. Consume exact supplied typed occupancy, pseudoattack, pin, check and
action-edit edges. Preserve the distinction between attacks and legal moves.
Use shared recurrent learned message/update operators for 1/2/4/8 iterations,
with competition-sensitive aggregates and action/value readouts. Piece identity,
orientation and action support belong to the exact ingress contract. This is
explicit tactical computation, not another renamed square attention layer.

Hypothesis: iterating over attacker/defender and discovered-line relations can
learn useful exchange/refutation reasoning more efficiently than compressing
the board once into eight tokens. Borrowed ideas are algorithm-step supervision
from [Neural Execution of Graph Algorithms](https://arxiv.org/abs/1910.10593) and
learned message/aggregate operators from
[NBFNet](https://proceedings.neurips.cc/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html).
Bellman–Ford is not chess minimax. Chess-specific operator composition and its
benefit are the experiment.

Controls use the same ingress/readout: one step versus recurrent steps; typed
versus untyped edges; trace auxiliary loss on/off; full exact-child evaluation
versus parent/action processing. Optional SEE/quiescence trace targets are
declared bounded algorithm outputs, not perfect minimax truth. Teacher values
and future search results never enter observed-state inputs.

Approximate ordinary message-passing cost is
`R * ((P+64+K)*D^2 + E*D)` before action-specific graph construction. This is a
cost model, not a measured speed claim. Dense action overlays, edge construction
and GPU scatter/reduction can erase the expected benefit.

### B. Event-driven persistent object memory — extend A after quality

Maintain branch-local piece memories and persistent identities. Exact move events
change moved/captured/promoted objects and attack rays; propagate messages through
affected relations, with periodic full refresh. Compare the same trained processor
re-encoded at every node, no-memory updates, additive incremental features and
fixed refresh intervals. Test castling, EP, promotion, discovered rays, king-context
changes, branch isolation and long-path drift.

Memory ideas borrow from [Temporal Graph Networks](https://arxiv.org/abs/2006.10637);
exact incremental accumulation has precedent in
[NNUE](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md).
A nonlinear local update is not equivalent to full re-encoding. Changes can
propagate globally; locality and refresh savings must be demonstrated.

### C. Search-aware adaptive computation — conditional extension

Use causal completed-search context, with state/epoch/perspective, to predict
ranking-error risk and select more operator iterations, latent transport or full
refresh under a compute budget. Compare always-refresh, always-transition,
fixed-interval and fixed-computation controls. Measure calibration, missed tactical
refreshes and strength/time. Learned risk cannot omit legal moves or certify
alpha-beta pruning. [PonderNet](https://arxiv.org/abs/2107.05407) supplies an adaptive
computation precedent; chess-search refresh allocation remains our hypothesis.
Do this only after A is useful and B has a controlled drift envelope.

## Stop recomputing exact facts

Current `Diagram.edges` constructs dense blocker contractions; conditioned
transitions repeat successor diagrams. `SymbolicResidualAnchor` still calls the
ordinary encoder, so native compact channels do not eliminate topology rebuilding.
These are concrete duplication, not an inferred GPU counter bottleneck.

Build per-source attack bitboards once for an admitted exact state, then derive
counts and sparse typed edges for all consumers. A 64-u64 topology payload is 512 B;
two 64x64 boolean planes would be 8192 B before tensor overhead. This is a storage
comparison, not measured runtime or a claim that every source is occupied.
The current twelve compact channels cannot reconstruct source-target relations;
this needs an extended typed topology seam. A counts-only model is a separate
trained variant. First verify full production, then incremental repair of changed
occupancy rays, special moves and piece types against full recomputation.

Do not start another dispatch/wave-width optimization cycle now. Existing
matched-dispatch gains are 0.2–0.45% and retained-prefix gains about 5% on larger
work. Captured FP32 is the current batched inference reference; keep INT8/FP16
controls for actual model/search batch shapes after a quality winner exists.

## Admission and execution order

1. Implement experimental bounded GPU search and tactical operator mechanics
   behind separate seams. No production promotion on mechanics alone.
2. Hold classical evaluation fixed for search correctness and tactical tests;
   compare equal-time decisions and direct paired games before scheduler tuning.
3. Admit fresh independent root families with the exclusion ledger. Declare one
   teacher-query protocol per target; retain terminal utility outside ordinary CP.
   Record label instability and tactical/quiet strata. Earlier consumed validation
   can explain failure but cannot serve as a fresh final evaluation.
4. Train exact-child and recurrent-action controls on identical data. Report
   convergence and data-scaling curves. Start with value and complete-root ranking
   objectives; auxiliary board reconstruction is an explicit on/off experiment.
5. Predeclare a bounded development sweep of recurrence, width, learning rate,
   ranking weight, trace weight and residual shrinkage with paired seeds. Freeze
   choices before fresh evaluation; account for training and inference time.
   The sweep budget is not yet sealed and no HPO is claimed.
6. Test a frozen neural winner in a fixed search at one second per move. Then
   investigate persistent memory and adaptive computation, retaining exact-child
   and fixed-budget controls throughout.

2FFS remains a distinct stochastic-tree research variant. Its slow-oracle and
bias assumptions do not turn deterministic learned scores into confidence
certificates. Lean work should next target completed-bound/abort composition,
dependency cancellation and topology repair under explicit premises. Existing
cache/selection certificates do not prove chess-policy optimality or GPU latency.

Ownership: main worktree owns GPU reference/search contracts and integration;
neural worktree owns the tactical processor and training studies. Root coordinates
commits and hardware windows; an independent reviewer checks search/topology
boundaries. Reuse admitted evidence; no duplicate search/model implementations.
