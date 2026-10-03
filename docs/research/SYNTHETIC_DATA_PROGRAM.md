# Synthetic data program — 2026-10-03

Train the offline models as long as useful data and optimization improve them;
select deployment candidates by decision quality at an inference budget. Cheap
synthetic rules data can be regenerated almost indefinitely. Strong strategic
labels, independent game families and coverage of difficult decisions remain
separate resources. Repeating one teacher's errors is not more supervision.
This is the generation design and first-pack proposal. The
[reachable mechanics generator and pilots](../SYNTHETIC_MECHANICS.md) now exist;
tactical trace production, strategic labeling and the large training study
still need implementation and execution.

The [algorithmic redesign](ALGORITHMIC_REDESIGN.md) defines the bounded search
reference and recurrent tactical operator. The neural worktree's
[scaling study](../../../chen-neural/docs/SCALING_STUDY.md) owns model sizes,
optimization and execution declarations. The simulator owns legality, terminal
values, history and exact topology throughout.

## Label types and what they establish

| Label class | Examples | Meaning and training use |
| --- | --- | --- |
| Exact current-state facts | Legal action IDs, next state, attacks, pins, checkers, rule counters | Simulator facts under the declared rule/history contract; useful mechanics and auxiliary targets, not evidence of winning play |
| Exact restricted algorithm outputs | Fully completed same-square legal exchange tree, deterministic completed finite-horizon minimax with declared leaves | Exact for that artificial policy/problem, not game-theoretic chess value |
| Completed search bounds | Exact/Lower/Upper from the bounded alpha-beta/quiescence reference | Bounds under identical state, history, evaluator and continuation semantics; abort supplies no score target |
| Stronger teacher estimates | Finite-node Stockfish complete-root action scores and rankings | Teacher search estimates with provenance; ordinary CP is not an exact minimax value or certified confidence interval |
| Outcomes | Completed game W/D/L, or separately admitted tablebase result | Game outcome depends on both policies; tablebase meaning depends on metric and draw rules. Keep these distinct from CP and restricted tactical labels |

Targets and future traces live in the supervision record. They never enter
observed-state features, action priors or inference memory. A known attack bitboard
can be a causal input; reconstructing that same bitboard measures copying of a
prior, not tactical reasoning. Keep mechanics pretraining and decision learning
metrics separate. Terminal and mate examples use typed targets rather than
arbitrary large ordinary-CP regression values.
Restricted exchange/quiescence values use task-specific outputs. They need
not equal or bound a stronger teacher's strategic score, even on the same board;
do not train one scalar to satisfy incompatible horizon/evaluator targets.

## Generate reachable states and retain their history

Start ordinary games from the standard initial position. Generate replayable
legal playouts using a mixture of classical-policy self-play, stronger-teacher
play, bounded-temperature alternatives and explicitly uniform legal moves.
Uniform-random play is one diversity source, not the target playing distribution.
Use opening diversity, phase/material buckets and both movers; sample at most
eight separated observation roots per source game for the first pack. Cap games
at256 plies; capped games have no terminal outcome label. Save the entire legal
prefix even when only the reversible history is needed by a consumer.

For rare events, use legal prefix search and rejection sampling: guide play
toward castling, EP, promotions/underpromotions, discovered attacks, pins,
checks, repetition and high halfmove clocks, then retain the prefix witness.
Local counterfactuals are legal moves and legal reply sequences from an admitted
root, not pieces pasted onto an arbitrary board. Authored FEN fixtures without
a reachability witness remain a separate mechanics/OOD suite. Locally legal
random boards do not certify legal reachability or meaningful repetition history.

A family contains its source game, all sampled roots, legal successor groups,
counterfactual branches, trace states and augmentations. Give each family a split
before expensive labeling. Deduplicate admitted observations and planned child
states across splits using canonical board/action identities as well as exact
history-sensitive identities. Resolve duplicate families before label inspection;
do not merge every game merely because they share the initial position. Record
sampling policy and inclusion probability where known, so oversampled hard
strata can be distinguished from natural-distribution evaluation.

Use the existing spent-input/family exclusion registry, including planned and
rejected old inputs; do not reopen closed neural holdouts. Labels may be generated
for a sealed test pack, but training/development code cannot read them. Fresh
challenge families and an unmined natural-distribution test both remain closed
until a frozen selection. Failure-driven mining uses TRAIN/development only.

## Cheap exact supervision: the large-volume layer

For every admitted root, export complete legal support and the exact effect of
each action: piece relocation, capture/victim, promotion, castling rook edit,
EP edit, rights, clocks, mover, hash and terminal/history changes. Store variable
legal counts with masks; never truncate high-branching roots to preferred moves.
Do not count each child as an independent family.

Supply outgoing pseudoattack bitboards by source square, including empty-square
targets and the first occupied blocker. Derive incoming/outgoing counts and
typed piece attack/support edges from this same topology. Pins and current-mover
king checkers are separate facts: pseudoattacks are not legal moves, and pinned
pieces still have pseudoattacks. Bind object slots to squares and retain canonical
action IDs, node/hash/history/epoch and feature schema. The native producer must
admit content and overflow, not just tensor shapes.

Useful exact training tasks include action-edit prediction, attack-ray changes
after a move, newly exposed attacker/defender relations, check-evasion support,
and special-move event composition. Sample unchanged facts as negative controls
alongside changed facts; otherwise a model can predict "everything changed".
For incremental models, supervise both the edit and agreement with a full exact
recompute over legal multi-ply sequences. Store intermediates as labels, not
additional future input. Exact input facts need not all be decoded at deployment.

## Exchange and quiescence traces: teach computation

Define a legal exchange task explicitly. Apply a legal initial capture/promotion,
then enumerate legal recaptures onto its occupied destination square, recomputing
attackers after each capture. Allow declining the exchange in a non-check state.
Record participant identities, king/pin legality, material delta, newly exposed
slider rays and bottom-up max/min decisions. This is a restricted exchange game,
not unrestricted tactics. If check requires moves outside this task, classify it
as unsupported or switch to the separately declared full-evasion policy; never
claim a partial recapture list exhausts check evasions. EP's initial off-square
victim and promotion values must be explicit. An exhausted exchange tree has an
exact restricted value; a16-ply or node-cap interruption has an abort/missing
value, not an exact SEE target.

For quiescence use the admitted bounded reference: captures/promotions outside
check, all evasions in check, no stand-pat in check, full terminal/draw/history
handling. Start with depth2, qply cap16 and32,768 entered states per root job.
An unresolved q continuation at the cap aborts; preserve reason and charged work.
Completed bounds refer to the declared search policy. Do not turn Lower/Upper
into point values or treat a pruning return as a fully evaluated subtree.

Trace records contain task-local state/action IDs, mover frames, legal/tactical
support, enter/return events, alpha/beta, stand-pat eligibility, completed bound
tags, child negation, incumbent changes and aborts. Supervise intermediate
attacker sets and completed backup decisions with trace loss on/off controls.
DFS event count is not the neural recurrent-step count; define an explicit
projection onto local exchange stages or dependency rounds, and test different
trace lengths. Ground-truth intermediate states fed during teacher forcing are
a named training control; deployed recurrence receives only its own state plus
current exact observations.

For a small, diverse subset, a no-cutoff traversal checks completed scores and
narrow-window inequalities independently. Add capture ordering only as a named
trace policy. Vary legal move order on TRAIN to detect memorization of canonical
DFS order; compare values under the same completed continuation semantics.
The current reference exports result packets, not these full training traces;
trace production and independent admission must be implemented before this tier
can run. The [bound composition model](../../proofs/SearchBounds.lean) assumes
sound completed bounds; it does not certify the trace emitter.

## Strategic supervision: pay for complete decisions

Label a smaller selected subset with a pinned stronger Stockfish teacher. Start
with complete-root MultiPV over every legal action at262,144 requested nodes,
Threads1, fixed Hash and cleared per-query state; record teacher binary/net,
settings, node/depth/work, legal PVs and every score emission. Use the existing
[ordinary packet protocol](ORDINARY_SCORE_PACKETS.md): one complete ordinary
emission at a common reported depth, no borrowed ranks or point labels from
bounds. This is uniform packet provenance, not a uniform completed child horizon.
Missing/ineligible rows leave a teacher-label mask while exact facts survive;
report acceptance by stratum rather than silently replacing difficult families.

Keep action-value regression, pairwise ranking, chosen-action regret and optional
WDL objectives separate. Convert all children into the parent mover's score
frame. Use complete support for ranking; do not show the model only teacher top-K.
Near-tie buckets use the declared teacher CP scale: gaps0–25,26–100 and>100CP;
retain ties and avoid imposing an arbitrary one-hot winner. These thresholds are
initial sampling definitions, not calibrated confidence boundaries.

Increase the teacher budget4× on a preselected TRAIN/development calibration
subset before choosing a production label tier. Measure best-action changes,
pairwise reversals and gap changes. Disagreement is a stability diagnostic,
not a confidence interval. Never infer a Chen material-scale conversion from an
unrelated standalone versus parent query; retain query kinds and frame metadata.
Tablebase/mate/decisive evidence goes to its typed task, not ordinary CP. Completed
self-play outcomes, if collected later, are kept separate from teacher scores.

## Mine useful failures, including quiet defense and sacrifices

Use current engine losses to identify motifs and write replayable diagnostics;
they do not become a fresh test set. Exact roots already spent in evaluation or
older corpora remain in that diagnostic ledger. Generate fresh TRAIN games with
the same failure mechanisms and mine new on-policy positions. Future public
diagnostic roots may enter a separately declared TRAIN-only correction shard,
but cannot be called fresh holdout evidence or bypass the exclusion contract.

Freeze the mining engine/model per round. Compare its chosen move with stronger
complete-root labels; retain the chosen move, best competitor and refuting legal
reply sequence. Prioritize high regret, unsupported quiet defenses, checked
static leaves, sacrifice-like material losses, changing discovered rays and
large value reversals. Mix hard examples with ordinary roots and easy negatives;
a corpus containing only blunders teaches the wrong deployment distribution.

Quiet-defense candidates are non-capturing, non-promoting legal moves that a
stronger analysis favors over apparently attractive alternatives. Sacrifice
candidates have an adverse immediate restricted exchange/material signal but a
better stronger-teacher continuation. These are teacher/miner-defined strata,
not exact claims about a winning tactic. Preserve the refutation and teacher
budget, and include failed-sacrifice/irrelevant-quiet-move controls.

All generated counterexamples and their near neighbors stay in the originating
family/split. Do not mine model errors on TEST. Stop a mining round when its
fixed queue/resources end; publish queue dispositions rather than pretending
only successful labels existed. A new round trains a new offline checkpoint.

## Legal transformations

Action and topology relabeling must commute with exact transitions. Rank
reflection with color/mover exchange is useful only with a full mapped legal
prefix and castling/EP/history/hash handling. If a standard-start witness is
unavailable, keep the transformed sample in an explicit equivariance task.
File reflection is not a general symmetry with orthodox castling; initially
restrict it to states with no castling rights, certify transformed legal support
and transitions, and declare transformed reachability separately. Never rotate
or reflect pawn directions, rights or EP casually. Do not assume strong-teacher
scores are bit-exact under a transformation; requery them or use exact-label
equivariance only. Original/transformed roots share one family.

## First pack: proposed P1, then scalable increments

P1 begins with a CPU mechanics pilot of256 TRAIN families, no closed-test access,
then an8192/1024/1024 TRAIN/development/TEST family declaration. Propose eight
observations per family: up to65,536/8192/8192 exact roots plus complete action
effects. These are ceilings, not claimed accepted counts. Early terminal games,
duplicate exclusion and failed strata reduce yield; freeze replacement policy
and any pre-generated reserve list before teacher results. This volume is a
first measured increment, not a limit on offline overtraining.

| P1 TRAIN family allocation | Families | Generation/selection intent |
| --- | ---: | --- |
| Broad reachable play |4096|Half classical/stronger-policy play, half temperature/randomized legal deviations; opening/middlegame/endgame coverage |
| Tactical and special-event play |2048|Checks, exchanges, pins/discoveries, EP/castling/promotion and repetition/clock events |
| Fresh error-driven play |1024|New on-policy refutations, quiet-defense and sacrifice candidates, with matched negative controls |
| Endgame and near-tie play |1024|Legal low-material continuations and stronger-teacher gap buckets |

Allocate development/test families independently with the same broad target
mixture and separately report natural-distribution versus challenge results.
Rare-target quotas are generation goals; exact achieved counts must be reported.
Near-tie acceptance is determined by the declared teacher procedure, not model
errors. Keep the cheap root bank and costly-label subset as separate records.

Initial expensive ceilings: one depth2/q16/32,768-node trace job per family
(8192/1024/1024 jobs), and4096/512/512 complete-root teacher queries at262,144
nodes. The latter is about1.07B requested TRAIN teacher nodes; q TRAIN ceiling
is about268M entered states. These distinct node units cannot be added as equal
work. The4× teacher calibration uses128 TRAIN and32 development roots, no TEST.
Select its roots before baseline teacher outcomes. Aborts/ineligible labels
remain missing and retain exact data; no unbounded rescue queries.

Before launching P1, time the256-family pilot and publish labels/sec, accepted
family rate, query work, bytes/root and projected wall time. Set actual worker,
wall, disk and cleanup bounds from that measurement; do not assume old small-pack
throughput scales linearly. Store exact state/move/topology plus compact variable
action rows rather than float-expanded features for every successor. Dense
64x64 graphs and identical board planes need not be stored repeatedly. Streaming
shards and reproducible regeneration separate resident training size from total
offline data volume. Implement and verify the generator and trace/teacher
adapters, then execute the bounded pilot before admitting the larger pack.

## Learn to convergence and measure the right scaling axes

Use nested TRAIN family banks at 1×/2×/4×/8×/16× P1, with added increments from disjoint new seeds
with unchanged closed holdouts and source distributions. Repeated exposures,
legal children and augmentations are not new independent families. Independently
vary model capacity, recurrent steps, data volume and teacher quality; do not
change all four and attribute the result to model size. Fix inference budgets
for deployment comparisons and include exact-child versus parent/action controls.

For each size, run declared development learning-rate/regularization and
auxiliary-loss tuning, including trace on/off and decision-only controls. Train
until stratum-aware development decision metrics plateau across repeated checks,
then test whether fresh TRAIN increments improve them. Report loss/decision
curves against unique families, optimizer steps, repeated exposures and actual
training/label-generation seconds. A safety cap before plateau means "not
converged", not "architecture failed". Equal short-update runs are learning-curve
controls; an offline model may be deliberately overtrained to save inference.

Predeclare plateau/selection rules in the executable study, retain multiple
seeds and the chosen checkpoint, and evaluate its sealed TEST once. Measure
complete-root regret, gap-bucket ranking, quiet-defense/sacrifice performance,
special-move/history correctness and longer-than-training trace generalization.
Then compare frozen winners at measured producer+inference+search budgets and
paired equal-time play. Teacher agreement, restricted-algorithm fidelity and
actual playing outcomes remain separate axes. No fitted scaling exponent is
credible from the existing small, short studies alone.

## Online adaptation has a separate contract

Offline training and new data-mining rounds use backpropagation. Deployment
adaptation here means branch-local recurrent/object memory, completed causal
search evidence, extra computation or refresh decisions with frozen weights.
It must retain state/history/mover/epoch identity and cannot consume future
teacher labels. Online experience can be logged for a later offline pack; it
does not silently update weights or contaminate a held-out evaluation. Training
the adaptation policy is a later declared experiment after an offline quality
signal and drift/branch-isolation controls.
This state adaptation is distinct from online training. A non-gradient head
update, if explored, needs its own causal target, update cost, reset/split rules
and frozen-head comparison; it does not replace the offline scaling study.

## Primary precedents and limits

- [Neural Execution of Graph Algorithms](https://arxiv.org/abs/1910.10593)
  supervises algorithm steps, supporting trace targets rather than only final
  outputs. Its graph results do not establish chess minimax generalization.
- [CLRS](https://arxiv.org/abs/2205.15659) supplies generated algorithm tasks and
  standardized evaluation. Borrow explicit task semantics and extrapolation
  tests, not an assumption that more synthetic instances guarantee reasoning.
- [Amortized Planning with Large-Scale Transformers](https://arxiv.org/abs/2402.04494v2)
  studies Stockfish-annotated chess with data/model/target ablations; its
  [released implementation](https://github.com/google-deepmind/searchless_chess)
  provides state/action-value controls. This supports larger offline distillation
  studies, not a transfer of its rating or optimal data ratio to Chen.
- [Stockfish NNUE training documentation](https://github.com/official-stockfish/nnue-pytorch/blob/master/docs/nnue.md)
  distinguishes search evaluation from game-result supervision. Keep both label
  sources and their weights explicit instead of calling either exact truth.
- [DAgger](https://arxiv.org/abs/1011.0686) motivates labeling states induced by
  the learner to address distribution shift. Our offline failure-mining rounds
  borrow that mechanism; its guarantees require assumptions not established
  for this bounded chess teacher or finite dataset.
