# Falsifiable chess latent experiments

The goal is a strong gfx1030 engine that can explore resident neural states and
learn from resident experience. The current `square`, `relation` and `plan`
models are experimental scaffolds for that program. Exact chess rules generate
legal states and adjudicate outcomes. Learned dynamics estimate strategic state;
they do not decide what moves become legal. A 2500 Lichess result requires
measured play and a named time-control population.

## Prior art and novelty boundary

The [AlphaZero chess report](https://arxiv.org/abs/1712.01815) establishes self-play
policy/value learning plus search as a relevant baseline. It does not imply
our resource budget or smoke data can reproduce its playing strength.
[MuZero](https://arxiv.org/abs/1911.08265) already combines latent transition,
policy/value predictions and tree search; an action-conditioned latent operator
is therefore prior art. [EfficientZero](https://arxiv.org/abs/2111.00210) studies
sample-efficient model-based learning and motivates comparing consistency
objectives, without transferring Atari performance claims to chess.

[AlphaGateau](https://arxiv.org/abs/2410.23753) applies graph representations and
edge-aware attention to chess. Chess graphs and attack edges are not a novelty
claim here. [Karvonen's chess world-model study](https://arxiv.org/abs/2403.15498)
shows board representations and player-skill-related latent interventions in
chess language models. Board probes and a skill vector are also prior art.
Our hypotheses concern chess-specific composition, certified latent exploration
and plan persistence under controlled compute. This literature pass is not an
exhaustive novelty search; no “first architecture” claim is justified.

## Implemented controlled variants

All variants use the same feature/action schema and shared reconstruction,
policy, value and action-conditioned transition heads. `square` supplies a
global-summary square-MLP baseline. `relation` adds occupancy-blocked pseudoattack
and geometric attention. `plan` adds a small shared bottleneck of learned plan
tokens to that relation model. The latter currently aggregates a static board;
it is not yet persistent plan memory or evidence of tactical understanding.

Their latent is 64×width, with one token per canonical square. An action names
source/destination/promotion, combines endpoint features, updates all square
tokens with explicit source/target flags, then changes viewpoint. Branching and
rollout APIs permit resident candidate-state exploration. The present smoke
trains one-step dynamics. Exact board, castling, EP and halfmove targets expose
state drift instead of hiding it behind value-only supervision. Full repetition
history is absent and must be added before draw-sensitive evaluation.

## H1: legal diamonds constrain strategic dynamics

Collect two **fully legal** four-ply sequences from a common state whose exact
endpoints match, including side, rights, effective EP and clocks. Begin with
independent quiet knight/bishop moves and exclude captures, pawns, rights
changes, checks and repetition-dependent terminations. One ordering might move
White knight A, Black knight B, White knight C, Black knight D; the alternate
ordering swaps each side's independent moves. Confirm every intermediate move
and endpoint using the rule oracle. Never assume arbitrary chess moves commute.

Train both orderings with endpoint latent consistency, in addition to ordinary
one-step supervision. The hypothesis is that these chess “legal diamonds”
reduce multistep drift and improve budget-limited search because the operator
learns composition rather than memorizing single transitions. Compare against
the same architecture/data with the diamond loss disabled and a matched extra
one-step-data baseline. Measure exact endpoint reconstruction, rule-state error,
eight-ply latent consistency and paired-match score at equal wall time. Reject
if gains disappear after equalizing sample/compute or if tactical errors rise.
Implementation and endpoint corpus remain future work.

## H2: persistent threat plans survive opponent interference

Replace the static plan-token aggregation with recurrent plan memory carried
through transitions. A plan token should predict a concrete tactical or positional
event at a horizon: king-ring pressure, defended passer advance, overloaded
defender removal, material conversion or an attack-ray change. Use exact
rule-derived event targets and supervised legal continuations. A token predicts
completion, interruption and horizon; the opponent's move updates or invalidates
it instead of keeping a stale advantage claim.

Compare persistent tokens against static tokens and equally sized square-only
memory. Ablate event labels, token persistence and interruption supervision
separately. Measure event calibration on held-out games, rollout recovery after
opponent defense, plan-diversity collapse, tactical solve rate and paired-match
score. Probe interventions by suppressing a claimed threat token and measuring
the corresponding legal candidate ranking change. An interpretable picture
without intervention/search evidence does not validate this hypothesis. This
is a research proposal; current `plan` implements only the static control.

## H3: trust gates make latent rollout cost useful

A learned dynamics model can produce a fictitious favorable position. Train a
trust estimator on simulator-verified branches: predict occupancy/rule error,
value error and horizon-conditioned uncertainty. Let search choose between
continuing a cheap latent branch and refreshing from exact state encoding.
Always use simulator legality, history and terminal checks for actual nodes.

Compare always-refresh, never-refresh and calibrated selective-refresh at equal
time and equal legal-node budgets. Report risk/coverage curves, error on
captures/castling/EP/promotions, refresh rate, batch occupancy and wall time.
Reject if a trust threshold selected on validation fails under tactical shift
or if backend scheduling cost consumes the presumed savings. A legality mask
alone cannot certify a latent value estimate. The trust head and hybrid search
integration are future work.

## H4: coherent skill control in plan/policy space

Train a skill-conditioned policy from clearly separated human rating strata and
keep the strong search/value branch as a reference. A lower-skill condition may
choose among plausible plans and apply a measured candidate-quality floor;
temperature and search budget are explicit experimental controls. Hold out games
by player and chronology when evaluating human-style prediction, since player
identity can become a shortcut.

Compare conditioning against temperature-only and budget-only controls. Measure
paired offline score, entropy, tactical error type, positional consistency and
monotonic strength across levels. Keep separate calibration for each target time
control. Report measured levels with uncertainty, not invented rating numbers.
The earlier world-model skill interventions establish related prior art; the
question here is whether plan-conditioned calibration produces stable adjustable
play in a resident search engine.

## Experiment registration and promotion

Freeze input manifests, checkpoint hashes and exact configuration before each
run. Use identical dataset fingerprints, game-group splits, seed lists and
minibatch schedules for architecture comparisons. Report both equal-parameter
and equal-time results, end-to-end ingestion/training cost, peak VRAM and
search integration latency. Dense GPU kernels can lose to smaller CPU search
when queues are empty; measure P50/P95 at real batch-size distributions.

A model advances through mechanics tests, held-out task metrics, tactical
counterexamples, offline paired games, calibrated offline anchors and eventual
Lichess confirmation. “Finite loss,” “legal policy,” “better reconstruction” and
“faster batches” are distinct pieces of evidence. None individually proves
playing strength. [NEURAL_TRAINING.md](NEURAL_TRAINING.md) defines data provenance,
replay lifecycle and statistical/rating evidence requirements.
