# Closed legal destination-exchange supervision

`neural/exchange_targets.py` produces a concrete restricted tactical task for the
[recurrent operator](TACTICAL_OPERATOR.md). It imports no neural model or GPU.
Targets, child states and branch traces are **supervision only**, separate from
the observed current-state/action inputs and Stockfish strategic score head.

For every legally replayed nonterminal TRAIN root, enumerate **all** legal
captures and promotions in ascending parent canonical action order. Each initial
action fixes its occupied destination square. Thereafter enumerate every legal
capture onto that square, recomputing legality after every capture. This handles
pins, king safety, reopened slider rays, underpromotions and EP's initial
off-destination victim. No least-attacker shortcut, pruning or fabricated null
move is used.

A player may stop the artificial exchange only outside check. In check, declining
is forbidden. If any legal evasion lies outside the restricted recapture support,
the entire initial-action target is unsupported. Actual checkmate/stalemate,
current-state repetition/fifty-move/insufficient-material outcomes are separately
typed and produce no ordinary CP regression label. Full legal move-prefix history
is retained during traversal. A 16-ply cap (including initial action), 4,096 states
per action, 250,000 global states or wall interruption produces no CP target;
partial trace backup values are removed. An execution-failed pilot admits no
dataset. No missing root/action is replaced.

Fully exhausted trees maximize/minimize exact material **delta relative to the
initial parent position**, using P100/N320/B330/R500/Q900/K0, always in the initial
parent mover frame. Decline evaluates the current material delta. Trace nodes
also record its signed current-mover counterpart. Canonical ties prefer decline,
then the lowest parent action ID. This is exact only for this declared artificial
game, not unrestricted tactics, chess minimax, SEE under another policy or SF CP.

Each record retains family ID/split/full source moves/root ply, complete initial
action IDs and missing masks. Per action it records EP victim, promotion suffix,
initial edits, immediate material delta and the closed return/correction when
admitted. Nodes carry legal current-mover action IDs, parent-oriented recapture
IDs, replayable branch suffix, FEN/hash/history/terminal status, child references,
decline/check eligibility and completed backup decisions. `dependency_round`
is bottom-up tree height; DFS visit count is not neural recurrent-step count.
Authored FEN tests are separate mechanics fixtures, not reachability evidence.

The closed [256-family TRAIN pilot](../neural/evidence/exchange-target-pilot-2026-10-03/README.md)
contains 1,940 roots and 3,864 initial actions: 3,175 closed targets, 604 unsupported
check-evasion targets and 85 typed terminals. It visited 7,244 states in 4.404 seconds
on CPU 3; no cap was exhausted. There are 15 initial EP actions and 672 promotions.
Of the closed targets, 1,344 have recapture continuations and 1,172 differ from
immediate material gain (465,080 total absolute CP difference). These are target
yield/complexity measurements, not learned quality or engine strength. Eight CPU
fixtures pass, including a separate small exhaustive no-cutoff oracle, both
movers/order invariance, forced recapture in check, king/pin legality, special
moves and no-point abort semantics.

Next learning requires fresh whole-family TRAIN/development/TEST admission and
development-only convergence tuning. Compare shared weights at 1/2/4/8 recurrent
steps, typed/untyped messages, trace loss on/off, and exact-child re-encoding at
declared inference work. Report closed-task return error, correction error,
branch/decline accuracy and complexity strata against the immediate-material
baseline. Future trace states cannot be fed to deployment as input; teacher
forcing, if used, is a named training control. Wider converged models remain
eligible under the [scaling study](SCALING_STUDY.md). This current TRAIN-only pilot
is not a fresh holdout or an architecture-selection result.
