# Observed move transport and strategic residuals

The legal-diamond experiment found that similar learned endpoints could be a
consequence of weak action effects. This follow-up introduces exact local slot
transport, a separately learned residual, and explicit discrimination between
two legal actions from the same position. **Transport improves action separation
in the measured smoke, but does not establish better semantic predictions.**
One-ply latent error and action-delta error remain worse than the previous
transition control, and four-ply reconstruction remains weak.

The architecture is a chess-specific composition hypothesis, not a claim that
structured dynamics, contrastive prediction or hybrid rule/neural planning are
globally new. The learned-model and representation prior art in
[NEURAL_ARCHITECTURES.md](NEURAL_ARCHITECTURES.md) still applies. Exact piece
algebra is implementation of chess rules, not a neural world model.

## Ownership and causal contract

`neural/dynamics.py` defines independent components without changing the existing
encoder, training/checkpoint infrastructure or rules engine:

* `move_effects(board, move)` verifies a standard-chess legal move at CPU ingress
  and records source, destination, promotion, captured square and castling rook
  source/destination in the current mover's canonical frame. Chess960 is rejected.
* `transport_slots(slots[B,64,C], effects)` copies source to destination, resets
  source/captured square, transports the castle rook, then flips ranks to the next
  mover. It is differentiable with respect to the transported values. EP removes
  the separately named pawn square. Promotion and color identity depend on the
  representation being transported.
* `transport_pieces(features, effects)` applies promotion identity and switches
  own/opponent piece channels after slot transport. It reproduces all 12 exact
  successor piece planes for validated legal moves. Rights, EP targets, clock,
  repetition history and terminal adjudication remain simulator-owned.
* `AnchoredState(observed, strategic)` keeps the simulator observation separate
  from the neural square state. `StrategicTransport.predict(strategic,
  previous_observed, effects)` never receives the successor observation. A learned
  perspective transform and residual use the transported state, source piece,
  promotion embedding and affected-square flags. `transition` returns a new
  anchored state containing the supplied authoritative successor observation.

The exact observed state is reliable because it comes from the simulator. That
is not predictive evidence for the neural strategic component. The current
strategic target is an existing frozen square encoder, whose features include
local board information; this is not a demonstrated disentanglement into pure
strategy and occupancy. No trained value, policy, threat-plan or outcome claim
belongs to this prototype.

Observed successor features enter losses and diagnostics only. A causal test
replaces those features with zeros and verifies unchanged strategic prediction.
Four-ply evaluation supplies the real **previous** observation and legal special
move effects at every ply; it is a hybrid observed rollout, not wholly imagined
neural play. The simulator continues to retain the actual history and legal node.

## Controls that distinguish the mechanism

The experiment holds fixed encoder initialization, data, schema, sample schedule,
optimizer settings and number of updates. The encoder is frozen so the target
cannot become easier by drifting or collapsing during predictor training. A
piece readout and predictor parameters are trained. The active parameter counts
are matched to the nearest hidden unit:

| Control | Rule inputs and predictor | Trainable parameters at width 16 |
| --- | --- | ---: |
| `baseline` | Existing action-conditioned transition | 3,056 |
| `observed_residual` | Same observation/effect conditioning as transport; rank flip and learned residual, no slot movement | 3,048 |
| `transport` | Exact local slot movement plus learned perspective/residual | 3,064 |
| `transport_contrast` | Transport plus explicit legal sibling delta and margin objective | 3,064 |

The same-observation residual control isolates transport from the advantage of
additional rule metadata. Parameter counts differ by at most 0.3%; elapsed time
and computation are not exactly equal. The first timed GPU loop includes initial
optimizer/kernel startup, so these timings cannot rank steady throughput.

All controls optimize normalized successor latent refresh plus occupied-aware
piece classification on predicted and exact encoded states. Classification
adds a fixed empty-square class to the 12 piece logits and weights that class
at 0.25. This addresses the previous BCE incentive to predict empty planes.
BCE remains a diagnostic, so lower BCE alone cannot support better piece identity.
The contrast control additionally matches the difference between legal sibling
latents to the exact re-encoded difference and penalizes predictions closer to
the wrong sibling target. An action-blind or swapped predictor fails this loss;
mere separation without correct direction still does not satisfy it.

## Corpus and evidence boundary

Each training root is reached by legal random play from the standard initial
position and supplies two distinct legal actions, exact successors and complete
source history. Held-out roots and successors reject any canonical model-input
identity present in training. Four-ply legal-diamond validation rejects overlap
of every intermediate encoded state with training. These identities account for
color/rank canonicalization, raw EP and the clock semantics of schema 1.

The standalone report freezes all histories, FENs, move effects and dataset,
schedule and source fingerprints. Declared sparse stress roots separately test
castling, EP, queen/knight promotion, capture promotion, and actual check evasion
by either capturing the checking queen or moving the king. They are untrained
counterexamples rather than game-strength tasks. Exact piece transport is tested
on both colors, all promotions, all standard castles and ordinary captures.

The measured gfx1030 float32 run uses ROCm PyTorch `2.10.0+rocm7.0`, seed
`20260930`, 64 training roots, 16 held-out roots, two actions each, width 16,
batch 8 and 64 updates. Four disjoint diamonds supply rollout validation.

| Control | Held-out latent error | Legal action separation / exact encoded separation | Action-delta error | Occupied-piece accuracy |
| --- | ---: | ---: | ---: | ---: |
| baseline | 0.01841 | 5.2% | 0.00252 | 45.7% |
| observed residual | 0.02286 | 1.3% | 0.00247 | 46.4% |
| transport | 0.02299 | 34.2% | 0.00308 | 46.0% |
| transport + contrast | 0.02259 | 30.2% | 0.00288 | 43.2% |

Transport makes different legal actions produce more different learned states,
but the increased displacement is partly wrong: its delta error is worse than
baseline. Contrast reduces that error relative to transport, while still failing
to beat baseline. Four-ply latent error is approximately 0.0353, 0.0339, 0.0360 and
0.0354, respectively; occupied-piece accuracy there is 5.6%, 0%, 0% and 0%.
The exact-encoding readout reaches roughly 45% on held-out occupied pieces, so
the frozen random representation/readout is itself a weak semantic reference.
These diagnostics do not prove strategic understanding or a stronger engine.

This follow-up improves the experiment: action identity, correctness of the
action effect, exact observed state and learned state fidelity now have separate
measurements. The next architecture should train useful observed/strategic
representations on real tactical or outcome supervision, and predict the
strategic residual relative to an explicit observed encoding. Longer-horizon
tests must include captures, promotions and rights changes. Larger latent
separation alone is not a promotion gate; a future model must beat refreshed
observed and full-refresh controls on held-out semantic tasks and paired play.

## Reproduce

```sh
python -m unittest neural.tests.test_dynamics -v
CHEN_DYNAMICS_GPU=1 python -m unittest neural.tests.test_dynamics -v
python -m neural.dynamics --device cuda --samples 64 --steps 64 \
  --batch-size 8 --width 16 --seed 20260930 \
  --output results/neural/dynamics/gpu-final.json
```

`CHEN_DYNAMICS_GPU=1` enables the actual device parity test. Tests prove slot
identity/gradient transport, exact special-move piece effects, retained corpus
replay, no successor shortcut, optimizer updates, legal-sibling discrimination
and float32 CPU/GPU prediction/gradient agreement within `3e-4`. They do not
claim identical long optimizer trajectories across CPU and GPU. Saved experiment
results are ignored by Git. The compact authoritative GPU report is tracked at
[dynamics-gfx1030-2026-09-30.json](../neural/evidence/dynamics-gfx1030-2026-09-30.json),
with the [seven-test log](../neural/evidence/dynamics-tests-gfx1030-2026-09-30.txt)
and [source hash manifest](../neural/evidence/dynamics-source-sha256-2026-09-30.json).
It preserves every support metric, corpus fingerprints and regeneration settings;
the complete local report additionally freezes all source histories and FENs.
The models have no registered production checkpoint or Rust UCI inference path.
