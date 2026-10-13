# Root-action strategic residual transport

This experiment predicts a legal action's successor finite-search value from
the root observation, its root latent and the action. It preserves the existing
square/relation/plan encoders and all older production modules. Exact local
piece transport, material and blocker-aware pseudoattacks are declared chess
priors; a learned latent correction predicts the remaining score component.
Observed successor features never enter `predict()`.

The previous value-head study found that all 27 learned models trailed material
on heldout mean CP error. Residual heads improved some pairwise concordance
without improving mean selected-child regret. The earlier transport study
increased action separation while worsening the direction/error of the latent
delta. This experiment therefore tests action-specific **semantic prediction**:
absolute successor score, same-material sibling score differences, correct
action intervention direction and selected-child regret. Latent distance alone
cannot support promotion to an engine evaluator.

## Model and move contract

`neural/strategic_transport.py` owns the new module:

```python
model = StrategicPredictor(base, mode="transport", anchor_kind="material",
                           freeze_encoder=False)
z = model.encode(root_features)             # [B,64,D]
prediction = model.predict(root_features, actions, root_latent=z)
fresh = model.refresh(successor_features)   # diagnostic branch only
```

Roots are schema-1 float32 `[B,18,8,8]`, and actions are resident int64 `[B]`
canonical indices `(source*64 + target)*5 + promotion`. Promotion values are
0/1/2/3/4 = none/knight/bishop/rook/queen. Both squares are in the **root mover's**
frame. A black root already flips ranks with xor56 and swaps piece colors.
Supplied cached latents must have been encoded from those same roots; shape and
device checks cannot certify their causal origin.

Prediction returns `latent [B,64,D]`, exact `pieces [B,12,8,8]` and scalar
`material_cp`, `residual_cp`, `value_cp`, `utility [B]`. Every output is in the
**successor mover's** frame. Thus lower child CP is better for the root mover,
and the root-side action estimate is `-prediction["value_cp"]`.

`derive_effects(roots,actions)` produces source/target/promotion/captured/
rook-source/rook-target int64 vectors. It recognizes EP from the own pawn,
diagonal empty target and raw-EP root plane; standard castles from the own king
on canonical e1 moving to c1/g1; and promotion from the action. It assumes
already certified **legal standard-chess** input. Range/content/legality must be
validated at simulator ingress; padding, null actions and Chess960 are outside
this contract. Hot-path validation inspects tensor metadata without downloading
contents. No successor board packet or teacher score is consulted.

`successor_pieces(roots,effects)` transports pieces, removes captures including
EP, moves the rook for castling, replaces promoted pawn identity, flips ranks
and swaps colors. Promotion classes use resident equality rather than
`F.one_hot`, whose content checks triggered host scalar extraction in the CPU
dispatch oracle. Exact successor rights, EP, clocks, fullmove and repetition
context are not reconstructed. Root rule planes remain encoder inputs; learned
prediction must account for their successor influence, while simulator state
continues to own legality, history and terminal adjudication.

## Learned correction and ablations

The transport mode moves the root's learned square slots through the known
local move algebra. A learned perspective transform, local action context and
squarewise residual then produce a next-mover latent. Context includes source,
destination and capture-site root latents, source piece identity and promotion.
Five square flags mark source, target, capture site and both rook sites. The
capture-site flag may name an empty target on a quiet move; piece identity and
occupancy covariates distinguish it from an actual capture.

Eight deterministic square covariates accompany the residual: next own/enemy
pseudoattack counts, changes from oriented root counts, next own/enemy occupancy
and pseudoattack exposure of each king. Attack counts use exact ray blockers;
endpoints count as attacked/defended even when occupied by a friendly piece.
Pins/check legality, special-move legal attacks and history are excluded.
Counts/exposure divide by eight as a fixed scale. These covariates give the
network a substantial handcrafted tactical prior. Improvements using them do
not by themselves demonstrate that the encoder learned tactical relations.

The CP readout consumes the mean of 64 predicted latent slots and starts at zero
without consuming random numbers. Float predictions are unrounded/unclamped:
`residual_cp=800*head(mean(latent))` and
`value_cp=material_cp+residual_cp` for the material anchor. Piece values are
100/320/330/500/900/0, own minus enemy in the child frame, with the same ±29000
static material bound used by the CPU-search/HIP evaluators. The zero-anchor
ablation uses `value_cp=residual_cp` while still returning the exact material
diagnostic. `utility=tanh(value_cp/800)` is a compressed search heuristic, not WDL.
`static_scores()` applies nearest-even rounding, ±29000 clipping and int32;
terminal and mate scores remain simulator responsibilities. No deployment or
neural strength claim is supplied by this API.

| Control | Learned prediction inputs |
| --- | --- |
| `transport` | Moved root slots, action context, effect flags, exact attack/occupancy covariates |
| `conditioned` | Oriented root slots without local movement; same remaining inputs |
| `action_blind` | Oriented root slots only; learned correction independent of action |
| Frozen encoder | Same transport operator, with encoder parameters and incoming root latents detached |
| Zero anchor | Same operator, omitting material from the predicted scalar |
| Full refresh | Actual successor encoded through the shared CP readout; separate observation diagnostic |

Every mode constructs identical state keys and consumes the same initialization
RNG. A study can clone one common state into every mode. Active optimizer
parameters differ explicitly: action-blind excludes unused local correction
components, and a frozen encoder receives no gradient. Architecture parameter
counts and compute differ; shared initialization is not a claim of equal
parameter budgets or equal wall time. Modules take ownership of their supplied
base, so variants need independent clones.

On same-material siblings, the action-blind control produces exactly tied values
and zero learned score delta. On captures/promotions its material anchor can
still differ by action; it is specifically a **learned residual blindness**
control. Shuffling legal actions at evaluation should replace only the learned
residual while retaining the true action's material anchor, to isolate strategic
sensitivity. The conditioned control isolates the slot transport contribution
from the action/pseudoattack metadata advantage.

## Targets, causal controls and acceptance

The controlled study is owned separately. Root→child actions must be joined to
the child's own finite-depth teacher label, not the child's subsequent action.
An exact two-ply child score is supervision, not a game outcome. Mate-band and
terminal handling must be declared independently. A full-root regret claim
requires every legal root action to have an eligible child label; dropping
individual children would create a misleading partial policy comparison.

The planned common objectives combine normalized-CP SmoothL1, same-material
sibling score-delta prediction and latent refresh/delta against an externally
frozen initial encoder. A moving shared target could collapse or drift to make
the prediction task easier. A frozen random target is stable but remains a weak
semantic reference, so only actual heldout teacher-score/action performance can
justify a strategic prediction claim. Full refresh remains a separate control
and cannot silently enter root-action prediction.

Use fresh train/validation/test source families and reject canonical feature
overlap across roots and successors. The spent prior heldout corpus is not a
fresh test set. Report whole-root coverage, same-material teacher-gap bins,
parent-grouped action shuffle, blocker and side-to-move interventions, latent
state variance and value/ordering errors. No-action or wrong-action predictions
must fail direction-sensitive tests; larger latent separation alone is not a
useful result. Exact pseudoattack changes in blocker tests are mechanics, while
teacher error/regret under those interventions tests learned semantic behavior.

Prior art already establishes task-relevant learned dynamics in
[MuZero](https://arxiv.org/abs/1911.08265), models selected by value relevance in
[the value-equivalence principle](https://arxiv.org/abs/2011.03506), and future
representation prediction in [SPR](https://arxiv.org/abs/2007.05929). SPR uses an
EMA target; this study's fixed initial target is a deliberately simpler control.
The explicit chess move algebra, canonical slot transport and strategic residual
composition are hypotheses new to Chen. No global-first claim follows from
these references, and this prototype does not reproduce their training systems.

## Checkpoints and verified scope

`save_strategic_checkpoint(model,path,provenance)` writes a separate
`chen_strategic_transport_checkpoint` kind with exact model/mode/anchor/freeze,
feature/action schema, effect/attack algebra, score-frame/scale and target
semantics. All original base weights and deterministic buffers remain archived.
`load_strategic_checkpoint(path,device="cpu")` uses restricted CPU-first loading,
a 128MiB file limit, exact configuration/envelope/state keys, tensor shape/dtype/
finite checks, and exact equality for all deterministic buffers before device
transfer. Provenance is primitive JSON metadata. This trusted local boundary
does not promise protection from every malicious resource-exhaustion archive.
Older deployment/value/teacher kinds do not accept this checkpoint kind.

```sh
python -m unittest neural.tests.test_strategic_transport -v
# Only after an exclusive GPU slot has been assigned:
CHEN_STRATEGIC_TEST_GPU=1 python -m unittest neural.tests.test_strategic_transport -v
```

CPU verification covers both-color castles/EP, all promotions/capture promotions,
24 legal random actions, exact child piece/material frames, common mode state,
canonical color symmetry, blocker-aware pseudoattacks against python-chess,
no successor ingress, cached roots, no hot-path host scalar extraction, exact
action-blind ties and finite optimizer gradients. Synthetic ±120cp sibling
labels exercise updates/signs and frozen encoder invariance; they are mechanical
test targets, not chess teacher or strength evidence. Restricted checkpoint
roundtrips cover all encoders, with malformed schema/frame/bool/dtype/finite and
deterministic-buffer corruptions rejected. GPU parity is an opt-in scoped test;
no GPU execution or study result is claimed by the CPU test completion.
