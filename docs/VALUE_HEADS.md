# Value-head controls for the existing latent encoders

`neural/value_heads.py` changes the value readout around the existing square,
relation, and plan encoders. It preserves `models.py`, its geometric/attack
buffers, square latents, plan tokens, policy heads, and learned transition code.
The experiment asks whether value training and its output parameterization can
beat a declared material prior on finite-depth search labels. It does not claim
a new neural architecture or playing strength from adding a scalar head.

## What failed, and what remains uncertain

The [previous teacher study](../neural/evidence/search-teacher-gfx1030-2026-09-30.json)
used 3872 training and 1048 heldout parents, a two-ply remaining classical
teacher horizon, and 256 joint value/policy updates. Heldout value errors were
308/317/314 cp for square/relation/plan, versus 191 cp for current material.
Exact teacher action accuracy was 49.6/48.4/48.2%, versus 54.7% for the one-ply
material-greedy control. These are measured negative results.

Several mechanisms could explain them; that experiment does not identify one
cause. The value loss regressed compressed `tanh(cp/800)` utility while reporting
inverse-transformed CP error, and shared an encoder with chosen-action CE.
The bounded three-logit expectation must also learn piece-count information
through a mean-pooled, normalized latent representation. Training was short,
single-seed, and used random legal-game roots rather than diverse human games.
None of these possibilities establishes that relation attention is ineffective.

The new study removes policy CE and uses the same ordinary-only
`SmoothL1(value_cp/800, teacher_cp/800)` across all heads. Compressed utility is
a separate diagnostic. Initialization, active parameter counts, sampled data,
optimizer, gradient clipping, and heldout family scope remain explicit controls.
This isolates a useful readout/training comparison; it does not fully isolate
every encoder, prior, or optimization effect.

## Adapter and score semantics

`ValueAdapter(base, head_kind)` owns the supplied `ChessLatent`. Construct each
variant from an independent copy or identically initialized base. The constructor
does not clone implicitly or consume random numbers for new readouts. Passing
the same mutable base to multiple variants would couple their training.

All heads read `base.encode(features).mean(1)` from float32 schema1
`[B,18,8,8]` input. Their outputs are float32 `[B]`:

| Head | Differentiable `value_cp` | Initialization |
| --- | --- | --- |
| `three_logit` | `800 * atanh(clamp(softmax(logits)[2]-softmax(logits)[0], ±(1-1e-6)))` | Existing trained/seeded three-logit weights |
| `direct_cp` | `800 * Linear(pooled_latent)` | Zero weights and bias; starts at 0 cp |
| `material_residual` | `material_cp(features) + 800 * Linear(pooled_latent)` | Zero weights and bias; starts at exact material |

Forward returns `value_cp`, `utility`, `material_cp`, and `residual_cp`. For direct
and residual heads, `utility=tanh(value_cp/800)`. The control returns its exact
existing softmax expectation. `residual_cp=value_cp-material_cp` consistently
means a correction relative to material; for `direct_cp`, it is not a separately
predicted residual parameter. Training CP is unrounded and unclamped; the
three-logit control retains its inverse-utility epsilon clamp.

`static_scores(features)` separately rounds nearest-even, clamps to ±29000, and
returns int32. These are current-mover static heuristic scores. The simulator
provides legality, mate distance, terminal draws, and repetition adjudication.
Teacher mate-band rows are excluded from the value loss by the harness. A raw
scalar prediction outside the static band remains a training diagnostic, and
must be clamped before search consumes it.

The existing control can saturate at utility ±1; its inverse clamp then caps CP
near 5800 and the clamped branch has zero gradient. A direct CP readout continues
to have a finite linear CP derivative even when its diagnostic `tanh` utility
saturates. The unit test demonstrates this mechanism with deliberately extreme
biases. It is not evidence that the earlier trained checkpoints reached that
regime or that saturation caused their measured failure.

New zero-initialized readouts first learn their own weights. Their first-step
encoder gradient is zero; subsequent updates transmit gradients through the
unchanged encoder. `active_parameters()` exposes only encoder and selected value
head parameters. The old policy, reconstruction, and transition parameters
remain archived but receive no value-only gradient or optimizer state.

## Exact material prior

The anchor counts friendly-minus-enemy pieces in schema1 planes 0..11 with
PNBRQK weights `[100,320,330,500,900,0]`, then applies the static ±29000 clamp.
Those are the actual values in Rust `src/eval.rs` and HIP evaluation. Rust's raw
`Material.evaluate` precedes the search clamp; the anchor matches CPU-search and
HIP static material. Counts and integer-valued sums are exactly representable
in float32 for binary 8×8 boards. It does not infer pieces from the latent.

Black input already swaps friendly/enemy planes and flips ranks, so no additional
side sign is applied. Rights, raw en-passant, halfmove, and fullmove do not affect
material. Promotions change the represented piece; en-passant removes its captured
pawn; castling preserves material. Independent python-chess count/mirror oracles
and legal special-move fixtures verify these effects. The API's shape/dtype checks
are metadata-only; exact binary-board semantics assume canonical producer input.

Material is a strong handcrafted prior. Any initial residual-head advantage over
zero/three-logit initialization follows from that prior by construction. Evidence
for learned value needs heldout improvement over its own initial material score.
Evidence for richer relational reasoning additionally needs same-material or
material-neutral sibling tests, advantage/regret controls, and encoder ablations.
Changing the head does not by itself establish such reasoning.

For legal child ordering, a child score uses the child's mover. The parent prefers
the *lower* child-mover score, equivalent to maximizing its negation. Study
ordering must declare whether it uses raw float predictions or rounded deployed
scores, how ties are handled, and whether terminal/mate actions are omitted.
An ordinary nonterminal subset is a diagnostic rather than a full engine policy.

## Tensor artifact contract

`save_value_checkpoint(adapter, path, provenance)` and
`load_value_checkpoint(path, device="cpu")` use a separate
`chen_value_adapter_checkpoint` kind. The exact envelope contains
`checkpoint_version=1`, `strength_evidence=false`, `adapter_config`, primitive
`provenance`, and `state_dict`. Config records head_version1, feature_schema1,
the complete encoder configuration, head_kind, mean64 pooling, CP scale800,
inverse epsilon1e-6, residual scale800, material values, initialization,
unclamped float training CP, static limit/rounding, mover perspective, finite
search target kind, and simulator terminal responsibility.

Loading is bounded at 128 MiB, restricted with `weights_only=True`, and CPU-first.
It checks exact keys, supported config and bounds, tensor shapes/dtypes, finite
floating state, and every deterministic buffer before strict state loading and
device transfer. This includes material weights and the encoder's orientation,
geometry, attack patterns, and between-square buffers. Schema integers reject
boolean aliases. Provenance identifies the actual corpus/search teacher, split,
optimizer, seed, head experiment, and limits of the result.

These checkpoints are not accepted by the frozen deployment_version1 exporter
or production Rust three-logit loader. Deploying an adapter requires a separate
explicit artifact contract and native eager/script/parity/copy audits while
preserving the same simulator responsibilities. Do not change metadata on an old
archive to impersonate a new value head. This stage supplies training adapters
and tensor checkpoints, rather than new engine wiring.

## From value controls to action-sensitive latent research

The head study is a diagnostic step for the broader latent program. Planning
models that predict decision quantities already have extensive prior art:
[MuZero](https://arxiv.org/abs/1911.08265) learns reward, policy, and value through
latent dynamics. [EfficientZero section 4.1](https://arxiv.org/html/2111.00210v2)
adds projected consistency between predicted and refreshed observations, including
multi-step transitions. Chen's exact chess simulator provides richer deterministic
supervision, but neither scalar readouts nor consistency alone are globally new.

The next falsifiable composition should combine the existing local transport/
strategic residual split in [ACTION_DYNAMICS](ACTION_DYNAMICS.md) with these value
heads. Its tests should ask:

1. **Does a legal action predict a strategic difference beyond material?** Select
   sibling moves with equal exact material change but at least a predeclared
   teacher gap, such as 25 cp. Train root-latent/action predictions of signed
   successor residual value; refreshed successor features are targets rather
   than predictor inputs. Compare with a material-only tie, action-blind model,
   and fixed encoder. Include poisoned captures and tempo/blocker counterexamples.
2. **Do imagined and refreshed strategic latents agree without collapse?** Add
   projected stop-gradient successor consistency plus exact piece/rule refresh
   and value targets. Report action interventions, variance across different
   board states, and correct-versus-counterfactual action discrimination.
   Spatial variance alone can come from position embeddings and is insufficient.
3. **Does composition survive a longer legal path?** Use oracle-certified diamonds
   and ordinary two-to-four-ply trajectories. Separate exact transported observed
   state from learned strategic state; use each ply's canonical action frame.
   Compare imagined rollout and complete refresh on residual value, tactical
   auxiliary targets, and action ordering. A small crosspath consistency loss
   can indicate insensitivity, so reconstruction/refresh and interventions must
   accompany it. Preserve clocks and history caveats explicitly.

For a residual head, any material anchor used at a predicted successor must be
produced by admissible exact move algebra/rules or a declared learned transport,
not silently read from a withheld successor target. The strategic latent still
has to predict the correction. The corpus and loss weights should be fixed before
comparing scalar-only versus action/refresh objectives. Heldout families and
canonical board aliases need the same exclusion protocol as the value study.
This is a proposed chess-specific composition within Chen, not a global-first
claim. It becomes evidence only after the interventions and controlled runs pass.

## Verification scope

Six CPU tests passed in 1.918 seconds in the isolated Torch environment. They
cover all nine encoder/head combinations, no-RNG/no-state-change initialization,
exact old-head utility/static score equivalence, same-loss gradient activation,
independent material count/color symmetry, en-passant/promotion/castling/capture
effects, quiet material ties, saturation behavior, nine checkpoint roundtrips,
and malformed schema/weights/buffers/nonfinite-state rejection. No GPU work was
performed by this architecture agent. The methods agent owns hardware checks,
multi-seed training, corpus fingerprints, ordering diagnostics, and its separately
tracked study evidence. CPU mechanics tests establish no teacher or playing gain.
