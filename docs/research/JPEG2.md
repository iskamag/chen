# JPEG2 direct channel policy checkpoint

The new runtime chooses among legal actions with a trained heterogeneous-channel
policy. It does not delegate to Chen search, score successor `Position` objects,
run minimax/PVS, or use Tapered as its decision core. This is an implemented
learning/inference interface; strength requires trained artifacts and independent
matches, and no Elo or breakthrough claim follows from this checkpoint.

The 256 float observables use seven computational bases: action roles (32),
material counts (16), relational pressure (64), destination attacker/defender
balance (32), slider rays (48), knight/pawn topology (32), and king/overlapping
blocks (32). The exact feature names and scales are available through `schema`.
Root state is encoded once, legal counterfactual edits and attack planes come
from the codec, and other typed relations read a sparse edit view. No successor
`Position` is constructed by feature extraction. Geometric attackers include
pinned pieces; these are facts for learned factors, not legal exchange proofs.

Model v1 has a separate ReLU projection for each channel, a learned additive
coarse logit, explicit elementwise products between selected channel projections,
and a learned residual interaction network. Its decision is the legal argmax.
Actual independently trained additive and parameter-matched flat controls can
use the same runtime with dynamic model groups/projection widths and disabled
residuals. Post-hoc `--base-only`, `--pair-products-off`, and `--omit-channel NAME`
are available for diagnosis. Disabling pair products still leaves nonlinear
cross-channel computation in the residual MLP. They must not be presented as independently trained
controls. There is no claim that evaluating this fixed circuit lazily is a new
reasoning algorithm.

A secondary shared-relation check certifies immediate mating moves before the
final choice. It can override learned logits only for a proven mate-in-one.
Unknown terminal obligations never prune a quiet move. Raw logits and choices
are independently observable **before** this helper, so its contribution cannot
be confused with the learned policy's contribution.

```sh
cargo build --release --manifest-path experiments/jpeg2/Cargo.toml
experiments/jpeg2/target/release/chen-jpeg2 schema
experiments/jpeg2/target/release/chen-jpeg2 export positions.tsv output-directory
experiments/jpeg2/target/release/chen-jpeg2 predict --model model.json --input positions.tsv --out logits.jsonl
experiments/jpeg2/target/release/chen-jpeg2 diagnose --model model.json --fen 'FEN'
experiments/jpeg2/target/release/chen-jpeg2 uci --model model.json
```

Input rows are `id|fen` with an optional third field containing the physical
legal UCI inventory separated by spaces/commas. Export preserves physical UCI
coordinates (including all promotion choices), while normalizing feature
orientation to side-to-move rank flip and color swap. Outputs are row-major
little-endian `features.f32`, position `index.jsonl` with offsets/move lists,
and `manifest.json` with channels, feature names, scales and SHA-256 hashes.
Runtime and training export call the identical feature function.

The public library entrypoint is `chen_jpeg2::run(args)`. `diagnose` also exposes
all pre-helper legal logits for independent Rust/PyTorch comparison. UCI owns
history ingress and asynchronous stop handling; history transition calls are
isolated from feature/model inference. The learned policy is history-blind
beyond its FEN observables; UCI retains and reports repetition context but does
not claim a learned repetition-avoidance policy. No missing-model fallback to
old heuristics/search exists: a model file is required.

Checkpoint verification: typed attacker counts for every target and piece type
match materialized-child oracle answers on legal actions in start, Kiwipete,
promotion, both-color en-passant and castling fixtures. An independent python-
chess test validated 128 roots / 2,494 physical legal actions and exact equality
of all 256 features under color/rank mirroring. Model tests demonstrate an
interaction changing the action ranking and verify its ablations/disabled
residual control. Training export completed 3,873 TRAIN/DEV roots / 135,073 legal
actions; sealed test data was not opened by this engine implementation.


## Explicit support-message schema v2

`export INPUT OUTPUT --support` appends 64 features as the eighth channel,
`support_obligations[256,320)`. V1's first 256 floats and strict fingerprint stay
unchanged. Model files select 256 or 320 features explicitly; normalization,
projection dimensions and feature-schema fingerprint must match. There is no
silent model reinterpretation or fallback between schemas.

The appended operator preserves source identities in a source-to-target support
graph. Absolute-pin rays remove off-ray support edges. Target demand sends a
message to each eligible defending source; a second message returns competing
source load to its supported targets. Root-to-action edge mapping exposes
abandoning a sole defender and changing load on shared defenders. These
identity-sensitive relationships were unavailable from v1's aggregate attacker
counts. The operator's fixed demand conventions are **features**, not exchange
proofs or a hand-scored move policy; the trained policy decides how to use them.

Implementation scope is precise: the root graph is cached once, but each action
**copies 64 symbols and reconstructs a counterfactual graph**. It creates no
successor `Position` or game tree, but is not simultaneous graph inference over
all alternatives and does materialize counterfactual symbol boards. Telemetry
reports `successor_position_objects=0` and the actual number of
`counterfactual_graph_rebuilds`. Feature/graph construction is included in encoder
time, not attributed to neural inference.

V1 semantic schema SHA-256:
`890278fadee8d12099254c2e1edc428a84157977ddffe5a32335f9b690ed8d5f`.
V2 semantic schema SHA-256:
`d70273878dad4b215e50d13bf24c3c045c8eaffb43a2a6e6f974244f419fd5d2`.

The matched bank-2 export contains 5,745 TRAIN/DEV roots and 202,783 actions.
Every v2 row's first 256 floats and complete move-index file equal the v1 export
exactly. All values are finite. Independent color/rank mirroring over 128 roots
and 2,494 actions differs by at most 2.38e-7 across 320 features. Four support
regressions cover shared-source identity despite equal target counts, pin-ray
eligibility, sole-support abandonment, and sparse versus materialized graph
agreement. These checks establish feature semantics, not increased strength.

First v1 model interface checks also matched Rust against independent PyTorch
on 16 roots / 570 actions: maximum full-logit error 1.43e-6, all argmax choices
equal. Independently trained additive and flat controls loaded with their own
group counts/widths and also matched within 1.43e-6. The interaction model's
learned residual changed 9 of the first 16 pre-helper choices; this demonstrates
an actual decision contribution, **not** an advantage over the trained controls.
The initial interaction model did not beat those controls on held-out DEV loss.
Strength comparisons and graph contribution remain experimental requirements.
