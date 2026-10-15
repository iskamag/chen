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
are available for diagnosis; disabling pair products still leaves nonlinear cross-channel computation in the residual MLP. They must not be presented as independently trained
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
