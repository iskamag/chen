# Neural training and evidence boundary

The `research/neural` worktree contains three interchangeable latent models and
a real device-resident replay/training smoke. The target is eventually **2500
Lichess**, with most evaluation performed offline and a calibrated adjustable
playing level. No current neural checkpoint has measured chess strength. The
present synthetic run verifies legal ingestion, resident minibatches, trainable
heads, finite gradients and comparable experiment inputs.

## Run the controlled smoke

Use the existing ROCm Torch container, leaving its `/work` project alone. Copy
this worktree's `neural/` package into an isolated directory such as
`/tmp/chen-neural-methods/project/neural`. Install the optional ingestion oracle
in a separate target directory:

```sh
docker exec cpp-rocm python -m pip install --target /tmp/chen-neural-methods/deps chess==1.11.2
docker exec cpp-rocm mkdir -p /tmp/chen-neural-methods/project
docker cp neural cpp-rocm:/tmp/chen-neural-methods/project/
docker exec -w /tmp/chen-neural-methods/project -e PYTHONPATH=/tmp/chen-neural-methods/deps cpp-rocm python -m unittest discover -s neural/tests -v
docker exec -w /tmp/chen-neural-methods/project -e PYTHONPATH=/tmp/chen-neural-methods/deps -e CHEN_TEST_DEVICE=cuda cpp-rocm python -m unittest discover -s neural/tests -v
docker exec -w /tmp/chen-neural-methods/project -e PYTHONPATH=/tmp/chen-neural-methods/deps cpp-rocm python -m neural.train --device cuda --models square relation plan --samples 192 --steps 8 --width 32 --depth 1 --output /tmp/chen-neural-methods/smoke.json
```

`cuda` is Torch's API spelling for ROCm devices. Record Torch/HIP versions and
`gcnArchName` from the report; acceptance requires the observed device to be
gfx1030 rather than inferring it from a command line target. The same generated
record fingerprint and sampling sequence are shared by all model variants.
Parameter counts differ; this first comparison fixes width/depth/data/steps,
not parameter count or total compute. A real architecture benchmark needs both
parameter-matched and equal-time variants, multiple seeds, and held-out games.

The initial hardware run observed Torch `2.10.0+rocm7.0`, Torch HIP
`7.0.51831`, `cuda:0`, and architecture `gfx1030`. This Torch environment differs
from the separate HIP compiler/runtime version; record both when integrating
the Rust kernels. HIP's reported 30 multiprocessors represent workgroup
processors on this RDNA2 device; hardware-agent profiling reports 60 physical
CUs and 120 SIMDs. Avoid treating that 30 figure as a physical-CU count.

The first 192-position/8-step smoke generated four legal trajectories. Every
architecture consumed record fingerprint
`863923c05e93a25649240fc9afdc36acb44d6f8964006c031d95e1e83dc8c069`
and identical minibatch fingerprint
`2ea0656c9a6e735ebd210eff8e99d2a4dcfe7e12803844b047110c725cc872c5`.
A repeat produced exactly matching first/final losses and fingerprints. This
verifies this seeded run on this backend; deterministic algorithms were not
enabled, so it is not a cross-version determinism guarantee. Model timings are
short smoke measurements containing first optimizer allocation and warm-up
effects; they must not rank production architectures.

Each run saves model snapshots by default under the ignored
`results/neural/checkpoints/` directory; `--checkpoint-dir` changes that location
and `--no-checkpoints` suppresses artifacts. A checkpoint contains architecture,
configuration, schema version, CPU tensor `state_dict`, corpus provenance,
training settings and explicit `synthetic_legal_smoke_checkpoint` status with
`strength_evidence=false`. The report includes its path and SHA256. It is an
inference artifact for integration experiments, not a strength-qualified model.
`neural.train.load_checkpoint(path, device="cuda")` returns `(model, metadata)`.

The loader uses [Torch's restricted tensor loader](https://docs.pytorch.org/docs/stable/notes/serialization.html)
with `weights_only=True`, validates schema/evidence/configuration and exact state
keys/shapes, rejects nonfinite weights and oversized smoke files, reconstructs
on CPU, then transfers to the requested device. It has no unrestricted-pickle
fallback. Round-trip tests compare all outputs for all architectures and reject
schema mismatches, missing/nonfinite weights and arbitrary object payloads.

## What enters VRAM

`data.generate_synthetic` plays deterministic, seeded, legal random games using
`chess==1.11.2`. Each record contains the actual FEN, chosen legal move, exact
successor FEN, game id and simple heuristic target move. The optional library
is a development oracle; it is not an engine dependency. Generation uses CPU
rules, then transfers tensors at one explicit ingestion boundary. Future GPU
self-play should supply the same schema directly from resident exact states.

The schema is version 1 and matches `features.py`:

| Field | Shape per row | Type | Meaning |
| --- | --- | --- | --- |
| `features`, `next_features` | 18×8×8 | float32 by default | Own/enemy pieces, castling rights, EP target, halfmove clock; next row uses next mover's perspective |
| `legal_mask` | 20,480 | bool | Exact legal actions in current canonical orientation |
| `actions` | scalar | int64 | Actual transition action |
| `target_actions` | scalar | int64 | Immediate capture/promotion/check/mate heuristic target |
| `material_proxy` | scalar | float32 | `tanh(own_minus_enemy_material / 8)` |
| `terminal` | scalar | bool | Exact successor termination, including claimable draws |
| `game_ids` | scalar | int64 | Source trajectory for group splits |

Action `((from*64+to)*5+promotion)` has promotion codes none, N, B, R, Q.
For Black, squares flip rank with xor 56 and piece colors swap into own/enemy
planes. This intentionally large, unambiguous action vocabulary includes many
impossible actions; exact masks exclude them. Current rows are nonterminal,
while successors can be terminal. Repetition history is **not** in these 18
planes. Exact rules own repetition and termination; full-strength value inputs
will require a versioned history/context extension to distinguish aliased
positions with different repetition claims.

`DeviceReplay` allocates a bounded ring on the requested device, validates exact
shapes/dtypes/device and current-action legality before mutation, then overwrites
oldest rows. Oversized insertions retain the newest capacity records. Monotonic
sequence ids distinguish a replacement from the old record at the same physical
index. `clear()` invalidates rows without reallocating or reusing identities.
Gathered minibatches own storage and survive later ring overwrites. Sampling
uses resident `torch.randint` and `index_select`; no host arrays, scalar reads or
device-to-host copies feed the minibatch. Ingestion validation intentionally
synchronizes scalar checks. The sampler's no-host path is checked with a Torch
dispatch audit. Training logs and small audit fingerprints are read after the
timed loop. Ingress drops autograd history to prevent replay retaining the
producer's computation graph.

Dense masks currently cost 20,480 bytes per row. Float32 current/successor
features cost 9,216 bytes per row; all fields plus sequence identity cost 29,733
bytes per record. A 100,000-row ring therefore allocates about 2.77 GiB before
models, optimizer, minibatches and activations. Capacity is explicit and an
optional byte budget rejects oversized allocations. Packed masks or variable
legal-action lists are a planned scalability experiment; no memory compression
is claimed yet. Single-owner writes/readers on one ordered stream are supported;
concurrent streams require external events/ordering.

## Objective and interpretation

The smoke optimizes legal heuristic policy cross-entropy, expected-value MSE
against the material proxy, current/successor binary board reconstruction,
castling BCE, EP-category CE, halfmove-clock MSE and normalized latent consistency
with a stop-gradient exact-successor encoding. The dynamics operator is
action-conditioned and flips into the next mover's orientation. Terminal
successors still supply exact board/rule reconstruction; no policy is sampled
from them.

The three WDL output channels are merely a parameterization for the proxy loss:
their softmax win-minus-loss expectation is fitted to material. They are **not
calibrated outcome probabilities**. The immediate heuristic ignores recaptures,
positional play and search. All reported accuracy/loss metrics are on training
data. This setup must not establish an Elo claim or select a production model.
The training script records synthetic status explicitly in every report.

For a strength run, replace proxy supervision with search visit distributions,
versioned teacher evaluations and/or completed-game outcomes. Keep label source,
teacher hash, search budget, perspective, mate handling, game termination and
claim policy in the dataset manifest. Bootstrapped labels must remain separated
from independently evaluated targets. Reanalyze older games when the teacher
changes, rather than silently mixing policy generations.

## Data acquisition and leakage plan

No large download or public dataset is included. A first bounded pilot can use
the [Lichess open database](https://database.lichess.org/), whose ordinary game,
puzzle and evaluation exports are CC0. Its broadcast section separately uses
CC BY-SA 4.0; keep broadcast material in a distinct provenance partition. The
evaluation export has varying search depths/node counts and incomplete FEN
clock/history fields. It should train position tasks with explicit missingness,
not pretend to reconstruct draw-sensitive trajectory outcomes.

Before download, record URL, retrieval timestamp, exact file checksum, export
date, license/category, chosen filters and parser version. Stream compressed
records through bounded CPU staging into resident shards. Keep original game
ids and full history when available. Policy labels from human games describe
human decisions; engine evaluations describe a particular teacher. Neither is
automatically optimal. Handle resignations, disconnections and time losses
separately from board-terminal wins.

Split games as units first, then audit exact-position collision across splits:
hash pieces/side/rights/**effective** EP and draw context separately from the
raw FEN. Group transpositions and mirrored/color-canonical duplicates. Common
opening positions may collapse most games into one connected component: retain
game-level chronology, exclude shared opening prefixes from held-out scoring,
and publish collision counts rather than asserting clean splits from a random
row split. Reserve a later-month chronological test and an independent opening
suite; never tune on the final match openings. Puzzle solutions overlap source
games, so use their `GameUrl` and position hash when blocking cross-source
leakage. The puzzle FEN is *before* the opponent's first listed move; apply that
move before training the solver policy.

Outcome training needs strata for phase, rating, time control, resignation,
result and player color. Cap positions per game to avoid long draws dominating,
retain hard negatives and natural error rates, and report target-population
weights alongside any oversampling. Evaluate draw calibration and tactical
precision separately; a model can improve average proxy loss while becoming
more exploitable in forced lines.

## Strength and level acceptance

Most experiments run offline paired games from fixed opening seeds with colors
reversed. Archive engine/checkpoint hashes, device, complete configuration,
time/node budget, stopping criteria and all PGNs. Report W/D/L, failures,
paired-score uncertainty, and relative rating with a confidence interval.
Unexpected protocol/illegal-move losses stay in the score and receive a separate
failure count. Estimate uncertainty by opening pair rather than treating paired
games as independent Bernoulli trials. Predeclare fixed sample size or a valid
sequential test; stop-on-good-results invalidates an ordinary interval.

An offline engine-pool Elo is a relative scale. It does not prove 2500 Lichess.
Use separately calibrated offline anchors tied to the desired Lichess time
control, then reserve actual Lichess games and rating uncertainty for eventual
confirmation. Account operation and its connection workflow are future work.
The target must name blitz/rapid/classical/bullet and the rating population.
Until calibrated evidence exists, publish measured offline scores with their
opponent and budget instead of labeling a checkpoint “2500.”

Adjustable level should be a reproducible policy/configuration: skill-conditioned
policy, bounded temperature, candidate-quality floor and/or search budget. Train
on skill strata and calibrate each setting offline against the same anchor panel.
Measure rating, tactical error distribution, move entropy, latency and game
length. Verify monotonic strength with uncertainty; a larger temperature does
not automatically correspond to a particular rating. At low levels use
coherent learned alternatives rather than arbitrary forced illegal/random
blunders. The current prototype exposes no calibrated level settings.
