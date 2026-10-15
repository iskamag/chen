# Direct channel policy training

This is CPU-only training for Chen's experimental direct action policy. It
learns from offline Stockfish labels and scores every legal root action. The
deployed model has no search, child-position value backup, or teacher process.
Rich redundant topology and relationship features are the computational basis;
compression ratio is not an objective. The engine separately supplies an exact
mate-in-one helper.

`prepare.py` validates legal moves, replays supplied histories, checks score
frames, and splits connected opening-prefix/game/transposition families before
fitting. It admits historical TRAIN provenance only, preserves earlier split
allocations when extending a bank, and excludes new families that bridge frozen
splits. Fitting opens TRAIN and DEV only. Each append-only producer input is
copied as an exact newline-complete consumed prefix, with byte count and SHA256.

Teacher packets are distinguished from hard best-move labels. Mixed-depth
MultiPV packets provide hard supervision only. Coherent packets provide soft
targets normalized over observed actions only; omitted actions remain unknown.
Forced actor-choice labels stay outside that soft support. Large-regret pairs
require a gap of at least 200cp and both reported depths at least eight. Mate
outcome classes are handled separately. Forced 64k-node and reference 128k-node
scores are approximate, unequal-budget teacher estimates, not proven regret or
confidence bounds.

`model.py` implements independently fitted typed additive, typed channel/product,
and parameter-matched flat controls. The interaction policy projects each channel
to 32 features, adds unary action potentials, then combines channel embeddings
and seven explicit elementwise channel products through a small residual MLP.
The 320-feature version appends the defender-identity/obligation channel to both
the base and residual inputs. It retains the same seven explicit product pairs;
the new channel has no extra product slot. This is one fixed direct neural pass,
not progressive inference or a shared alternative-world algorithm.

Install official CPU-only Torch, NumPy, and python-chess into an isolated venv.
Then prepare labels, ask the Rust exporter for features, and fit:

```sh
python prepare.py --old-train OLD_TRAIN_PROVENANCE.json \
  --hard-bank HARD_LANE0.jsonl HARD_LANE1.jsonl \
  --soft-bank COHERENT_LANE0.jsonl ON_POLICY_LANE0.jsonl \
  --out PREPARED
chen-jpeg2 export PREPARED/positions.tsv FEATURES --support
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  python fit.py --labels PREPARED --features FEATURES --out FIT \
  --architecture interaction --cpu 0
python -m unittest discover -s . -p test_boundaries.py
```

Run from this directory. Omit `--support` for the unchanged 256-feature schema.
Model artifacts require an exact schema fingerprint. `bind_schema.py` only
documents the initial cold-binding event; normal fits bind directly from the
export manifest. Every checkpoint is independently evaluated from its actual
deployment JSON with NumPy, and the engine team checks Rust logits and choices.

The current objective is 0.3 hard CE + 0.2 observed-support soft CE + 0.05 Huber
score-gap loss + 1.0 large-regret pair softplus. DEV checkpoint selection uses
that weighted objective, with pair loss averaged over available pairs. All
architectures use the same immutable labels, split, seed and objective. No
runtime labels, successor evaluations, or stock-engine scores enter features.

## Results and limits

The first 3,138-root fit was weak: its interaction model lost all four DEV games
against stock Chen. It also failed to beat independently fitted additive/flat
controls on teacher CE and accuracy. This is a negative baseline.

The next complete admission contains 7,585 TRAIN roots, 1,877 DEV roots, and a
closed 1,809-root TEST partition. It includes all 4,795 on-policy teacher roots
before deduplication. The frozen export contains 297,817 legal actions for TRAIN
and DEV; the 256-feature prefix is bit-exact in the 320-feature export. TRAIN has
2,615 qualified large-regret pairs and DEV 767. Earlier anchored records retain
their original labels; their old packets were not retroactively expanded.

| Model | Parameters | DEV objective | Top-1 | Bad-pair ordering errors | Selected bad action / pairs |
|---|---:|---:|---:|---:|---:|
| Initial 256 interaction, evaluated later | 37,442 | 3.516 | 27.44% | 25.95% | 13.17% |
| Retrained 256 interaction | 37,442 | 3.287 | 23.49% | 21.25% | 4.56% |
| Retrained 320 interaction | 41,602 | 3.267 | 22.16% | 21.51% | 4.82% |
| Retrained 320 additive | 41,665 | 3.168 | 22.32% | 19.04% | 4.17% |
| Retrained 320 flat | 41,539 | 3.138 | 24.03% | 17.86% | 3.65% |

The new objective/data reduce selection of known catastrophic teacher negatives.
The flat control performs better than the explicit interaction architecture on
the selected DEV criterion. These experiments do not establish a benefit from
the support graph or typed products. Graph versus 256 also changes parameter
count by 4,160; that comparison alone cannot attribute causality to the graph.

Partial-label regret has policy-dependent coverage and cannot rank overall
strength. On the same 129 ordinary complete-support DEV roots, mean regret was
82.0cp for the initial model, 105.3cp for retrained 256, 105.0cp for retrained 320,
and 83.4cp for flat 320. Thus the known-negative improvement does not establish
an overall teacher-regret improvement. Fresh forced-choice probes and actual
games remain the strength evidence, reported by the parent experiment.

Real 320-feature Rust/PyTorch checks passed on 16 roots/570 actions: identical
move inventories, order, and argmax; maximum logit error 1.43e-6 for interaction
and 2.98e-6 for flat. Four admission/serialization boundary regressions pass,
including partial append records, all-mate packets, and forced-label isolation.

The local asset inventory found 233 historical TRAIN families/8,477 all-legal
action edges, not a large ready-made strong policy corpus. An external retirement
archive was inspected through its manifest only: 386,600,789 compressed bytes,
1,283 entries, 1,200,359,592 uncompressed bytes. It contains duplicated 88.5MiB
real-game PGNs plus restricted exchange/runtime checkpoints. No archive was
extracted and no existing checkpoint was promoted as a strong whole-board policy.
New coherent and on-policy teacher expansion was therefore necessary.
