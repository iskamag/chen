# Legal diamonds: executable composition experiment

H1 in [LATENT_EXPERIMENTS.md](LATENT_EXPERIMENTS.md) now has an oracle-certified
corpus, a differentiable objective, three controls, and a measured gfx1030 smoke.
The smoke found **no meaningful additional benefit from the crosspath term**.
It exposed an important failure mode: initially similar endpoints can reflect
weak action effects rather than correct composition. This is synthetic mechanics
evidence, with no strength or global architectural novelty claim.

## What the corpus certifies

`neural/diamonds.py` samples reachable roots by independently playing legal
games from the standard initial state. It samples a four-ply quiet path A, then
tries path B with each color's two moves swapped. The initial fixture is
`Nf3 Nf6 Nc3 Nc6` versus `Nc3 Nc6 Nf3 Nf6`. The generator accepts a sample only
after replaying both paths with python-chess; presumed commutation is insufficient.

Every intermediate move must be legal, noncapturing, nonpawn, nonking,
nonpromotion, nonchecking and preserve castling rights. Every node must be
nonterminal and outside check/repetition/draw-claim boundaries. Roots with
halfmove clocks at least 90 are conservatively rejected. The retained root
history is checked for any repeated position; every path node rejects twofold
repetition, current or next-move threefold/fifty-move claims, and automatic draw
conditions. Endpoints must have identical six-field FENs, including clocks,
side, castling and raw EP, plus identical effective-EP FENs and current draw
context. Four quiet plies necessarily clear any root EP target.

The two full repetition-history maps still differ because their intermediate
positions differ. This experiment certifies matching current unclaimed context
and excludes draw-boundary histories; it does not erase path history or prove
that all future continuations have identical repetition outcomes. Exact search
must preserve each actual path's history.

Records freeze origin FEN, retained root moves, root FEN, both UCI paths and all
intermediate FENs. Features use **raw FEN EP targets**, matching schema 1 in the
ordinary replay generator. Endpoint equivalence also checks effective legal EP.
Each action is encoded in its own ply's side-to-move frame. The held-out corpus
rejects overlap with any training intermediate state by hashing the actual
canonical encoded tensor, so fullmove-number and color/rank equivalents cannot
evade the split. This remains a small synthetic split, with no opening or
material-class generalization claim.

## Objective and controls

`diamond_losses(model, features[B,2,5,18,8,8], actions[B,2,4])` performs no host
extraction. It encodes the exact path states and runs both learned four-ply paths.
The controls share architecture, initialization seed, corpus and device sample
schedule:

| Control | Optimized objective |
| --- | --- |
| `ordinary` | Exact-state reconstruction plus one-step successor board/rules and detached latent target |
| `refresh` | Ordinary objective plus four-ply detached exact-endpoint latent refresh and endpoint board/rule reconstruction |
| `diamond` | Refresh objective plus normalized latent agreement between the two learned endpoints |

The source exposes each component separately. Reconstruction includes pieces,
castling, EP and clipped halfmove clock. Refresh targets use a detached online
encoder; an EMA target is an independent future experiment. There is no policy,
outcome or teacher-value learning in this standalone composition experiment.
Equal sample schedules do not equalize backward-graph cost or wall time.

Diagnostics report endpoint board BCE, overall and occupied-piece classification,
rights accuracy, EP accuracy and clock MAE. They distinguish square/spatial
latent variance from variance across different root states. A legal-action probe
compares each path's first move: these two moves are legal from the same root
and lead to different exact states. It reports learned successor separation,
exact re-encoded separation, and their ratio. A low crosspath error alone cannot
establish action understanding or prevent collapse; position embeddings alone
can keep spatial variance nonzero.

## Measured smoke and its implication

The gfx1030 run used float32 ROCm PyTorch `2.10.0+rocm7.0`, 16 training diamonds,
4 held-out diamonds, seed `20260930`, width 16, depth 1, batch 4 and four optimizer
steps per control. All nine model/control combinations completed with finite
losses. Initial metrics were identical within each architecture's controls.
There is one seed and very little training, so this run cannot rank architectures.

| Model | Held-out latent refresh error: ordinary | Refresh control | Diamond control | Diamond occupied-piece accuracy |
| --- | ---: | ---: | ---: | ---: |
| square | 0.0549151 | 0.0463032 | 0.0463030 | 26.4% |
| relation | 0.0284625 | 0.0186934 | 0.0186932 | 2.4% |
| plan | 0.0380972 | 0.0327655 | 0.0327655 | 1.6% |

The endpoint-refresh control reduces its own latent error. Adding the diamond
term changes that error by approximately `2e-7`, `2e-7` and `2e-8`, respectively.
Crosspath disagreement was already roughly `4e-6` before training, and
occupied-piece reconstruction remains poor. Lower BCE can accompany worse
occupied-piece accuracy because empty squares dominate. EP endpoint accuracy
is an easy all-absent task in this quiet-only corpus. None of these results
supports a useful composition or playing-strength advance.

The legal first-move probe strengthens that caution: diamond-control learned
successor separation is only 2.4%, 7.2% and 5.6% of exact re-encoded separation
for square, relation and plan. This measures weak action effects relative to
each model's own encoding, rather than proving a calibrated semantic distance.
Nonzero spatial or across-state variance does not repair this discrepancy.

The next experiment should explicitly supervise legal sibling action effects
or compare their latent deltas against exact re-encoding, then test longer
horizons and captures/rights changes under matched data and compute. Diamond
loss should be promoted only if it improves held-out state fidelity beyond the
refresh control while preserving action discrimination and tactical performance.
A future persistent-plan architecture can consume these certified transpositions;
this initial objective does not implement persistent threat plans.

## Reproduction and evidence

```sh
python -m unittest neural.tests.test_diamonds -v
python -m neural.diamonds --device cuda --samples 16 --steps 4 \
  --batch-size 4 --width 16 --depth 1 --models square relation plan \
  --output results/neural/diamonds/gpu-final.json
```

The report includes source-file SHA256s, both complete frozen corpora and their
fingerprints, sample-schedule hash, all losses/diagnostics and peak allocation.
The authoritative local report is `results/neural/diamonds/gpu-final.json`;
`results/neural/diamonds/tests.txt` records the six tests. These artifacts are
ignored experimental results. Earlier `gpu-smoke.json`/`cpu-smoke.json` are
superseded pre-audit runs. The final run uses corpus hashes:

* Train: `8122a29dd0facabacac42dd47a75fe7909106fd7f6811cb2b21e7aa7292b4050`
* Held out: `b26e01d64345377761aae90841ed0dba0193604433be1249f24fa422d96d65c5`

Tests replay every generated move and root history, verify complete endpoint
equality, raw EP/schema consistency, action orientation, split identity,
reproducibility, equal initialization metrics and separate differentiable
composition gradients. They reject dependent knight moves, pawn commutation,
repeated histories and the fifty-move boundary. The report's first timed control
includes initial GPU optimizer/kernel setup; its elapsed time is not a steady
throughput comparison. No neural UCI integration or Elo match is part of this run.
