# Baseline validation — 2026-09-30

Toolchain: Rust/Cargo 1.98.1, Linux x86-64. Engine dependencies: none.
The initial checkout has no commit yet; the benchmark harness archives source
and records compiler, machine, binary/suite SHA-256, and commands.

## Automated checks

```sh
cargo test --release
cargo clippy --all-targets -- -D warnings
cargo fmt --check
python3 -m py_compile scripts/bench.py scripts/oracle.py
```

All **24 Rust tests** passed. Coverage includes six reference perft positions,
castling transit attacks, pinned en passant, check evasion, underpromotion,
FEN rejection/round trips, deterministic playout and hash recomputation,
complete undo, mate/stalemate, draw claims, TT collisions/replacement/mate
rebasing, alpha-beta against exhaustive minimax, feature ablations, bounded
cancellation, evaluator-stack balance on abort, evaluator/policy replacement,
static-score bounds, and a separate UCI process with repeated immediate stops.
Strict Clippy and formatting checks passed. Both Python scripts compiled.

Additional release CLI runs:

| Position | Depth | Nodes |
| --- | ---: | ---: |
| Start position | 5 | 4,865,609 |
| Kiwipete | 4 | 4,085,603 |

## Independent oracle and protocol check

`scripts/oracle.py` was run with `chess==1.11.2` (python-chess distribution
1.999) in an isolated QA virtual environment, using seed 20260930.
It passed **100 independent legal-move and depth-2 divide comparisons** across
special-move fixtures and eight random playouts. Random positions were also
replayed through UCI and compared by full FEN, including the raw en passant
field. A python-chess UCI session completed **60 legal self-play plies** at depth
2. The QA library is external development tooling, separate from the engine.

## First controlled ablation

```sh
python3 scripts/bench.py --depth 4 --repeats 5 \
  --candidate-args='--no-tt' --out results/final-baseline-vs-no-tt
```

Both variants use tapered evaluation, quiescence and move ordering, and a
fresh 16 MiB table budget per position. Total nodes include quiescence.
Scores, best moves, and node counts were deterministic within each variant.
All six root scores and best moves agreed between variants:

| Position | Score (cp / internal mate scale) | Best move | TT nodes | No-TT nodes |
| --- | ---: | --- | ---: | ---: |
| Start | 0 | b1c3 | 2,107 | 2,518 |
| Kiwipete | -18 | e2a6 | 104,100 | 126,319 |
| Rook ending | 117 | b4f4 | 1,600 | 2,094 |
| Middlegame | 8 | d4c5 | 20,351 | 18,588 |
| Promotion | 880 | a7a8q | 791 | 781 |
| Mate in one | 29,999 | f7g7 | 29 | 29 |

The mate fixture terminates after depth 1. TT effects on ordering can increase
work on individual positions; this result is an initial diagnostic. Raw timing
samples, paired ratios, metadata, and the source archive live in the ignored
`results` directory. Timings are local observations with uncontrolled system
noise; no performance acceptance threshold was declared for this smoke run.

## Evidence limits

These checks establish the tested rule/search/protocol behavior. No Elo rating,
long self-play match, SPRT decision, neural inference, training, SIMD kernel,
parallel search, GUI integration session, or cross-platform run was performed.
The logo SVGs were parsed, rendered, and visually inspected at full size and
64 px. Future architecture changes should rerun the relevant oracles and retain
separate cost and playing-strength evidence.
