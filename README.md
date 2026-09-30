<p align="center"><img src="assets/chen.svg" width="720" alt="Chen, a chess-coded two-tailed catgirl mascot" /></p>

# Chen

A chess engine written from scratch in Rust, and a laboratory for controlled
architecture experiments. No runtime dependencies. The current engine is a
classical baseline. This `research/neural` branch hosts independent neural
experiments; the main `../chen` worktree owns current GPU engine integration.

The research includes [square/relation/plan architectures](docs/NEURAL_ARCHITECTURES.md),
[legal-diamond dynamics](docs/DIAMOND_EXPERIMENT.md),
[observed move transport](docs/ACTION_DYNAMICS.md), and
[compact resident replay](docs/COMPACT_REPLAY.md). The latest combined suite
passes 48 tests on gfx1030. Synthetic mechanics and mixed/negative dynamics
results are recorded explicitly; no neural playing-strength claim is made.

```sh
cargo build --release
cargo test --release
cargo clippy --all-targets -- -D warnings
./target/release/chen                 # UCI on stdin/stdout
./target/release/chen perft 5         # 4,865,609
./target/release/chen divide 3
./target/release/chen bench --depth 4 # JSON Lines
python3 scripts/bench.py --repeats 5 --candidate-args='--no-tt'
```

Load `target/release/chen` as a UCI engine in a chess GUI. A short session:

```text
uci
isready
position startpos moves e2e4 e7e5
go depth 4
# Wait for bestmove, or send stop to interrupt.
quit
```

UCI supports `position startpos|fen … moves …`, iterative `info` with PV,
`go depth|nodes|movetime`, clocks/increments/`movestogo`, `go infinite`, `stop`,
`ucinewgame`, and `quit`. Search runs on a worker so `stop` and `isready` stay
responsive. Options: `Hash` (MiB), `Eval` (`tapered` or `material`), `UseTT`,
`Quiescence`, and `MoveOrdering`. `searchmoves`, pondering, Chess960, MultiPV,
and multiple search workers are future work; unsupported `go` parameters
produce an `info string` error.

## Components

| Module | Owns | Experiment seam |
| --- | --- | --- |
| `core` | Bitboards, attacks, FEN, legal moves, special moves, hashes, make/unmake, perft | Replace attack kernels or move generation against the rules oracle |
| `eval` | Side-to-move centipawn score and evaluator state | `Evaluator` trait; material/tapered baselines; reset and make/unmake hooks |
| `search` | Iterative deepening, fail-soft alpha-beta, check-aware quiescence, mate/draw handling, ordering, cancellation | Explicit `SearchConfig`; isolate a search policy here |
| `tt` | Bounded direct-mapped cache, bounds, replacement, generations | Compare layouts at equal memory budgets |
| `main` | UCI commands, worker lifetime, clocks, CLI/JSONL output | Keep protocol out of board/evaluator experiments |

The legal generator filters pseudo-legal moves by king safety. Sliding attacks
scan rays. Undo saves the complete position; deterministic Zobrist hashes update incrementally,
with full recomputation available as an oracle. These
are auditable reference implementations to compare against optimized kernels.
The evaluator uses material, simple piece placement, bishop pairs, and tapered
king placement. Ordering uses the TT move, captures/promotions, killers, and
quiet history. Search currently has no selective pruning.

Positions use `a1 = 0`. `make_move` accepts a legal move; validate external
moves through `parse_uci_move` or `legal_moves`. FEN parsing checks structural
and local invariants, rather than proving reachability. Search owns game
history; pass previous hashes, excluding the root, to `Searcher::search`.
Use `set_config` and `set_evaluator` to change experiments; both invalidate cached
scores and ordering state. Clear its atomic stop handle before starting another
job after cancellation. Static scores are clamped to ±29,000, reserving a separate
range for mate distances.

Search treats available fifty-move and threefold claims as draws, after checking
for mate. Insufficient material uses conservative dead-position detection.
Quiescence searches all check evasions, captures, and promotions, with a hard
128-ply safety bound. No tablebases are included. Transposition identity includes
the reversible history and halfmove clock to protect draw-sensitive results;
this deliberately limits sharing, leaving a measurable optimization question.
TT scores normalize mate distance. A returned PV can end at a cached bound;
it is not guaranteed to reach the reported depth.

## Experiments and the neural layer

`bench` uses a fresh searcher and TT for every suite position. Select
`--eval material|tapered`, `--no-tt`, `--no-qsearch`, `--no-ordering`, `--hash MB`,
and `--suite FILE` independently. Suites are `name | FEN` lines. JSONL includes
configuration, FEN, completed depth, score, best move, total/quiescence nodes,
elapsed nanoseconds, NPS, cancellation status, and PV. Mate scores use the internal ±30,000 scale.
Total nodes include quiescence nodes.

`scripts/bench.py` warms both binaries, alternates paired runs, checks within-
variant determinism, saves raw results and run metadata, and reports paired time
ratios. Supply `--base` and `--candidate` to compare preserved binaries. It does
not build binaries. Timing results measure the local machine and workload;
playing strength requires separate paired games. The harness archives tracked and untracked, nonignored source files alongside
the dirty patch, including an uncommitted initial checkout.

An incremental evaluator gets `reset`, `after_make`, and `after_unmake` callbacks
for its own accumulator stack. The rules kernel does not know about a network.
Start with a scalar full-refresh oracle, specify features and serialization,
then compare incremental state and SIMD output bit-for-bit. This branch contains
neural model/training experiments and resident replay; production Rust inference
and a trained playing-strength evaluator remain open.

Optional independent QA uses python-chess in a separate development environment:

```sh
python3 -m venv /tmp/chen-oracle
/tmp/chen-oracle/bin/pip install chess==1.11.2
/tmp/chen-oracle/bin/python scripts/oracle.py
```

See [the validation record](docs/VALIDATION.md), [the experiment protocol](docs/EXPERIMENTS.md) and
[the verified reading shelf](docs/REFERENCES.md) for books, original papers,
NNUE references, and the optimization roadmap. Correctness tests, cost
measurements, and playing-strength matches are separate evidence.

Original vector mascot: [full logo](assets/chen.svg) · [square mark](assets/chen-mark.svg).
