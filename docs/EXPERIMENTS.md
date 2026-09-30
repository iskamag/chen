# Making Chen experiments comparable

This is the proposed laboratory protocol. The first engine provides the board,
search, classical evaluator, UCI adapter, and deterministic benchmark foundation.
The experiments below are a roadmap; they are not implemented results or a
playing-strength estimate.

## Boundaries to preserve

The board owns chess rules and state transitions. Search owns its stack,
transposition table, repetition context, limits, and telemetry. An evaluator
scores a position from the side to move; checkmate and draw adjudication stay in
search. UCI translates commands into these components and owns protocol output.
Benchmarks call the same engine components as UCI.

Keep experiments explicit: choose an evaluator and a small search configuration
at construction, give each variant a stable name, and record that configuration
with every result. A runtime toggle suits cheap policies; a generic implementation
or separate build suits kernels where dispatch changes the cost. Use a shared
input/output contract for either. Add abstractions when a second implementation
demonstrates the need.

## Three independent gates

1. **Correctness:** legal move sets and perft; state round trips; incremental
   hash versus recomputation; mate/stalemate/draw fixtures; bounded search
   differential tests. Compare score, and allow different equally good root
   moves. Perft verifies movement rules, not search or playing strength.
2. **Cost:** same positions, same work budget, release binaries, repeated paired
   measurements. Report elapsed time and node counts together; a higher nodes/s
   rate can accompany weaker search. Measure perft, evaluation throughput, and
   full search separately.
3. **Playing strength:** paired UCI self-play with saved PGNs, fixed clocks,
   openings, adjudication rules, and statistical settings. A tactical suite is
   diagnostic evidence; match outcomes measure the tested conditions.

For rule-preserving kernel changes, require identical legal move sets and
fixed-depth scores with selective pruning disabled. For heuristic search changes,
score differences can be expected; retain tactical regressions and self-play as
separate evidence.

## Reproducible result envelope

Every run should retain the engine commit and dirty diff, binary SHA-256, build
profile, Rust compiler version, CPU model/features, operating system, thread
count, affinity policy, hash budget, evaluator and search configuration, suite
hash, and random seeds. For a neural run add network hash, feature schema,
quantization version, and dataset/training provenance.

The benchmark's JSONL rows are the position-level payload. Put the run envelope
beside them until the harness carries these fields directly. Record the exact
command and node-counting convention, including whether quiescence nodes count.
Do not assume elapsed time is deterministic because node counts are.

For each position record best move, score, completed depth, nodes, elapsed time,
and abort status. Future telemetry can add evaluation calls, TT probes/hits/cutoffs,
first-move cutoffs, quiescence nodes, re-searches, and maximum ply. Clear the TT
between independent cases or explicitly declare a warm-table experiment. Separate
fixed-depth, fixed-node, and fixed-time results.

Warm up both binaries and alternate AB/BA order for several repetitions. Report
paired time ratios across positions and their spread rather than choosing the
best run. Use both equal-time and equal-node comparisons when evaluating a more
expensive evaluator. Keep tuning positions and validation positions separate;
save disappointing experiments too.

## Prioritized experiments

| Order | Experiment | Controlled comparison | Evidence to retain |
| --- | --- | --- | --- |
| 1 | Board transition cost | Copying a board versus make/unmake with undo state | State/hash round trips, allocation counts, perft and search time |
| 2 | Sliding attacks | Ray scan versus lookup tables; optionally BMI2 PEXT on supported CPUs | Exhaustive relevant-occupancy differential tests, table size, cache counters, CPU-specific timings |
| 3 | Move storage and ordering | Heap vectors versus stack buffers; full sort versus staged selection | Legal move equivalence, allocations, first-move cutoff rate, fixed-depth time |
| 4 | TT layout | Direct-mapped versus clustered entries and replacement policies at equal bytes | Mate-distance/bound tests, hit/cutoff rate, nodes, cache misses, elapsed time |
| 5 | Search windows | Full-window alpha-beta versus PVS and aspiration windows | Root-score agreement without selective pruning, fail/re-search counts, time |
| 6 | Ordering heuristics | Ablate killers and history separately; add SEE ordering as another variant | Nodes at equal depth, tactical regressions, paired games |
| 7 | Selective search | Null move, late-move reduction, futility or delta pruning one at a time | Zugzwang/check/promotion fixtures, re-search counts, independent self-play |
| 8 | Classical evaluation | Ablate tapered placement; add mobility, pawn structure, or king safety separately | Eval cost, feature ablations, fixed-time paired games |
| 9 | Neural evaluation | Refresh-only scalar network, then incremental accumulators, then SIMD | Bit-exact accumulator/inference oracles, eval/search cost, paired games |
| 10 | Parallel search | Single-thread baseline versus a specified sharing strategy | Equal total hardware budget, scaling, nondeterminism, thread-safety checks, paired games |

Each experiment gets one declared hypothesis, one primary metric, and an
acceptance condition written before the run. Preserve the simple implementation
as an oracle when replacing it. Search-window experiments are motivated by
[Knuth–Moore](https://www.sciencedirect.com/science/article/pii/0004370275900193)
and [Plaat](https://liacs.leidenuniv.nl/~plaata1/papers/thesisg.pdf); the ordering
and kernel variants above are Chen's proposed experiments.

## Self-play and SPRT

Use a runner that starts each selected opening twice with colors exchanged.
Pin the runner version, opening file and hash, opening order/seed, both binary
hashes, time control, concurrency, hash allocation, and adjudication settings.
Begin with protocol compliance and a fixed small match to detect crashes,
illegal moves, time losses, and pathological draws. Do not infer an Elo rating
from that smoke test.

[Fastchess's manual](https://github.com/Disservin/fastchess/blob/master/man.md)
supports this shape of future local experiment. The paths below are inputs to
prepare, not supplied files. This example has not been run:

```sh
fastchess \
  -engine cmd=./artifacts/chen-base name=base proto=uci \
  -engine cmd=./artifacts/chen-candidate name=candidate proto=uci \
  -each tc=10+0.1 option.Hash=32 \
  -openings file=./artifacts/openings.epd format=epd order=random \
  -srand 20260930 -rounds 100000 -repeat -concurrency 1 \
  -sprt elo0=0 elo1=5 alpha=0.05 beta=0.05 model=normalized \
  -pgnout file=./artifacts/match.pgn append=false \
  -log file=./artifacts/match.log append=false engine=true
```

The bounds are illustrative, in normalized Elo, and must be chosen before the
test. Ending at the game cap without crossing a likelihood boundary means the
test is inconclusive. A pass concerns the chosen hypotheses and test conditions;
it does not establish an absolute rating. Keep paired outcomes and use a model
that accounts for pairing. Fishtest uses a pentanomial model and generalized
SPRT; its [mathematical documentation](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-Mathematics.html)
describes the distinctions.

Repeated tuning on one opening suite selects for that suite. Recheck accepted
changes on held-out openings and a longer clock. Passing estimates are selected
samples; summing their reported Elo gains biases the progress estimate. Hardware
mix can also change neural/classical comparisons, as discussed in the
[Fishtest FAQ](https://official-stockfish.github.io/docs/fishtest-wiki/Fishtest-FAQ.html).

## Neural evaluator boundary to build later

The existing `Evaluator` trait has `reset`, `after_make`, and `after_unmake`
hooks. They can support a refresh-only evaluator or a state stack. The current
move callbacks receive the resulting position and move; a future efficient
delta interface should also expose changed pieces or cached parent state.

First implement a complete scalar refresh evaluator behind this scoring
boundary. Define feature indexing, orientation, accumulator widths, saturation,
output scaling, and file schema before adding incremental updates. Store model
weights immutably and share them; let each search worker own its evaluator stack.
The following is a proposed interface expansion, not today's API:

```text
refresh(position) -> evaluator state
push(before, move delta, after) -> child evaluator state
evaluate(position, evaluator state) -> side-to-move score
pop() -> restore parent evaluator state
```

Move deltas must identify every removed and added piece: captures, en passant's
different capture square, promotion replacement, and both castling pieces. A
king move can remap king-relative features; refresh the affected perspective
instead of applying an invalid delta. Define null-move behavior separately.
Unmake restores exactly the parent state even when search aborts.

Require incremental accumulators to equal a full refresh after every move in
seeded legal walks and special-move fixtures. Require optimized inference to
equal scalar inference under the declared integer arithmetic. Preserve these
oracles across architectural experiments. This proposal follows the incremental
principle documented in [Stockfish's NNUE reference](https://official-stockfish.github.io/docs/nnue-pytorch-wiki/docs/nnue.html),
while leaving Chen's architecture and model format open.

Train and validate separately from runtime inference. Split datasets by game or
source to reduce position leakage; retain dataset manifests, labels, loss,
random seed, training configuration, and quantized network hashes. Offline loss
is a screening signal. Accept an evaluator through equal-time self-play after
correctness and throughput gates pass.
