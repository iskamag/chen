# CPU search scratch reuse, 2026-10-15

The CPU alpha-beta implementation now retains child principal-variation buffers
per ply and shares one move-ordering scratch vector. This removes transient
allocation work while retaining the existing search policy, move ordering,
quiescence, terminal handling and budgets. The GPU path was not executed.

Each node takes its own ply's child-PV vector, clears it before each sibling,
copies an improving line into the caller's PV, and restores the scratch vector
on ordinary return or cancellation. Recursive descendants use different ply
slots. Search reset intentionally retains scratch capacity; the added regression
compares complete published traces against fresh searchers after interrupted
tactical, promotion, terminal and opening searches, including PVS/aspiration.

Move scores are still computed once. An explicit original-index tie break gives
the same ordering as the previous stable cached-key sort; the new unstable sort
requires no temporary allocation. Ordering finishes before descent, allowing
one scratch vector to serve every ply. This introduces no pruning or selective
search and changes no defaults.

## Reproduction and scope

The native timing control uses `chen bench` on `benches/positions.txt`, depth64,
200,000 nodes, fresh searchers and TT for each position, PVS off/on independently,
with two warm-up rounds and alternating baseline/candidate order. Compare every
position's completed depth, score, best move, node/qnode counts, cancellation and
PV before interpreting timing. Mate terminates before the requested work budget
and its tiny timings should not be interpreted as a throughput measurement.

`cargo run --release --example cpu_search_allocations -- 100000` is a separate
allocation-counting diagnostic over four positions and both PVS settings. Its
global allocator counts allocation/reallocation calls only during each search;
searcher/TT construction and formatting are excluded. The atomic counter changes
timing, so native engine observations are the latency evidence.

The baseline is `c4c90961e87778dee811da9d78873a07ab205876`. Allocation comparisons
compile this same example against both revisions. All measurements are CPU-only
on an AMD Ryzen 5 3600 with rustc1.99.0 (`b940084d7`), release ThinLTO and one
codegen unit. They establish scoped implementation equivalence and allocation/
latency observations; they do not establish a playing-strength improvement.

## Observations

The final native control was pinned to CPU3 on a shared host. Across six paired
timed rounds, median baseline/candidate speedup from complete suite-time totals
was **1.0225× with default PVS off** and **1.0218× with PVS on**. Individual
candidate/baseline round ratios ranged from 0.9611 to 1.0384 off, and 0.9592 to
1.0307 on, so the small timing gain should retain this variance qualification.
All 12 position/PVS signatures matched in every observed round, including both
warm-up rounds. No builds ran in this lane during that final control.

Allocation/reallocation calls at 100,000 visited nodes fell by **29.8–42.4%**:

| Position | PVS | Baseline calls | Candidate calls |
| --- | --- | ---: | ---: |
| Start | off | 202,530 | 138,217 |
| Start | on | 205,532 | 136,504 |
| Kiwipete | off | 180,257 | 103,815 |
| Kiwipete | on | 171,688 | 101,668 |
| Rook ending | off | 203,606 | 132,998 |
| Rook ending | on | 206,963 | 130,557 |
| Middlegame | off | 156,200 | 107,939 |
| Middlegame | on | 151,830 | 106,587 |

The [compact raw evidence](evidence/cpu-search-20261015.json) retains binary,
source, suite and runner hashes; all per-position timing observations; exact
search signatures; and allocation counts. Native runner invocation:

```sh
python3 scripts/compare_cpu.py \
  --base BASELINE_CHEN --candidate target/release/chen \
  --suite benches/positions.txt --rounds 6 --nodes 200000 --cpu 3 \
  --out /tmp/chen-search-comparison
```

Validation passed: all 13 search integration tests, all four internal search
tests, `cargo fmt --all -- --check`, and `cargo clippy --all-targets -- -D warnings`.
The bundled opening-book test uses the pinned TSV cache already present in the
main checkout; an initial offline build without that cache failed only that test.
