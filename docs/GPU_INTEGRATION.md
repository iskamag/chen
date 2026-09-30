# GPU engine integration

`BatchEvaluator` receives ordered positions and produces matching ordered scores;
`SearchBackend` owns search scheduling and budgets. `CpuBatchEvaluator` and
`HipBatchEvaluator` implement the same contract. `Searcher` wraps the original
alpha-beta implementation; `FrontierSearch` supplies bounded full-width minimax
with level-synchronous leaf batching. `StrengthPolicy` controls resource levels.
These are explicit alternative algorithms/backends, rather than replacing CPU
search with one expensive GPU call per leaf.

## Run on the existing container

```sh
scripts/rocm_run.sh test
scripts/rocm_run.sh bench --backend hip-frontier --depth 3 --batch-size 4096
scripts/rocm_run.sh uci
```

The wrapper builds Rust with `--features hip` on the host, copies only task files
into a unique container directory, compiles HIP for actual gfx1030, sets
`CHEN_GPU_LIBRARY`, runs, and cleans its directory. It needs a compatible host
Linux binary and the already running `cpp-rocm` container. The hardware tests
are ignored in ordinary `cargo test`; this wrapper runs them explicitly and
fails if the actual device/backend is absent. There is no passing skip or CPU
substitution in that hardware test path.

In the UCI session set `Backend` to `hip-frontier`. Available values are
`alphabeta`, `cpu-frontier`, and `hip-frontier`. `Eval`, `BatchSize`,
`MaxPositions`, and `Level` select the evaluator, batch capacity, tree capacity,
and resource level. GPU library creation or execution errors are reported.
Creation failure keeps the requested configuration unavailable and prevents
CPU substitution; select another valid backend to recover. failed search produces an
explicit error plus `bestmove 0000`, which the legality harness treats as failure
on a nonterminal board.

CPU-frontier is the direct search oracle. Compare it against HIP-frontier at
identical depth/batch/capacity/evaluator; compare alpha-beta with quiescence off
for score equivalence. Tie-breaking can differ between search algorithms.
`MaxPositions` limits nodes stored per iteration; capacity exhaustion preserves
the last completed depth and reports `stopped=true`. Requested depth can exceed
completed depth, and must be checked before interpreting comparisons.

## Semantics and measurement

Both frontier implementations adjudicate mate before claims, retain exact
repetition context, and clamp static scores outside the mate range. They leave
the caller's root unchanged. Frontier has no quiescence, TT, selective pruning,
or move-ordering heuristic; those alpha-beta switches do not apply to it. JSONL
records their *effective* settings, along with backend, capacity, completed
depth, node count, evaluated positions and HIP event intervals.

Frontier total nodes count every visited position, including repeated iterative
deepening visits and unevaluated terminal nodes. The CPU oracle expands legal
moves and reduces minimax on the host. Only batched static evaluation runs on
the GPU today. Its event intervals exclude Rust packing and include possible
host enqueue gaps; full search `elapsed_ns` includes expansion, packing,
transfers, dispatch and reduction, but not backend construction. Ordinary NPS
is algorithm-specific work, not a strength metric.

The HIP ABI uses an evaluation packet, not full search state. Contexts own
persistent device/pinned staging buffers and are movable with exclusive
ownership. Linux `dlopen` keeps the optional GPU dependency separate from the
portable rules/search build. ABI layouts, length/capacity errors, actual GPU
parity, and thread transfer are tested. Native rule-state/frontier/replay arenas,
neural decoding and batched policy-guided search remain the next integration
steps. See [GPU_RESEARCH.md](GPU_RESEARCH.md) for the hardware/literature ledger.
