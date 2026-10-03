# GPU engine integration

`BatchEvaluator` receives ordered positions and produces matching ordered scores;
`SearchBackend` owns search scheduling and budgets. `CpuBatchEvaluator` and
`HipBatchEvaluator` implement the same contract. `Searcher` wraps the original
alpha-beta implementation; `FrontierSearch` supplies bounded full-width minimax
with level-synchronous leaf batching. `StrengthPolicy` controls resource levels.
These are explicit alternative algorithms/backends, rather than replacing CPU
search with one expensive GPU call per leaf.

`RulesBackend` exchanges ordered full-rule packets and legal transitions.
`ResidentTree` owns arena state and operations, while `ResidentSearch` supplies
one scheduler for `HostTree` and `HipRules`. The latter runs legal generation,
hash-preserving transitions, adjudication, evaluation and reduction in HIP
kernels. [GPU_RULES.md](GPU_RULES.md) specifies its ABI and measured scope.

## Run

The portable engine does not require ROCm. These experiments require Linux,
an AMD gfx1030 GPU, and a ROCm environment with `hipcc` available. Build directly:

```sh
cargo build --release --locked --features hip
bash gpu/build.sh
CHEN_GPU_LIBRARY="$PWD/gpu/build/libchen_gpu.so" ./target/release/chen
```

Select a HIP backend through UCI, for example
`setoption name Backend value hip-resident`, before searching.
The library path must point to the built shared library.

Alternatively, use the wrapper with an existing ROCm Docker container. Set
`CHEN_ROCM_CONTAINER` to its name; the historical default is `cpp-rocm`.
The wrapper does not create or provision the container.

### Container wrapper

```sh
scripts/rocm_run.sh test
scripts/rocm_run.sh bench --backend hip-frontier --depth 3 --batch-size 4096
scripts/rocm_run.sh bench --backend hip-resident --depth 3 --max-positions 250000
scripts/rocm_run.sh gpu-perft 4 --max-positions 250000
scripts/rocm_run.sh uci
```

The wrapper builds Rust with `--features hip` on the host, copies only task files
into a unique container directory, compiles HIP for actual gfx1030, sets
`CHEN_GPU_LIBRARY`, runs, and cleans its directory. It needs a compatible host
Linux binary and the already running `cpp-rocm` container. The hardware tests
are ignored in ordinary `cargo test`; this wrapper runs them explicitly and
fails if the actual device/backend is absent. There is no passing skip or CPU
substitution in that hardware test path.

In the UCI session set `Backend` to `hip-resident`. Available values are
`alphabeta`, `cpu-frontier`, `hip-frontier`, `cpu-resident`, and `hip-resident`. `Eval`, `BatchSize`,
`MaxPositions`, and `Level` select the evaluator, batch capacity, tree capacity,
and resource level. Resident search uses arena capacity rather than `BatchSize`.
GPU library creation or execution errors are reported.
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
moves and reduces minimax on the host. For `hip-frontier`, only batched static
evaluation runs on the GPU. Its event intervals exclude Rust packing and include possible
host enqueue gaps; full search `elapsed_ns` includes expansion, packing,
transfers, dispatch and reduction, but not backend construction. Ordinary NPS
is algorithm-specific work, not a strength metric.

The classical scoring ABI uses an evaluation-only packet. Contexts own
persistent device/pinned staging buffers and are movable with exclusive
ownership. Linux `dlopen` keeps the optional GPU dependency separate from the
portable rules/search build. ABI layouts, length/capacity errors, actual GPU
parity, and thread transfer are tested.

`hip-resident` uploads the root and previous hashes once per iterative-deepening
iteration, retains the tree in VRAM, and downloads small statistics plus the
root score/PV. The host still controls depth, budgets and dispatch. Rust legal
generation supplies an emergency root move and handles terminal roots at the
boundary. Device work is level-synchronous full-width minimax; it has no TT,
quiescence or selective pruning. A single thread handles each position and a
single-thread prefix pass assigns child ranges: these are correctness baselines,
with substantial optimization room. The arena retains every level and a fixed
256-move buffer per slot, so capacity cost is material.

Rules ABI 1 carries 128-byte states, distinct from the 104-byte evaluation-only
packet. Classification mode and evaluator are fixed until reset. Repeated leaf
classification/evaluation preserves counts; mixing perft/search or evaluators,
evaluating expanded nodes, or expanding scored static leaves fails explicitly.
Capacity exhaustion admits no new children and preserves the last completed
search iteration. Exact node caps still permit admitted leaves to be scored.
Arena event spans accumulate since reset and include enqueue gaps, but exclude
root/history uploads and debug/result downloads. End-to-end `elapsed_ns`
includes scheduler and transfer overhead, excluding context construction.

Direct HIP-to-Torch ingestion now exports resident transitions into Torch-owned
VRAM and admits them to compact replay in `../chen-neural`. The
[export contract](GPU_REPLAY_EXPORT.md) covers runtime identity, epochs, allocation
bounds and completion. A combined suite runs 62 tests on actual gfx1030:

```sh
python3 -m venv /tmp/chen-oracle
/tmp/chen-oracle/bin/pip install chess==1.11.2
CHEN_QA_PYTHON=/tmp/chen-oracle/bin/python scripts/rocm_run.sh neural-test
```

`CHEN_NEURAL_WORKTREE` can override the sibling checkout. Torch is supplied by
the existing container; the wrapper copies the neural sources and development
oracle into its isolated task directory. Root/history ingress happens on the
host; subsequent exported board payload, replay sampling and training stay on
the GPU. The later [search teacher](GPU_TEACHER.md) supplies exact backed-up finite-tree
scores and chosen actions. [Native neural inference](GPU_NEURAL.md) uses trained
square/relation/plan static encoders as resident leaf evaluators. Both are
experimental: the first heldout comparison trails material. GPU self-play,
stronger training and a policy-guided scheduler remain open. See [GPU_RESEARCH.md](GPU_RESEARCH.md) for
the hardware/literature ledger.
