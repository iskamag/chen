# Two-fidelity minimax research direction

Chen now has an isolated CPU reference controller in `src/two_fidelity.rs` and a
runnable synthetic profiler in `examples/two_fidelity_profile.rs`. It implements
2FFS from [Chen and Chen, arXiv:2606.01708v2](https://arxiv.org/abs/2606.01708v2),
particularly Appendix B.3 Algorithms 2–5. It does not replace the UCI engine's
search, add a GPU kernel, or establish a chess rating.

## Model and boundaries

`FiniteTree` supplies a finite balanced tree rooted at node 0. The root maximizes;
players alternate by depth. Values always use the root player's perspective.
Construction rejects cycles, transpositions, unreachable nodes, unbalanced leaf
depths, malformed envelopes, and invalid numerical/resource configuration. The
CPU reference limits depth to 128 and has configurable node, round, cost, and
floating scale bounds. Tree topology is already resident on CPU; value oracles
are queried only as nodes become exposed.

`FastOracle` supplies a deterministic estimate whose absolute minimax error is
bounded by declared `B[h]`. The envelope must be nondecreasing and `B[0] = 0`, so
leaf estimates are exact. `SlowOracle` supplies independent sigma-sub-Gaussian
samples of the same node's minimax value, including at internal nodes. A known
uniform slow bias bound can inflate slow intervals. `OracleContract` records
these premises and their provenance; `justified` is the caller's declaration,
not a runtime check or proof. The result preserves that declaration separately
from its stopping reason.

Ordinary chess rollouts estimate a policy value, generally not minimax value.
Classical evaluation, neural evaluation, deeper alpha-beta search, and Stockfish
reported score quantization currently provide neither the required deterministic
fast envelope nor the required mean-correct slow oracle. A future adapter using
those heuristics must declare `justified = false` and remain **uncertified** until
its premises are established. There is no such chess adapter in this module yet.
A one-second UCI search and a fixed-confidence identification problem are also
different termination contracts.

## Controller mechanics

The controller retains both local evidence and descendant evidence. Each slow
sample updates an incremental mean and intersects its interval with every prior
slow interval. The effective interval intersects this running intersection, the
fast interval, and the minimax backup of expanded children. Expansion does not
retire samples collected at the expanded node; later local sampling remains
available at that node.

Confidence is allocated uniformly over all non-root nodes: `delta_v = delta /
(node_count - 1)`. The implemented radius is

```
beta(n, delta_v) = sigma * sqrt(2 * log(8 * n^2 / delta_v) / n)
```

with the optional slow bias bound added afterward. Two-sided sub-Gaussian tails
and a union over sample counts give failure probability at most
`sum_n 2*delta_v/(8*n^2) = delta_v*pi^2/24 < delta_v`. This radius decreases with
sample count. Its minimal sample count at target `rho/4` is found by doubling and
binary search, with integer overflow producing an infinite recursive budget.
A known bias floor that prevents local certification likewise leaves recursive
expansion available.

This deliberately differs from the [author's numerical implementation](https://github.com/PeterLauLukChen/2FFS/blob/f364fdb2b899e1b704ebb78cf47aab54f6458136/method/twoffs/twoffs.py),
which uses `log(pi^2*n^2/(6*delta_v))`. Applying the ordinary *two-sided* tail bound
to that expression spends `2*delta_v` across sample counts. The Rust version
pays both tails explicitly. It is a valid radius instantiation rather than a
byte-for-byte replication of the author's sampling trajectory. The author's
approximate endpoint ties and `1e-14` width floor are also not copied.

Each scale has monotone lower/upper completion flags and separate local and
recursive spending counters. Bottom-up certificate evaluation latches observed
certificates, including certificates discovered while scanning scales. The zero
tolerance certificate implements the finite/infinite unresolved-side distinction;
the bounded floating implementation returns `PrecisionLimit` if a needed finite
scale cannot be represented or reached within the configured cap.

The outer loop compares the leader's lower side with the challenger's coarser
unresolved side. Selector cases follow current endpoint witnesses. Comparison
cases exclude children outside the scale margin or already same-side complete,
then select the strongest remaining blocker. That blocker is refined on its
opposite endpoint first, then on its same endpoint, always at a scale no finer
than the parent's. Ties use deterministic node/side ordering.

A resolver invocation performs at most one expansion or slow sample. Its
recursive budget is `(remaining_depth + 1)^2 * Gamma(scale)`. Descendant spending
is charged to every active ancestor obligation. An inherited cap that blocks an
action propagates upward; exhaustion of the node's own recursive budget permits
a local slow sample. Interval/certificate state is refreshed after every positive
action before any new descent.

Ordinary stopping requires a root action whose lower endpoint is at least every
competitor's upper endpoint minus epsilon. Exact ties can satisfy that condition;
the paper's unique-best premise is needed for its exact finite-stopping theorem,
not for finite-output validity. An empty interval immediately returns the fixed
first root action with `ConfidenceFailure`, before further certificate work. That
is the paper's exceptional confidence-event exit, **not an ordinary interval
certificate**. Cost, round, and precision exits return no recommendation. Oracle
errors/nonfinite values return errors. These exits cannot be relabeled as wins or
successful fixed-confidence identifications.

The implementation uses ordinary `f64` arithmetic, not directed-rounding interval
arithmetic or a formal numeric proof. Near machine precision, theoretical
real-arithmetic guarantees require additional numerical analysis. Reported
synthetic tests demonstrate behavior on their fixtures; they do not establish
empirical coverage for arbitrary stochastic or chess adapters.

## Profiling and GPU seam

Run the isolated reference and checks:

```sh
cargo test --offline --lib two_fidelity
cargo clippy --offline --lib --example two_fidelity_profile -- -D warnings
cargo run --offline --release --example two_fidelity_profile -- 5 4 5
cargo run --offline --release --example two_fidelity_profile -- 5 8 5
```

The profiler builds finite minimax trees, constructs a valid depth envelope,
queries synthetic exact means plus seeded symmetric bounded noise, and reports
regret, oracle cost, sample counts, expansions, refresh/certificate visits, scale
counts, and separate controller/oracle timings. Its exhaustive baseline ignores
internal heuristic values and backs up exact fast leaf values. Its CPU oracle is
a cheap array lookup, so a reduced abstract query count need not reduce wall time.
Neither this benchmark nor the paper's synthetic experiment is a chess strength
measurement.

`FastOracle::evaluate_batch` receives every child of an atomic expansion in one
ordered request. A GPU implementation can evaluate that batch, return all values,
and preserve the controller semantics. Default behavior is serial CPU queries.
The reference slow action requests one sample and waits before updating state;
blindly collecting many slow samples or pursuing stale blockers changes the
algorithm. Large neural batches should start with multiple independent trees,
or use a separately named asynchronous algorithm with its own tests and claims.
There is no GPU execution evidence for this controller yet.

Five seeded runs at each of depth 5 / branching 4 (1,365 nodes) and depth 5 /
branching 8 (37,449 nodes) all selected the true optimal action. Profiling exposed
full-topology scans of nodes that had never been exposed. The current controller
visits only its sorted exposed nodes in refresh/certificate/storage passes and
reuses one preallocated certificate scratch buffer. Construction still validates
all topology once. This optimization preserves every recorded positive-action
trace byte, sample/cost count, and answer field across all ten paired runs.

| Synthetic case | Full-topology median | Exposed-tree median | Ratio | Certificate slots scanned, median |
| --- | ---: | ---: | ---: | ---: |
| D=5, b=4 | 9.792 ms | 6.815 ms | 1.44x | 870,870 -> 53,050 |
| D=5, b=8 | 1,330.677 ms | 742.852 ms | 1.79x | 185,522,346 -> 4,407,370 |

These are CPU 2 pinned, sequential block measurements, not isolated crossover
benchmarks. The first exposed-tree variant spends a median 97.5% of measured
wall time in scale evaluation. Cheap exhaustive lookup minimax is much faster
in wall time despite querying all 37,448 non-root values. Query savings alone
therefore do not justify a GPU controller. Next optimize affected-ancestor refresh
and compact scale storage, comparing against the frozen baseline's exact traces.
Kernelizing tiny lookups would not address the measured bottleneck. See
[evidence comparison](research/evidence/two-fidelity-cpu-2026-10-02/comparison.json)
and its underlying per-run CSV and traces.

## Reusing completed scale certificates

The next bounded CPU optimization caches which scales have been evaluated since
an evidence refresh. A bottom-up pass reads completed child certificates before
computing each parent; same-scale prior latches are monotone. With unchanged
local/effective intervals and exposed topology, repeating that pass cannot change
its result. Oracle-cost counters do not enter the certificate predicate. Every
refresh clears the valid-scale set, including expansion and slow-sample updates;
ordinary certificate scratch can still be reused for zero-tolerance queries.

All ten positive-action trace files remain byte identical to the exposed-tree
baseline. Sample, expansion, cost, answer and refresh counters also match. The
600 independent irregular/noisy fixtures and all 12 controller tests pass.

| Synthetic case | Exposed-tree baseline | Cached-scale variant | Ratio | Median scale passes |
| --- | ---: | ---: | ---: | ---: |
| D=5, b=4 | 6.013 ms | 2.667 ms | 2.25x | 518 -> 174 |
| D=5, b=8 | 766.957 ms | 287.009 ms | 2.67x | 4,159 -> 1,491 |

These are five seeded runs per case on CPU 2, in sequential baseline/candidate
blocks. They are a new measurement series, not ratios formed from the earlier
series. Frozen standalone source modules, timed binaries, profiles, action
traces, build bindings and checks are in
[the scale-cache evidence](research/evidence/two-fidelity-scale-cache-2026-10-02/comparison.json).
Remaining work includes compact scale storage and affected-ancestor refresh.
This remains a synthetic CPU controller measurement.

## Source provenance

Reviewed paper v2 is dated 29 September 2026. Downloaded PDF SHA-256:
`10469e6d7480b24408c32a2618c22f612dcd908abf2c4588fa17916ec75ce8ef`.
Author repository revision: `f364fdb2b899e1b704ebb78cf47aab54f6458136`.
Its `method/twoffs/twoffs.py` SHA-256:
`3b6a28b774bce2a4ee75c3104dd40f284b2f069ba6dc142998dfba25b05fcc46`.
The Rust source is an independent implementation with the explicit radius and
finite-resource differences above. Frozen run sources, tests, profiles, action traces, and build bindings are under
`docs/research/evidence/two-fidelity-cpu-2026-10-02/`. Twelve unit tests include
sixty independently backed-up synthetic minimax fixtures, side-certificate
endpoint checks, exact ties/epsilon, retained non-root samples, both cap cases,
known slow bias, malformed input, and confidence/resource failure exits.

An [independent source and algorithm review](research/evidence/two-fidelity-review-2026-10-02/independent-review.json)
additionally checks 600 irregular, reordered and noisy fixtures, exposed-node
intervals, the archive's build bindings and exact paired traces. It finds no
required source correction. This validates the tested implementation and its
declared numerical scope; it does not verify real-world oracle assumptions.
