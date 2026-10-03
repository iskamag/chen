# CPU causal profiling with Coz

Actual Coz profiling completed successfully. Fixed-line virtual acceleration
of the `piece_at` bitboard probe predicts lower visited-node periods in this
stationary CPU workload. Whole-search counters are too coarse here to support
a precise total-search gain. This identifies a candidate for implementation
and clean timing; it is not an achieved CPU optimization.

The optional `causal-profile` feature adds CPU progress markers without changing
normal engine builds. This experiment targets the alpha-beta/Tapered search
with the default SWAR attacks. It does not profile GPU kernels or neural
inference.

The [stationary example](../examples/coz_search.rs) repeatedly clears the
searcher and searches the same Kiwipete position to exactly 200,000 visited
nodes. Each repetition asserts identical completed depth, score, best move,
node/quiescence counts and full PV, and verifies the original position was
restored. Move-clock limits are disabled. These assertions constrain the
profiled workload; they are not an independent move-legality or strength test.

Three marker types are present:

| Marker | Meaning |
| --- | --- |
| `cpu_256_visited_nodes` | One throughput unit per 256 visited CPU nodes |
| `cpu_completed_fixed_node_search` | One throughput unit per completed, checked fixed-node search |
| `cpu_fixed_node_search` begin/end | Search interval excluding reset and post-search assertions |

Coz performs virtual-speedup experiments on selected source lines and relates
them to progress rates. This measures an optimization hypothesis rather than
only attributing hot samples. Its [official documentation](https://github.com/plasma-umass/coz#using-coz)
requires debug information and distinguishes throughput markers from paired
latency markers. The local Coz CLI/runtime is version 0.2.5; the Rust marker
crate is 0.1.3. The causal Cargo profile inherits optimized release settings
with debug information enabled.

## Bounded experiment and provenance

The frozen executable has SHA-256
`4f405455ca8fe2845f71758214f2c0e0abebe07c45ee6dd0014b75aab18dedc1`.
The generated dependency file, relevant Rust sources, Cargo manifest/lockfile,
compiler fingerprint, marker crate source, profiler CLI/runtime and debug
section listing are bound in `results/coz-cpu-2026-10-03/source-binding.json`.
Code provenance is `b3f3709` plus the root-owned optional Coz changes; the later
`b0b3d8f` HEAD adds proof material. Treating HEAD alone as this build's complete
source identifier would hide the uncommitted marker changes.

Root observed a successful up-to-date build validation with this command:

```sh
CARGO_TARGET_DIR=results/coz-build taskset -c 3 cargo build \
  --profile causal --features causal-profile --example coz_search
```

Its executable hash was unchanged before and after. The original successful
recompilation was observed, but its exact earlier argv was unavailable; the
binding records that distinction rather than reconstructing an invocation as
an actual log.

Runtime was reserved for CPU5 after the one-second match retired and
root explicitly released the CPU window. CPU11, its SMT sibling, is avoided.
Both the workload and Coz helper threads inherit the CPU5 mask. The declared
runs are a four-search plain pilot (20-second watchdog), an 80-search scoped
scout (45 seconds), and an adaptively declared fixed-line confirmation (45
seconds). Confirmation length targets roughly 30 seconds from pilot speed,
clamped to 80–1000 searches. No system permissions or kernel settings change.

The runner launches isolated process groups, captures stdout/stderr and exact
argv/environment overrides, checks stationary completion, and verifies process
retirement and unchanged executable/runtime hashes. All failures remain evidence.
The fixed-line confirmation retains Coz's randomized speedups, including real
zero-speedup controls. There is no fallback to the lowest observed nonzero
speedup as a baseline.

## Interpretation rules

The raw Coz duration is its effective experiment duration after virtual-delay
accounting, not a clean wall-clock timing of an implemented optimization. For
each source line, progress point and virtual speedup, sum effective durations
and marker visits. Compare the resulting period `Σ duration / Σ visits` with
that same line's actual 0% period. Report a prediction only when both groups
have at least five visits and positive selected-line sample exposure. Retain
counts so sparse observations remain visible.

The [profiler implementation](https://github.com/plasma-umass/coz/blob/master/libcoz/profiler.cpp)
records selected-line experiments, throughput visits, latency records and
cumulative sample snapshots. The independent analyzer uses the latest
cumulative count per line; summing snapshots would double-count samples.
Latency records remain raw rather than claiming an independently validated
Little's-Law estimate. The whole-search progress point is the primary complete
work unit; node-marker curves are a secondary within-search view.

This bounded single-position study can rank a hypothesis, not establish a
universally useful optimization or confidence interval. Selected source-line
effects can include inlined code and call-site attribution; they are not
necessarily an exclusive function cost. Sampling cadence, profiler/marker
overhead, helper-thread contention, optimized debug mapping and uncontrolled
package clocks limit interpretation. A promising prediction still needs a
separate implemented candidate and clean equal-trajectory timing comparison.

## Actual runtime outcome

Root released CPU5 only after the 16-game GPU match terminated and its engine
processes retired. All three runs finished normally, with empty stderr,
unchanged executable/runtime hashes, unchanged `perf_event_paranoid=2` and
absent process groups 5325/5491/5788. CPU5 was explicitly released immediately
after confirmation. The window was reserved against other GPU/perf runs.

| Run | Checked searches | Wall time | Emitted Coz experiments |
| --- | ---: | ---: | ---: |
| Plain pilot | 4 | 1.919s | — |
| Scoped scout | 80 | 40.008s | 8 |
| Fixed-line confirmation | 80 | 40.018s | 8 |

The scout observed 139 source lines but had no line with both a nonzero
intervention and a 0% control. It supports no causal gain claim. The subsequent
line choice was explicitly adaptive: the most sampled admitted source line,
`src/core.rs:376`, also had a sampled0% experiment. That line probes one of
the 12 piece/color bitboards in `Position::piece_at`. The confirmation was
declared before launch and used that line for all randomized experiments.

Confirmation emitted two 0% experiments and one experiment at each nonzero
level below. All have positive selected-line sample exposure. These are period
reductions relative to the two controls, not implemented speedups:

| Virtual line acceleration | 256-node marker period reduction | Completed-search marker period reduction |
| --- | ---: | ---: |
| 15% | 2.12% | 12.19% |
| 75% | 7.11% | 16.54% |
| 85% | 7.10% | 6.99% |
| 90% | 8.33% | 17.65% |
| 95% | 8.48% | 18.31% |
| 100% | 8.01% | 17.93% |

The two controls contain 13,309 node-marker visits and 16 completed searches
over 8.000094 effective seconds. Nonzero experiments contain 6,662–6,726 node
visits but only 8 or 9 completed searches. Every fixed search has 781 node
markers; an experiment begins and ends inside searches. The complete-search
counter therefore has substantial boundary quantization: one extra completion
changes an 8-count window by 12.5%. In this run the resulting coarse curve
disagrees with the node curve at several levels. Retain both raw curves and
prioritize finer stationary-work counts for this hypothesis; do not quote the
18.31% whole-search point as an expected implemented improvement. Each nonzero
level has only one experiment, so even the finer curve remains exploratory.

A useful next CPU candidate is a measured change to piece lookup, such as an
empty-square/occupancy shortcut or direct per-square representation. That
candidate must account for representation maintenance in every move and
preserve EP, castling, promotions and trajectory. This milestone changes no
lookup implementation and establishes no achievable gain or strength result.

[Compact evidence](evidence/coz-cpu-2026-10-03/) retains raw profiles, commands,
assertion/retirement records, source bindings, analyzer and a compressed runtime
archive. The independent analyzer replay is byte identical. Its 14 confirmation
period points also match the pinned official CLI formula to numerical tolerance,
while applying stricter actual 0%-control admission. The complete-search curves
are deliberately retained rather than silently dropping the disagreement.

Re-analyze from the repository root without running the workload:

```sh
python3 docs/evidence/coz-cpu-2026-10-03/analyze.py \
  docs/evidence/coz-cpu-2026-10-03/confirmation/profile.jsonl \
  --output /tmp/chen-coz-confirmation-analysis.json
```

Profile-path metadata differs when using another input path; the numerical
curves and counts should match. Full runtime reproduction must use a separately
reserved CPU window and the frozen runner/declaration, not overlap clean timing
or rely on these virtual predictions as timing benchmarks.
