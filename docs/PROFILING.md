# Measured CPU and gfx1030 search profiles

Latest decision: the [matched causal dispatch intervention](CAUSAL_PROFILING.md)
finds only 0.2%/0.45% observed graph-launch savings with identical GPU work and
fences. Prior phase shares below locate hypotheses; they do not by themselves
predict whole-program gains. Avoid another graph-launch tuning cycle now.

The frozen fusion profile identified **GPU ancestor backup and selection**, and
**CPU move application and legal generation**. The actual one-second profiles
are archived in [the summary](../gpu/evidence/engine-profiling-gfx1030-2026-10-02/summary.json)
and [hashed evidence manifest](../gpu/evidence/engine-profiling-gfx1030-2026-10-02/manifest.json).
These observations locate work; they do not establish stronger play.

## Workload and provenance

The frozen Rust executable is `a31be5d1…`, with CPU alpha-beta, Tapered,
quiescence, ordering and a 16 MiB transposition table. The GPU arm uses the
hardware-validated classical fused library `04447c45…`, HIP/ROCm 7.2.4 and the
actual RX 6800/gfx1030. Both receive `go depth 64 movetime 1000`; the GPU uses
batch 32, candidate limit 1024 and capacity 250,000. This is a different
workload from the earlier fusion latency comparison's batch 8/node limit 2000.

The three positions are the opening position, Kiwipete as a tactical middlegame
workload, and a rook ending. There were 21 separate UCI processes: three plain
CPU searches, nine CPU sampling runs, three separate CPU counter runs, three
plain GPU searches and three separately instrumented GPU searches. Every
process retired successfully. Independent python-chess replay checked all 935
reported PVs and all 21 selected moves. The authored positions are not a
tactical accuracy test or a representative strength distribution.

All workloads run on CPU5; the concurrent one-second strength match uses CPU0.
Those are separate physical cores, with SMT siblings 11 and 6 avoided. Shared
cache, package power and GPU clock state remain uncontrolled. Plain runs and
instrumented runs are separate. CPU/GPU node counts and scores cannot be used
as an equal-work speedup comparison: their schedulers and trees differ.

## CPU instruction-pointer sampling

`perf` was absent from the host. Official Artix `perf 7.2.8-2` and libpfm
packages were extracted under `/tmp`, without installing system packages.
The permission probe succeeded with `perf_event_paranoid=2`. Nine searches
recorded `cycles:u` at 199 Hz with DWARF stacks. The summary uses only flat
instruction-pointer symbols from the single search worker: **1,792 samples,
zero lost samples**, with startup/main-thread samples excluded. Recorded
callchains include unresolved or implausible frames, so inclusive callchain
attribution is not admitted. Percentages weight each sample's cycle period;
they are sampled attribution, not exact instrumented function timers.

| Flat worker cycle attribution | Opening | Kiwipete | Rook ending |
| --- | ---: | ---: | ---: |
| `Position::make_move` | 35.19% | 30.52% | 26.27% |
| Ray `sliding_attacks` | 16.80% | 26.56% | 32.75% |
| `Position::legal_moves` | 14.70% | 12.39% | 9.06% |
| `Position::attackers_to` | 10.14% | 10.37% | 12.01% |
| Effective EP identity | 8.13% | 7.89% | 5.20% |
| Tapered evaluation | 6.39% | 4.02% | 3.20% |
| Transposition-table functions | 1.00% | 0.00% sampled | 1.81% |

Separate whole-process hardware counters ran without multiplexing. IPC was
2.69/2.44/2.38, branch-miss ratio 0.73%/1.22%/1.03%, and generic cache misses
0.194/0.240/0.288 per thousand instructions, in table order. Startup is
included in these counter totals. They neither identify a particular cache
level nor prove that branches or memory are irrelevant.

The source explains why rules work is prominent. [Legal generation](../src/core.rs:403)
builds pseudo-legal moves, applies each to a full copied position, and tests
king safety. [Quiescence](../src/search.rs:404) generates all legal moves before
retaining captures/promotions outside check. [Move application](../src/core.rs:497)
already maintains an incremental Zobrist key; replacing full hash recomputation
is therefore not an available optimization. Effective EP identity performs
legal king-safety checks, which must survive changes to hashing or move testing.
Flat samples do not separate legality-testing calls from searched-move calls.

## GPU dispatch and host API traces

`rocprofv3 1.1.0 --runtime-trace --stats --output-format csv` records real
kernel, HIP API and allocation activity. The analysis identifies the search
worker by the fused completion kernels and excludes main-thread backend
construction and option changes. It reconstructs statistics from raw CSV,
rather than adding profiler domain percentages. Host waits overlap device
execution, so their durations must not be added to kernel durations.

| Instrumented search | Opening | Kiwipete | Rook ending |
| --- | ---: | ---: | ---: |
| Classified positions | 121,965 | 142,540 | 90,473 |
| Published advances | 136 | 108 | 171 |
| Device dispatch union | 947.91 ms | 964.97 ms | 949.24 ms |
| Backup/completion dispatch share | 78.46% | 79.29% | 77.59% |
| Selection dispatch share | 18.43% | 17.37% | 18.89% |
| Child rules/evaluation dispatch share | 2.42% | 2.81% | 2.44% |
| Worker stream synchronizations | 411 | 327 | 516 |
| Summed synchronization API duration | 950.27 ms | 968.38 ms | 952.25 ms |
| Search-worker allocations/frees | 0 | 0 | 0 |

The GPU's reported minimum frontier depth remains 2, while its maximum reaches
18. That is a mixed tree, not a uniformly searched depth-18 result. Plain
GPU response intervals were 1001–1009 ms; the deadline is checked between
completed advance/publication cycles and can overshoot by one such cycle.

[The profiled backup](../gpu/evidence/selective-fusion-gfx1030-2026-10-02/sources/fused/chen_selective.inc:96) used one device thread scanning
every retained old node backwards, then reducing the marked parents' children.
Completion duration closely follows arena size: Pearson correlations are
0.993–0.997. Mean completion time in the first versus last quartile rises from
1.35 to 9.98 ms in the opening, 2.00 to 12.11 ms in Kiwipete, and 1.09 to
7.54 ms in the ending. Correlation alone does not prove causation, but the
full-arena serial scan is directly visible in the pinned source.

Fusion provides one fence per advance; every published advance additionally
calls `finish`, whose snapshot and PV extraction add two fences. The observed
counts are `3 * advances + 3` including reset. Repeated publication is visible
overhead, but the profiled critical path was mostly the long serial kernels:
their execution overlaps the approximately 95–97% of worker time spent inside
stream synchronization. Removing waits without reducing that device work would
not remove the dominant cost.

## Ranked experiments from the frozen fusion profile

1. **Sparse GPU ancestor backup.** Record/deduplicate ancestors touched by the
   selected batch and process only those in descending allocation index, or
   use ordered ply waves with parallel sibling reductions. Preserve first-child
   ties, exact open/min/max metadata, complete-sibling admission and error
   publication. This targets the measured 78–79% dispatch share. Validate every
   step against the pinned serial implementation before alternating clean
   equal-work timings; an Amdahl upper bound is not a predicted speedup.
2. **CPU direct legal generation and capture/promotion generation.** Reduce
   candidate full-position `make_move` work using check/pin masks or a dedicated
   king-safety scratch path; replace ray walking behind a checked attack-table
   implementation. Quiescence can generate captures/promotions directly, but
   it still needs correct quiet-move existence for stalemate and complete
   evasions while in check. Preserve EP discovered checks, castling legality,
   promotion variants, clocks and repetition identity. The four main rules
   symbols account for roughly 77–80% of sampled worker cycles.
3. **GPU selection structure.** Cache or index eligible child choices so the
   single-thread root-to-leaf selector avoids repeated sibling scans. Preserve
   reservation semantics and the batch's pre-step values; replacing it with
   a global leaf priority changes the search and requires a separate strength
   experiment. The measured selection share is 17–19%.
4. **GPU publication cadence.** Reuse completed stats and extract PV less often
   while retaining final/cancellation/error publication and UCI responsiveness.
   This addresses 327–516 fences and many synchronous scalar copies per search,
   after the dominant serial work is reduced. Measure it independently of
   backup changes to keep the causes comparable.

Optimizing the cheap evaluator, tuning the CPU TT first, or micro-optimizing
GPU child legal generation has substantially less measured headroom in these
runs. Neural inference and training have a different cost balance and need
their own profiles.

## Reproduction

[profile_engine.py](../scripts/profile_engine.py) requires external binary
hashes, a JSON FEN suite, an unused output directory and an argv instrumentation
prefix. Prefixes contain `{stem}` for per-workload trace/perf filenames. The
frozen declarations and exact invocations are in the raw-record archive. No
shell interpolation is used for the prefix. Each UCI response and retirement
has a bounded deadline; a failure ends that invocation without replay.

[analyze_engine_profile.py](../scripts/analyze_engine_profile.py) independently
replays moves/PVs, filters worker samples/API records, checks phase counts and
sample loss, reconstructs unions and emits the summary. It consumes exported
flat perf streams and raw rocprof CSV; no GPU is required for analysis.

```sh
mkdir /tmp/chen-profile-replay
tar -xzf gpu/evidence/engine-profiling-gfx1030-2026-10-02/raw-records.tar.gz \
  -C /tmp/chen-profile-replay
PYTHONDONTWRITEBYTECODE=1 results/qa-venv/bin/python scripts/analyze_engine_profile.py \
  /tmp/chen-profile-replay --output /tmp/chen-profile-replay-summary.json
```

The retained `.perf.data` files allow further inspection with the pinned
executable and matching symbols. The portable exported leaf records are the
summary's actual sampling input. Trace instrumentation, a small suite and
uncontrolled shared resources limit generalization; no rating claim follows.

## Sparse-backup follow-up

The subsequent [classical sparse ancestor experiment](GPU_SELECTIVE_SEARCH.md#classical-sparse-ancestor-backup)
has [separate actual parity, fixed-work and trace evidence](../gpu/evidence/selective-sparse-ply-gfx1030-2026-10-02/summary.json).
It preserves the preceding profile as historical evidence. Equal-work six-position
median time fell 5.55% at 2,000 nodes/batch 8 and 48.66% at 50,000 nodes/batch 32.
Separate one-second traces moved completion from approximately 78% of dispatch
duration to 15–19%; repeated serial child selection now accounts for 71–75%.
Those traced arms complete different amounts of work, so their dispatch shares
locate the new bottleneck rather than measure matched-work kernel speedups.
