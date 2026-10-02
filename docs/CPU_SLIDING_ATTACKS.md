# Controlled CPU SWAR sliding attacks

The Rust candidate improves the six-position fixed-node search workload by
**1.167×** in this study. The aggregate perft improvement is smaller, **1.013×**,
and the middlegame perft case regresses. Root review adopts SWAR for the default CPU core, retaining the ray control
with `cargo build --features ray-attacks` for comparable experiments. The GPU result
is separately documented in
[RDNA_BITBOARDS.md](RDNA_BITBOARDS.md).

[src/attacks.rs](../src/attacks.rs) provides occupancy-only rook and bishop
attacks using three Kogge–Stone propagation stages. It masks file wrapping,
includes the first occupied square, and excludes the origin. It owns no rules,
evaluation, history, search state, or lookup allocation. The controlled candidate
copy redirects the private core `sliding_attacks` function to these helpers and
exports the module only in that copy. Every other source file is unchanged.

## Verification and build bindings

Both frozen copies passed **56 release tests**. Each example executable passed
**4,878,976 checks** comparing the actual helper and its core wrapper against an
independent square-walking oracle. Coverage enumerates all relevant rook/bishop
occupancy subsets at every square, both values of the origin occupancy bit, and
100,000 additional random occupancies in both directions. An independent
python-chess oracle checked **256 roots and 6,975 child legal-move count maps per
arm**, with both movers, castling, promotions, en passant, pinned en passant,
mate and stalemate. These measured candidate copies did not edit production or neural dependency
files. Subsequent root adoption changes the CPU dispatch and adds the compile
feature; historical native-corpus baseline witnesses retain their archived
source/binary identities.

The experiment sources are frozen under
[cpu-sliding-swar-ryzen3600-2026-10-02](../gpu/evidence/cpu-sliding-swar-ryzen3600-2026-10-02/).
Builds use the same release profile, compiler, and
`RUSTFLAGS='-C debuginfo=1 -C force-frame-pointers=yes'` in both arms, on a Ryzen
5 3600. These flags support meaningful frame-pointer sampling; the timing claim
is about these declared builds. Binary gzip payloads preserve the original
executable bytes and SHA-256 hashes.

The workload example received a numeric-literal grouping correction after
measurement to satisfy Clippy. The seed value and algorithm are identical;
the original measured source copies and initial Clippy failure remain archived.
The current example passes Clippy with warnings denied. The helper source itself
is byte-identical to the measured candidate.

## Measurements

[cpu_sliding_probe.py](../scripts/cpu_sliding_probe.py) pins both executables,
the suite and perf tool. It warms both arms, then runs balanced forward/reverse
orders with four samples per arm. Every worker is pinned to logical CPU 2.
Other agents completed their CPU timing tasks before this reservation; shared
cache, frequency and power remain uncontrolled. All owned child processes
retired normally, and both perf captures report zero lost samples.

Perft uses start depth five and depth four for the other five positions.
Fixed-node search uses a fresh default searcher/TT, Tapered evaluation, depth
cap 64 and a 200,000-node cap. All non-timing result fields match between arms,
including perft counts, search scores, PVs, nodes and quiescence nodes.

| Six-position fixed work | Ray walk median (ms) | SWAR median (ms) | Ratio |
| --- | ---: | ---: | ---: |
| Perft | 542.030 | 535.219 | 1.013× |
| 200,000-node search | 2,018.104 | 1,729.688 | 1.167× |

| Position | Fixed-node search ratio | One-second median node ratio |
| --- | ---: | ---: |
| Start | 1.024× | 1.015× |
| Kiwipete | 1.189× | 1.176× |
| Rook ending | 1.309× | 1.298× |
| Middlegame | 1.007× | 0.989× |
| Promotion | 1.447× | 1.430× |
| Forced mate | 1.355× | 1.000× |

One-second searches are additional throughput telemetry, with two samples per
arm. They can finish different iterations, with different scores or PVs, and
are not compared as equivalent work. The forced-mate case finishes early with
29 nodes in each arm, so its tiny latency ratio has little practical meaning.
Middlegame perft is 4.9% slower in the candidate; keep this visible when reviewing
adoption rather than reducing the decision to a single aggregate number.

Separate one-second sampling profiles use `cpu-clock` at 997 Hz and frame-pointer
call chains. Ray walking accounts for 25.09% of baseline self samples; the new
core sliding wrapper accounts for 20.31% of candidate samples. `make_move`
accounts for 29.72% and 31.52%, respectively. These percentages come from
different timed searches and are diagnostic attribution, not equal-work cycle
counts. Hash/EP recomputation and repeated legality checks remain worthwhile
next profiling targets.

The [cold stdlib audit](../gpu/evidence/cpu-sliding-swar-ryzen3600-2026-10-02/audit-report.json)
checks frozen source hashes, all **96 fixed-work rows**, the independent move
oracle, process retirement, and saved sampling reports. This study establishes
correctness and measured performance on its workloads; it makes no rating or
strength claim. Root owns production adoption and commits.

Root adoption passes all 56 release tests with both the default SWAR backend and
`--features ray-attacks`. The feature changes CPU attacks only; GPU experiments
retain their independently pinned libraries.
