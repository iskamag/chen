# Round 1: block composition, topology bands, and query-specific equivalence

This branch tests three concrete factorizations of the proposed JPEG-like
representation. They are exact geometry/constraint kernels, **not playing
policies**. Mate recognition, move-choice quality and Elo are N/A. An attack
union or an evasion-destination mask does not prove checkmate.

The principal result is negative for full attack maps: the strongest ordinary
whole-board control is much faster. A narrower positive result is that a cached,
progressively refined king-constraint query beats an ordinary target-ray query.
The equivalence measurements explain why: preserving all geometric answers
admits much less merging than preserving one particular obligation.

## Implemented mechanisms

* **Microblocks:** all `13^4 = 28,561` raw 2x2 piece arrangements map to a
  packed 92-bit descriptor: two 36-bit expanded leaper stencils, colored
  orthogonal/diagonal slider emitters and occupancy. The attack query consumes
  the leaper projection; slider channels retain their own precise source and
  blocker projections. This is operator factoring, not a complete position
  dictionary or a new legal-move generator.
* **Bands:** two-row, eight-file source bands use a factored 16 KiB table for
  pawn, knight and king consequences. Source-rank masks index byte tables and
  their outputs compose through OR. These source pairs are non-overlapping.
  Ranks, files and both diagonal families redundantly overlap at board squares,
  but a separately maintained staggered 2x8/8x2 band store is **not implemented**.
* **Pyramid:** four 2x2 microdescriptors compose into each 4x4 macroblock's
  exact expanded leaper consequence. Four projected macroblock summaries are
  retained. A sparse edit updates only ancestors whose consumed semantic
  projection changed; slider effects continue through their separate channels.
  This is a real cached 2x2 -> 4x4 composition, not a renaming of the band query.
* **Boundary transfer:** all 46 H/V/diagonal lines compose two four-cell
  finite-state transducers. State is an incoming active ray; occupancy kills
  propagation and matching slider emitters start propagation. Each direction
  uses a 512-entry table of `i32`. A separate first-two-blockers projection
  composes direct checker, absolute-pin and capture/interposition obligations
  across the four lines incident to each king.
* **Progressive constraints:** if no matching enemy slider emitter shares any
  king line, the query proves that slider check/pin decoding is unnecessary.
  Otherwise it refines through first-two-blocker summaries. Leaper source
  identity is retained independently, because an aggregate attack map discards
  information needed to identify a checker.
* **Explicit SIMD:** AVX2 executes eight independent line transducers at once,
  using four `vpgatherdd` operations plus vector shifts, ANDs and ORs. A scalar
  path uses the same `i32` tables. Runtime feature checking guards AVX2. This is
  vectorization across topology lines, not the earlier root-alternative lanes.

The current encoded state is 392 bytes; the cached pyramid state is 456 bytes.
The total payload of **all** tables instantiated by this experiment is 583,952
bytes (570.3 KiB), including the 4 KiB conventional control ray table. The band
path's band/transducer/expansion/first-two payload is 122,880 bytes (120 KiB).
The micro dictionary alone occupies 456,976 bytes (446.3 KiB). Geometry vectors
and allocator metadata are additional. This is deliberately redundant
computation storage, not smaller serialization than bitboards.

## Exactness and interpretation

Release tests exhaust all 65,536 valid eight-cell occupancy/colored-emitter line
states against an independent ray walker, including the explicit AVX2 path.
They also exhaust every microstate at every 2x2 board location: 456,976 cases,
including edge clipping and all leaper types/colors.

On the frozen DEV corpus, 727 roots and **all 16,721 legal child transitions**
passed independent in-crate full attack-mask and king-obligation checks.
Sparse patches reproduce rebuilt line codes, micro codes, source planes and
king coordinates. Transition edits include 5 three-square EP updates and 28
four-square castling updates; promotions and captures are included. The
ordinary Chen rules supply legal moves and apply transitions. This branch does
not independently claim complete move legality, castling transit legality,
en-passant evasion sufficiency, repetition or rule-clock adjudication.

The shared python-chess checker independently passed all six variants on DEV
(558,336 individual square/color attack predicates and 17,448 additional
obligation masks). The parent also independently certified the frozen kernels
on reserved CERT: 659 roots, six variants, 506,112 attack predicates and 15,816
additional obligation masks. The subsequent harness-only change added timing
warmup/minimum duration; the geometric kernels did not change. The parent owns
the final replay of the final executable hashes.

For each color, `evasion` is all 64 squares when not checked, the checker square
plus interposition squares for a single slider check, or the checker square for
a single leaper check. Checker constraints intersect, producing zero for double
check on legal roots. This is an obligation on ordinary **non-king destination
squares**, not a complete evasion predicate; EP and king moves need their full
transition semantics.

## Measurements

DEV medians from three alternating portable/native pairs on CPU 4, Ryzen 5
3600. Both builds use thin LTO and one codegen unit. Native additionally uses
`-C target-cpu=znver2`. Every workload warms up for at least 3 ms and is timed
for at least 15 ms. `--iters` is a lower bound on sweeps. The initial short,
unwarmed probes showed startup frequency drift and are not the table below.

The ordinary attack control propagates **whole sets of slider sources** with
Kogge–Stone bit operations and computes all leaper attacks with word operations.
The ordinary constraint control uses eight precomputed king rays and nearest
two blockers via bit scans. It does not loop over 128 target predicates or
construct temporary boards. The complete packet has exactly the same output
on both sides: both full attack maps and both kings' checkers/pins/evasion masks.

| Native variant | Attack query | Build + attack query | Complete packet query |
|---|---:|---:|---:|
| Ordinary control | 35.7 ns | 36.2 ns | 82.5 ns |
| 2x2 micro, scalar | 572.6 ns | 1282.8 ns | 617.7 ns |
| 2x8 bands, scalar | 305.0 ns | 990.9 ns | 324.4 ns |
| 4x4 pyramid, scalar | 279.9 ns | 1145.2 ns | 316.5 ns |
| 2x2 micro, explicit AVX2 | 621.4 ns | 1303.2 ns | 664.0 ns |
| 2x8 bands, explicit AVX2 | 322.4 ns | 1010.4 ns | 351.6 ns |
| 4x4 pyramid, explicit AVX2 | 305.3 ns | 1161.5 ns | 346.4 ns |

Portable attack-query medians were 36.7 / 609.4 / 332.9 / 288.0 ns for control,
scalar micro, scalar bands and scalar pyramid respectively. Native targeting
helps some paths but does not reverse the architecture ranking. Static
disassembly of the actual measured binaries confirms four `vpgatherdd` sites
in the hot explicit SIMD function in both builds. The scalar transducer has no
vector instructions in either build. This verifies the intended implementation;
it is not a sampled dynamic-instruction or cache-miss profile.

For the **king-constraints-only** workload, native ordinary target rays take
42.2 ns for both kings, full first-two decoding takes 73.7 ns, and progressive
projection takes **26.0 ns**. Portable values are 46.2 / 86.5 / 29.7 ns. DEV
requires refinement in 240 of 1,454 king queries (16.5%); 113 queries have a
nontrivial slider check or pin constraint. This is a cached-query win, not a
whole-decision win. Copy + exact sparse patch takes 78.3 ns on this workload,
before any query; roughly five repeated same-state constraint queries would
be needed to amortize that cost against this control. Diff construction from
the ordinary board is excluded from patch timing and the raw reports say so.

Across 16,721 DEV legal transitions, the pyramid refreshes zero macroblocks on
7,971 transitions (47.7%), one on 6,759 and two on 1,991: **0.642 macroblocks per
transition**. These counts describe leaper consequences only; ray state still
needs its sparse updates. The measured copy/patch/pyramid-attack operation
takes 473.2 ns versus 33.8 ns for the ordinary attack query on the corresponding
already-made child. Caching saves microblock decoding but loses overall here.

## What compressed, and what did not

These exhaustive equivalence classes are of the **named operator answers**,
not of chess positions, plans, minimax values or continuations:

| Input domain | Distinct projected answers |
|---|---:|
| 28,561 raw 2x2 microstates, combined leapers + ray emitters + occupancy | 28,507 |
| Same microstates, ray/occupancy projection only | 4,096 |
| Same microstates, leaper-union projection only | 1,999 |
| 65,536 valid whole-line source/blocker states, both attack unions | 6,921 |
| Fixed king + seven five-symbol cells (78,125 cases), joint line check/pin/evasion | 14, 24, 40, 48, 48, 40, 24, 14 by king index |

Almost every combined microstate stays distinct: a general-purpose dictionary
merges almost nothing. Selecting the operator first produces large equivalence
classes. In particular, a fixed-king line obligation fits into at most 48 answer
classes, although the seven remaining cells have 78,125 classified inputs.
The finite boundary interface can discard everything behind an already
decisive blocker. This is stronger evidence for **query-conditioned reasoning
objects** than for a universal compressed board ID.

This experiment does not demonstrate merged adversarial continuations, a
searchless policy, a tactical-strength improvement, or a hardware-independent
speedup. Its positive finding concerns a precise constraint query. Most time
in the broader paths is spent extracting/expanding outputs that ordinary word
operations produce very cheaply. The AVX2 gather design adds decoding costs
and is slower on this CPU; simply widening the same work is insufficient.

## Next architectural branches

1. **Target-conditioned obligation codes.** Retain the tiny exact answer class
   for each king line and compose capture/interposition/refutation constraints
   directly. Update these codes only when a move can alter the first relevant
   boundary message. Avoid ever producing full attack maps unless an unresolved
   king move or tactical query demands them. Join these obligations with the
   reply/certificate branch; equivalence must preserve the exact named query,
   and unresolved cases must refine rather than guess.
2. **Actual staggered H/V bands with query-selected boundaries.** Maintain both
   phases of overlapping 2x8 and 8x2 bands, and choose the phase containing a
   target king zone. Compare retained boundary metadata/update fan-out against
   this fixed source-pair factorization. Current results do not test that idea.
3. **SIMD over demanded constraints, not full-output line expansion.** Batch
   independent target-obligation queries in a structure-of-arrays layout and
   use small factor tables or Boolean planes to avoid random gather + scalar
   expansion. Compare useful answers per cache line and complete certified
   tactical outcomes, not just a wider variant of this losing attack kernel.

## Reproduction and evidence

The standalone crate is `experiments/radical_round1/bands`. It changes no Chen
production source. Build and execute with:

```sh
cargo test --release --manifest-path experiments/radical_round1/bands/Cargo.toml
cargo build --release --manifest-path experiments/radical_round1/bands/Cargo.toml
taskset -c 4 experiments/radical_round1/bands/target/release/chen-radical-bands \
  --corpus results/radical-round1/corpus/dev.tsv --out /tmp/bands.jsonl --iters 100
```

`--all-only` times aggregate workloads, while retaining all semantic checks.
`--answers-only` writes per-root/per-variant full mask answers to stdout for
independent replay. `run_matrix.py` alternates two frozen binaries and retains
raw runs; `inspect_isa.py` records actual hot-function instructions and binary
hashes. `evidence-summary.json` is the compact tracked DEV evidence. Raw evidence
is under ignored `results/radical-round1/bands/`, especially `matrix/`,
`dev-portable-warmed-families.jsonl`, `portable-isa.json`, `native-isa.json`,
`dev-independent.json` and the parent-created `cert-independent.json`.

Final portable binary SHA256:
`c30fd87951568a3d83008f503062a3ba5842153654f1226ce0859d8592d7e238`.
Final native binary SHA256:
`7a5a835c9f2990275ef785cfffa4dbf08a04d65f256a4d4ac91f3ba61b5f08ed`.
