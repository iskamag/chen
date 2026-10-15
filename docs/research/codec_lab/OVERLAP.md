# Overlap, integer lifting, and exact occlusion boundaries

This experiment isolates a cross-boundary interaction: a slider in one block
makes occupancy in another block consequential, even when that second block has
no slider and produces no local attack. Conditional operator composition answers
that interaction exactly; independently computed local attacks and overlap
reduction do not. It is a small geometry laboratory: no engine, search, training,
or GPU is linked. Representation size is a secondary implementation measurement.

The protocol was committed as `f0c64a9` before execution; measured source was
committed as `dc450df` before execution. Protocol SHA256:
`ae26607be428ecc73eb05e39b8189d07833a818eaff37fbbf6cabae532a04276`.

## Connection to the codec mechanism

The [JPEG XR transform designers' paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/spie2008_hierarchical_lbts.pdf)
describes offset overlap/core supports, hierarchical composition, and integer
lifting. Its section 3.4 implements reversible steps as integer shears, with the
inverse reversing the steps. Those mechanisms motivate two separate controls:
local supports spanning hard boundaries, and a reversible signed re-encoding.

Our toy transform is **not** the JPEG XR POT/PCT or a conforming codec. It applies
two integer shears across each four-cell boundary, followed by reversible
two-point Haar/S lifting. For a pair `(a,b)`, the latter stores
`d=b-a; s=a+floor(d/2)`. Its inverse recovers `a=s-floor(d/2); b=d+a`.
All detail coefficients are retained in the exact variant. The loss ablation
zeros the detail coefficients before inversion. Empty, blocker and slider are
mapped to signed values `0,-1,+1`.

Neither ordinary overlap nor averaging inherits MDCT's deliberately constructed
alias-cancellation relationship. Our local visibility answers contain no designed
alias terms, synthesis window, or cancellation identity. Here overlap adds
support and redundancy. The lifting control preserves information; it does not
by itself simplify the nonlinear first-blocker query.

## Frozen question and independently expressed answers

A line has eight or sixteen cells in `{empty, opaque blocker, slider}`. Sliders
look in both directions, attacking empty cells through the first occupied cell,
including that occupied endpoint. They never attack their own square. Output is
the complete directed source-to-target relation: one target mask for every
possible source. Attack unions could conceal source-identity mistakes and are
not the validation contract.

The direct oracle walks outward independently from every actual slider. The
directional control instead sweeps targets left-to-right and right-to-left,
carrying the nearest occupied slider. All comparisons check every output mask.

We exhaust all `3^8=6,561` eight-cell lines. Sixteen-cell tests contain 20,000
frozen-seed ternary random lines and 376 constructed lines: every single slider,
every distinct slider/blocker pair, and slider pairs on an otherwise empty line.
The latter deliberately include long clear rays and all hard-edge placements;
their error frequency is not a natural chess distribution. Sixteen cells are a
boundary-composition stress domain, not a literal single chessboard ray.

## What boundary information suffices

Consider these two-block compositions, where `S` is a slider and `B` an opaque
blocker. In both cases the left block's independent source-zero answer is
`0x0e` (targets 1,2,3), and the right block's independent local answer is zero.

| Left block | Right block | Independent local OR | Exact composed source-zero answer |
|---|---|---:|---:|
| `S...` | `....` | `0x0e` | `0xfe` (targets 1 through 7) |
| `S...` | `.B..` | `0x0e` | `0x3e` (targets 1 through 5) |

The coupling is concrete. The left operator sends **an incoming source-zero
ray**, making the right block answer a different question from its own local
attack query. The empty right block passes the ray and emits all four targets.
The blocked right block emits only its first two targets and terminates it. No
new local slider was introduced, yet composition adds four or two attack bits
and changes whether the relationship can continue into another block.

For a source at zero and a target at six, the exact predicate is
`[cell0=slider] * product(j=1..5)[cellj=empty]`. The target itself may be occupied.
This is contextual gating across supports, not a sum of independent attack
strengths. A boundary entry therefore stores a **conditional operator**, not just
the block's answer in an empty external context. Redundant overlap can expose
more such interactions to a local operator; explicit transfer composes them
across arbitrarily many blocks without extending every window to the full line.

A four-cell lookup entry contains:

- The four internal source-to-target masks.
- The prefix/suffix target masks visible from immediately outside either side,
  through the first occupied cell inclusive.
- Whether the block is empty, and its first/last occupied slider identity when
  that occupied piece is a slider.

A directional incoming state is either inactive or an external slider's source
identity. An empty block passes that identity and emits its whole prefix/suffix
mask into that source row. A nonempty block emits only through its first blocker,
then replaces the outgoing state with its last occupied slider, or deactivates it
if that piece is a non-slider blocker. The opposite sweep is symmetric.

This is sufficient because every visible external ray is terminated exactly by
the first occupied cell. Only the last occupied cell can determine what leaves a
block in the direction of travel. Induction over blocks proves the sweep invariant
and the complete relation. Block composition is ordinary composition of these
state/output transducers; the implementation uses two sequential sweeps, not a
hierarchical parallel scan.

For sixteen-cell source-identified outputs there are seventeen carry values:
inactive or one of sixteen source identities. An attack union could use a single
active/inactive bit but would answer a weaker question. This is not a minimality
proof for every possible representation.

## Actual ablations

`hard4_local` ORs four-cell local answers at starts 0,4,8,12. The staggered variant
adds starts 2,6,10, so its four-cell windows have stride two. The OR reconstruction
keeps every locally established relation. The mean reconstruction averages each
source-row/target vote over all windows covering that target, then requires a
strict majority. Windows lacking that source contribute zero. This precisely
specified averaging rule is an ablation, not a proposed optimal estimator.

| Variant | Wrong complete relations / 6,561 exhaustive8 | / 20,000 random16 | / 376 constructed16 |
|---|---:|---:|---:|
| Direct source oracle | 0 | 0 | 0 |
| Full directional band | 0 | 0 | 0 |
| Full eight-cell LUT | 0 | N/A | N/A |
| Hard four-cell local | 4,880 | 19,558 | 364 |
| Staggered local OR | 992 | 7,813 | 356 |
| Staggered mean/majority | 5,936 | 19,910 | 372 |
| Four-cell boundary composition | 0 | 0 | 0 |
| Exact lapped lifting, inverse, boundary composition | 0 | 0 | 0 |
| Detail-discarding lifting, inverse, boundary composition | 6,122 | 19,915 | 365 |

Staggered OR fixes every distance-one relation crossing a hard four-cell edge:
the hard variant misses 4,374 such relation bits in exhaustive8; OR misses zero.
However, OR still misses 852 relation bits at distance at least four. No four-cell
window contains both ends of those relations. Mean/majority removes valid votes
as well; local answers are not signed reconstruction residuals. Both local OR
variants have false negatives only. The detail-discarding transform also creates
false positives. All exact lifting round trips pass.

Concrete witnesses make the limits clearer:

- For `S.......`, the true source-zero mask is `0xfe` (targets 1 through 7).
  Hard local and staggered OR return `0x0e`, majority returns `0x02`, and the
  detail-discarding reconstruction returns zero. The retained-detail path is
  exact. The single slider and the completely empty line have identical retained
  low-pass coefficients in this toy transform, so no decoder of those coefficients
  alone can answer both correctly.
- `S...B...` and `S....B..` produce identical **per-window local visibility
  answers** for every stride-two window: only the first window contains a slider,
  and both blockers are outside it. Their correct source-zero masks differ:
  `0x1e` versus `0x3e`. No sum, average, or other decoder of those local answers
  alone can recover the missing distinction. Raw overlapping input blocks do
  distinguish the positions; the limitation concerns the local answer interface,
  not information loss in a complete redundant tiling of the raw line.

## Storage and whole query classes

The operational four-cell LUT has 81 entries of nine packed bytes: **729 bytes**.
Its entries yield 74 distinct current-position boundary operators. This modest
equivalence does not preserve arbitrary future edits: the class experiment keeps
the original ternary codes when calculating replacements.

The full8 LUT has 6,561 entries. Its physical `[u16;16]` rows occupy 209,952 bytes;
a specialized eight-cell packed format would need 52,488 bytes. The analogous
full16 table would have 43,046,721 entries and need 1,377,495,072 packed bytes. It
was not allocated. LUT construction is outside query timings; code extraction,
composition and any lifting/inverse are inside.

For each of 632 roots (all 376 constructed lines plus the first 256 random lines),
we request all three replacements at all sixteen cells, including no-ops: 48
complete relations. Both rebuilding and shared-descriptor variants match the
direct oracle on all **30,336 queries**, with zero wrong whole query classes.

The shared variant extracts four original codes once, obtains the original four
descriptors, then changes one lookup code per query using a ternary digit delta.
It performs 52 local table lookups rather than 192, and four initial code
extractions rather than 192. It still performs 48 complete boundary compositions.
It is shared local work over an exact query set, not a symbolic answer to all
possible positions in one operation.

## Exploratory matched CPU costs

Portable `target-cpu=x86-64`, Rust 1.99.0/LLVM 23.1.1, release opt-level 3/thin LTO,
Ryzen 5 3600, CPU4 affinity. Seven repetitions; medians below include output
materialization protected by `black_box`. No explicit SIMD kernel is implemented
and no SIMD speedup is claimed. Per-relation timing uses 6,561 exhaustive8 inputs
or the first 4,096 random16 inputs; query-class timing uses 256 random16 roots.

| Complete relation | Eight cells, ns | Sixteen cells, ns |
|---|---:|---:|
| Direct source oracle | 40.6 | 92.1 |
| Directional band | 60.4 | 158.3 |
| Full8 LUT | 5.9 | N/A |
| Boundary4 LUT | 15.2 | 39.3 |
| Lifting + inverse + boundary4 | 37.1 | 66.8 |

For a complete 48-query class: independent direct oracles take 2,864ns; rebuilding
boundary composition takes 798ns; shared descriptor edits take 786ns. Most of the
gain here comes from the small exact boundary operator, while the extra sharing
only saves about 1.6% in this run. Different call sites and within-root correlation
also affect compiler optimization and branch prediction. These are narrow CPU
kernel measurements, not engine throughput or chess strength.

The useful result is the exact conditional interaction between an external ray
and another block's stopping/transmission geometry. Its small lookup is merely
one implementation. Ordinary overlap fixes some edge interactions but does not
replace cross-block occlusion composition.
The reversible signed transform is faithful yet adds work when decoded before
the same query. The current class mechanism shares code/lookup work but barely
improves an already cheap exact query. A next distinct experiment could reuse
prefix/suffix transducers across all edits; that is not implemented here.

## Reproduce

```sh
python3 experiments/codec_lab/overlap/run.py --out /tmp/overlap-rerun \
  --verify experiments/codec_lab/overlap/results/correctness.json
```

`--out` must be new. The runner pins CPU4 and portable target flags, builds offline
with the committed lockfile, compares every deterministic result, and writes
separate timing and provenance files. Two local runs produced the same
correctness SHA256:
`69e3e33ba71d58863d5f12d223de2ad8c524161e3998714bdbba10d96d6f7bb8`.
Frozen compact outputs are under `experiments/codec_lab/overlap/results/`.
