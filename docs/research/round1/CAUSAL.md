# Round 1: persistent causal channels

This branch studies exact attack consequences and demand-driven king-attack
predicates. Mate recognition, move selection and playing strength are **N/A**.
It does not claim a complete JPEG engine or a new chess search algorithm.

## Hypotheses and alternatives

The proposal's useful causal claim is testable: occupancy events should touch
only channels that can observe them; typed piece events should affect only their
source contribution. A channel whose observable summary stays equal should not
propagate. Fewer affected sources need not mean less wall time on a modern CPU.

Four persistent architectures are implemented independently of Chen's orthodox
attacks and move generator:

- **Dirty line channels:** 46 overlapping rank/file/diagonal channels retain
  eight-bit occupancy codes and two-color attack-union contributions. Typed
  source changes and occupancy edits invalidate appropriate lines. Leaper
  contributions are separate. Per-target contribution counts preserve other
  attackers when one channel stops contributing.
- **Source counts:** each source retains its contribution; target reference
  counts preserve union deletion exactly. Coarse invalidation touches every
  matching slider on an incident dirty line. Fine invalidation inspects only
  pre/post first blockers. Indexed invalidation uses compressed old occupancy
  and slider-source codes to look up observing sources directly.
- **Source OR tree:** the same sparse source edits update a persistent binary
  OR-summary tree. Propagation stops at the first unchanged internal summary.
- **Deferred dense reduction:** source-leaf edits remain sparse, but unions
  are reduced densely on demand. This is an intentional hardware ablation:
  dense vectorizable work can beat more selective branch-dependent propagation.

A separate **proof-support predicate cache** retains king-attack answers. A true
predicate depends only on one witness and its unobstructed ray; a false predicate
retains every inspected leaper exclusion and first-blocker boundary. Unrelated
symbol edits leave the proof valid. This implements a small, exact observable
obligation, without pretending that attack predicates determine tactical value.

## Exact event interface and stopping rule

The mechanism receives up to four `(square, old symbol, new symbol)` edits.
Castling, en passant and promotion therefore use the same event interface.
Occupancy changes and typed source changes are distinct events. Updates mutate
persistent channel state directly; the finalized source invalidator does not
copy a full board or construct a successor `Position`. Inverse packets restore
state through the same update mechanism.

The indexed invalidator needs only **old** visible sources. If an unchanged
slider's contribution changes, consider the nearest changed occupancy square
along that ray. Every earlier square is unchanged, so that changed square was
inside the slider's old visible interval. Querying visibility from the edited
square therefore finds the observing slider. Every edited source is dirty
regardless of occupancy. This also covers multiple blockers changing at once.
A regression removes two aligned blockers in one four-edit packet.

The 8 x 256 visibility LUT and line-to-board expansion LUT total about 98 KiB
with geometry. These are exact geometry tables, not tactical tables. The large
expansion table is a cache-cost concern; it is not L1-resident as a whole on
this machine. Sparse event updates allocate no heap storage.

## Work and control boundaries

There are three separate cost modes:

1. Fresh ingress starts from a Chen `Position`, includes symbol encoding and
   persistent-state construction, then produces both complete attack unions.
   The strong scalar control computes the same unions from piece bitboards and
   SWAR slider fills.
2. Sibling events receive precomputed packets and cached parent channel state;
   they apply a child and its inverse, retaining the child output. Cache clones
   are included once per root, parent-state construction is excluded. The scalar
   control receives precomputed child `Position` values and computes each child's
   full unions once. This intentionally gives the scalar control its strongest
   state-ingress assumption; timings are not isolated equal-instruction kernels.
3. Trajectories apply one event per step without inverses, amortizing root
   initialization across an episode. Both controls receive precomputed ingress
   (packets versus child states). Packet construction from two authoritative
   states is measured separately; it is never hidden inside the reported event
   cost.

Full-union variants produce both 64-bit color masks and are checked individually
against all 128 Boolean attack questions, outside timing. Timed consumption uses
`black_box` on the two-mask result. King-only variants answer both kings' attack
predicates; they perform a smaller obligation and are compared to each other.
All legal children make own-king safety zero, so the opponent-king predicate is
also consumed to avoid a constant-zero benchmark. Invalid pseudo-legal children
are additionally verified to cover genuine self-check refutations.

The timed intervals exclude FEN parsing, legal move generation, child creation,
packet extraction and oracle checking. Fresh ingress includes encoding; the
separate ingress-packet record counts board encoding and diffing explicitly.
The geometry tables are initialized before timing. The allocator counter reports
actual allocation calls in each sibling/trajectory interval. Local timings use
CPU 1 affinity on a shared Ryzen 5 3600; they do not establish GPU behavior.

## Results

The final DEV workload has 727 roots, 16,721 legal sibling events and 3,210
shared trajectory events. Four alternating portable/`znver2` process pairs use
20 repetitions per workload, CPU 1, reverse method order on odd rounds, and no
builds between measurements. These are medians in nanoseconds; event counters
remain compiled into the measured kernels. Sibling events include inverse
updates; the scalar control receives a child state for free as explained above.

| Complete two-color attack unions | Portable sibling roundtrip | Portable trajectory | znver2 sibling roundtrip | znver2 trajectory |
| --- | ---: | ---: | ---: | ---: |
| Strong scalar SWAR control | 61.8 | 63.3 | 56.2 | 58.7 |
| Full source-cache rebuild | 1,095.7 | 1,240.7 | 1,135.7 | 1,298.0 |
| Coarse source counts | 955.6 | 620.6 | 975.8 | 641.2 |
| Fine pre/post source counts | 668.0 | 416.4 | 649.5 | 402.3 |
| Indexed source counts | 283.1 | 161.0 | 267.2 | 153.7 |
| Indexed OR tree | 246.6 | 141.9 | 230.1 | 130.6 |
| Indexed deferred dense reduction | **181.1** | **120.4** | **187.1** | **113.4** |
| Dirty-line count cache | 559.1 | 336.5 | 560.7 | 339.7 |

Indexed invalidation is 2.36x faster than pre/post ray discovery in sibling
source-count updates and 2.59x faster in trajectories. Fine and indexed methods
recompute the same 2.83 sources/event on average, versus 3.80 coarse siblings
and 3.90 coarse trajectory sources. About 2.70 source observables actually change.
The gain comes from finding the dependencies more cheaply, not pruning a different
set of sources. Source counts must touch about 17.4 changed target bits per
sibling and 14.7 per trajectory step. OR-tree events visit about 15.7 internal
nodes and write about 18 summaries/event; fewer observable changes do not make
those dependency chains free.

Deferred dense reduction beats eager tree propagation despite reducing 128 leaf
words for the two-color union. Disassembly of the separately preserved
`dense_union` function confirms SSE2 `por` on XMM registers in the portable
binary and AVX2 `vpor` on YMM registers in the `znver2` binary. This is actual
compiler-generated SIMD, not a claim that widening `u64` improves dependency
control. Whole-CPU targeting improves the deferred trajectory median by 5.8%,
but its sibling median worsens by 3.3%; the scalar control also improves. No
isolated AVX2 speedup is established. The best native trajectory variant remains
1.93x slower than its strong scalar control, and its sibling roundtrip is 3.33x
slower under the stated asymmetric ingress assumptions.

| Both king-attack predicates | Portable siblings | Portable trajectory | znver2 siblings | znver2 trajectory |
| --- | ---: | ---: | ---: | ---: |
| Direct demand ray queries | 118.5 | 126.7 | 122.2 | 126.9 |
| Proof-support lazy cache | 110.9 | 113.5 | 118.0 | 117.6 |
| Orthodox demand control | 34.8 | 34.7 | 30.9 | 32.7 |

Only 58.6% of sibling predicate queries and 47.4% of trajectory predicate
queries need recomputation with the lazy cache. This reuse gives a modest gain
against the direct ray walker and still loses heavily to efficient orthodox
target queries. The false-negative interpretation would be "dependency caching
is pointless"; the supported finding is that this *cheap obligation* does not
pay for the present proof construction and maintenance costs.

Fresh ingress is particularly unfavorable: portable complete union computation
is 52.6 ns scalar, 1,561.9 ns source-count construction, 1,567.7 ns line-count
construction, and 1,684.9 ns source-tree construction. Event packets extracted by
encoding/diffing two authoritative states cost a further 65.2 ns/sibling pair or
84.9 ns/trajectory pair in the portable runs; event timings do not include this.
State footprints are 816 bytes source counts, 1,504 bytes dirty lines, 2,744
bytes OR tree, 1,704 bytes deferred reduction, and 112 bytes lazy predicates.
Fresh constructors initialize those state arrays; persistent event updates do
not reinitialize them. Every measured sibling/trajectory interval reports zero
heap allocation calls through the instrumented global allocator.

Validation checks each complete attack mask, never only a checksum. DEV internal
checks cover 39,417 forward transitions (legal siblings, pseudo-legal self-check
refutations and 4,062 generated trajectory steps), totaling 5,045,376 Boolean
attack questions, plus 3,210 shared trajectory events / 410,880 questions. Source
count, tree, flat and line variants are compared individually on legal siblings;
all are compared individually on every shared trajectory event. Lazy predicate
answers are checked individually on generated/shared trajectories; their sibling
timing checksum is aggregate-only. Four release tests also cover overlapping
attacker deletion, simultaneous blocker removal, witness invalidation and special
rule/trajectory differentials.

Independent python-chess 1.11.2 checks all 17,448 DEV root/child boards, or
2,233,344 Boolean answers, exactly. The emitted masks assert source/count/tree/line
agreement before external checking. The parent subsequently audited reserved CERT
roots and every legal child independently: 15,493 boards / 1,983,104 Boolean
answers; the parent's internal CERT run also passed 35,722 forward transitions
and 3,036 shared trajectory events. CERT was not used to select this branch's
variants. These checks establish exact attack consequences, not move-selection
quality, a mate solver, or a strength result.

[Compact DEV evidence](../../../experiments/radical_round1/causal/evidence/dev-summary.json)
binds source, corpus and executable hashes, all aggregate timing replicates,
allocation counts and event counters. Large per-family raw outputs and separate
SIMD disassemblies remain under `results/radical-round1/causal-final/` in the
primary checkout. The final source SHA-256 is
`e67693c29966a12d91abc79071ebb5febc147078bbd66ed4e1ffd22fff47f443`.
Portable executable: `524dd7521570a592a13e709c5e082fc2c21d24634f523c4ba12bb5a18208ed73`;
znver2 executable: `52957677020be848777f455369ef2ccac9dcb098a34ed09bbbdb887fb1eb7a1f`.

The causal hypothesis is partly supported: exact observing-source events and
unchanged-summary stopping substantially reduce the original update costs. Its
performance extension is rejected for these obligations: even the strongest
implemented persistent variant loses to full recomputation. This is a bounded
negative result on attack unions and king-attack predicates, not a rejection of
compressed adversarial reasoning.

## Proposal coverage and limits

Implemented: first-class exact movement-channel codes; typed semantic/occupancy
packets; persistent contributions; sparse reversible updates; old-visible-source
LUT transitions; unchanged-summary propagation stopping; independently checkable
Boolean obligations; proof-support demand caching; a dense SIMD reduction
ablation. No orthodox attack or move generation call appears inside these
mechanisms; orthodox code is confined to ingress, validation and named controls.

Missing: interlaced macroblocks, lossy/learned descriptors, continuation-class
merging, opponent-reply quantification, nontrivial tactical descriptors, and
complete move selection. The source-summary heap groups square indices, not
semantic macroblocks. A short-horizon attack predicate is not a mate proof.
No Elo or tactical-solving score applies to this primitive-only branch.

## Falsifiable next branches

- Replace the roughly 98 KiB line expansion tables with compact per-family
  deposits or BMI2/AVX2 channel decoders. Require lower total event cost on both
  sparse and promoted-piece stress families, not just smaller lookup tables.
- Co-design inverse packets with an undo log of changed summaries. Compare
  recomputing inverse consequences with restoring recorded summaries, including
  bytes written and episode memory, while retaining exact overlap deletion.
- Maintain proof-support obligations for a *set* of alternative replies,
  deduplicating equal dependencies; use exact refutation certificates as output.
  Attack-mask maintenance alone is already too cheap a scalar workload to
  justify expensive causal bookkeeping.
- Combine sparse edits with vector batches across independent trajectories.
  Compare useful per-core throughput against dense SWAR batches and the packed
  alternative architecture, with ingress/output-layout costs included.


## Reproduction

From the branch checkout:

```sh
cargo test --release --manifest-path experiments/radical_round1/causal/Cargo.toml
cargo clippy --release --manifest-path experiments/radical_round1/causal/Cargo.toml -- -D warnings
cargo build --release --manifest-path experiments/radical_round1/causal/Cargo.toml
RUSTFLAGS='-Ctarget-cpu=znver2' cargo build --release --manifest-path experiments/radical_round1/causal/Cargo.toml --target-dir experiments/radical_round1/causal/target-znver2
```

The executable takes `--corpus roots.tsv --transitions transitions.tsv` and one
of `--verify-only`, `--answers-only`, or `--bench-only`. Add `--out file.jsonl`,
`--reps 20`, `--reverse-methods`, or `--method flat_indexed` as needed. Answers
are full root/child masks with FENs, suitable for `verify_answers.py`. DEV final
pairs used:

```sh
python3 experiments/radical_round1/causal/benchmark.py \
  --corpus /home/iskam/src/vibe/chen/results/radical-round1/corpus/dev.tsv \
  --transitions /home/iskam/src/vibe/chen/results/radical-round1/corpus/dev-transitions.tsv \
  --out /home/iskam/src/vibe/chen/results/radical-round1/causal-final \
  --cpu 1 --reps 20 --rounds 4
```

The committed source and compact summaries contain no GPU execution requirement.
