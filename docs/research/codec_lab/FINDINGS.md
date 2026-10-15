# What the interaction experiments show

Follow-up: [latent depth without a move tree](LATENT_DEPTH.md) tests repeated
algebraic closure, synchronous constraint messages, and learned recurrence on
the same fixed input.

This round studies how components combine. It does not optimize compressed
storage or build a chess bot. Three agents work in isolated worktrees; the
parent supplies an actual MDCT control and independent checks.

## Alignment changes the relation

The phase lane computes all relative-offset correlations of two fields directly
from their cross-spectrum. Moving a knight source from a1 to b1 while keeping a
target on c2 preserves the separate Fourier magnitudes but changes whether the
knight attacks it. Relative phase carries the alignment.

Full signed/complex composition matches all 225 requested outputs on all 4,288
cases. The experiment performs the interaction in the transformed domain; it
does not reconstruct each field and then ask the geometrical question. A
fixed unblocked linear ray operator still fails on blockers: the correct
interaction law matters, not just retaining signs.

[Phase experiment](PHASE.md)

## A component can become consequential only through context

Consider two four-cell blocks:

    S... | ....
    S... | .B..

The right block has no slider and therefore emits no independent local attack
in either case. The left local attack is also identical. When an incoming ray
is passed between blocks, however, the right block either transmits it or
stops it at B. The complete source-zero masks are 0xfe and 0x3e.

The exact interaction is a gate: a distant source-target relation requires
every intervening cell to be empty. Explicit boundary composition passes
26,937 complete line cases and 30,336 substitution queries per variant.
Staggered local OR repairs short boundary relations but misses long rays.
Reversible lifting followed by decoding does not add an interaction mechanism.

[Overlap experiment](OVERLAP.md)

## Pairwise interactions need not determine collective possibilities

Each target contributes its usable resource choices. Compose those choices
by polynomial multiplication, with x_d squared equal to zero: using a
unit-capacity resource twice makes that assignment term vanish.

For four labelled resources a,b,c,d, compare:

    (a+b)(a+c)(a+d)(a+b) = 2abcd
    (a+b)(a+c)(b+c)(a+b) = 0

These systems have the same two choices per target and the same complete
matrix of pairwise target overlaps. Yet the first has two compatible complete
assignments and the second has none. An independent direct coefficient-domain
check gives partial assignment counts 1,2,3,4,2 versus 1,2,3,2,0.

The important operation is the compatibility product. A signed Walsh basis
is a way to express that operation, not the source of its semantics.
The transformed compatibility operator matches independent explicit assignment
counts on all **5,505,024** graph/demand/availability cases in the frozen
three- and four-defender domains. It composes target operators and reads the
joint count directly, without reconstructing incidence or calling the oracle.

The alignment intervention isolates the effect of coupling: swapping two
resource identities for only one target preserves every individual target's
coverage, but changes feasibility in 17,440 of 1,048,576 unconditional
four-defender queries. Relabelling the resources consistently across every
target, with any unavailable-resource condition remapped too, changes no
assignment count. With a fixed unavailable resource, the independent swap can
also change effective individual coverage; those cases are reported separately.

Pairwise Gram information answers the feasibility query throughout the smaller
three-defender domain, but not the larger domain: there are 288 ambiguous
four-defender cases with all four targets demanded and no removal. The
four-way counterexample above explains why pairwise agreement is insufficient.

[Shared-resource experiment](IDENTITY.md)

## The codec analogy has an actual cancellation example

The parent reproduces MDCT overlap-add on 578 signals at each of two block
sizes. Individual block alias residuals reach about 2; composition cancels them
to below 2.1e-14. Discarding coefficient signs or mismatching synthesis windows
breaks reconstruction. This establishes the signal-processing mechanism,
not its transfer to chess.

[MDCT control](MDCT.md)

## Architectural implication

The useful question is which **composition law** makes a chess interaction
easy to calculate. Bilinear correlation, ordered blocker gating, and
shared-resource compatibility answer different questions. A coordinate
transform can expose or simplify such an operation; invertibility alone does
not create it. These results do not yet establish one algebra covering their
combination, move timing, legal counterplay, or move selection.

The frozen [round note](../CODEC_COMPOSITION_LAB.md) and lane protocols retain
the initial questions, controls, source references, and subsequent steering.

## Verification

The parent reran every lane. Phase and MDCT deterministic outputs match
byte-for-byte; overlap correctness matches its frozen JSON; identity results
match after removing only diagnostic wall-time fields. An independent audit
also uses pair enumeration, a complete MDCT analysis-matrix identity, closed
ray predicates, and direct coefficient-domain resource composition. Astra
separately reviewed the MDCT formulas and the transformed resource creation
operators, including normalization and conditional identity handling.

The lab retains small source and results, with no game replay media. See
[commands](../../../experiments/codec_lab/README.md),
[parent audit](../../../experiments/codec_lab/parent-audit.json), and
[replay inventory](../../../experiments/codec_lab/parent-replays.json).
