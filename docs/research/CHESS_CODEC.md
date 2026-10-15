# Counterfactual chess codec

The current implementation includes [linear channel transfers and shared
geometry](CODEC_TRANSFER.md), plus [whole-set queries and a selectable static
policy](CODEC_POLICY.md). The measurements below retain the initial implementation
and the discovery of its weak comparator; the refinement report owns the newer
near-parity result against the strong scalar control.

This executable CPU experiment changes the unit of work from one successor board
to a relation of sparse alternatives. It constructs exact legality and all-square
attack consequences together, without constructing or making a `Position` for
any candidate. A decision procedure can query those consequences directly.

It is a concrete part of the proposed "JPEG of chess engines": overlapping
movement channels, small exact tables, sparse changes, and computation inside a
compressed set of possibilities. It is **not** a claim of a new competitive chess
algorithm, historical novelty, or a replacement for game-theoretic reasoning.
Actions are still enumerated. The experiment removes the per-action full-board
transition/attack loop, not the need to resolve adversarial continuations.

## Representation and algebra

`CodecPosition` stores the board as 64 four-bit symbols. Four overlapping channel
families cover ranks, files, and both diagonals: 46 lines of at most eight squares.
Each line has an eight-bit occupancy symbol. An 8 x 256 byte table returns the
visible interval around a slot, including its first blocker in each direction.
Knight, king and directed pawn channels use their own local topology.

An action is a packet of at most four square edits. Promotion replaces its moving
symbol, en passant removes a third square, and castling moves four squares. An
unchanged lane and separate castling-transit lanes retain obligations that a
final-board test alone would miss.

Those sparse packets are transposed into bit planes. A bit in a square's
occupancy/type plane denotes one alternative. Sixty-four alternatives therefore
share one machine word; no lane contains its own full successor board.

For an ordered ray segment, the boundary descriptor is `(occupied, attacker)`.
`occupied` means some square in that segment blocks the ray in that alternative;
`attacker` means its first occupied square is an appropriate hostile slider.
For near segment A followed by far segment B:

```
occupied(A then B) = occupied(A) | occupied(B)
attacker(A then B) = attacker(A) | (~occupied(A) & attacker(B))
```

This operation is associative and has the empty segment as identity. The same
formula composes 64 distinct counterfactual rays simultaneously. The eight ray
answers and local pawn/knight/king channels produce an exact attack plane for
every target square and both colors. King safety, checking moves, starting-square
safety and castling transit safety are intersections/queries in that relation.
Pinned moves and en passant discoveries consequently need no separate pin rules.

`MoveRelation` retains the Boolean all-square attack answers. Consumers can ask
`attacked_after(candidate, color, square)`, read sparse patches, and select a move
without constructing successor boards. Attacks use chess's geometric attack
meaning: a pinned piece still attacks squares. They do not themselves establish
that an opponent's capture is legal.

The alternative set is compressed, but the implementation does not claim smaller
total storage than ordinary bitboards: line topology is redundant, scratch planes
include piece identity, and the retained output is deliberately richer than a
legal move list. Scratch plane storage is proportional to the number of 64-lane
words, not one `Position` per alternative. The current implementation recomposes
all target channels; incremental cache reuse is future work.

## Evidence

Run:

```
cargo test --release --test codec
cargo test --test codec
cargo run --release --example codec_probe -- 512 10
```

The original four tests cover boundary associativity and first-blocker semantics, packed
board identity, exact square patches and every color/target attack answer for all
candidates in special-rule fixtures, deterministic legal walks, cross-word lane
sets, and known independent perft counts. Fixtures include pinned en passant,
removing a checking pawn by en passant, castling through check/out of check,
promotion and underpromotion, double/check evasions encountered in walks, mate,
stalemate and a locally valid many-queen position. Known depth-three counts are
8,902 (start), 97,862 (Kiwipete), and 2,812 (rook/pawn ending). Only the tests and
probe use the orthodox kernel to construct oracle successors; production
`src/codec.rs` does not call `make_move`, `legal_moves` or the orthodox attack code.
Both debug and release tests passed on 2026-10-15. The refinement adds focused
linear-work and target-exclusion tests; the whole-set query interface has
separate cross-word/padding controls.

The probe compares identical Boolean attack questions for every candidate,
including illegal candidates. Both scalar controls receive precomputed candidate
packets for free and make one successor for each. The original **query control**
asks `attackers_to` about all 128 color/target combinations. The stronger
**piece-union control** computes occupancy once per child, unions pawn shifts,
precomputed leaper masks and SWAR slider rays, and retains two ordinary attack
bitboards per candidate before answering the same 128 Boolean questions. The
codec timing includes encoding, proposal construction, sparse plane construction,
the full relation and queries. A separate orthodox legal-only timing makes the
workload difference explicit.

Before timing, every candidate's complete two-color attack masks are compared
individually between piece union, the codec and the query API. This catches
differences that could cancel in a checksum. A focused example test repeats that
comparison on castling, both colors' en passant, promotion and multi-word roots.

A deterministic 512-position corpus with 10 repetitions contains 15,865 candidate
packets, 31,825 sparse edits including transit lanes, and 512 lane words. Each
position's retained attack result occupies 1,024 bytes in this corpus. All four
runs produced attack checksum 10,396,840 and legal checksum 141,810.

On an AMD Ryzen 5 3600, Rust 1.99.0, default release settings, the original
query-control observations were:

| Run | Codec full relation, us/position | Scalar queries, us/position | Queries / codec | Orthodox legal only, us/position |
| --- | ---: | ---: | ---: | ---: |
| 1 | 13.27 | 49.22 | 3.708 | 1.65 |
| 2 | 17.07 | 63.86 | 3.741 | 2.19 |
| 3 | 16.83 | 62.50 | 3.713 | 2.09 |
| 4 | 17.09 | 63.21 | 3.699 | 2.08 |

Those local wall-clock observations used shared machine load, with no core
isolation or controlled thermal/frequency conditions. They establish a gain over
the repeated-query API, whose occupancy and ray work is redundant for this
Boolean output. They do not establish a gain over an efficient orthodox attack
producer. The stronger control below corrects that comparison.

Four corrected probe runs pinned to CPU3 on the same shared host produced:

| Run | Codec, us/position | Scalar piece union, us/position | Codec / union |
| --- | ---: | ---: | ---: |
| 1 | 14.39 | 3.59 | 4.004 |
| 2 | 14.41 | 3.59 | 4.013 |
| 3 | 14.74 | 4.14 | 3.561 |
| 4 | 16.65 | 3.36 | 4.958 |

These are diagnostic runs with fixed method order (codec, repeated queries,
piece union, legal only), not isolated frequency/thermal controls. Complete
mask equality was checked outside timing before every run. The stronger scalar
control retains its output before consumption, including its vector allocation;
the gap is not an artifact of streaming one attack mask and discarding it.
The [raw observations](evidence/codec-piece-union-20261015.json) bind the corrected
example and executable hashes. Reproduce the focused check with
`cargo test --release --example codec_probe`; strict example Clippy and formatting
also passed.

The original codec was roughly four times slower than scalar piece union on
this workload; this rejects a general full-relation speedup claim for that
version. The [subsequent refinement](CODEC_TRANSFER.md) reduces codec runtime
about fourfold and reaches approximate scalar parity, without a scalar speed
advantage claim. Its
retained all-square answers are a transposed representation, not an output-size
gain over two ordinary attack bitboards per candidate: those masks need 16 bytes
per candidate, while codec padding needs 1,024 bytes per 64-lane word. The full
codec relation is also substantially more expensive than Chen's legal-only path.
The 512-position timing corpus fits into one lane word per position; separate
exactness tests exercise multiple words. None of these measurements establish
search throughput, match strength, GPU behavior or an engine-level speedup.

## Boundary of the result

The exact interface is now executable: independent sparse consequences can be
composed jointly and inspected without unpacking one full state per action. This
supports searchless policies and a different place to spend reasoning effort.

It does not solve opponent refutation, terminal/draw history, exchange legality,
or long-term value. An attack plane does not distinguish a defended piece from a
profitable capture. A learned or symbolic decision layer must preserve these
boundaries rather than treating local attack arithmetic as minimax. Composing
multi-action, cross-channel refutations without exploding the alternative set is
the next research question; this implementation makes no claim to have solved it.
