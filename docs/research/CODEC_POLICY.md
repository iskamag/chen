# Searchless codec policy

`codec-policy` is a runnable CPU decision procedure for the
[counterfactual codec](CHESS_CODEC.md), following the user's
[shared proposal](https://chatgpt.com/share/6ad06b5b-0c50-83eb-94a6-40105b5812af).
It selects directly from the composed legality/attack relation. Its relation
and preference computation do not construct candidate `Position`s, call the
orthodox legal generator, or descend a move tree. Ordinary alpha-beta remains the default engine.

Run the experiment with the normal release binary:

```text
uci
setoption name Backend value codec-policy
position startpos
go depth 1
```

Or inspect its JSON telemetry over the fixed suite:

```sh
cargo build --release --locked
./target/release/chen bench --backend codec-policy
```

The requested search depth and node count do not increase reasoning: this is one
bounded relation composition plus static preference reduction. It reports
**depth 0, nodes 0, qnodes 0**, with alternatives, legal alternatives, scored
alternatives, lane words, sparse edits and boundary compositions separately.
`go infinite` also returns after that work. Stop/time are checked between
preference reductions; relation construction itself is bounded but currently
noninterruptible. A stopped reduction retains a legal fallback or the best
preference already inspected. Source position and history are unchanged.

## Questions over whole alternative sets

The codec also exposes `legal_words()` and `attacked_words(color, square)`.
Bit *i* denotes candidate *i*, with parent, transit and padding lanes removed.
This is an interface for reasoning directly over alternative classes. For example:

```text
legal & attacks(enemy_king) & ~enemy_attacks(target)
```

selects every legal checking action after which a chosen square is not attacked.
Each AND/NOT combines as many as 64 alternatives without decoding their boards.
The target predicate describes a square; it does not imply that a piece remains
there or establish a winning move. `examples/codec_query.rs` runs the query and
only decodes selected action identities for display:

```sh
cargo run --release --example codec_query -- e1 \
  'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'
```

`tests/codec_query.rs` checks complete packed-word answers and intersections
against ordinary successors, including multiple words, castling auxiliary lanes,
empty legal sets and masked padding. These normalized word queries complement
the per-alternative Boolean API; they expose the compressed relation rather than
requiring a downstream consumer to unpack every condition.

## Preference mechanism

The first preference component is the exact existing Tapered score after each
alternative. A root aggregate carries white-frame middlegame/endgame totals,
uncapped phase and bishop counts. Each patch subtracts old and adds new terms at
its two to four changed squares. Bishop-pair bonuses and the phase cap are
applied after the edits. Castling, en passant and underpromotions use the same
mechanism; no successor evaluation or geometry is recomputed.

The second component queries the retained two-color attack planes. For each
side it finds the largest exposed piece: full material value if attacked and
undefended, one eighth if attacked and defended. The preference in the mover's
frame is:

```text
exact Tapered child score
- own largest exposure
+ enemy largest exposure / 4
+ 12 if checking
```

These pressure coefficients are explicit, untrained hypotheses. An attacked
piece is not necessarily legally capturable or lost: pinned attackers still
attack squares, defenders can be pinned, and exchanges depend on continuations.
This policy does not detect candidate checkmates or draw transitions and does
not solve opponent refutations. A concrete negative control is the fixed suite's
`7k/5Q2/6K1/8/8/8/8/8 w - - 0 1`: the current policy selects `g6f5`, although
`f7f8`, `f7e8`, `f7h7`, and `f7g7` immediately mate (checked with python-chess).
The runnable policy therefore does not establish even a mate-in-one solver. Only the current root's mate/stalemate, draw
clock, insufficient material and repetition are classified before preferences.
An optional opening book is supported, after root-terminal classification.
Book lookup delegates to the existing book implementation, whose opening-name
annotation makes a selected-move transition; that metadata work is outside the
codec counterfactual counters.
The policy owns its tapered patch evaluation; selecting a different evaluator
reports an error rather than silently changing the mechanism.

Root terminal scores, book choices, heuristic scores and interrupted fallbacks
have distinct `score_kind` telemetry. Heuristic values stay outside mate range;
the root's proven checkmate retains the normal mate score. A single selected
move is an action recommendation, not a searched principal variation.

## Verification

`tests/codec_policy.rs` differentially checks sparse static scores against full
Tapered evaluation after orthodox transitions, including both castlings, both
colors of en passant and promotion, bishop-pair changes and promoted phase
above 24. Seeded legal walks extend those controls. Other tests cover legal
selection, zero tree telemetry, terminal-before-draw ordering, repetition,
immediate stop, zero time, board restoration and backend reuse. The UCI test
selects the backend, checks its telemetry and legal output, rejects an
incompatible evaluator and switches back to alpha-beta. These tests passed on
2026-10-15.

`examples/codec_oracle.rs` and `scripts/check_codec_oracle.py` provide an
independent python-chess check of complete legal sets and checking effects.
The script retains the input corpus, actual engine output, source/binary hashes
and exact coverage counts under an ignored result directory. The final independent run admitted 15,641 positions and
414,731 legal actions, including 22,399 checking effects,
263 castlings, 75 en passant captures and
1,432 promotions. Complete move sets and all check flags agreed.
This checks rules, not strength. Reproduce with a Python environment containing `chess==1.11.2`:

```sh
cargo build --release --example codec_oracle
python scripts/check_codec_oracle.py \
  --oracle target/release/examples/codec_oracle \
  --games 128 --plies 128 --out results/codec-oracle
```

The next algorithmic question is how to compose opponent refutations and
interaction-dependent preferences inside the channel relation. Static pressure
alone is not a solution to that question. This executable mechanism makes its
inputs, exact consequences, approximation and costs available for that work.

The [integrated checkpoint](CPU_IMPROVEMENTS_20261015.md) records final source
verification and the stronger scalar performance comparison.
