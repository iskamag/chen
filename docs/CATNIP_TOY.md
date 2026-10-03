# Distracted Chen: the δ–θ toy

This is a playable **deliberately silly rollout backend**, `catnip-toy`, not the
serious [batched attack/refutation design](research/CATNIP_SEARCH.md). It is
unrated. Neither its jokes nor its score imply strong chess.

Several cats follow different root moves. They notice knights, forward motion,
captures, promotions and checks, plus a reproducible hash-derived sparkle. A
move's `fish_probability` is represented by a bounded attention score: it is a
heuristic lure, **not a calibrated probability**. The implementation calls it
`fish()`; it never samples a real probability distribution.

- **θ, minimum fish attention:** a nonterminal move below θ is ignored. θ=0
  considers all visited moves; θ=∞ ignores every nonterminal move. A legal
  fallback still exists. Mate and draw classification happens before attention.
- **δ, drift before Chen notices:** each rollout remembers its root-player score.
  A new score replaces the remembered score only after its absolute drift reaches
  δ centipawns, or when that visited position is terminal. With huge δ, Chen
  follows its original impression much too long.
- **Fail-meow:** a score change was too small to notice, or a cat found no
  interesting continuation and went to sleep.
- **Fail-zoomies:** a drift of at least δ was noticed. At δ=0 even zero drift
  counts; this is a joke counter, not a pruning diagnostic.

Each side of a rollout chooses its most alluring visited legal move, not its
best evaluated move. Attention, rather than minimax, routes the cats. After a
bounded number of plies, Chen picks the root cat with the best remembered score.
It can happily follow a knight around while ignoring a queen hanging elsewhere.
This is neither alpha-beta with a check bonus nor an exact search replacement.

## Run it

In UCI:

```text
setoption name Backend value catnip
setoption name CatnipDelta value 80
setoption name CatnipTheta value 0.55
position startpos
go movetime 1000
```

`CatnipTheta` also accepts `inf` for perfect boredom. UCI prints `fail-meow`,
`fail-zoomies`, ignored moves and explored ply in an `info string catnip` line.
The browser offers Catnip as a selectable mood with δ and θ sliders.
A bounded CLI probe is `cargo run --release -- bench --backend catnip --depth 6
--nodes 1000 --catnip-delta 80 --catnip-theta 0.55` (on one line).

## API and limits

```rust
use chen::catnip::{CatnipConfig, CatnipSearch};

let search = CatnipSearch::new(CatnipConfig {
    delta_cp: 80,
    theta: 0.55,
    seed: 0x4348_454e,
    walkers: 8,
    max_plies: 6,
})?;
```

`CatnipSearch` implements `SearchBackend`, including cancellation via
`stop_handle()`. `stats()` returns counters, maximum visited ply and peak retained
walkers. Invalid negative δ/θ, NaN θ, or walker/ply capacities outside 1..32 are
rejected. Positive infinite θ is permitted. It owns a stateless tapered evaluator;
changing a learned evaluator is not part of this toy's interface.

Default state is at most eight root walkers and six selected moves per rollout.
The caller's depth can lower this cap. Child classification is charged as one
node; the nonterminal root is charged as one node. Root bootstrap generates legal
moves and checks terminals even with zero depth, zero nodes, zero time or cancellation, so
there is always a legal fallback if one exists. Terminal root bootstrap is not
charged. Time and cancellation are checked between child classifications, so a
single legality/terminal operation can overrun a deadline slightly. A node cap
is never exceeded. No root mutation survives search.

A single incremental root list is retained, not an exponential tree. Each
rollout has a bounded PV, current board and history. Root history is restricted
to the reversible draw window; appended rollout history is at most 32 positions.
The seed and board/move identity determine attention. Runs with the same node
budget/configuration are reproducible; time-limited runs can stop at different
places on different machines.

## Honest scores and terminals

Visited positions are checked for checkmate, stalemate, insufficient material,
100-halfmove draw and third repetition before heuristic pruning. Checkmate has
priority over the move clock. Historical positions passed to search exclude the
current root, as in the other backends. Only visited children are classified;
resource exhaustion can leave an immediate mate elsewhere unexamined.

Root checkmate and an observed root mate-in-one can produce a UCI mate score.
A later mate encountered in a sampled rollout **does not prove a forced mate**:
that result is clamped to the static centipawn range. The remembered score and
PV are therefore heuristic root-player centipawns, with legal replayable moves.
An exact terminal child can still be followed by a heuristic root decision:
visiting one reply does not establish its parent's minimax value.

`SearchResult.depth` stays zero because this algorithm completes no uniform
minimax horizon. `stats().deepest_ply` reports explored ply, not a proven depth.
`qnodes` and `hashfull` are zero; `selective` is absent. There is no TT and no
`Exact`, `Lower` or `Upper` bound publication. An unfinished budget-limited toy
result is a heuristic recommendation with `stopped=true`, not a completed-depth
claim. The first generated legal root move remains the fallback if every cat
ignores every nonterminal root child.

## Checks

`tests/catnip.rs` covers legal PV replay and root restoration, fixed-budget
reproducibility, bounded node use, θ=∞ distraction, δ-dependent noticing,
invalid configurations, precancellation/zero time, root repetitions and terminal
precedence. These establish mechanics, not quality or Elo. The serious CPU PVS
backend remains the default in integrations.
