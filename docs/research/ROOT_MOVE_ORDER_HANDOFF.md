# Next bounded GPU candidate: move ordering

Source review and integration-seam coordination are complete. **No ordering
helper was implemented, no ordering tests were executed, and no GPU ordering
measurement exists.** Work stopped at the requested wrap-up boundary.

The recorded recoverable ABI4 Kiwipete control entered 238658 nodes in 1 s but
completed no child, with 15 q-cap failures, 33 paused children and 0 retries: no
Exact incumbent existed to tighten a failed child's window. Ordering is a
useful next search-policy experiment, rather than evidence that more node
throughput alone repairs this failure.

## Policy and native seam

The existing CPU `Searcher::ordered` in [src/search.rs](../../src/search.rs)
puts a hash/PV move first, then captures with priority
`1_000_000 + 10*victimValue - attackerValue + promotionValue`, then quiet
promotions at 900000, then killers/history. Values are `[100,320,330,500,900,0]`.
The king's ordering value 0 follows that existing CPU policy; it is not an
exchange-safety claim. A different king value or promotion formula must be a
separate declared policy, not a hidden evaluator change.

The first isolated helper should provide canonical/off, capture-promotion,
and capture-promotion plus optional previous-PV-hint controls. It must return
a deterministic **permutation**, never remove or insert moves. Cache priorities
once; stable equal-priority ordering preserves the original generator order.
Copying the CPU capture/promotion formula provides an explicit initial control,
without adding TT, killers, history, SEE, pruning or evaluator changes.

Native moves encode A1=0, promotion 0/1/2/3/4 as none/N/B/R/Q, and reserved 0.
`ChenRulePosition.boards` uses white piece indices 0..5 and black 6..11.
En-passant has an empty destination: a legal pawn EP capture must be scored
with a pawn victim from the actual behind-target square, not as a quiet move.
Capture promotions include their promotion bonus. Quiet promotions precede
quiet nonpromotions; equal promotion priority preserves the generator's native
Q/R/B/N order. Castling remains quiet. A PV hint matches exact from/to/promotion
bytes in the current admitted legal list; an absent hint is ignored and never
manufactures a move. Previous-depth hints affect scheduling only, not bounds.

The coordinated ABI4 seam is `pool_workers_kernel` in
`gpu/experiments/root_retry.hip`: inside the `!frame.entered` path, after
successful complete `legal_moves` generation and after any qsearch tactical
filtering. In check, retain **all** evasions. Terminal/draw/static completion,
stand-pat cutoff and resource-cap abort paths keep their existing semantics.
Sort only a surviving expandable frame; persist its order across pauses rather
than re-sorting after `frame.cursor` has advanced. A fresh failed-attempt retry
may initialize a fresh ordered frame. Ordering cost can extend the unchecked
entry phase, so elapsed-cap overshoot needs measurement after integration.

Preserve `p.actions` and `child.canonical_index` in immutable native root
generation order. Root execution priority uses a separate index permutation;
DFS-local move arrays may be reordered. Root certificates and earliest-ID ties
use the original ordinals, never execution slots or completion order. Local
tied PV witnesses may change while remaining legal. The existing recoverable
ABI4 worker/control must remain unchanged; integrate a separately named variant.

Reviewed retry sources in the GPU agent worktree had SHA256:
`root_retry.hip` = `a91a37c26372034104869b4d7e959912dcbe86b949be22a09bc2c62fd5e3b43d`,
`root_retry.h` = `3b13cb41ab8ab2eeec662b441364c48f907adc40febbe582e62c07558da0f368`.
These are source references, not a new runtime admission or copied evidence.

## Focused acceptance checks and next comparison

Before any device experiment, check quiet/capture priority, both EP orientations,
capture and quiet promotions/underpromotions, castling, present/absent PV hints,
equal-priority stability, empty/single/256-move boundaries, duplicate-identity
multiset retention, and unchanged input state. Exhaustively enumerate small
priority lists/permutations and assert every original ordinal appears exactly
once; compare ordering against an independent stable-sort reference. Explicit
disabled policy must return the original order.

Then compare a separately pinned ordered worker against unchanged ABI4 on the
same finite-horizon exhaustive fixtures and narrow-window bounds. Ordering may
change work counts, tied PV witnesses and budget/q-cap abort outcomes; those are
not score-equivalence assertions. Completed bounds must remain sound, aborted
work must publish no bound, and canonical root ties must remain unchanged.

The next actual 1 s Kiwipete control should hold evaluator, q policy/caps, root
history, main-depth schedule, total node budget, epoch/deadline policy and retry
admission fixed, varying ordering only. Report entered/charged nodes, child
Exact/Upper completion, first-incumbent time, q/stack failures, retries, completed
root depth and publication/cancel behavior. Seed failure remains a failed
iteration, not a heuristic Exact result. Only after complete useful searches
exist should a separate equal-time match test any strength benefit.
