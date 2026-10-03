# Catnip: batched attack and refutation search

Status: **design only, 2026-10-03**. No new engine backend, SIMD kernel, GPU
execution, training, speedup or strength result is claimed here. The experiment
asks whether searching related moves together can buy stronger decisions per
second, rather than merely increasing raw node throughput.

## Start from the actual code

`src/frontier.rs` already implements level-synchronous full-width minimax with
batched leaf evaluation. It copies positions into a tree, gathers another vector
for each evaluation batch, and rebuilds that tree for each depth. It is a useful
control, not a new candidate to reimplement.

`src/selective.rs` already implements bounded best-open minimax, with complete
legal sibling sets and host/device `SelectiveTree` arenas. Its mixed-depth scores
are deliberately not full-horizon alpha-beta bounds. `src/backend.rs` supplies
`SearchBackend` and `BatchEvaluator`; the default CPU batch evaluator is a scalar
loop. `src/rules.rs` supplies exact batched rule packets. `Position` and `Undo`
currently store whole states. Existing resident GPU arenas establish a device
ownership path; reuse their rule packets and history contracts.

The proposed difference is **move-family computation plus explicit refutation
obligations**, not another generic frontier. Keep alpha-beta as the reference
and replace candidate allocation and the sibling execution unit independently.

## What searches together

For each selected parent, retain its board and tactical features once. A sibling
packet contains move deltas, not independent copies of every parent's features:

```
Parent: exact board, history handle, feature epoch, cached tactical features
Move delta: from, to, promotion, captured square/piece, rook relocation,
            piece-plane changes, occupied-square XOR, rights/EP/clock/hash changes
Obligation: parent handle, move ordinal, target depth, window, score frame,
            search generation, status, continuation handle
```

Make special moves explicit: en passant changes three occupancy squares;
castling moves two pieces; promotion changes piece identity. State and repetition
history remain separate. A history handle must retain path identity, not just
board identity. No interning of draw-sensitive bounds across unrelated paths.

Compute material/phase/placement deltas from the move and parent. Reuse unchanged
piece attacks. Recompute attacks of moved/captured pieces and every slider whose
empty-board geometric ray intersects changed occupancy; changing one blocker can
open a distant rook or bishop. Recompute affected pin/check relations, with a
full king-safety fallback initially. Piece support counts must account for added
and removed attacks, not just OR a union attack mask. The exact recomputation
control remains available for every feature.

Use one 64-bit bitboard per SIMD lane across four sibling positions on AVX2. Pack
the same piece plane across lanes; broadcasting parent planes and applying lane
deltas replaces repeated gathers from separate `Position` objects. SWAR still
operates within each lane. This does not promise that variable slider iteration
or legality becomes branchless: keep active-lane masks and count wasted lane
work. Scalar plus shared deltas is a separate control from actual SIMD.

Do not retain full board copies plus dense attack maps plus neural graphs for
every node by default. Retain compact deltas and hot decision metadata; materialize
boards for active packets. Bound delta-chain length with periodic checkpoints,
and measure replay cost before selecting that interval. An unchanged feature can
be borrowed only while its parent allocation and epoch remain valid.

## Attack and refutation obligations

1. Obtain a completed incumbent at a shallow horizon using current alpha-beta.
   Generate the complete legal root action list once; batch a cheap sibling
   feature/evaluation pass. Rank all actions, without silently removing quiet
   moves. Give every action a shallow baseline or retain its unresolved status.
2. Maintain root **challenger jobs** and the incumbent's **refutation jobs**.
   Challenger jobs test whether another move can beat the incumbent; refutation
   jobs seek the opponent reply most likely to overturn an attractive attack.
   A sacrifice gets defender search, not just extra optimistic attacker rollout.
3. Fill a packet with distinct obligations. Prefer sibling families sharing a
   parent feature cache, then compatible depth/check-state work, subject to a
   maximum wait. Do not manufacture low-value work merely to fill SIMD/GPU lanes.
4. Execute bounded continuation slices and batch ready evaluations. A job can
   suspend at an evaluation boundary and keep its stack/state; no repeated root
   search solely because a neural batch was unavailable. At a beta cutoff,
   cancel or retire dependent speculative obligations; charge work already done.
5. Back up changed results through affected ancestors only. Store compact
   parent/child handles and frontier indices rather than scanning the whole
   retained topology after every sample. Handle generations reject stale returns.

At the root, start with a concrete allocation control: one incumbent validation
slot, two strongest challengers, and one rotating unvisited/quiet candidate in a
four-job dispatch. Slot identities describe obligations; SIMD leaf batches can
contain ready leaves from those jobs. The ratio is an initial experimental
parameter, not a tuned answer. Periodically service every unresolved action.

The wild arm ranks attack/refutation jobs by predicted **decision change per
unit cost**: likely root margin reversal, latent feature disagreement, sacrifice
depth, forcing-reply indicators, and measured completion cost. These are
allocation hints. Start with deterministic features, then learn a scheduler from
paired traces if the deterministic candidate earns its complexity. Neither a
disagreement score nor a forcing-move label is a value bound.

For comparison, a conservative Jamboree-like arm first resolves the best-ordered
child, batches null-window challenges to the remaining children, and fully
re-searches fail-high challengers. This gives batching a serious pruning-aware
control, rather than comparing only with exhaustive breadth-first search.

## Bounds and publication

Attach every result to a **declared search horizon and leaf policy**. With fixed
depth, exact terminal handling, and a fixed evaluator, unpruned alpha-beta bounds
refer to that finite search's value; they do not certify chess game-theoretic
value. If quiescence is enabled, the stated horizon includes its declared policy.

Keep `Exact`, `Lower`, `Upper`, `Pending` and `Aborted` distinct from heuristic
priority scores. An interval `[L,U]` negates to `[-U,-L]`. A completed MAX backup
uses `[max L_i, max U_i]`; a MIN backup uses `[min L_i, min U_i]`. Missing legal
children retain unbounded endpoints. A pending job is never a zero evaluation.
Mate distance, canonical move ties, repetition and cancellation use the existing
completed-bound contract in `SEARCH_BOUND_CONTRACT.md`.

A null-window challenge against an incumbent threshold can establish a fail-low
upper bound or a fail-high lower bound. A fail-high is not a new exact winner:
perform the necessary full-window verification. Returned bounds keep their
original windows when the incumbent changes; the scheduler decides which need
renewed work. Stale thresholds cannot change a result's kind.

The exact comparison arm publishes only a fully covered winning action: its
lower bound dominates every competing upper bound at the same target horizon.
On timeout use the last completed incumbent, with its actual horizon. An
alternative mixed-depth experimental recommendation must have separate metadata
and cannot write an `Exact` TT entry. LMR/null-move heuristics, if added,
must be separately toggled; their scores are not proofs of the unpruned horizon.

## Chen on catnip

The intended personality is fast development, active knights, changing attack
routes, and sacrifices backed by explicit defense searches. It should not be
implemented by blindly adding centipawns to checks or pawn pushes.

For an optional style level, define `tau` in evaluator centipawns. Among root
actions verified to satisfy `L_a >= max_b U_b - tau` at the same horizon, select
the most active: multiple distinct pieces attacking the enemy king ring,
development, and forcing continuation diversity. Name pseudo-attacks and legal
threats separately. If no such choice is verified, play the incumbent. At `tau=0`
this is only a tie preference; positive `tau` intentionally spends a declared
finite-horizon score tolerance. It guarantees neither low Elo loss nor safety
against deeper tactics. Exclude moves disproven by an opponent mate result.

Compare style as its own arm. Report attack frequency, sacrifices retained after
defender verification, opening diversity, blunders, and strength—not only checks
per game. A principled style feature can later be learned from user-rated games;
there is no established model of the fictional character's chess play.

## Small, comparable implementation milestones

| Arm | Scheduling | Sibling computation | Purpose |
| --- | --- | --- | --- |
| A | Current alpha-beta/PVS | Current scalar | Reference |
| B | Same search/order | Scalar shared exact deltas | Isolate data reuse |
| C | Pruning-aware batched challenges | Full scalar materialization | Isolate scheduling overhead |
| D | Same as C | Delta packets, SIMD width 4 | Isolate vectorization/reuse |
| E | Attack/refutation allocation | Same packet implementation | Test decision allocation |
| F | Same as E | Same execution, `tau` style gate | Test personality cost |

First milestone: material/placement sibling-delta evaluator and exact tactical
feature parity on legal sibling sets, including EP, castling, promotions, pins,
double check and discovered attacks. Do not start neural training to test it.
Second: four-job bounded root challenger execution with scalar evaluation and
existing completed-bound publication. Only then add SIMD and the wild scheduler.

For exact arms, compare fixed-horizon values and canonical best moves against an
unpruned reference. Freeze evaluator and pruning controls. At fixed nodes and
at one second record completed depth/decision, time, instruction/cycle counts,
branch/cache events, bytes copied/replayed, feature invalidations, batch fill,
queue wait, stale/cancelled work, and peak retained state. Strength runs use paired
openings and colors. Change one axis at a time; tune scheduling on development
positions before freezing a held-out set. End-to-end decision quality decides
whether a higher-throughput implementation is useful.

The later full-GPU arm retains board checkpoints, deltas, history, queues and
neural residual features on the device. The CPU transfers root/config/history
once and reads compact results. Compare against CPU packets before claiming
GPU superiority; the current resident search's weak incumbent formation is not
solved by changing where its queue lives.

## 2FFS and prior work

2FFS is present in `src/two_fidelity.rs`, exercised by tests and
`examples/two_fidelity_profile.rs`; it is **not selected by UCI or used by the
rating evaluation**. `FastOracle::evaluate_batch` evaluates complete sibling
expansions. The current controller retains ordered map/set scale state and
performs exposed-tree certificate/refresh scans; compact scale storage and
affected-ancestor invalidation remain meaningful candidates. Existing trace
preservation and profiling controls in `docs/TWO_FIDELITY.md` must remain.

The paper's fast-error envelope and independent slow samples concern node
minimax values. Neural scores, alpha-beta searches at different depths, or
ordinary rollout outcomes do not establish those premises. Catnip may borrow
the leader/challenger obligation idea, but it uses deterministic search bounds
for its exact arm and does not claim 2FFS confidence guarantees. Batched stale
slow samples would change the sequential reference controller's semantics.

These sources establish relatives, not novelty or Chen performance:

- [Kuszmaul, Synchronized MIMD Computing, chapter 4](https://people.csail.mit.edu/bradley/papers/Kuszmaul94.pdf):
  pruning-aware parallel alpha-beta/Jamboree. Catnip's conservative control uses
  this first-child and parallel-challenge pattern.
- [Cazenave, Batch Monte Carlo Tree Search](https://arxiv.org/abs/2104.04278):
  batched neural evaluations and distinct inference-cache/search-statistics
  structures. Catnip is not MCTS and does not import its visit-value semantics.
- [Guez et al., Learning to Search with MCTSnets](https://proceedings.mlr.press/v80/guez18a.html):
  learned simulation/search components and vector backups. A learned obligation
  allocator is a possible later relative, not an implementation of MCTSnet.
- [Chen and Chen, Two-Fidelity Best-Action Identification for Stochastic Minimax Tree](https://arxiv.org/abs/2606.01708):
  heterogeneous oracle costs and conditional identification guarantees. Keep
  those assumptions visible rather than relabeling heuristic chess intervals.

The potentially interesting combination is sibling feature reuse, explicit
attack/defense competition, asynchronous verification and a controlled style
gate. Each mechanism has relatives; no literature-wide novelty claim is made.
