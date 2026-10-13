# Exact tactical input reference

`neural/tactical_inputs.py` supplies the existing [tactical operator](TACTICAL_OPERATOR.md) from legally replayed standard chess. It is a CPU admission/reference adapter, not a resident GPU emitter, learned evaluator or strength result. Importing its compact extraction helpers needs python-chess but no Torch. Tensor emission imports Torch lazily.

```python
from neural.tactical_inputs import ReplaySpec, compact_from_replay, tactical_config, to_tactical_batch
from neural.tactical_operator import TacticalOperator

state = compact_from_replay(ReplaySpec(moves=('e2e4', 'e7e5')),
                            node_id=7, root_epoch=3)
batch = to_tactical_batch([state], max_edges=4096)
model = TacticalOperator(tactical_config(width=32, steps=4))  # untrained
output = model(batch)
```

Default replay starts at standard chess's initial position and includes every move. A different valid FEN requires `explicit_fresh_seed=True` and halfmove zero: it explicitly begins a new history, without inventing pre-FEN repetition. This does not establish historical reachability of a handcrafted seed. Illegal/null moves, unknown history, invalid seeds, replay past adjudicated terminals and forged compact packets reject. Chess960 is unsupported. Geometry follows [python-chess's official rules API](https://python-chess.readthedocs.io/en/latest/core.html): slider pseudoattacks include the first occupied blocker; pinned pieces retain pseudoattacks. Absolute pins, their pinners and checkers are separate evidence.

## Compact contract and history

Canonical squares flip ranks with XOR56 when Black moves. Piece codes are zero for empty, 1..6 for mover P/N/B/R/Q/K and 7..12 for opponent. `source_attacks[64]` contains unsigned canonical target bitboards at canonical source indices; empty sources are zero. `pinned[2]` is mover/opponent; `pinner_by_square[64]` is -1 except at an absolutely pinned piece, where it identifies the canonical pinner square. `checkers` identifies only pieces checking the current mover's king. Castling rights and raw EP square remain explicit compact sidecars.

History counters are halfmove clock, occurrences of the current position in retained reversible history, retained length including current, and `ply_from_search_root`. The default last counter zero declares a new search root. When comparing the synthetic pilot's game-continuation record, explicitly pass its `root['ply']`; it declares the standard initial position as continuation root. Native-compatible uint64 position hashes use Chen's Zobrist layout and legal-capture-effective EP. Context SHA binds replay, history, node ID, root epoch and search ply; a string alone certifies nothing. `admit_compact` reconstructs replay truth and compares the complete packet.

Mate/stalemate are adjudicated first, then current halfmove≥100, current threefold and insufficient material. A terminal draw can retain geometric legal actions; callers must obey `state.terminal` before using an action choice. Model CP heads do not replace terminal adjudication. No one-move-ahead draw claim is fabricated.

Complete legal actions are sorted by `(source*64+target)*5+promotion`, where promotion is 0/n1/b2/r3/q4. Each action carries parent-oriented changed-square `(square,old,new)` edits, EP victim distinct from destination, capture/castling/check flags and classical material after that known legal move. Castling edits include the rook; EP edits include the removed pawn; promotions include every legal underpromotion. The four geometric channels and action IDs use the same orientation as the parent, including when the successor's mover changes.

## Sparse relations and features

Pieces occupy up to32 slots sorted by canonical square, with explicit `piece_squares`; square nodes start at32 and action nodes at96. All14 declared relations are emitted when present: occupancy and reverse occupancy; piece→square pseudoattack; four own/enemy attack/support piece pairs; pinned→king; pinner→pinned; checker→mover king; moving piece→action; destination square→action; victim piece→action; action→every changed square. Empty-target pseudoattacks are retained. There is no dense adjacency or invented reverse relation. Optional edge capacity rejects overflow instead of dropping evidence. Complete legal-ID coverage is independently supplied to structural tensor admission.

The versioned `chen_tactical_exact_v1` schema is FP32:

| Token | Ordered channels |
| --- | --- |
| Piece16 | Own/opponent piece-code one-hot12; file/7; rank/7; absolute-pin bit; mover-checker bit |
| Square12 | Own/enemy incoming counts clipped8 /8; own/enemy outgoing counts clipped8 /8; own/enemy pin bits; checker bit; capped material phase/24; halfmove/100, repetition/3, retained length/256, search ply/256, each clipped1 |
| Action12 | Source file/rank /7; target file/rank /7; capture, EP, castle, gives-check; promotion suffix/4; victim material/900; mover material/900; post-action parent material CP/32000 |
| Edge4 | Receiver−source file/rank /7; receiver old/new piece codes /12. Action node coordinates equal destination; action-effect edges carry the actual edit. Other edges carry unchanged current receiver codes. |

Phase weights are P0/N1/B1/R2/Q4/K0 capped24. Material uses P100/N320/B330/R500/Q900/K0. Current value prior is mover material difference; action prior is the exact post-move material difference in the **parent** mover frame. They are causal classical heuristics, not teacher scores, WDL values or future PV context. Child effects/check geometry are calculated from the already-known legal action, without labels or future search. Raw EP and castling-right sidecars certify rules/action construction but are not separately projected into this feature version; the complete legal-action/effect graph exposes currently actionable consequences. A future richer version must change the schema explicitly.

## Native boundary and checks

`topology_packet` produces CPU int64 storage with uint64 bit patterns matching the independent native topology ABI1. `admit_native_topology` compares source attacks, pins, checkers, pinners, position hash, node ID and epoch against replay truth. Native output needs authoritative board, history, complete actions/effects and object mapping in addition to those topology words. The additive [native tactical emitter](GPU_TACTICAL.md) now exports this complete schema directly into caller-owned device tensors. `neural/native_tactical.py` performs device/runtime/epoch/output ownership admission and returns the exact terminal sidecar with the model batch. Initial hardware checks compared 45 complete graph rows and performed a resident forward/backward check; four additional states verify bit-identical old symbolic channels/history. The wrapper synchronizes and allocates snapshots; allocation-free or captured inference remains open. The compact topology-only API remains [documented separately](GPU_TOPOLOGY.md).

Focused fixtures cover independent scalar-ray attacks, both movers, pinned pseudoattacks, all14 relations, pins/pinners/checkers, EP/castling/promotions, history draws, complete action IDs, forged packets, capacity rejection, signed ABI words and actual sparse forward/backward. With `CHEN_SYNTHETIC_SHARD` explicitly supplied, a three-root TRAIN-only smoke replays the first family's prefixes and binds topology/history/action edits to the pilot, then runs a small CPU forward. It does not read teacher labels, train, optimize parameters or assess chess quality.

```bash
PYTHONPATH=/tmp/chen-gpu CHEN_SYNTHETIC_SHARD=/path/to/families.jsonl.gz \
  HIP_VISIBLE_DEVICES= ROCR_VISIBLE_DEVICES= CUDA_VISIBLE_DEVICES= \
  OMP_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 \
  python -m unittest neural.tests.test_tactical_inputs -v
```
