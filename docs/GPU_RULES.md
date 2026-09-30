# Device rules and resident search arena

Chen now executes legal chess move generation, move application, incremental
hash updates, leaf adjudication/evaluation, and minimax reduction on gfx1030.
The host schedules stages; boards and search edges remain in a preallocated GPU
arena between stages. The separate batched-evaluation ABI remains version 1.
The new rules ABI is independently versioned as `chen_rules_abi_version()==1`.

This implementation is a transparent correctness baseline: one GPU thread owns
one board. Sliding attacks step along rays; pseudolegal moves are filtered by
applying them to a private board and checking the mover's king. It matches Rust's
move enumeration order, including queen/rook/bishop/knight promotions. No CPU
move expansion is hidden behind the device calls.

## ABI and ownership

`gpu/chen_rules.h` is authoritative. The 128-byte position packet contains:

| Offset | Field | Meaning |
|---:|---|---|
| 0 | `u64 boards[12]` | White P/N/B/R/Q/K, then black; A1 is bit 0 |
| 96 | `u32 side` | 0 white, 1 black |
| 100 | `u32 castling` | WK/WQ/BK/BQ bits 1/2/4/8 |
| 104 | `i32 ep_square` | Raw square, -1 absent |
| 108 | `u32 halfmove` | 0..65535, saturates at Rust's u16 maximum |
| 112 | `u32 fullmove` | 1..4294967295, saturating increment after black |
| 116 | `u32 reserved` | Must be zero |
| 120 | `u64 hash` | Chen's exact Zobrist identity |

The move packet is four bytes: from, to, promotion, reserved. Promotion codes
are 0 none, 1 knight, 2 bishop, 3 rook, 4 queen. Reserved must be zero.
Ingress validation runs on GPU and checks disjoint boards, one nonadjacent king
per color, pawn ranks, castle-home pieces, EP consistency, counters, reserved
fields, and recomputed hash. These are Rust's local FEN invariants; neither
implementation proves historical reachability of arbitrary FENs.

Hash keys use the same SplitMix function, seed `0x4348454e00000000`, and offsets
as `src/core.rs`. EP contributes to repetition identity only if at least one
legal EP capture exists, including discovered-check and pinned-pawn cases.
Actual transitions update hashes incrementally; ingress recomputation supplies
an independent arithmetic check. Side, rights, EP, captures, rook movement,
promotions, and counter saturation are preserved.

A context has exclusive caller ownership and serialized calls. Each host-thread
device operation selects the context's device. GPU buffers and pinned staging
live until destruction, which drains the stream. The same shared library must
remain loaded until the context is destroyed. There is no CPU fallback or
architecture override. This build rejects a non-gfx1030 device.

Generic `legal` returns 256 move slots per position, a meaningful count, and
check/insufficient-material flags. Generic `make` validates each requested move
against GPU legal generation before applying it. Empty batches succeed. These
calls invalidate the resident arena; reset before returning to tree operations.
On any failure, caller outputs are discarded.

## Resident stages

1. `reset` uploads one root and previous-position hashes. History excludes the
   current root and is bounded to 65535 entries. Arena nodes carry parent IDs,
   incoming moves, depth, child ranges, score state, and best-child IDs.
2. `classify` generates legal moves on GPU and adjudicates terminals. Search
   mode checks mate/stalemate before halfmove, insufficient-material, and
   threefold draws. Repetition scans the device parent chain plus uploaded
   previous-root history within the reversible-move window. Perft mode ignores
   draw claims and material adjudication.
3. `expand` checks phase legality, classifies the current level, counts children,
   assigns nonoverlapping ranges, and writes child boards with incremental
   hashes. All these stages run on device. The initial prefix pass uses one GPU
   thread; this is an explicit optimization target.
4. `evaluate` classifies depth-limit leaves before scoring nonterminals with
   Material or Tapered. Scores are side-to-move centipawns clamped to +/-29000;
   mate uses -30000 plus ply. Terminal nodes retain exact outcomes.
5. `reduce` processes a parent level after all its children have scored. Each
   thread computes max of negated child scores and preserves deterministic
   first-best tie breaking. `finish` traverses best-child IDs on GPU and copies
   only the root result and PV to the host.

The first classification/expansion/evaluation locks perft or search mode until
reset. The first evaluation locks its evaluator selector until reset. Repeated
classification/evaluation does not increment position counts twice. Expanded
nodes cannot be reclassified/evaluated, and a statically evaluated nonterminal
cannot subsequently be expanded. Before reduction writes any scores, a read-only
GPU pass verifies one ply, expanded or scored terminal parents, complete child
scores, and no children within the submitted parent span. The Rust scheduler
owns level admission and traversal order.

All stages complete synchronously. The host may inspect a small count/statistics
record to decide whether to admit the next level; it does not download the
frontier's board packets. Debug `download` exists for parity verification and is
not part of normal search. There is no exposed device-pointer or DLPack export
in rules ABI 1.

## Capacity and interruption

Arena capacity counts all retained tree levels. Each board has 256 reserved move
slots; a position producing more legal moves returns capacity failure instead
of truncating. Child-count validation precedes range assignment and expansion,
so arena overflow does not commit child boards or ranges. Return codes are 0
success, -2 bounded capacity, and -1 runtime/packet/phase failure. A failed mixed
range operation may have classified other valid nodes; general failures promise
discarded outputs, not complete rollback. Reset before abandoning a generation.

The host retains the previous completed iterative-deepening result. Admission
checks enforce node/time/stop budgets at stage boundaries, and leaf work already
admitted can finish at an exact node cap. GPU stages cannot be interrupted in
the middle by this ABI. This is a bounded soft-deadline design; cancellation
latency includes the current stage. Device and CPU frontier schedulers can do
different amounts of partial work before stopping; compare completed depth and
useful work separately from raw visited counts.

Preallocated device payload is approximately `1340*capacity + 524320` bytes;
pinned host payload is `1160*capacity` bytes. At capacity 250000 this is roughly
320 MiB device and 277 MiB pinned host storage, excluding runtime/module and
allocation granularity. The 256-slot move array dominates. Compressed/ragged
edges and level reuse are future experiments with the same overflow semantics.

Statistics expose arena size, uniquely classified positions, and evaluated
nonterminals cumulatively since reset. `kernel_ms` also accumulates since reset;
it measures HIP event intervals, including possible enqueue/dispatch gaps.
Root upload and result/debug downloads are excluded from that metric. Generic
batch calls separately report H2D, kernel, and D2H event intervals. Whole-engine
wall time remains the measure that includes scheduling and host staging.

## Executed evidence

The independent oracle ran on actual gfx1030 with strict HIP compilation:

```sh
gpu/build.sh /path/to/libchen_gpu.so
python3 gpu/rules_probe.py /path/to/libchen_gpu.so \
  --positions 256 --capacity 250000 --max-depth 4
```

The script requires the optional development oracle `chess==1.11.2`. For the
container run, its pure-Python package was copied from the existing QA venv into
the isolated `/tmp/chen-gpu/chess` directory; no system package was installed.
The source/library/compiler/oracle fingerprints and records are committed in
`gpu/evidence/rules-gfx1030-2026-09-30.json`.

Verified in that run:

- Legal move sets and check/material flags for 256 fixture/trajectory positions.
- Every legal transition from those positions: 7090 exact 128-byte packet and
  hash comparisons against independently applied Python rules.
- Six standard perft families at depths 0..3, including castle/check/EP/promotion
  cases; start position additionally reaches depth 4 with 197281 leaves and
  206604 retained arena positions.
- Device-resident depth-3 minimax from start position: score 36, PV `b1c3 b8c6
  g1f3`, 9323 classified positions, 8902 evaluated leaves.
- Explicit repetition history, idempotent evaluation, mode/evaluator locks,
  evaluated-node expansion and mixed-level reduction rejection, illegal
  move/hash errors, empty batches,
  insufficient PV capacity, and arena capacity-20 failure with root preserved
  and subsequent classify/evaluate recovery.

Start-position depth-4 perft took about 4.2 ms wall time in this single
unprofiled correctness run. This observation has no paired CPU baseline,
repetition/variance study, or controlled power-state record. It is not a speedup
claim. Exact Rust ordered-move, legal-state, resident-search, and budget tests
are separate evidence supplied by the integration suite.

This establishes an executed device rules/search path with bounded storage.
It does not establish exhaustive rule correctness, neural engine inference,
selective-search strength, or the requested 2500 Lichess performance. Direct
device-produced Torch replay has separate, later
[export evidence](GPU_REPLAY_EXPORT.md); it does not change this rules benchmark.
