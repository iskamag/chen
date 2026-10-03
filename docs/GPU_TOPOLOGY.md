# Compact native pseudoattack topology

`chen_rules_export_topology` exports exact rule topology from an existing GPU
search arena into caller-owned device buffers. It accompanies the unchanged
symbolic ABI1 rather than extending that layout. The native exporter does not
construct a dense attack matrix, download a board or allocate device memory per
call. The neural adapter can consume these masks as sparse directed edges; that
integration is a separate step, with no inference-speed or chess-strength claim
from this correctness milestone.

## Contract

The declarations are in [chen_symbolic.h](../gpu/chen_symbolic.h); the additive
implementation is in [chen_symbolic.inc](../gpu/chen_symbolic.inc).
`chen_topology_abi_version()` returns 1. `ChenTopologySink` is 80 bytes and uses
64-bit device addresses. The export reuses the 24-byte `ChenSymbolicExport`
result. The existing `ChenSymbolicSink` remains 72 bytes.

```c
int32_t chen_rules_export_topology(
    ChenRulesContext *context, uint64_t start, uint64_t count,
    uint64_t expected_epoch, const ChenTopologySink *sink,
    ChenSymbolicExport *result, char *error, uint64_t error_capacity);
```

Every output is contiguous, on the arena device, aligned to eight bytes and
nonoverlapping with other outputs or arena allocations. Each interval advertises
its entire `capacity`; a small requested count does not admit an overstated
allocation. The seven arrays occupy 1,072 bytes per exported position:

| Field | Element type and shape | Meaning |
| --- | --- | --- |
| `source_attacks` | `uint64[N,64]` | One outgoing pseudoattack mask per source square |
| `pinned` | `uint64[N,2]` | Absolutely pinned mover/opponent piece squares |
| `checkers` | `uint64[N]` | Opponent sources checking the current mover's king |
| `pinner_by_square` | `int64[N,64]` | Pinner square, or `-1` for an empty/unpinned source |
| `position_hash` | `uint64[N]` | Absolute simulator position hash |
| `node_id` | `uint64[N]` | Arena index `start + row` |
| `root_epoch` | `uint64[N]` | Current root epoch |

Torch `int64` tensors can hold the unsigned masks and identities without changing
their bit patterns. A high bit is therefore a negative signed value, not an
invalid mask.

## Chess semantics and canonical indexing

Both source indices and target bits are in the current mover's rank orientation:
white-to-move is unchanged, black-to-move maps square `s` to `s XOR 56`. The hash
stays in absolute simulator coordinates. Both colors' source masks are present;
color comes from the authoritative board supplied by the existing snapshot or
transition packet. Piece-object slots must be mapped explicitly to canonical
source squares; they are not the 64 source indices.

The exporter reuses `symbolic_outgoing`, `symbolic_pinned`, and the native exact
`attackers` helper. Slider attacks include every empty square up to and including
the first occupied square, even when that blocker is friendly. Pinned pieces
retain pseudoattacks. Pawn masks contain diagonals only, including empty targets;
they do not contain forward moves or en-passant actions. King masks do not contain
castling. Empty source squares produce zero masks. These are not legal-action
masks, defended-piece counts, or pin-filtered attacks.

Pins and checkers are distinct evidence. `pinner_by_square` covers pinned pieces
of either color and records the first enemy slider beyond that piece along its
king ray. `checkers` concerns only the current mover's king. Consumers should
preserve empty-square attack edges and distinguish pseudoattack, occupancy,
pinner/pinned, checker/king and complete legal-action relations. Action IDs,
en-passant victim effects, castling rook effects, full state features and
history bindings remain supplied by the existing ABI1 snapshot/replay APIs.

## Ownership and failure

The caller serializes arena access, retains all output allocations through the
call and completes prior writers before export. The existing private HIP stream
finishes synchronously before success; the caller may then read or retain the
snapshot independently of reset or arena destruction.

Host admission checks range/search mode, exact epoch, ABI size/version/reserved
field, device, capacity, pointer alignment, allocation extents and disjointness.
The existing device preflight checks the resident rows before any output writes.
Admission/preflight rejection leaves payload and result metadata unchanged. A
runtime failure during the output kernel may leave partial payload, which the
caller must discard; result metadata is published only on success. A valid empty
request with capacity zero and null addresses succeeds with zero rows.

## Executed validation

The strict `gpu/build.sh` gfx1030 build (`-Wall -Wextra -Werror`) passed. The
focused [test](../gpu/test_native_topology.py) then passed four actual GPU tests
covering 755 independently checked rows:

- Complete 64-source masks, both pin planes, all pinners, checkers and identities
  against Python-chess, with both mover orientations, friendly blockers, pinned
  knights/sliders, double checks and terminal/draw-clock fixtures.
- Reachable roots plus every legal child of the sampled and special-move roots,
  including castling, en passant and promotion.
- Deep history, unchanged search counters, independent snapshot lifetime, and
  the unchanged ABI1 seven exact topology channels on 24 mirrored fixtures.
- Twelve rejection cases with untouched payload/result, empty export and recovery.

The correctness process exited zero after 2.93 seconds; its host process retired
and the container process scan found no remaining probe. Timing was not studied.
The native compiler was HIP 7.2.4/Clang 22; the executed Torch 2.10.0+ROCm7.0
process resolved `libamdhip64.so` to Torch's runtime, with
`hipRuntimeGetVersion = 70051831`. This interoperability statement is supported
by this executed test, not by assuming the compiler and loaded runtime match.

Source, library, oracle/runtime bindings, exact argv, raw output, retirement and
limitations are recorded in the
[evidence packet](../gpu/evidence/native-topology-gfx1030-2026-10-03/README.md).
This packet establishes export correctness on its fixtures; it does not establish
exhaustive position coverage, sparse neural integration, throughput or playing
strength.
