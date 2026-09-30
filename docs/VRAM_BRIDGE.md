# Resident rules producer to Torch replay

`neural.bridge.RuleArena` publishes existing resident HIP parent→child
transitions into Torch-owned compact tensors. The arena receives a CPU root and
previous-position hash prefix once, expands legal search levels on the GPU,
and exports a selected child span without downloading state payloads. The
consumer admits those tensors to `CompactReplay`, samples device features, and
uses the existing models and training loss. This is a bounded producer/consumer
mechanics implementation; it does not implement GPU self-play or establish
playing strength.

The detailed [ownership contract](VRAM_BRIDGE_CONTRACT.md) and
[compact record schema](COMPACT_REPLAY.md) describe the shared boundary. This
binding uses rules ABI 1 and the independent `gpu/chen_replay.h` ABI 1. It rejects
unsupported ABI versions and missing symbols; there is no CPU fallback.

Before loading the producer, the adapter promotes Torch's actual bundled
`libamdhip64.so` with `RTLD_GLOBAL`. It then compares native bound addresses for
`hipPointerGetAttributes`, `hipMemGetAddressRange`, `hipMalloc`, `hipFree` and
`hipStreamSynchronize` against that runtime. Every identity must match before
context creation. This guards against mixed HIP runtimes with conflicting
allocation ownership; matching version strings alone is insufficient. Native
copy APIs are intentionally outside this identity set so copy diagnostics can
interpose them.

## Use

Build the actual gfx1030 library with `gpu/build.sh` in the engine checkout,
then select it explicitly or with `CHEN_GPU_LIBRARY`. The FEN root adapter
requires the optional python-chess development oracle. Applications that already
have a correctly hashed `RulePosition` packet can pass it without python-chess.
The Torch build must support ROCm; its device API uses the spelling `cuda`.

```python
from neural.bridge import RuleArena, position_from_fen
from neural.compact import CompactReplay

root = position_from_fen(
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1")
with RuleArena(capacity=4096) as arena:
    arena.reset(root, history=())  # previous hashes only; root excluded
    first = arena.expand()
    second = arena.expand(first["span"])
    records = arena.export(second["span"], game_id=1)
    replay = CompactReplay(400, arena.device)
    records.append_to(replay)
# Arena storage has retired; the replay and exported snapshots remain owned.
batch = replay.sample(32)
```

Expansion returns statistics plus a `ResidentSpan(context_uuid, epoch, start,
count)`. Export and non-root expansion require this typed selector. Use
`span.subset(offset, count)` for a bounded subset. UUID checks reject another
context's selector, and Python/C epoch checks reject a selector retained across
reset, even when its numerical node IDs have been reused. The C expected-epoch
preflight also protects against native invalidation outside the adapter. The
wrapper exposes no generic arena operations or raw arena pointers.

`expand()` initially selects the current root. After expanding it, repeat root
expansion is an explicit phase error; choose an unexpanded returned span instead.
Only search-mode, expanded, nonterminal parents with legal children can produce
records. Children may themselves be expanded. The exporter computes successor
terminal state locally when needed without changing node classification. Empty
terminal spans are valid expansion results; the Python exporter rejects empty
requests rather than publishing uninitialized rows.

## Exact fields and labels

The 136-byte sink passes 14 writable device addresses and bounded metadata.
Nine fields enter compact schema directly: current/next pieces and rule state,
sorted legal actions/counts, executed action, material proxy and successor
terminal flag. Five resident sidecars preserve current/next simulator hashes,
parent/child node IDs and root epoch. The 24-byte return contains row count,
epoch and an export event span, all control metadata.

Boards are absolute white PNBRQK then black PNBRQK signed-int64 bit patterns,
with A1=0. State int32 columns are side, absolute castling rights, raw EP (-1
absent), halfmove and fullmove. Unsigned clocks and hashes retain their bit
patterns; fullmove values above 2³¹ remain negative int32 until decoded with
`int64 & 0xffffffff`. Halfmove is bounded to Rust's u16 domain and both clocks
saturate exactly as the simulator does. Raw EP is retained even when ineffective
or pinned; repetition hashes use effective legal EP instead. Sparse action
indices use the current parent's canonical side and are sorted, unique, padded
with -1. Legal bounds must be integers 1..256; overflow fails without truncation.

The adapter constructs four additional fields on-device: `target_actions` is
a clone of `actions`, history handles are -1, and `game_ids` use the supplied
identity. Policy targets mean **executed-action imitation**, not a tactical or
best-move teacher. The value proxy is
`tanh(material_evaluator0_score / 800)`, from the current mover's perspective.
Piece weights in centipawns are 100/320/330/500/900/0; the simulator evaluator
clamps the score to ±29000. Neither target represents a game result or calibrated
WDL. The manifest names these exact semantics.

Successor terminal follows `simulator_current_position_draw_policy`: legal
mate/stalemate first, otherwise current halfmove≥100, insufficient material or
current third hash repetition. This differs from prospective claim checking
such as Python's `is_game_over(claim_draw=True)`. Retained root hashes plus node
ancestry drive repetition. The manifest records root packet/hash-history
digests, prefix count, context UUID and epoch. The caller is responsible for
truthful history provenance; an empty prefix does not prove a played game's
history was empty. Replay handles remain absent and the 18-plane neural input
still lacks repetition context. Hash-based adjudication inherits the simulator's
hash-collision assumption.

Sidecars belong to the returned `ExportedTransitions`, rather than the current
ring schema. Ring admission copies only the compact fields and assigns monotonic
sequence IDs. Applications retaining provenance after ring overwrite need a
bounded external sidecar/history owner and explicit admission mapping; the
adapter does not silently reinterpret node IDs as persistent history handles.

## Lifecycle and stream ordering

`RuleArena` holds the loaded library and serializes operations with a lock.
Use a context manager or explicit idempotent `close`; use after close fails.
A failed reset invalidates Python's old provenance. Native errors raise
`BridgeError(operation, code, message)` and output buffers are discarded before
replay admission. Capacity preflight protects child allocation commits; a
runtime device fault still requires retiring the context, not retrying a
possibly damaged stream.

Each export allocates fresh, contiguous Torch base tensors. Python verifies
shape, dtype, exact logical storage bytes, zero storage offset, alignment and
nonoverlapping intervals without reading their contents. HIP independently
checks actual device/type/allocation ranges and private-arena overlap. These
checks serve different purposes: a HIP allocator block can exceed a tensor's
logical storage. Raw pointer access remains a trusted same-process interface;
concurrent frees or writes violate its ownership contract.

The adapter drains the Torch device before HIP writes. The synchronous native
export drains its private stream before returning. The adapter then constructs
auxiliary labels on the current Torch stream and drains the device again before
publication. `records.append_to(replay)` drains prior ring work, performs
validated admission, and drains the completed source reads before returning.
Publication and admission completion fences also run in `finally` when
auxiliary allocation or admission fails after queued work has begun.
This permits export on stream A, admission on stream B, then immediate source
retirement. Direct `replay.append` continues to require the existing ordered
stream contract. Concurrent manipulation of the same replay remains unsupported.
These conservative barriers deliberately trade overlap for explicit ownership.
Torch documents the [raw tensor address](https://docs.pytorch.org/docs/2.10/generated/torch.Tensor.data_ptr.html)
and [device synchronization](https://docs.pytorch.org/docs/2.10/generated/torch.cuda.synchronize.html);
HIP defines [stream synchronization](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/reference/hip_runtime_api/modules/stream_management.html).

## Copy evidence and validation

The production binding does not bind or call the rules download API. Root/history
upload is explicit setup. Export transfers only status/summary telemetry to the
host; `CompactReplay.append` also reads scalar validation reductions. Therefore
this synchronous path does not claim zero **all** D2H bytes. Payload includes
boards, metadata, sparse legal support, hashes and per-row labels/identity;
those arrays remain resident during publication, admission and sampling.

The test suite wraps export/admission with a TorchDispatchMode that rejects
device-to-CPU tensor conversions while permitting declared scalar control.
Sampling/loss/gradient checks use a stricter mode that also rejects scalar host
reads. Deliberate `.cpu()` and `.item()` violations demonstrate those checks
fail. Native HIP copies are outside Torch dispatch visibility; source inspection
and separately instrumented native traces are needed to substantiate that part
of the path. Independent oracle downloads happen only in the explicitly marked
verification region after producer completion.

Run CPU/schema tests and actual binding tests separately:

```sh
python -m unittest neural.tests.test_bridge -v
CHEN_BRIDGE_TEST_GPU=1 CHEN_GPU_LIBRARY=/path/to/libchen_gpu.so \
  python -m unittest neural.tests.test_bridge -v
python -m neural.bridge --steps 4 --batch-size 32 --seed 20260930 \
  --producer-revision MAIN_COMMIT --gpu-source-root /path/to/gpu \
  --output results/neural/bridge-gfx1030.json
```

The smoke uploads one initial root, expands 20 first-ply and 400 second-ply
children, exports the second-ply records, admits them, closes the arena and
trains a width-16/depth-1 relation model for four steps. It reports setup,
resident expansion, allocation/export/label barriers, validated admission and
training separately. Source/library hashes and producer revision are recorded;
source hashes are necessary when the producer revision has dirty export changes.
Diagnostics download only scalar losses and sampled sequence IDs after the
measured region. A single un-warmed bounded run supports mechanics validation,
not throughput or strength claims.

## Captured execution

All 14 tests passed on actual gfx1030 in 3.593 seconds, including seven CPU
schema/negative tests and seven hardware tests. The separate CPU run skips the
seven hardware tests. Captured [GPU test log](../neural/evidence/bridge-tests-gfx1030-2026-09-30.log),
[CPU test log](../neural/evidence/bridge-tests-cpu-2026-09-30.log),
[smoke report](../neural/evidence/bridge-smoke-gfx1030-2026-09-30.json) and
[smoke console log](../neural/evidence/bridge-smoke-gfx1030-2026-09-30.log)
retain exact commands/provenance through the accompanying source manifest.
The run used Torch 2.10.0+rocm7.0 / Torch HIP 7.0.51831; the producer was compiled
with the separately installed HIP compiler and bound to Torch's verified runtime.
The producer report names main commit `45f3fb6` plus hashes of its dirty export
sources and the compiled library. Python source hashes identify the binding,
decoder, models and trainer used in that run.

The 400-row replay payload was 320,400 bytes. The un-warmed four-step run
reported 2.17 ms initial root ingress, 1.24 ms resident two-level expansion,
9.60 ms output allocation/export/label publication, 309 ms replay
allocation/admission, and 1.012 s sampling/decode/training. These startup-dominated
phase measurements include barriers and initial kernel/optimizer work. The
exporter's narrower HIP event span was 0.238 ms and excludes pointer inspection
and host telemetry. Their scopes differ; these values do not establish producer
throughput. Total toy loss was finite, 4.270 initially and 4.168 finally, without
any strength inference.

The hardware tests cover exact sparse legal support, both mover orientations,
castles, all promotion variants, EP/pinned EP, check evasion, mate/stalemate and
fifty-move successors; they preserve bit 63 and saturated uint32 fullmove values.
They compare the 20 first-ply and 400 second-ply source transitions, permit
expanded successors, reject stale selectors with numerically reused IDs, and
check native stale preflight leaves output unchanged. A real knight-cycle
fixture starts at ply six with six retained previous hashes, expands a non-root
parent at ply seven, and exports the ply-eight third-repetition successor.
All three models match independently packed feature/support loss and gradients,
and one optimizer step changes parameters. Side-stream retirement and injected
partial publication/admission failures check completion on the failure paths.

The matching native library has a separate
[export copy diagnostic](../../chen/gpu/evidence/replay-gfx1030-2026-09-30.json) and
[raw diagnostic log](../../chen/gpu/evidence/replay-gfx1030-2026-09-30.jsonl) in the engine
checkout. Nine marked successful C export calls each transferred two 16-byte
Summary records to the host (32 bytes total), with zero H2D/D2D bytes through
the two interposed APIs, `hipMemcpy` and `hipMemcpyAsync`. A deliberate 128-byte
payload download through each API was detected (256 bytes total). This supports
absence of state payload download within those C calls. It does not trace
Torch replay admission/sampling or runtime-internal/other transfer APIs, and
the inspection region is excluded. Both reports use library SHA-256
`333e3427c23c9de726387e62cc85f28bf20f8af27d8d8abc48882e586c101444`
and producer source hashes matching this binding's smoke. The complete path
therefore has scoped native copy evidence plus Torch dispatch/source evidence,
rather than a universal zero-byte trace claim.

The final combined actual-GPU run passed all 62 neural tests without skips
([combined log](../neural/evidence/combined-bridge-tests-gfx1030-2026-09-30.txt)).
The main checkout records the source-identical wrapper build and the six Rust
hardware tests rerun after this extension in
[the integration report](../../chen/gpu/evidence/bridge-integration-2026-09-30.json).
Its separately compiled library has its own fingerprint; source identity does
not imply byte-identical build artifacts.
