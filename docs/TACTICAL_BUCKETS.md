# Complete native tactical edge buckets

The named `native_tactical_bucket_v2` Torch producer exports directly into small
edge buckets. A bucket overflow publishes no completion metadata and returns a
retryable capacity failure. The wrapper discards all failed tensors and retries
a larger bucket under the same arena lock. Success always represents a complete
graph and complete legal action support; no board, edge payload, or slot-count
download chooses the bucket.

This implements the direction motivated by [the earlier padding intervention](TACTICAL_PADDING.md).
ABI1 remains the default conservative producer, with its original 192-byte sink,
version1 export entry, feature layout, and E>=4992 admission. Its frozen evidence
was not modified. The separately named API2 uses the same feature schema/layout
but permits E>=1. Default retry buckets are `(512,1024,2048,4096,6144)`; applications
must explicitly select the new producer. Captures must be keyed by admitted
batch/bucket shape; this wrapper does not automatically manage a capture cache.

## Capacity and ownership contract

`chen_tactical_bucket_abi_version()` returns 2.
`chen_rules_export_tactical_bucket()` requires a sink tagged version2 and returns:

| Status | Meaning | Ownership/action |
| --- | --- | --- |
| 0 | Complete graph; completion rows/epoch published | Owned tensors may be consumed |
| -2 | Retired edge-bucket overflow | Discard every output; retry larger disjoint storage |
| -1 | Schema/range/object/move/runtime failure, with this call retired | Discard every output; propagate failure |
| -3 | Stream retirement could not be established | Retain allocations/context/library; stop using arena; retire process |

The writer reports a distinct device edge-capacity error 7. Other object/move
capacity failures cannot be fixed by growing E and map to -1 in API2. On overflow,
tensor payloads can be partial, including masks/counts; they are never an admitted
packet. Completion metadata stays untouched on every failure. Retried success
recomputes the entire graph from the same state/history/epoch.

The new failure helper explicitly confirms stream retirement before allowing
failed allocations to be discarded. On -3, the Torch wrapper quarantines all
output tensors, the native context handle, arena and loaded library/runtime,
and closes arena admission so normal close/destruction cannot free that context.
Process-level retirement is required; retry or continued use is forbidden.
This branch was tested with a CPU fault fixture, not an injected GPU driver fault.
The successful and capacity-failure device paths retired normally in the actual
GPU run.

Native emission and Python allocation/emission/snapshot construction share helpers
with ABI1. Exact geometry, action effects, terminal authority, retained history,
source identities and snapshot ownership remain unchanged. The native API still
does no per-call device allocation; Torch allocates fresh output tensors.

## Actual whole-producer pipeline measurement

Actual gfx1030 FP32 runs use all 20 device-produced children of the standard
initial board; B1 uses the first native child. All active inputs are preserved.
Every row fits E512 with zero retries (263 active edges for B1, maximum 270 for
B20, 5261 total for B20). The same untrained checkpoint per width, with random
nonzero residual heads, is reused across arms and batches; processor steps 4.
Eager/captured/fresh-pipeline predictions pass rtol1e-5/atol1e-4 and canonical
IDs/masks match. No labels, optimizer or training are involved.

Each measured call allocates and exports a **fresh native snapshot**, performs
resident copies into the captured input buffers, replays the model, and
synchronizes completion. The table reports median pipeline wall milliseconds
from six balanced forward/reverse three-arm rounds. The ratio is the median of
paired ABI1/control-to-bucket observations, not a ratio of inference-only times
or necessarily a ratio of the displayed medians.

| Width | B | ABI1 E6144 ms | API2 E6144 ms | API2 E512 ms | Paired pipeline ratio |
| --- | ---: | ---: | ---: | ---: | ---: |
| 32 | 1 | 2.380 | 2.386 | 2.105 | 1.145x |
| 32 | 20 | 9.105 | 8.680 | 3.519 | 2.500x |
| 128 | 1 | 3.154 | 3.253 | 2.777 | 1.144x |
| 128 | 20 | 27.801 | 27.505 | 6.082 | 4.563x |
| 256 | 1 | 4.226 | 4.300 | 3.138 | 1.333x |
| 256 | 20 | 58.761 | 58.871 | 11.848 | 4.974x |

API2 E6144 is an explicit control for the API/wrapper change; all three arms use
the same pinned native binary. Export wall medians remain approximately
0.83–1.05ms at these shapes, including allocation/admission. Native preflight+
emission event medians are approximately 0.29–0.34ms. The large-batch effect comes
primarily from reducing padded model work; it does not remove export/fence costs.
B1 gains are much smaller. Retry latency is not estimated by this timing suite,
whose shapes fit the first bucket; overflow/retry is separately tested.

Reset/root expansion, model construction, checkpoint hashing, capture warmup+
setup and cold validation are **excluded from steady pipeline timing** and
reported separately in the [raw result](../gpu/evidence/tactical-buckets-gfx1030-2026-10-03/timing-results.json).
Model setup was 9.76/2.89/5.82ms at widths 32/128/256. Capture warmup+setup ranged
29–195ms across arms/shapes; at B20/width256 it was191.4ms for ABI1 E6144 and
44.1ms for buckets. These observations do not estimate an engine's startup cost
or a changing-state capture-cache hit rate.

The CPU5-affinitized semantic run completed 7 tests in 2.012 s, with 50 oracle row
comparisons (some repeated), plus sentinel/failure checks. It covers special
moves, both mover frames, all 14 relations, terminal priorities, repetition,
complete native child sets, capacity overflow, exact-E boundary, retry exhaustion,
and snapshots surviving source reset/failure. Six CPU wrapper fixtures cover
policy admission, same-source/lock retries, no failure publication, and quarantine.
Both GPU worker process groups were independently confirmed absent. No GPU job
overlapped this semantic/timing window; CPU-only preparation and clocks/shared
package power were uncontrolled. Timing is a small repeated same-opening mechanics
sample, not a learned-quality, Elo, general throughput or search-engine result.

## Replay

Native sources and the Torch wrapper live in separate worktrees. The evidence
contains their exact frozen sources, provider pins, native binary and commands.
With their matching revisions on PYTHONPATH and an actual gfx1030 ROCm runtime:

```sh
HIPCC=/opt/rocm/bin/hipcc bash gpu/build.sh /tmp/libchen_tactical_bucket.so
CHEN_GPU_LIBRARY=/tmp/libchen_tactical_bucket.so \
CHEN_GPU_LIBRARY_SHA256=<sha256-of-that-library> \
python3 -m unittest gpu.test_tactical_buckets -v
python3 gpu/profile_tactical_buckets.py \
  --library /tmp/libchen_tactical_bucket.so \
  --library-sha256 <sha256-of-that-library> --output /tmp/bucket-profile --pairs 6
python3 gpu/analyze_tactical_buckets.py /tmp/bucket-profile/results.json
```

Use fresh output directories and an exclusive bounded GPU window. The archived
supervisor retains the exact 120 s + 5 s watchdog, source/binary checks and process
retirement records used for the observed run. Native binary SHA256:
`bde1bac9bf8cb052b5d056765329c946a68f1a52ac0a90de63b172e221ff3c05`.
