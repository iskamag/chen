# Tactical edge-padding intervention

The native tactical observation uses conservative E=6144 capacity. The recurrent
Torch cell projects nodes once, but gathers source/receiver features, embeds
relations and computes messages/reductions across **every padded edge slot**.
Masking inactive messages preserves semantics; it does not avoid their work.

A same-checkpoint intervention on actual gfx1030 holds piece/action capacities,
all active graph/features, recurrence and precision fixed. It removes only the
trailing inactive edge slots. Observations are the complete20 native children
of standard initial chess; single-row runs use the first child. Active E is263
for that row, at most270 across the20 rows (5261 active edges total), versus
6144 per row processed by the conservative layout. No labels/optimizer are used.
Random nonzero heads expose processor differences rather than hiding them behind
zero-initial residuals. Full/tight predictions and each captured/eager pair pass
rtol1e-5/atol1e-4; canonical chosen IDs and masks match exactly.

## Measured inference

FP32, shared processor steps4. Each arm has three synchronized samples of three
calls, after warmup. Table shows median GPU event time in milliseconds; ratios
are full/tight. Raw event and synchronized-wall samples are retained in the
[evidence packet](../gpu/evidence/tactical-padding-gfx1030-2026-10-03/README.md).

| Width | Batch | Eager full / tight ms | Captured full / tight ms | Captured ratio |
| --- | ---: | ---: | ---: | ---: |
| 32 | 1 | 6.902 / 6.930 | 1.337 / 0.979 | 1.37x |
| 32 | 20 | 8.070 / 7.001 | 7.489 / 1.971 | 3.80x |
| 128 | 1 | 7.103 / 6.894 | 2.088 / 1.473 | 1.42x |
| 128 | 20 | 26.508 / 7.285 | 26.129 / 4.096 | 6.38x |
| 256 | 1 | 7.029 / 6.825 | 3.139 / 1.942 | 1.62x |
| 256 | 20 | 57.220 / 9.697 | 56.807 / 9.169 | 6.20x |

This is a large observed cost of padded arithmetic at these shapes. Eager
batch-one times barely improve: dispatch costs remain visible there. It does
not establish general speed ratios, hardware occupancy/cache utilization,
learned quality or engine throughput. Clocks, shared power and concurrent
CPU-only source/proof/dataset preparation were uncontrolled. Inputs/checkpoints
are untrained mechanics diagnostics on one reused opening's child set.

## Implementation direction

This intervention reads slot counts and creates tight contiguous copies **outside
inference timing**. The native ABI1 emitter still admits only E>=4992, and the
producer still allocates/synchronizes snapshots. Export, graph construction,
copy, checkpoint setup and capture costs are excluded. Therefore the measured
layout is not yet a deployable native compaction or complete producer/search
speedup. Offline learning can collate exact graphs at actual per-batch capacities
before upload and must disclose that boundary.

Next native candidate should export directly into bounded small edge buckets,
with device capacity checking, explicit failed-packet discard and complete retry
rather than truncated success. Scheduling/capture can then use the admitted
bucket shape without downloading boards or edge payloads. Compare the entire
producer/inference path and preserve conservative E6144 as a named control.
A count/shape admission or retry has a cost; report it rather than importing the
inference-only ratios as the pipeline result. Keep larger learned widths eligible.
