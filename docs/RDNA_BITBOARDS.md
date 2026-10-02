# gfx1030 bitboards: measured directions

The first isolated study favors branchless Kogge–Stone rook attacks over the
current style of walking rays. Keep ordinary 64-bit popcount intrinsics. Use
three-stage SWAR transpose when an 8×8 board transpose is actually required.
One wave per board is a poor default for this attack workload, even though it
successfully scalarizes the board occupancy. These primitive measurements need
the integrated comparison below to determine whether they justify adoption.

The controlled production comparison below is now complete: rook/bishop/queen
fill is correct, but the six-position search study shows **no meaningful whole
engine gain**. Retain the current default while optimizing frontier selection.

## Hardware sources and the neighboring project

Read AMD's [RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/),
[architecture slides](https://gpuopen.com/wp-content/uploads/2020/11/RDNA_Architecture_public-1.pdf),
and [machine-readable ISA](https://gpuopen.com/machine-readable-isa/).
The August 2026 official XML bundle includes RDNA2. The PDF RDNA2 ISA endpoint
failed during retrieval; the actual RDNA2 instruction definitions were read
from AMD's XML instead. Source hashes and the relevant instruction descriptions
are preserved in the evidence, rather than silently substituting RDNA3.

AMD describes native wave32, a 32-bit banked LDS organization, and scalarization
through read-first-lane. The architecture slides explain how independent
instructions and resident waves cover dependency latency. The measured code
objects here are wave32; their reported VGPR/SGPR counts are compiler metadata,
not measured hardware occupancy. [AMD performance guide](https://gpuopen.com/learn/rdna-performance-guide/),
[AMD architecture slides](https://gpuopen.com/wp-content/uploads/2020/11/RDNA_Architecture_public-1.pdf).

`../bitmatrices`, inspected at `603d2b9a64d9eaa30b939d3e95b50325d1ec3efc`,
provided useful concrete patterns:

- `hip/live_set_scalar.h`: ballots keep uniform masks in SGPRs; split 64-bit
  words into DWORDs for independent chains; interleave accumulators.
- `hip/live_set_dpp_reduce.inc`: explicit lane movement and OR reduction,
  with cross-half-wave exchange. Its comments document compiler trouble in
  that experiment; they are not a general claim about our compiler.
- `c/transpose_layout.c`: one-time layout conversion and independent
  accumulators amortize work when the graph remains resident.

The sibling was read without modification. Its graph peeling shortcut depends
on a descending monotone sequence and cannot simply be transferred to arbitrary
chess occupancies.

## Actual experiment

Source: [HIP microbench](../gpu/bitboard_microbench.hip),
[independent runner and oracle](../scripts/rdna_bitboards.py).
Evidence: [report](../gpu/evidence/rdna-bitboards-gfx1030-2026-10-02/runs/report.json),
[runtime ISA analysis](../gpu/evidence/rdna-bitboards-gfx1030-2026-10-02/isa-analysis.json),
[runtime disassembly](../gpu/evidence/rdna-bitboards-gfx1030-2026-10-02/runtime-isa.txt).

Actual device is RX 6800, `gfx1030`, HIP warp size 32. The raw JSON field `cu=30`
contains HIP's `multiProcessorCount` verbatim; it should not be interpreted as
the card's physical CU count. Compiler version is archived. No architecture
override was needed. Each ordinary lane owns one independent board; the
cooperative candidate owns one board per wave32 and each lane tests two
destination squares against a 32 KiB between-square table.

Inputs cover all 64 origins, empty/full boards, alternating patterns, rank/file
edges, single bits, sparse eight-bit boards, and seeded random boards. Every
iteration rotates and changes the occupancy and advances the origin. Outputs
XOR-reduce the results, so invariant input work cannot be hoisted out of the
iteration loop. The independent Python oracle computes transpose bit by bit,
uses integer bit count, and walks every rook ray. It does not import production
rules or any candidate device function.

All eight variants passed complete output comparisons for 16,384 boards at
1, 7, and 64 rounds, plus 64 and 1,024 boards at 64 rounds: **401,920 GPU outputs**.
The one-round run checks each primitive directly; longer runs also check the
recurrence. This is deterministic coverage, not exhaustive occupancy coverage.
The first cooperative attempt failed because `readfirstlane` returns signed
`int`; widening its low DWORD directly sign-extended bit31. A cast back to
`uint32_t` before widening fixed it. The failed source, outputs, and trace remain
under `failed-attempt/` and are excluded from the successful results.

Timing uses three warm-up launches per variant, then 15 HIP event samples in
alternating forward/reverse variant order, with 64 evolving operations per
board per launch. Inputs and outputs remain allocated; transfers and independent
CPU checking are excluded. GPU work was serialized with the other agent's
hardware studies. Medians below include the recurrence, reduction, and dispatch;
small batches can be dominated by these costs.

| Primitive | 64 boards (µs) | 1,024 boards (µs) | 16,384 boards (µs) | VGPR / SGPR |
| --- | ---: | ---: | ---: | ---: |
| Transpose bit loop | 14.440 | 13.961 | 34.241 | 22 / 12 |
| Transpose SWAR | 11.720 | 11.680 | 15.400 | 12 / 12 |
| Popcount64 intrinsic | 10.000 | 9.920 | 11.160 | 9 / 12 |
| Popcount two DWORD intrinsics | 10.080 | 9.920 | 11.080 | 9 / 12 |
| Popcount SWAR | 12.000 | 11.841 | 15.480 | 10 / 12 |
| Rook ray walk | 64.200 | 66.321 | 77.480 | 16 / 12 |
| Rook Kogge–Stone | 13.480 | 13.320 | 32.960 | 39 / 12 |
| Rook wave32/table/ballot | 33.200 | 32.960 | 224.002 | 8 / 20 |

All eight have zero private scratch bytes in the runtime metadata. The ISA was
extracted from the **timed executable's `.hip_fatbin`**, not merely from a second
compile assumed equivalent. Popcount64 and the explicit halves have identical
normalized instructions: two chained `v_bcnt_u32_b32` operations. Manual SWAR
popcount adds instructions and loses here. RDNA2 supplies a scalar 64-bit count
instruction for truly wave-uniform masks; this is a different mapping from
independent boards. [AMD RDNA2 machine-readable ISA](https://gpuopen.com/machine-readable-isa/).

## Controlled integration: correctness passes, selection dominates

The reusable candidate is [chen_sliding_swar.inc](../gpu/chen_sliding_swar.inc).
It implements all eight directions with three propagation stages, masks file
wraps, includes the first blocker, and excludes the origin. The experiment
replaced only `sliding` in a frozen copy of the already verified sparse-backup
source; `chen_selective.inc` is byte-identical between both arms. Production
`chen_rules.hip` was not changed.

The actual candidate include passed **2,439,488 host checks**: every relevant
rook/bishop ray occupancy at every origin, with both origin occupancy bits,
plus 100,000 random queen boards and disabled-direction cases. Both actual
gfx1030 libraries then passed the existing rules oracle with **1,024 roots and
19,574 exact transitions each**, full available perft fixtures including start
depth four (197,281), history/adjudication/boundary checks, and both ignored
selective tests comparing every native search step against portable Rust.

The [controlled comparison driver](../gpu/sliding_swar_probe.py) ran 72 search
rows in alternating arm order. Every score, PV, admission counter, stop reason,
and other non-timing field matched. A separate stdlib audit verified the saved
records without importing the producer. Clean timing medians sum six fixed-work
searches, with three repetitions per arm:

| Workload | Sparse baseline (ms) | SWAR candidate (ms) | Ratio |
| --- | ---: | ---: | ---: |
| Depth cap 4, 2,000 nodes, batch 8 | 134.066 | 133.827 | 1.0018× |
| Depth cap 64, 50,000 nodes, batch 32 | 809.710 | 808.456 | 1.0016× |

Those differences are below 0.2%; three samples do not establish an engine
speedup. The separate instrumented 50,000-node traces explain the limitation:

| Kernel, 355 calls per arm | Baseline total (ms) | Candidate total (ms) |
| --- | ---: | ---: |
| `selective_choose_classical` | 559.436 | 558.335 |
| `selective_complete_classical` | 115.870 | 114.270 |
| `selective_children_classical` | 56.807 | 47.552 |

Child processing improves locally in this trace, while frontier selection takes
74.51–75.67% of GPU time. Parallel selection is the next measured target.
Instrumented durations are diagnostic and are not the clean timing comparison.
The fused child kernel stays at 120 VGPRs, with SGPRs 70→66 and private scratch
552→408 bytes. The standalone classify kernel grows from 100 to 127 VGPRs,
with scratch 408→264 bytes. Metadata is taken from the actual second code object
in each shared library: `.hip_fatbin` contains two concatenated compiler bundles.
Inspecting only its first object would inspect the frontier evaluator rather
than the rules/search kernels.

The initial profiler launch failed cleanly because `libdw.so.1` was missing.
That entire attempt remains immutable. A separate profiling-only phase used
the already established profiler dependency path and successfully checked all
12 instrumented rows against the completed clean reference; clean benchmarks
were not repeated. Both phases retired their owned process groups.

Evidence: [cold audit report](../gpu/evidence/sliding-swar-gfx1030-2026-10-02/audit-report.json),
[frozen source declaration](../gpu/evidence/sliding-swar-gfx1030-2026-10-02/source-declaration.json),
[baseline kernel statistics](../gpu/evidence/sliding-swar-gfx1030-2026-10-02/profiles-repaired/trace-baseline/fixed-work_kernel_stats.csv),
[candidate kernel statistics](../gpu/evidence/sliding-swar-gfx1030-2026-10-02/profiles-repaired/trace-candidate/fixed-work_kernel_stats.csv).
This is an experimental candidate, not a new production default or strength
claim. Root owns adoption and commits.

## Adoption directions

1. Keep the shared fill implementation as a controlled candidate. Its diagonal
   integration and full rules checks now pass; repeat its whole-engine comparison
   after parallelizing selection before changing the production default. The observed rook kernel
   ratios are 4.76×, 4.98×, and 2.35×; they are not engine speedups.
2. Track register growth inside the complete move-generation kernel. The fill
   candidate uses 39 VGPRs here versus 16 for walking. Inline integration may
   change lifetimes and spills; measure the compiled integrated kernel rather
   than projecting occupancy from the small primitive.
3. Preserve independent board lanes for broad frontiers. The wave candidate's
   scalar occupancy emits SALU shifts/XORs, but spends 32 lanes per board and
   gathers table entries. At 16,384 boards it is 6.80× slower than fill. Low
   VGPR usage alone is insufficient evidence for a faster mapping.
4. Reserve cooperative waves for genuinely shared node work, such as reductions
   over a node's children or one selected parent's group. Keep shared masks and
   indexes uniform. A parent completion thread executing serial arena traversal
   will not become parallel merely by using a 64-bit SWAR instruction.
5. Retain `__popcll`; splitting it manually does not improve this compiler's
   emitted code. Use SWAR to replace branchy bit layout operations where the
   operation is required, and avoid introducing board transposes solely because
   the primitive is fast.

Reproduce by compiling the standalone file in `cpp-rocm`:

```sh
docker cp gpu/bitboard_microbench.hip cpp-rocm:/tmp/chen-bitboard-microbench.hip
docker exec cpp-rocm hipcc -O3 --offload-arch=gfx1030 \
  /tmp/chen-bitboard-microbench.hip -o /tmp/chen-bitboard-microbench
python3 scripts/rdna_bitboards.py --output results/rdna-bitboards-new \
  --sizes 64,1024,16384 --rounds 64 --repeats 15
```

Use a fresh output directory and an exclusive GPU slot. The runner preserves
raw inputs, every output, all timing samples, SHA-256 bindings, and failures.
