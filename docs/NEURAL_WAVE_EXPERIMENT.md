# gfx1030 neural wave32 / wave64 experiment

This is a separate native affine experiment. Production kernels, neural models,
checkpoints, and archived studies remain unchanged. The completed native study
passes numerical checks for both physical wave widths. It supports retaining
wave32 as the default; wave64 improves one scalar dynamic readout shape but
loses on most tested vector affine shapes. This is operator evidence, not a
whole-model or engine-strength result.

`gpu/experiments/neural_wave_stage.py` freezes both existing affine sources and
generates two full-wave mappings. Both variants use one **physical** wave per
input row/output-channel tile. For wave64 the weight stride rounds N up to 64,
the row/lane indices use 64, and reductions combine both halves explicitly.
Simply applying the compiler flag to production source is incorrect:
its `readfirstlane` would broadcast one logical32-row across two logical groups
in a physical 64 wave. N32 and N96 expose missing or duplicate output ownership.

A second trap appears in installed HIP shuffle headers: `__shfl_xor/down`
lowers directly to `ds_bpermute`, which the
[RDNA2 ISA manual, section10.4.4](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf)
limits to independent 32-lane halves even in wave64 mode. V4 reduces within
each half and merges lane0/lane32 using explicit `v_readlane_b32`. V3's full64
shuffle reduction was rejected by source/ISA inspection before execution.

The scalar N1 mapping partitions K across the physical wave. The dynamic rows
API retains sourceK tails and zero padding, zero-row scale1, nonzero absmax
times the rounded FP32 reciprocal127, FP32 tiny-scale clamp, round-to-even,
signed symmetric127 quantization, and FP32 bias/residual/output. Static INT8
uses four integer accumulators. FP16 uses packed `v_dot2_f32_f16` with FP32
accumulation and output. No CPU fallback is permitted.

The separate libraries retain original symbol names. Their packed layout is
part of the experimental caller contract; a wave64 library must **not** be
loaded into an existing wrapper that allocates Npad32. `chen_wave_resources`
is an additional experimental admission ABI. It queries actual kernel
attributes and occupancy for a requested block size before execution.

## Initial compiler evidence

Frozen candidate closure: `/tmp/chen-neural-wave-v4-2026-10-03` in `cpp-rocm`.
Builds use installed HIP7.2.4/clang22, gfx1030, `-O3 -shared -fPIC` and explicit
`-mno-wavefrontsize64` or `-mwavefrontsize64`. Compilation runs on CPU2;
code-object extraction and disassembly do not initialize the GPU.

| Kernel | Wave32 VGPR/SGPR | Wave64 VGPR/SGPR |
|---|---:|---:|
| INT8 packed vector | 16/33 | 16/33 |
| INT8 row-major vector | 12/26 | 12/26 |
| INT8 packed scalar | 11/22 | 11/22 |
| Dynamic INT8 vector | 17/28 | 17/29 |
| Dynamic INT8 scalar | 19/28 | 19/29 |
| FP16 half2 affine | 12/21 | 12/22 |
| FP32 affine | 12/21 | 12/22 |

Actual `.wavefront_size` is32 or64 respectively. All kernels have zero fixed
LDS, private segment and VGPR/SGPR spills. These numbers do not establish
achieved occupancy, memory latency or a speed advantage. The wider mapping
wastes output lanes for N32, whereas N64 can halve row/tile wave count.

The first staging attempt failed before reading any source because its repo
path was incorrect; the empty directory is retained. V2 compiled without the
resource-query ABI, and V3 adds that ABI without changing affine arithmetic.
V3 with its unsafe full64 shuffle is also retained as an unrun rejection.
Those preparation versions are not runtime results.

## Bounded numerical and timing probe

`gpu/experiments/neural_wave_probe.py` binds generated-source hashes, libraries,
its own source and actual loaded HIP provider. The compiler version and Torch
runtime are distinct: Torch may resolve the native library's HIP SONAME to its
own runtime. Runtime path/hash, `hipRuntimeGetVersion` and device properties
must be recorded by the executed probe, not inferred from hipcc.

Fifteen shapes include odd M, N1/31/32/33/63/64/65/96, sourceK18/20/22/45 tails,
K32/64/160/256, and representative M64/512/2048/8192/16384. Each variant is
checked against an independent CPU int64 dot and explicit FP32 scale/bias;
dynamic cases include half ties, zero, tiny and subnormal rows. Output buffers
start as NaN to expose incomplete ownership. Packed weights are independently
reconstructed, including channel padding. Residual is checked explicitly.

Parity precedes timing. The default9 balanced samples shuffle both widths and
methods in each sample. Eager events include repeated Python/ctypes dispatch
gaps. Captured24-dispatch replay provides a separate kernel-oriented timing
control, and synchronized wall latency is retained for both. Static activation
and weight preparation is excluded; fused dynamic quantization remains inside
its measured kernel. Graph capture and one-time packing are outside replay.
There is no whole-model or chess-strength inference from these operators.

## Counter admission

Installed `/opt/rocm/share/rocprofiler-sdk/counter_defs.yaml` explicitly defines
these metrics for gfx1030. Runtime enumeration and six separate collection
passes completed, but some collected metrics fail sanity checks (below):

| Counter/metric | Meaning or limitation |
|---|---|
| `GL2C_HIT`, `GL2C_MISS` | L2 hits/misses; misses include uncached reads |
| `L2CacheHit` | `100*sum(HIT)/(sum(HIT)+sum(MISS))` |
| `SQ_WAVES` | Started waves, returned per shader engine |
| `SQ_WAVE32_INSTS`, `SQ_WAVE64_INSTS` | Issued wave-width instruction counts; definitions label emulated |
| `SQ_INSTS_VALU`, `SQ_INSTS_SALU` | Issued vector/scalar ALU instructions |
| `SQ_LEVEL_WAVES` | Per-SIMD wave level requiring accumulation configuration |
| `MeanOccupancyPerCU` | Accumulated wave-level / active cycles / CU count |
| `OccupancyPercent` | Wave cycles / active cycles / CU count /32, expressed as percent |
| `SQ_WAIT_ANY`, `SQ_WAIT_INST_ANY` | Wave dependency/issue waiting counters |

Use actual `rocprofv3 --list-avail` after exclusive reservation, then separate
small PMC sets. A definition is not evidence that collection works with this
driver, runtime or permissions. Avoid asking for too many counters in one pass.
Do not substitute gfx9 TCC/TCP names for gfx1030 GL2C. No matching GL1/TCP hit
metric was found in the installed gfx1030 definitions; L1 hit-rate remains
unmeasured. Counter runs are separate from clean alternating timing, since
profiling may alter dispatch behavior. Zero cache denominator must be reported
as unavailable. Normalize instruction counts per output and account for the
counter's wave32-equivalent counting semantics before comparing widths.

AMD's [RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/)
describes register occupancy and wave-size tradeoffs; installed code objects and
actual measurements determine this candidate's outcome. AMD's
[rocprofiler SDK](https://github.com/ROCm/rocprofiler-sdk) is the counter source.

## Executed results

Root released the GPU after the one-second strength match and CPU Coz window
retired. The main probe completed with exit0: 15 shapes, **887,777 outputs per
operator path**, eight paths per shape, and nine balanced timing samples for
each eager/captured method. Packed-layout, ownership, independent CPU integer,
dynamic quantization policy, residual and FP16/FP32 reference checks pass.

The compiler is HIP7.2.4; the process actually resolves HIP to Torch2.10's
ROCm7.0 provider. `hipRuntimeGetVersion=70051831`, loaded provider
`/usr/local/lib/python3.12/dist-packages/torch/lib/libamdhip64.so`, SHA256
`c13a873d6444aa9c32d846feaa888de2108a5ed3c15f8a1b7708c5742a894e81`.
Both widths query16 active128-thread blocks per reported multiprocessor for
all tested kernels. This is the API admission limit, not achieved occupancy.

Median captured event latency in microseconds, with24 native dispatches per
capture replay:

| M / N / sourceK | Method | Wave32 | Wave64 |
|---|---|---:|---:|
| 64 /32 /20 | Static INT8 +residual | 3.63 | 3.66 |
| 64 /32 /20 | Dynamic INT8 +residual | 5.34 | 5.40 |
| 64 /32 /20 | FP16, FP32 accumulation/output | 4.38 | 4.38 |
| 64 /64 /160 | Dynamic INT8 +residual | 17.91 | 18.52 |
| 64 /64 /160 | FP16 | 12.82 | 13.09 |
| 32 /1 /256 | Dynamic INT8 +residual | 5.36 | 4.56 |
| 8192 /32 /22 | Dynamic INT8 +residual | 27.20 | 43.55 |
| 8192 /32 /22 | FP16 | 10.26 | 11.18 |
| 16384 /32 /32 | Static INT8 +residual | 14.89 | 16.60 |
| 16384 /32 /32 | Dynamic INT8 +residual | 54.48 | 102.81 |
| 16384 /32 /32 | FP16 | 22.86 | 27.19 |

The scalar dynamic readout has a paired median wave32/wave64 speedup near1.18;
it remains a separate shape-specific candidate. In the largest object shape,
wave64 dynamic quantization approaches twice the latency. This physical64
mapping uses twice as many threads for N32 while only half the output lanes
store. FP16 here starts from preconverted half activations and omits residual,
while dynamic INT8 starts from FP32 and includes quantization/residual. These
operator times do not establish an end-to-end precision replacement gain.
Prequantized static INT8 also excludes activation quantization.
No broad wave64 switch is justified. The archive retains eager and synchronized
wall samples as well as captured values, and the audit recomputes paired ratios.

### Counter results and limits

Runtime `--list-avail` advertises the declared counters. All six processes
completed exit0 for M8192/N32/sourceK22, with three separate PMC groups per
width and filtered native dynamic/FP16/FP32 kernels. This collection status does
**not** establish correct counter semantics:

* `SQ_WAVE32_INSTS` is zero for the verified physical32 code objects.
* `SQ_WAVE64_INSTS` is nonzero for physical32 but zero for physical64.
* `SQ_WAVES` medians near10,924 disagree with the exact8,192 waves launched.
* `MeanOccupancyPerCU` contains impossible values above1,000 and4,800.

These values are preserved and rejected as evidence for achieved occupancy or
instruction-width utilization. No automatic retry or metric substitution was
performed. API occupancy limits and ISA wave width are separate verified facts.
Profiler VGPR counts also reflect allocation granularity (dynamic24 for wave32
and20 for wave64), whereas metadata's17 is the compiler's used-register count;
the profiler reports SGPR128 for both. Keep those representations separate.

GL2C hit/miss counts are nonzero and repeatable. For dynamic affine, median
observed hits/misses are9,577.5/22,025 at wave32 and11,455.5/22,031 at wave64.
They remain raw profiling observations with counter validity limits; the wider
mapping's extra hits do not establish a cache bottleneck or compensate for its
clean measured latency loss. No L1 hit-rate claim is made.

All owned probe/rocprof processes terminated; an exact-argv `/proc` scan found
none remaining. GPU ownership was explicitly released before another study.
Preparation and execution archives are separate; the runtime archive binds the
preparation manifest rather than duplicating source/ISA payloads.

`gpu/experiments/neural_wave_audit.py` replays the compressed runtime archive
using only stdlib, validates library hashes/runtime admission/sample counts,
recomputes timing ratios, and reproduces the counter sanity failures. Its
terminal result is `PASS_WITH_COUNTER_LIMITATIONS`.

## Coz scope

Host `/usr/bin/coz` is0.2.5. Its CLI supports causal line experiments and
end-to-end measurement; sampling `--progress` is explicitly unsupported in
this version. Coz acts on sampled host execution and progress instrumentation,
not GPU ISA. Use it for CPU search or host orchestration boundaries, with a
defined completed-work progress point. GPU cache and occupancy claims require
device counters or a matched implementation experiment. The
[upstream Coz documentation](https://github.com/plasma-umass/coz) explains the
virtual-speedup/progress-point method. Root owns the separate Coz/strength work;
this experiment does not launch a competing profiler.
