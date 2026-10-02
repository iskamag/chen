# gfx1030 mixed INT8 affine operator

`gpu/chen_int8_linear.hip` is an experimental, allocation-free HIP C ABI. It
keeps integer dot products, FP32 scales/bias/output and an optional FP32 residual
on the caller's current stream. It does not select or deploy a trained model.
The separate neural wrapper owns tensor admission, lifetime and checkpoint
binding. Attention, normalization and the symbolic stream remain wider values.

AMD's [RDNA performance guide](https://gpuopen.com/learn/rdna-performance-guide/)
motivates wave32 mapping, uniform scalar loads and resource inspection. The
[official arithmetic builtin guide](https://rocm-handbook.amd.com/projects/amd-rocm-optimization-guide/en/latest/compiler-builtins/cross-arch/arithmetic-builtins.html)
describes packed signed-byte dot products and FP16 dot products with FP32
accumulation. The actual gfx1030 object emits `v_dot4c_i32_i8`; the FP16 control
emits `v_dot2_f32_f16`. This is RDNA2 dot-product code, with no WMMA/MFMA claim.
Four products in an instruction do not establish a fourfold engine speedup.

## ABI and layout

All functions return `int32_t(hipError_t)`. Dimensions are `uint64_t`; the final
argument is `hipStream_t`. The native layer rejects unsupported dimensions or
misaligned pointers before launch; Python must check the actual storage lengths,
device, dtype, contiguity, finite ingress and lifetime. It performs no CPU
fallback, allocations, synchronization or hidden default-stream work.

```cpp
chen_int8_linear(A, W, row_scale, weight_scale, bias, Y, M, N, Kpad, stream);
chen_int8_pack_weights(W, P, N, Kpad, stream);
chen_int8_linear_packed(A, P, row_scale, weight_scale, bias, Y, M, N, Kpad, stream);
chen_int8_linear_packed_residual(A, P, row_scale, weight_scale, bias,
                               residual, Y, M, N, Kpad, stream);
chen_int8_linear_dynamic_packed(A_float, P, weight_scale, bias, residual,
                                Y, M, N, Kpad, stream);
chen_int8_linear_dynamic_packed_rows(A_float, P, weight_scale, bias, residual,
                                     Y, M, N, sourceK, Kpad, stream);
```

The static input is INT8 `[M,Kpad]`, the row-major weight is INT8 `[N,Kpad]`,
and cached weights are INT32 `[Kpad/4,Npad]`, where `Npad=round_up_32(N)`.
The packer stores zero padding channels. Bias may be null; dynamic/residual
entry points also allow a null residual. Scales and output are FP32. Every
non-null pointer needs four-byte alignment. Limits are `M<=1048576`,
`1<=N<=4096`, and `4<=Kpad<=4096` divisible by four. The integer worst-case
sum fits INT32 throughout these limits.

The padded dynamic API accepts FP32 `[M,Kpad]` with zero K padding. The rows API
accepts contiguous FP32 `[M,sourceK]`, requires `Kpad=round_up_4(sourceK)`, and
reads tail elements as zero internally. It avoids an extra FP32 padding tensor
or dispatch for input widths such as 18, 22 and 45.

Dynamic quantization matches the separate neural `symmetric_rows` policy:
zero row scale one; otherwise `maxabs * FP32(1/127)`; clamp to FP32's smallest
positive normal value; true division by that scale; round to nearest, ties to
even; clamp to `[-127,127]`. The scalar reciprocal ordering matters. PyTorch's
[v2.10 division implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/native/cuda/BinaryDivTrueKernel.cu)
uses reciprocal multiplication for division by a CPU scalar. An initial strict
`maxabs/127` candidate failed at quantization half ties; the archive retains its
source, binary and partial results. A previously passing floor-1e-8 policy is
also retained as a different diagnostic variant, never relabeled as this policy.

## Executed validation and timing

The archive is
[`gpu/evidence/int8-affine-gfx1030-2026-10-02`](../gpu/evidence/int8-affine-gfx1030-2026-10-02).
Final native source SHA256 is
`7154af3253f5bf430cbdf6f987be9dbf3755ffc915ff2028a92da2f110ee1a44`;
shared-library SHA256 is
`4429e400dea73f6eb05275068dee9030e36744b0991541221a95360ae4b1db3b`.

Sixteen matrix shapes produced 674,337 outputs per checked path against an
independent CPU INT64 dot/dequantization oracle. Checks cover signed-byte edges,
zero rows, odd N, K tails, K=4096, static and packed weights, FP32 and packed-FP16
controls with FP32 accumulation, dynamic policy and FP32 residuals. Explicit
half-tie, tiny and subnormal input cases had zero maximum error. The final
runtime-admitted repeat passed all checks. Integer affine correctness establishes
neither whole-model quantization fidelity nor chess strength.

Representative final median event intervals in microseconds:

| M × N × sourceK | Prequantized INT8 | Fused dynamic INT8, no pad | FP16 control | Torch FP32 | Torch dynamic + INT8 |
|---|---:|---:|---:|---:|---:|
| 64 × 32 × 20 | 4.14 | 5.71 | 5.01 | 19.82 | 99.51 |
| 64 × 64 × 32 | 4.18 | 6.76 | 5.60 | 19.47 | 98.53 |
| 64 × 32 × 64 | 4.47 | 9.60 | 7.67 | 19.68 | 99.19 |
| 64 × 64 × 160 | 5.33 | 17.93 | 13.32 | 19.79 | 100.20 |
| 512 × 32 × 45 | 4.34 | 8.07 | 6.74 | 19.98 | 101.51 |
| 16384 × 32 × 32 | 12.16 | 52.97 | 27.13 | 34.66 | 104.08 |

These are resident affine calls with Python/ctypes launch gaps inside event
intervals. Five balanced-order samples contain 24 calls per method. Host
affinity is CPU2; the final repeat followed the other agent's CPU archive work,
but machine-wide power/cache remain uncontrolled. Weight packing and static
input preparation are excluded. The FP16 control receives prepared FP16 inputs
and emits FP32; it is not a fused FP32-to-FP16 whole-forward result. Torch FP16
matmul emits FP16 and converts to FP32, so it is a separate precision control.
The Torch dynamic column includes row reduction, scale selection/clamp,
division, rounding, conversion and dispatches. The native dynamic column fuses
those operations into its affine kernel. The rows API requires no input padding.

A separate rocprof trace binds actual kernel spans to dispatch/grid/resource
metadata; its timings are not mixed into the table. Cached packed weights do
not consistently beat row-major weights at these small widths. Fused dynamic
INT8 beats Torch affine dispatch for small matrices but loses to wider routes
at larger batches. Keep optimized FP16/FP32 routes where faster or numerically
needed; do not choose a format from memory savings alone. Whole semantic-forward
latency and TRAIN-only quantization fidelity need the separately frozen neural
diagnostic before adoption. No model was promoted by this operator study.

`torch._int_mm` was actually attempted and failed with
`HIPBLAS_STATUS_INVALID_VALUE`; this Torch package lacks its
`TensileLibrary_lazy_gfx1030.dat`. That provider failure is preserved. The native
operator remains GPU-only, with no replacement CPU inference route.

## Runtime and scheduling admission

The compiler was HIP7.2.4 / clang22, but the executed Torch2.10.0+ROCm7.0 process
resolved the native library to Torch's HIP provider. `hipRuntimeGetVersion`
returned `70051831`; `dladdr` and `/proc/self/maps` agreed on
`/usr/local/lib/python3.12/dist-packages/torch/lib/libamdhip64.so`, SHA256
`c13a873d6444aa9c32d846feaa888de2108a5ed3c15f8a1b7708c5742a894e81`.
The executed device properties show gfx1030, major10/minor3 and reported
multiprocessor count30. Interoperability is supported by these actual tests,
not by assuming the compiler and Torch runtimes are identical.

A separate native metadata query reported `cooperativeLaunch=1`,
`cooperativeMultiDeviceLaunch=1`, warp32, maxThreadsPerMultiProcessor2048,
maxBlocksPerMultiProcessor64 and regsPerMultiprocessor131072. That standalone
query used system HIP7.2.4. Occupancy API estimates for the earlier v2 kernels
were 32/16/8 blocks per reported multiprocessor at block sizes64/128/256.
These raw reported fields are not physical CU-count conversions, and no
cooperative launch, device-wide barrier or spin barrier was tested.

Final static row-major/packed kernels use 12/16 VGPRs and 26/33 SGPRs; dynamic
vector/scalar kernels use 17/19 VGPRs and 28 SGPRs. All report wave32, zero
scratch, zero LDS and zero register spills. Actual uniform activation loads,
four independent INT32 accumulators and packed DOT4 instructions are retained
in disassembly. These counts do not themselves prove throughput or occupancy
on a future combined engine kernel.

## Reproduction

Compile each frozen HIP source with the archived flags and gfx1030 target,
then run the frozen probe inside the Torch/ROCm container on an exclusively
reserved device. Its output path must be new. The CPU-only archive audit is:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 gpu/int8_affine_audit.py \
  gpu/evidence/int8-affine-gfx1030-2026-10-02
```

The audit verifies payload hashes, compressed raw binaries, final source/runtime
bindings and timing aggregation. It does not rerun hardware inference.
