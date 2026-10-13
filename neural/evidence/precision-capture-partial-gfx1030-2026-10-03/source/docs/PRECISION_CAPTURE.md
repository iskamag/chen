# Precision and dispatch controls

The closed 40-model INT8 diagnostic measured eager execution. Its hooks and GPU
event intervals include Python dispatch gaps; they do not measure register
pressure, cache hit rates or compute utilization. At batch one, the exact FP32
value-only square anchor takes about 524 microseconds versus 732 for fused INT8.
INT8's small affine gains therefore do not establish a whole-model gain.

The next diagnostic keeps eight fixed checkpoints: one paired-study seed
(20261021), each of the four anchors, and each conditioned transition. It reads
only TRAIN, without learning, calibration, quality selection or promotion. Every
precision variant is compared with the same checkpoint's exact FP32 value-only
output on all 1,864 observations or all 233 complete roots. CP/latent errors and
first-canonical action changes are numerical fidelity evidence, not strength.

| Variant | Affine operands/output | Other arithmetic |
| --- | --- | --- |
| FP32 value-only | FP32 | FP32 |
| Native fused INT8 | dynamically scaled INT8, int32 accumulation, FP32 output | FP32 |
| Torch FP16 affine | FP16 operands and output, converted back to FP32 | FP32 |
| Native FP16 affine | packed FP16 operands, FDOT2 FP32 accumulation/output/bias | FP32 |
| Torch autocast FP16 | eligible GEMMs/einsums follow Torch autocast, cache disabled | producing Torch autocast policy |

The native FP16 control uses the existing wave32 reference library and packs
weights once. Dynamic activation conversion and K-padding run on the GPU and
remain in forward timing. The Torch affine control also includes conversion;
its FP16-rounded output is a different numerical contract. Reduced precision
GEMM reductions and TF32 are disabled. The autocast control is explicitly mixed
precision, not a claim that every operation runs in FP16. The native FDOT2
instruction matches AMD's documented packed FP16 dot operation in the
[RDNA2 ISA guide](https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf).
Instruction throughput is only one part of the measured forward cost.

For batch 1, 32 and 256 anchors and one structurally selected complete-root
transition shape, each variant has three measured routes: eager, fixed-input
graph replay, and resident input-copy plus graph replay. Balanced forward/reverse
rounds record device event intervals and synchronized wall time. Graph replay
alone excludes input-copy cost; the third route includes GPU feature/action
copies. Model conversion and graph construction are separate setup costs. No
CPU features are reconstructed during inference. Capture is required and fails
closed; it never substitutes eager execution. Owned captured outputs are borrowed
and overwritten on replay, so consumers must finish before replay or copy them.

Capture checks require exact parity with each variant's own eager output and
exercise both original and rotated admitted position/action rows, including a
different observation for the batch-one anchor. These checks preserve each
variant's numerical meaning; FP16 and INT8 still have separate errors against
FP32. All source, native library, actual HIP provider and checkpoint pins belong
to a fresh declaration with a 600-second inner deadline and 660-second owned
watchdog. GPU-hidden CPU layout tests establish packing/admission behavior only.

The hardware agent separately compares physical wave32/wave64 mapping and records
VGPR/SGPR, LDS, scratch/spills and occupancy metadata. A register count is a
resource constraint, not proof of a bottleneck; dispatch traces and fixed-work
timings must explain the effect before adoption. New wave64 libraries require
their own matching layout wrapper. This diagnostic retains N-padding of 32 and
does not interchange those libraries.

If graph capture removes most of the difference, optimize launch ownership and
fusion first. If captured affine precision helps, retain the winning numerical
variant and measure producer + network + search together. If exact diagram and
attention work dominate, profile those operations before tuning GEMMs further.
No result here changes the selected-NONE quality gate or deploys a model.
