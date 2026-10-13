# Precision and dispatch controls

The closed 40-model INT8 diagnostic measured eager execution. Its hooks and GPU
event intervals include Python dispatch gaps; they do not measure register
pressure, cache hit rates or compute utilization. At batch one, the exact FP32
value-only square anchor takes about 524 microseconds versus 732 for fused INT8.
INT8's small affine gains therefore do not establish a whole-model gain.

The completed diagnostic covers eight fixed checkpoints: one paired-study seed
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

The first attempt is [closed with partial evidence](../neural/evidence/precision-capture-partial-gfx1030-2026-10-03/analysis.json).
Square/20261021 completed all five variants; relation capture then failed because
`ChessLatent.attack_edges` constructs a device tensor from a Python constant
inside forward. The attempt stopped at 22.104 seconds, with no retry or final
completion record. At batch one, square FP32 eager/replay wall medians are
560.0/154.7 microseconds and INT8 740.4/128.6. FP32 replay wins at batch 32 and
256; both FP16 affine routes and autocast lose here. Input-copy timings are
separate noisy observations and must not be subtracted to estimate copy cost.
Fifteen graphs pass exact own-eager output checks with changed admitted inputs.
Cold setup spans 3.62–9.45 ms including warmups, synchronization and capture.
No transition completed, so these results establish no action-choice fidelity.

The scoped repair stores that exact sliding mask in a nonpersistent module
buffer, which moves with the model; forward only selects its input dtype. CPU
fixtures require bit-exact blocked edges and encoder outputs against the actual
old archived source in FP32 and FP64, and strict old/new checkpoint schemas are
identical. A fresh continuation declaration admits exactly seven unfinished
keys and binds the failed declaration, terminal and completed square file by
fixed hashes. It excludes square/20261021 and permits no generic skip/resume.
No training, numerical-policy or timing configuration changes. The failed
source and partial record remain immutable; continuation was reviewed separately.

The separately reviewed seven-key continuation completed at 84.459 seconds
(86.262-second watchdog, exit 0), without replaying the square anchor. Joining
the two pinned records gives 80 graphs and 160 exact own-eager checks, including
80 changed-feature and 20 changed-action checks. This is one seed and eight
structural representatives, not a rerun of the 40-model study.

| Anchor, batch one | FP32 eager/replay wall µs | INT8 replay µs | Native FP16 replay µs |
| --- | --- | --- | --- |
| Square | 560.0 / 154.7 | 128.6 | 170.2 |
| Relation | 890.5 / 230.6 | 214.2 | 259.8 |
| Plan | 1236.0 / 299.7 | 274.3 | 337.9 |
| Threat | 1077.9 / 271.5 | 263.3 | 311.0 |

INT8 captured batch-one anchors beat FP32 by 1.03–1.20×, reversing the eager
result. FP32 wins all batch-32 and batch-256 anchors. Conditioned complete-root
FP32/INT8 replay ratios are 0.985–1.001, effectively tied in these observations.
Both FP16 affine controls and autocast lose to FP32 replay in all 16 contexts.
Cold capture setup spans 3.62–19.65 ms and includes three warmups and fences.
Capture removes repeated host dispatch and allocation from replay, but this
comparison does not isolate one launch cost or certify a cache/register cause.

Native FP16 preserves all 932 same-FP32 conditioned model-root choices, with
transition mean absolute CP errors 0.0114–0.0363 and maximum 0.1497. INT8 changes
23 of those choices; the largest changed FP32 best-versus-second gap is 0.4783
CP. The four models reuse the same 233 roots, so 932 is not an independent-game
sample. FP16 is a useful fidelity control here, not the fastest model variant.
Keep captured FP32 as the next integration reference, retain INT8 as a separately
measured small-batch option, and measure producer + inference + search before
choosing a deployment route. Full-GPU and symbolic residual integration remain
required; no quality gate or deployment admission changes.
