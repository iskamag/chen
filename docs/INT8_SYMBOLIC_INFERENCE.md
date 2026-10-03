# INT8 and symbolic residual inference

The new candidate is separate from the frozen semantic corpus study. The study
still uses its original FP32 model, training settings, data and selection gates.
`neural/int8_inference.py` defines a mixed INT8 inference variant; it is not
accepted by the existing three-logit deployment loader.

Every active linear uses symmetric INT8 weights with one FP32 scale per output
channel, dynamic symmetric activation scaling per row, zero padding to four
elements, INT32 accumulation and fused FP32 dequantization plus bias. Rounding
is nearest-even and the range is [-127,127]. Zero rows use scale one. The inner
scale for nonzero rows is maximum times FP32(1/127), clamped to FP32tiny;
this explicit reciprocal multiplication matches the frozen v3 native policy.
The inner dimension is bounded so worst-case products cannot overflow INT32. Attention,
LayerNorm, GELU, softmax, material algebra and residual additions remain FP32.
Dynamic scaling fits no data; any future fixed clipping range must be fitted
on TRAIN under a separate declaration.

The native wrapper requires a hash-pinned HIP library and actual loaded HIP
runtime path/hash/symbol ownership/API version, non-overridden gfx1030 ROCm,
contiguous GPU storage,
the current Torch stream and matching device ownership. It retains temporary
storage on that stream. It performs no host synchronization or CPU fallback.
Weights for N>=32 are packed once into coalesced INT32[Kpad/4,Npad] storage;
the scalar head keeps the row-major path. Native launch failures fail the
invocation. The caller remains responsible for a completion fence before
consuming or releasing results.

The v3 fused backend performs row reduction, activation quantization, integer
dot and dequantization in one kernel. Its rows ABI accepts the real source K,
so odd-width inputs need no FP32 padding allocation. All weights, including
the scalar N=1 readout, use padded coalesced storage for this backend. A separate
declared hybrid retains original FP32 weights and uses FP32 linear arithmetic
when the row count reaches its fixed threshold; hardware affine measurements
motivate testing this route, not a claim of a whole-model gain.

The paired FP32 reference uses the same frozen checkpoint and omits training
board/rule decoders. CPU fixtures show exact value parity for all four encoders
and all three transition modes, with nonzero readout weights. This does not
establish GPU parity or speed. The INT8 variant quantizes only linears used by
value inference; retained unused decoder weights do not enter the forward.

`neural/symbolic_residual.py` defines a separate learned architecture. Its typed
current-state input includes canonical exact state features, occupancy-blocked
pseudoattacks, pins, checkers, phase and four named history counters. Those
fields must be produced and certified by the authoritative simulator on GPU
and share a producer, state-context and epoch identity. Tensor metadata checks
do not certify their contents. This source does not yet implement the native
simulator producer or resident search integration.

The symbolic channels enter role attention and a learned value projection,
whose result is added to the eight semantic role tokens before value readout.
The projection starts at zero to preserve the frozen reference. CPU fixtures
show both a nonzero gradient into that projection and an actual value change
when a symbolic input changes after setting synthetic nonzero weights. These
are mechanical fixtures, not evidence of training or strength improvement.

An authoritative evaluated state can be an exact child node already created
by search. A hypothetical transition model retains its parent plus known legal
action contract. Future causal search-control or score residuals can use a new
typed version with explicit dependencies; teacher labels, future targets and
future PV information cannot become inference inputs.

Before adoption:

1. Freeze checkpoint, variant source, native library, exact device, shape set
   and measurement declaration. Validate the FP32 value-only wrapper against
   the unpruned frozen reference on GPU before changing arithmetic.
2. Check the INT8 kernel against independent integer accumulation, including
   K padding, tails, N=1, zero heads, nondefault streams and ownership failures.
   Inspect actual gfx1030 ISA; record whether Torch's own integer GEMM is usable
   on this ROCm build rather than assuming CUDA support transfers to ROCm.
3. Compare frozen FP32 and INT8 numerical error and complete-root first-canonical
   action agreement on predeclared TRAIN inputs. This is quantization fidelity,
   not a new heldout strength study. Fitted ranges require TRAIN-only calibration.
4. Measure activation reduction, quantization/padding, GEMM/dequantization,
   symbolic production/injection and whole inference with GPU events. Also
   measure synchronized end-to-end resident search cost, batch size, legal K,
   allocations and launch counts. A synthetic dot speedup is not an engine gain.
5. Implement the simulator-owned symbolic producer and verify all channels
   against exact independent rules/history fixtures. Train the injection and
   paired no-injection control only under a separately reviewed declaration.

The current CPU checks are `python -m unittest neural.tests.test_int8_inference`.
`neural/profile_int8_inference.py` supplies a separately pinned TRAIN-only
whole-model fidelity/latency diagnostic. It compares the same frozen checkpoints
against unpruned FP32 and exact value-only FP32, records maximal/mean CP and
latent error, complete-root first-canonical action agreement and changed-choice
near-tie gaps. Raw neural decisions exclude simulator terminal adjudication.
Its clean eager timings record device events and synchronized wall time; a
separate instrumented pass records linear shapes/costs. GPU event intervals
include host dispatch gaps. It does not claim native-node producer integration,
graph replay latency, learning, heldout quality, deployment or Elo.

The corrected [actual gfx1030 whole-model diagnostic](../neural/evidence/int8-whole-model-gfx1030-2026-10-03/analysis.json)
completed all 40 frozen models in 110.416 seconds, with the owned watchdog
terminal at 112.161 seconds. The missing-`python-chess` first attempt remains
[archived separately](../neural/evidence/int8-model-missing-chess-2026-10-02/README.md).
No model was retrained, calibrated, selected or promoted; only TRAIN was read.
FP32 value-only latent and CP results are exact for all eight anchors and 32
transition models. Eight fresh GPU-hidden CPU fixtures also pass.

INT8 mean absolute same-checkpoint error spans 0.716–2.992 CP for anchors and
0.472–1.824 CP for transitions; maximal errors are 13.977 and 7.180 CP.
Across 32 models evaluated on the same 233 complete roots, quantization changes
97 of 7,456 model-root choices. Those changed choices have an FP32 best-versus-
second gap no larger than 1.098 CP. These repeated model-root observations
describe fidelity, not independent games or quality improvements.

Fused INT8 is slower than the exact FP32 value-only route in every grouped eager
timing comparison: paired speed ratios span 0.700–0.880. At batch one, median
FP32→fused wall times in microseconds are square 524→732, relation 1004→1285,
plan 1222→1616 and threat 1040→1432. Fusing beats the multi-launch INT8 route,
but Python/native dispatch, tensor ownership and non-linear work still enter
these end-to-end eager intervals. Instrumented linear costs are a separate pass.
The fixed wider-row FP32 hybrid also remains an experiment, not a default.

Timing is the optimization objective, so this result does not justify replacing
the FP32 route. Keep the validated INT8 arithmetic for the next separately
declared captured/native graph comparison and symbolic producer integration.
Extra FP32 cached weights and scratch buffers are permitted. The full-GPU target
remains unchanged; this run measures neither graph replay nor engine search.
