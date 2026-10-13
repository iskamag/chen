# Multihead semantic transformer control

`neural/transformer_semantics.py` implements a separate arithmetic candidate.
It is outside the sealed four-encoder study and existing deployment loaders.
CPU mechanics are checked; training, native GPU execution, quality and playing
strength have not been measured for this candidate.

The same family now supports large width/depth/head descriptors, allocation-free
parameter/work estimates and explicit caller allocation budgets. The
[scaling study](SCALING_STUDY.md) defines~1M through~270M value-path bands and
separate data/training/inference axes; no large model has been allocated here.

The body uses 64 schema1 square tokens, learned absolute positions, pre-LayerNorm
multihead attention and residual GELU blocks. Every head has its own six static
geometry coefficients; optional four-channel blocked pseudoattack/support bias
uses the existing exact diagram. These priors are not legal-move or winning-threat
predictions. Eight role queries attend to the body using actual multiple heads,
then share the established board/maps/rights/EP/clock semantics and a material
plus zero-initial CP residual readout. There are no unused policy/transition heads.

The square-token direction is supported by [Chessformer, sections 3.1–3.3](https://arxiv.org/html/2605.19091v1#S3).
That paper generates dynamic attention-bias templates from compressed board
state and uses a source-destination policy head. This control implements static
geometric coefficients and the existing eight-role semantic bottleneck. It is
an architecture comparison for Chen, without claiming a paper reproduction or
transferring its reported strength to this engine.

`piece_keys_only` masks empty keys in body and role attention. It still computes
dense 64-square attention; timing must not call this token compaction. A shared
position-weighted EP context broadcasts information from the exact EP plane to
occupied keys, so an empty EP target survives this ownership control.
The separate symbolic-to-role attention retains all 64 symbolic square keys:
empty-square attack channels can be meaningful. Symbolic injection and piece-key
masking are therefore separate declared axes, rather than an information-matched
comparison of compact piece tokens.
This auxiliary attention has no position term and pools a bag of symbolic rows;
the square residual path supplies spatial reasoning through the position-aware
body. Any future spatial role-injection variant must be a separate arm.

Configured current-state symbolic channels have shape `[B,64,C]` and enter both
square and role residual streams through zero-initial projections. With copied
shared weights, this arm initially reproduces every no-injection output exactly.
The typed `forward_symbolic` entry requires the admitted symbolic state seam;
plain tensor forward is a training/arithmetic interface and cannot certify
history ownership, legal ingress or finite values. An exact producer must own
position/history provenance, orientation, stream lifetime and epoch consistency.
Teacher labels or unavailable future search information cannot enter this stream.
Exact initial parity assumes finite admitted inputs and a fresh zero-projection
arm. `copy_shared_state` copies shared tensors and leaves injection weights alone;
it does not reset an already fitted injection. The symbolic key projection first
receives a value-path gradient after its zero-initial value projection learns.

`value_only=True` omits semantic decoder work while preserving identical latent
and CP values, including a nonzero fitted CP head. Terminal adjudication remains
with the exact simulator. This candidate has no implicit transition dynamics:
the next study must include encoding exact legal children as a static control
against predicting their latents from the parent.

The seven CPU fixtures compare heads to a loop oracle, exclude masked keys,
check shared semantic loss gradients without host scalar extraction, establish
exact value-only and zero-injection parity, exercise nonzero symbolic residual
gradients, preserve empty-square EP information, and reject incompatible shared
state before any copy. They do not validate GPU residency or quantized inference.
Independent review caught the original finite mask sentinel's extreme-logit
failure and a mismatched-batch reshape. The corrected seam rejects incompatible
metadata and uses exact exclusion plus a tensor-gated zero-attention empty-row
fallback. Regression fixtures exercise both failures. [Recorded CPU checks](../neural/evidence/transformer-control-cpu-2026-10-02/review.json)
pin the corrected source and preserve stdout/stderr.

For the next declared study, vary dense/rule-biased/occupied-key bodies and
symbolic residual injection explicitly. Record active/stored parameters,
precision, source hashes, TRAIN-fitted calibration, complete legal sibling
groups, canonical tie handling, and setup/amortized/end-to-end timing. Matching
width/depth does not imply matching compute. Keep the old validation results as
development evidence; final TEST remains unopened. Compare FP32 and INT8 only
after mechanical native admission and frozen-weight fidelity checks. Timing
decides routing; extra cached FP32 weights or scratch memory are permitted.
