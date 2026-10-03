# Bounded resident semantic study

`neural/run_semantic_corpus_study.py` assembles the frozen phase mechanics in
`neural/semantic_corpus_study.py` into one executable study. The four encoders
(square, relation, plan, threat) share eight semantic roles and paired seeds.
Each seed/encoder trains one anchor, freezes its readout, then trains transport,
conditioned, action blind, and shuffled conditioned controls from identical
transition initialization. This produces eight anchors and 32 transitions.

The corrected actual gfx1030 attempt completed all eight anchors and 32
transitions in 142.158 seconds and reached validation. The paired joint gate
selected **NONE**; TEST remained closed. See the
[frozen results](../neural/evidence/semantic-corpus-study-manifest-2026-10-02/validation-results.json)
and [attempt record](../neural/evidence/semantic-corpus-study-manifest-2026-10-02/README.md).
This establishes execution and a negative bounded quality result, not playing
strength or a tuned architecture winner. Actual updates require both a root-reviewed
phase-kernel declaration and a separately reviewed outer suite declaration,
sealed against their explicit source and dataset hashes inside the runtime
where they will execute. CPU training is rejected.

The phase declaration's TRAIN and validation corpus pins must exactly match the
outer suite's manifests before a proposal is accepted. Final evaluation batches
and phase-boundary persistence are checked against the total wall ceiling;
crossing it cannot produce a successful complete-study artifact.

## Inputs and authority

The suite proposal binds all three immutable corpus manifests, the complete
neural/shared-resident source closure, the frozen phase declaration, one native
baseline/draw witness, and one absolute attempt-output directory. `prepare`
reads metadata and label-free simulator witnesses; it never evaluates heldout
quality. Review and seal both declarations before invoking `run`.

```sh
python -m neural.run_semantic_corpus_study prepare \
  --main-root /workspace/chen \
  --packs-root /workspace/chen/results/semantic-corpus/packet-v2 \
  --kernel-declaration /workspace/study/kernel-sealed.json \
  --baseline-witness /workspace/study/native-baselines.json \
  --baseline-sha256 EXTERNAL_WITNESS_SHA256 \
  --attempt-output /workspace/study/attempt-1 \
  --output /workspace/study/suite-proposed.json

python -m neural.run_semantic_corpus_study run \
  --declaration /workspace/study/suite-sealed.json \
  --declaration-sha256 EXTERNAL_SUITE_SHA256 \
  --output /workspace/study/attempt-1
```

The paths above illustrate staging; they are not evidence that those paths
exist in the ROCm container. The parent coordinator owns actual staging and
seal publication. Changing source paths, contents, corpus manifests, attempt
identity, or any declared protocol requires a newly reviewed declaration.

The native witness supplies exact CPU Material and Tapered scores for every
serialized state, state hashes, observation-to-state indices, and terminal
classifications obtained from the exact Rust history-aware oracle. It pins
CPU evaluator source and binary plus terminal oracle and build metadata. The
runner checks native hashes, feature/material parity and complete terminal
support. This is CPU baseline provenance; no HIP baseline parity is implied.
The first study accepts simulator draws only and rejects any mate before
training. A draw remains a legal candidate and receives exact utility zero
at decision time. Ordinary finite-search CP labels never encode mate utility.

## Execution and persistence

Only admitted resident TRAIN standalone observations fit the shared positive
material scale. Only TRAIN observations or complete TRAIN roots update model
parameters. There is no label-driven early stopping, CPU fallback, retry,
resume, or partial checkpoint promotion. Every phase uses the frozen schedule,
optimizer, loss coefficients, update counts and wall ceilings. The total wall
ceiling includes ingress, calibration, all phases, validation and any selected
TEST evaluation. Scalar predictions must remain finite during evaluation.

Each complete phase writes an owned CPU state dictionary to an exclusive
checkpoint path. A small `all-frozen-checkpoints.json` manifest pins the exact
canonical filenames, hashes and sizes of all eight anchor and 32 transition
files before a validation gate can admit validation labels. Its exact schema
requires every complete phase. Each file retains the 128 MiB cap; their combined
logical size may exceed it without creating another tensor bundle. Validation
uses the manifest's hash as the frozen checkpoint identity and reloads each
transition only after a bounded regular-file read verifies its individual hash.
Before granting validation access, the runner streams all 40 files through
restricted CPU loading, verifies their phase identity, exact architecture tensor
keys/shapes/dtypes, finite values and recorded state fingerprints, then discards
them. Symlinks, nonregular files and noncanonical joins are rejected. Validation sweeps every
standalone observation and every complete legal root for all paired arms.
The standalone error is distance outside the reporting interval, masked by
ordinary support. Root regret and agreement have equal root weights across
all legal-count batches, including tails. Raw material, the shared TRAIN-fitted
material scale, and unscaled native Tapered remain visible baselines.

Child role drift and the gap between parent-PV intervals and the frozen
anchor's exact-child value are recorded separately from selection. The latter
is a refresh-context disagreement, not a new Stockfish child query. The
baseline score vectors reside on the GPU during evaluation; batch baseline
inputs never enter learned branch forwards.

All sixteen encoder/control arms must have both seed results. Each seed must
strictly improve standalone interval error and root interval regret against
all three baselines while matching or improving optimal agreement. Eligible
arms are ordered by mean regret, mean scalar error, then encoder/control name.
If none pass, the durable result is `NONE`, and TEST remains sealed.

If an arm passes, the selected paired checkpoint bundle, validation result,
selection artifact and selection-source gate are frozen before TEST opens.
A durable consumption marker is written first. TEST then receives exactly
one evaluation pass for that selected two-seed configuration and visible
baselines. TEST never changes selection, parameters, thresholds or budgets.
An interrupted test pass cannot be retried through this attempt directory.

The [closed first GPU attempt](../neural/evidence/semantic-corpus-study-attempt1-2026-10-02/README.md)
completed all 40 phases, then failed because the former combined tensor bundle
was 137,594,521 bytes against the unchanged 128 MiB cap. No validation or TEST
gate opened. The manifest packaging repair requires new review and declarations;
it neither resumes that failed attempt nor admits its checkpoints for evaluation.

The result has no Elo estimate. Deployment into the modular engine and paired
playing-strength evaluation remain separate work. The relation and plan
controls are not a comprehensive modern transformer comparison; a separately
sealed multihead/token-ownership experiment is described by the frozen
`transformer_followup_proposal()`.

## Mechanical checks

```sh
PYTHONDONTWRITEBYTECODE=1 python -m unittest \
  neural.tests.test_run_semantic_corpus_study -v
```

Synthetic fixtures cover complete 8+32 execution, failure without heldout
promotion, NONE without TEST access, selected paired checkpoint persistence
before TEST admission, exclusive run identity, typed native witness support,
nonfinite predictions, ownership and unequal-batch metric aggregation.
