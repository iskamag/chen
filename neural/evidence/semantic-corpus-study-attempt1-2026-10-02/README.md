# Semantic corpus study: closed failed attempt 1

One root-sealed gfx1030 GPU attempt completed all eight anchor and 32 transition
training phases (3,072 anchor and 6,144 transition updates). Its owned-state
bundle was 137,594,521 bytes, exceeding the unchanged 134,217,728-byte stage-gate
limit. The runner failed before a validation gate, selection or TEST access.
No model-quality metrics, promotion or rating claim exist for this attempt.

The external watchdog retired the child with exit 1 in 115.370 seconds; both
owned processes were observed absent. No retry or resume occurred. Individual
checkpoints remain archived evidence of completed phases, not deployed models.
A cold CPU mechanical audit checks every file's finite tensors and state hash
against its GPU phase report without reading heldout labels.

`runtime-payloads.tar.gz` preserves exact staged source, all three immutable
split packs, native source/binary witness dependencies, declarations, root seals,
watchdog records/logs, all 40 individual checkpoints and the rejected bundle.
`runtime-members.json` pins every unique regular member. Absolute runtime paths
inside provenance records document the original execution; extracting elsewhere
does not itself authorize training or evaluation.

The initial preparation script failed a metadata comparison because it omitted
the already-declared study source hash. Its logs and corrected producer survive;
that happened before the single training attempt and did not consume GPU data.

A follow-up packaging change requires new review and declarations. It must retain
all individual file bounds, checkpoint freeze ordering and heldout gates rather
than increasing the global cap or treating this attempt as successful.
