# Same-model resident neural evaluation

Chen's first deployment backend preserves the trained square, relation, or plan
encoder and its existing three-logit scalar head. `neural/inference.py` exports a
frozen inference graph that accepts canonical float32 `[B,18,8,8]` features and
returns int32 `[B]` scores in the current mover's perspective. Unused policy,
reconstruction, and learned-transition heads are absent from this static graph;
the retained trained tensors are unchanged. The simulator continues to determine
legality, terminal states, repetition, and mate distance.

This is a version-pinned TorchScript reference backend. TorchScript is deprecated;
PyTorch's replacement deployment direction is `torch.export` and AOTInductor.
An AOTInductor artifact is a separate experiment until these exact architectures,
dynamic batch limits, score conversion, and gfx1030 operators pass parity tests.
See the [official AOTInductor deployment documentation](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html)
and [C++ API](https://docs.pytorch.org/cppdocs/).

## Scalar meaning

The search-teacher target is a finite-depth classical minimax score, converted to
bounded utility with `tanh(cp/800)`. The three logits provide
`u = softmax(logits)[2] - softmax(logits)[0]`. Their historical loss/draw/win names
do not make this calibrated WDL or a game outcome predictor. Deployment returns

```python
cp = torch.round(torch.atanh(u.clamp(-1 + 1e-6, 1 - 1e-6)) * 800)
cp = cp.clamp(-29000, 29000).to(torch.int32)
```

The exported graph performs the conversion, including nearest-even rounding.
Rust and C++ receive integer scores and perform no separate floating conversion.
Float32 saturation and the inverse clamp mean even extreme utility cannot
represent every centipawn value in the declared static score band. It cannot
produce mate scores. Exact simulator terminals override static inference:
checkmate uses the simulator's root-ply mate convention; stalemate, rule draws,
and retained-history repetition use its draw policy. Mate-band teacher rows
(`abs(cp) >= 29000`) are excluded from value regression and separately counted.
Their legal best actions may still supervise policy.

The present teacher construction expands a root to depth three and labels its
first-ply parents after full backup. Those parents therefore have a two-ply
remaining horizon. Sibling edge exports are deduplicated by parent ID: each
parent contributes one training vote. Neither a finite-horizon agreement result
nor successful UCI wiring establishes playing strength.

## Artifact boundary

`export_inference(model, path, target_kind=..., provenance=...)` snapshots the
model to CPU/eval, traces dynamic batch examples, and freezes with
`optimize_numerics=False`. Python shape guards cannot be relied upon inside a
trace; callers enforce the manifest shape before dispatch. The graph also has
to pass execution after device remapping, because tracing can retain
device-specific behavior. These limitations follow the official
[trace](https://docs.pytorch.org/docs/2.10/generated/torch.jit.trace.html) and
[freeze](https://docs.pytorch.org/docs/2.10/generated/torch.jit.freeze.html)
contracts. Frozen constants are relocated by load-time device mapping, rather
than a later module `.to()` call.

The archive is a standard TorchScript `.pt` containing an extra UTF-8 file
`chen_manifest.json`. Its `.pt.json` sidecar has exactly this envelope:

```json
{
  "manifest": {"deployment_version": 1},
  "manifest_json": "the complete exact JSON string embedded in the archive",
  "archive_sha256": "64 lowercase hexadecimal characters"
}
```

The abbreviated example above omits required manifest fields. The complete
manifest includes:

| Field | Required interpretation |
| --- | --- |
| `deployment_version`, `feature_schema` | Integers, both 1 |
| `format`, `torch_version` | `torchscript_static_reference`, exact exporting runtime version |
| `model_config` | `name` square/relation/plan; width 8..512, depth 1..32, plan_tokens 1..128 |
| `input_dtype`, `input_shape` | float32, `["B",18,8,8]` |
| `output_dtype`, `output_shape` | int32, `["B"]` |
| `max_batch` | Integer 1..4096; bounded deployment allocation |
| `target_kind` | Production: `finite_depth_minimax_cp_utility` |
| `scale_cp`, `epsilon`, `rounding`, `score_limit` | 800, 0.000001, nearest_even, 29000 |
| `perspective`, `terminal_responsibility` | side_to_move, simulator |
| `head_readout`, `strength_evidence` | softmax_logits_2_minus_0, false |
| `state_sha256`, `checkpoint_sha256`, `provenance` | Frozen source tensors, optional checkpoint hash, actual training/fixture provenance |
| `export_validation` | CPU parity batches and retained tracing warnings |

Material-proxy checkpoints and untrained architecture fixtures have separate
target kinds; they cannot silently become trained search-teacher artifacts.
The Python exporter supports them for scoped parity experiments. The production
Rust selector accepts the search-teacher kind. Schema integers reject booleans.

Archive size is bounded at 64 MiB; embedded manifest and sidecar at 64 KiB.
The loader verifies the pinned archive SHA, parses `manifest_json`, requires
that it equals `manifest`, checks all fixed semantics and bounds, and passes
the same archive bytes and exact manifest string to C++. C++ checks the embedded
string byte-for-byte. Using already verified bytes avoids reopening a path
between hashing and graph load. Hashes establish artifact identity, not safety
for an unknown executable graph. Only generated, trusted archives are allowed;
the restricted tensor checkpoint loader is a different boundary. See
[PyTorch serialization notes](https://docs.pytorch.org/docs/2.10/notes/serialization.html).

Export a trained teacher checkpoint with
`python -m neural.inference checkpoint.pt evaluator.pt --max-batch 256`.
The default path uses the restricted search-teacher checkpoint loader. Legacy
material-proxy export requires `--checkpoint-kind material-proxy` and retains
its distinct target kind. Full teacher lineage remains in the pinned checkpoint
and experiment report. The deployment sidecar retains a compact teacher summary,
training configuration, and hashes of complete checkpoint/teacher metadata so
that large lineage manifests do not overflow this bounded interface.

## Native runtime and ownership

The small C ABI in `gpu/chen_neural.h` owns an immutable loaded LibTorch module,
bounded Torch-owned scratch features, and its device. Rust owns that context and
the rules context for the whole search. A model-configured Rust CLI preloads and
retains the Torch-bundled HIP runtime before any backend context is created,
so both UCI option orders work. Direct library integrations must bind that runtime
before loading rules or creating their contexts; compare all
five runtime identity tokens before accepting interoperability:
`hipPointerGetAttributes`, `hipMemGetAddressRange`, `hipMalloc`, `hipFree`, and
`hipStreamSynchronize`. A system-HIP standalone Rust run and a Torch-bundled-HIP
run are distinct configurations. Compiler version alone does not identify the
active allocation/runtime implementation. The existing
[VRAM bridge contract](VRAM_BRIDGE_CONTRACT.md) explains the verified prerequisite.

Compile the plugin against the installed Torch headers/libraries, its C++11 ABI
setting, and the same bundled HIP runtime. Pin those dependency hashes alongside
the model archive. C++ and Rust communicate through a C ABI; no Torch C++ objects
or exceptions cross that boundary. Torch's C++ API is a dependency of this
backend, rather than a dependency of the generic search protocol.

A resident evaluation transaction is:

1. The rules arena classifies its selected nodes and adjudicates terminals using
   retained root history. Reject expanded nonterminal nodes and reuse previously
   scored rows according to the resident phase contract. Lock evaluator/model identity until
   reset; an error may not switch a partially scored tree to another evaluator.
2. While the arena is exclusively owned and immutable for this call, pass a
   borrowed, capacity-bounded device position span to the synchronous callback.
   The callback retains no arena pointer after return.
3. Decode pieces and rule state on device into Torch-owned contiguous features.
   Run the same frozen encoder/head on bounded chunks. Validate result dtype,
   shape, device, and static-score range before copying scores device-to-device
   into caller-owned output memory. Terminals retain their exact simulator scores.
4. Drain producer and Torch work on every success and exception path before
   scratch or borrowed pointers can be freed/reused. Return an error through
   bounded UTF-8 diagnostics. Commit node scores/SCORED phase only after a whole
   successful validated call; partial output writes on failure are uncommitted.

The first implementation uses conservative synchronization across the private
HIP stream and Torch's device work. Event handoff can be an optimization after
the baseline has failure-path and cross-stream evidence. There is no per-leaf
board, feature, or score payload download. Small validation/telemetry scalars,
initial model loading, root/history ingress, and final search output remain
explicit host boundaries. A claim about zero payload traffic requires a scoped
copy trace and a negative payload-copy control, not only source inspection.

## Canonical input and history

For black to move, exchange friendly/enemy piece planes and rank-flip squares
with `square ^ 56`; files stay fixed. Channels 0..5 are mover PNBRQK, 6..11 are
opponent PNBRQK. Channels 12..15 repeat mover K/Q and opponent K/Q rights across
all squares. Channel 16 marks raw en-passant, including an ineffective or pinned
target. Channel 17 repeats `min(halfmove,100)/100` in float32. Storage index is
`((batch*18 + plane)*8 + rank)*8 + file`. Fullmove and repetition history are
absent from model input and remain exact simulator state.

The same feature tensor can represent distinct repetition contexts. Therefore
the network is a board-feature static heuristic inside an exact simulator, and
teacher metadata must retain context UUID, root epoch, hash/history provenance,
root family, depth, and classical evaluator identity. Split training and heldout
families, then reject canonical-feature overlap across split roots and retained
parent states. Report excluded aliases and draw/mate rows separately. This
provides a defined heldout teacher test without claiming history-complete Markov
state or general chess strength.

## Comparable backends and verification

| Backend | Practical status and experiment boundary |
| --- | --- |
| Version-pinned LibTorch JIT | First same-model reference; dynamic batches and full trained architecture retained; runtime is comparatively large |
| `torch.export` + AOTInductor | Intended successor; compile the same encoder/head, check gfx1030 coverage, dynamic-shape constraints, runtime binding and score parity before selection |
| Native HIP implementation | Retain the same checkpoint tensors; implement LayerNorm, exact GELU, linear, softmax, reductions, pseudoattack blockers, relation attention, and plan attention; compare every layer and final integer scores |
| Separate inference process | Adds IPC/ownership/timeout complexity; host tensor IPC would violate resident payload goals, while HIP IPC needs an explicit allocation lifetime and stream protocol |

The evaluator callback stays interchangeable so these implementations can use
identical tree construction, corpus, tensor checkpoints, batch limits, and
terminal handling. Approximate kernels, changed activations, quantization, and
different networks are named experiments with fresh parity/strength evaluation.

CPU tests in `neural/tests/test_inference.py` train small tensor fixtures and
export/reload all three architectures at two width/depth settings. They compare
integer outputs at batches 1, 3, and 17; check unchanged original tensors; and
exercise schema, integrity, capacity, and conversion guards. These are artifact
mechanics tests. Before deployment acceptance, hardware tests must additionally
cover CPU-versus-GPU score tolerance/rounding boundaries, eager GPU-versus-JIT GPU
integer equality, all canonical special-rule features, the C++ callback,
exception fences, model/evaluator phase locking, native copy traces, and Rust/UCI
search with exact terminal fixtures. Pairwise backend agreement is separate from
whether a trained model improves heldout teacher agreement or playing strength.

On 2026-09-30 the scheduled gfx1030 inference suite passed all five tests in
20.670 seconds. Its 18 GPU cases cover square/relation/plan at width/depth
8/1 and 16/2, with batches 1/3/17. GPU eager and CPU-exported, frozen,
device-remapped graphs returned byte-identical int32 scores. Their maximum
CPU-versus-GPU difference was zero centipawns on four board/rule fixtures.
The [tracked evidence](../neural/evidence/inference-gfx1030-2026-09-30.json)
contains case metrics, exact source hashes, runtime/device metadata, and the
complete test log. The tensor updates and checkpoint lineage in these tests are
declared fixtures; no actual search-teacher learning or resident-copy claim is
derived from this report. Independent native callback and UCI tests must be
reported under their own evidence scope.

The final combined suite reran the hardened current sources on gfx1030 and
passed all 76 tests without skips. Main-worktree native tests additionally
compare 510 architecture-fixture rows and 255 actually trained rows against
eager GPU and frozen GPU outputs, with bit-exact canonical features. The
[integration report](../../chen/gpu/evidence/neural-integration-2026-09-30.json)
binds current sources, model fingerprints, runtime and separate test logs.
Earlier reports retain the source fingerprints from their own execution stage.
