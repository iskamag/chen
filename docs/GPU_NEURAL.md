# Trained same-model resident inference

`hip-resident` with `Eval=neural` runs a trained square, relation or plan encoder
and its scalar head inside the Rust UCI engine. HIP rules expand resident nodes;
the external evaluator reads their device packets, decodes schema-1 features
on GPU, runs the version-pinned LibTorch graph, and commits device scores before
resident minimax reduction. Policy and imagined latent transitions remain
separate research experiments. The current trained models underperform material
on heldout teacher positions; integration does not establish a strength gain.

## Train, export and select

The separate `../chen-neural` worktree owns the
[search-teacher pipeline](../../chen-neural/docs/SEARCH_TEACHER.md) and
[same-model deployment contract](../../chen-neural/docs/NEURAL_DEPLOYMENT.md).
Existing experimental checkpoints and deployment archives are retained under
that worktree's ignored `results/neural/` directory, with hashes in tracked
evidence. A clean clone requires regeneration or the matching local artifacts.

```sh
# In the neural worktree, with its Torch environment:
python3 -m neural.inference results/neural/search-teacher/square.pt \
    results/neural/deployment/square-inference.pt --max-batch 256

# In the main engine worktree, using the existing ROCm container:
CHEN_NEURAL_MODEL=../chen-neural/results/neural/deployment/square-inference.pt \
    scripts/rocm_run.sh bench --backend hip-resident --eval neural --depth 2
CHEN_NEURAL_MODEL=../chen-neural/results/neural/deployment/square-inference.pt \
    scripts/rocm_run.sh uci
```

In UCI, select `Backend=hip-resident` and `Eval=neural`. Levels still cap resources;
they are uncalibrated. `MaxPositions` bounds the arena. The archive's `max_batch`
bounds inference chunks; frontier `BatchSize` does not control resident neural
inference. Bench records the exact archive `evaluator_artifact_sha256`, allowing
different trained models to share the same rules, search scheduler and budgets.

The wrapper builds the native plugin against the container's installed Torch,
copies the archive/sidecar, and preloads Torch's actual HIP library before any
UCI backend creates a context. `CHEN_ARTIFACT_DIR` captures the actual model,
manifest, Rust executable, native plugin, rules library and sources. Direct
native runs additionally set `CHEN_NEURAL_LIBRARY` and `CHEN_NEURAL_HIP_RUNTIME`
or use the runtime-path file produced by `gpu/build_neural.sh`.

## Ownership and failures

The independently versioned [external evaluator ABI](GPU_EXTERNAL_EVAL.md)
passes borrowed contiguous device packets and writable scratch scores to a
synchronous callback. Their pointers stay within that invocation. The native
plugin owns a persistent feature tensor and the loaded module. It completes
all GPU uses before temporary outputs retire, including exception paths. Rules
commit no static score until every chunk succeeds and every required output is
in range. Rules retain exact terminal scores, independent of neural predictions.
Context/evaluator identity is fixed until reset; Rust drops the arena before
the inference context and keeps both libraries live until their contexts die.

Rust reads bounded archive bytes once, verifies SHA256, validates the sidecar's
exact embedded JSON, feature/model configuration, scalar interpretation and
checkpoint fingerprints, and passes those same owned bytes to native loading.
The native loader compares embedded manifest bytes and loads frozen tensor
constants directly onto the requested GPU. These are trusted generated graphs,
not a sandbox for arbitrary executable TorchScript archives.

Torch's bundled HIP runtime must bind before plugin/rules loading. Both native
contexts compare all five allocation/query/synchronization function tokens with
the producing runtime. A mismatched instance fails before pointer exchange.
When `CHEN_NEURAL_MODEL` is configured, Rust preloads and retains the producing
runtime for the entire CLI session, before any backend creates a HIP context.
Both UCI option orders therefore bind consistently; the wrapper also preloads
that runtime. There is no silent CPU substitution.

## Score and timing meaning

Teacher scores are finite-depth classical minimax values, not game outcomes.
The learned utility is `softmax(head)[2]-softmax(head)[0]`, supervised against
`tanh(cp/800)`. Inference clamps utility with epsilon `1e-6`, applies atanh×800,
rounds nearest-even, and clamps to ±29000. Epsilon also bounds the representable
inverse utility to roughly ±5800 cp. Neural static scores never declare mate.
The feature schema omits repetition history; exact simulator history still
governs terminal adjudication. Learned values are consequently an approximation.

Callback-inclusive `kernel_ms` is an event interval including host dispatch,
Torch scalar validation and barrier gaps, not pure GPU-active time. Resident
`h2d_ms`/`d2h_ms` fields remain uninstrumented zero placeholders; they do not
prove zero transfers. `elapsed_ns` measures search, excluding model/context
creation, while still including first-use runtime work during search. Cold runs
must not be interpreted as steady-state throughput.

The native copy diagnostic covers `hipMemcpy`, `hipMemcpyAsync` and
`hipMemcpyWithStream` during callbacks. The third API reveals Torch scalar
traffic invisible to the earlier two-API replay diagnostic. Recorded downloads
are scalar-sized (at most eight bytes each), and there are no board/feature
uploads through these APIs. Small first-call scalar transfers and device-to-device
score/constant copies are retained in the raw records. Injected 128-byte downloads
through all three APIs are detected. This is a scoped API audit, not a complete
trace of every runtime API or PCIe transaction.

TorchScript is deprecated. This implementation provides same-model reference
inference for the installed runtime. The callback seam permits independently
verified AOTInductor or native HIP replacements. See the
[official deployment documentation](https://docs.pytorch.org/docs/stable/torch.compiler_aot_inductor.html).

## Verification checkpoint

The [integration report](../gpu/evidence/neural-integration-2026-09-30.json)
records separate source/artifact snapshots rather than rewriting older test
fingerprints. All 76 combined neural tests and six Rust hardware tests pass on
gfx1030. The portable/feature-enabled host suites pass 37/41 tests; seven harness
tests pass. Native callbacks match 510 architecture-fixture rows and 255 trained
rows against eager and frozen GPU scores, with bit-exact decoded features.
An independent Python minimax oracle matches all 18 trained-model/position
cases and 11982 iterative leaf evaluations, including exact terminal scores.
The model-bearing wrapper also runs all six depth-two benchmark fixtures.

One full diagnostic run timed out during a game. Four fresh-process isolation
combinations and a complete rerun passed, including two 20-ply games with the
unchanged 30-second watchdog. The cause remains unexplained; the partial failed
log is retained alongside the successful rerun. Long-play reliability remains
an open gate.

These results establish the tested numerical/protocol mechanics. The first
training run trails material on heldout teacher metrics; it establishes neither
a playing-strength improvement nor the eventual Lichess target.
