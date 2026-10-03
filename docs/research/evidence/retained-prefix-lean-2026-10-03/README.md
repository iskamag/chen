# Retained-prefix finite proof evidence

This is a CPU-only Lean/source-review milestone. It consumes the committed
retained-prefix experiment's source binding and summary without revalidating
that GPU archive or rerunning hardware. It compiles the old imported dependency
but audits only the 20 new `Chen.Retained` theorems.

- `verification/` contains frozen imported/new modules, generated `.olean`
  artifacts, exact commands, compiler outputs and actual new-theorem axiom audit.
- `cold-replay/` retains a separately compiled replay's commands, verification
  record and relevant outputs. The new axiom-audit output is byte identical.
- `sources/` freezes the new proof, verifier and research document plus the
  pinned historical selector/reference and imported certificate.
- `prior-evidence/` copies the committed hardware summary and source binding
  for attribution, not as a new hardware validation.
- `development/` preserves two failed development sources and recorded causes.
  Their JSON records are contemporaneous execution summaries, not raw compiler
  stdout/stderr transcripts; the successful runs have raw outputs.
- `source-binding.json` records source origins, hashes and replay scope.
- `manifest.json` inventories every other payload by relative path, SHA-256 and
  byte length. It excludes itself to avoid a recursive digest.

Reproduce with a fresh output directory from the repository root:

```sh
python3 proofs/verify_retained_prefix.py --output /tmp/chen-retained-prefix-proof-replay
```

The source hashes and audit output should match. Absolute output directories,
timestamps and command records differ across invocations. This certificate is
not a HIP refinement, memory-model/bitvector proof, latency optimum, neural
batch-size equivalence or chess-strength theorem. See the frozen research
document for each premise and the proposed next decision.
