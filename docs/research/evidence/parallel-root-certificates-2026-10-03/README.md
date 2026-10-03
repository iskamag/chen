# Parallel root certificate audit

CPU-only Lean 4.27.0 run: 23 new theorems, warnings as errors, actual axioms
limited to propext and Quot.sound. No HIP execution/refinement or timing claim.
Copied SearchBounds is the existing imported dependency; historical archives
were not re-audited. Dependency compilation naturally emits its existing eight
#print diagnostics, but verification.json inventories only new theorem audits.

Final actual output came from /tmp/chen-root-certificates-final-v3-2026-10-03.
commands.json retains exact argv/cwd/environment overrides and terminal exits.
The generated olean files are reproducible and omitted from this compact record.

Replay from the repository root with a fresh output directory:

```sh
python3 proofs/verify_root_certificates.py --output /tmp/chen-root-certificate-replay
```

manifest.json inventories and hashes every payload; it excludes itself.
