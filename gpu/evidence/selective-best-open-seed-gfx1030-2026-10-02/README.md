# Isolated backup best-open seed, RX6800/gfx1030

This archive retains the experimental candidate; it does not select it in production. Baseline is committed packed-cache library `a47b8c4e…`; candidate is `f71eabee…`, compiled from the old frozen rules closure. Only `chen_selective.inc` differs in the two eleven-file closures. Private metadata grows 16→24 bytes, with the corresponding owned allocation/alias interval; public ABI is unchanged.

`summary.json` reconstructs initial portable/native/mechanism parity, 72 fixed-work rows, 12 plain one-second observations, 6 isolated ROCprof runs and actual timed-library code objects. `paired-summary.json` is the separately declared 12-pair large-suite timing confirmation with exploratory intervals. No correctness tests are repeated in that confirmation because sources/binaries are unchanged. Warmups are logged and excluded from paired estimates. GPU reservations are separate from neural diagnostics and epoch experiments; CPU5 is pinned, shared system cache/power uncontrolled.

`raw-records.tar.gz` is a compressed, relocatable record tree. `raw-member-index.json` binds every regular member. `manifest.json` uses `payload_sha256` and `payload_bytes` for the exact outer inventory; external committed baseline/Rust artifacts are pinned in `artifacts.json`. Source comments, the actual failed missing-followup attempt, and the failed cold replay that resolved the venv symlink to system Python are retained; corrected offline reconstruction passes. Build observations record the successful execution-tool commands/empty diagnostics rather than inventing build subprocess timestamps.

Offline reconstruction, using an interpreter with `python-chess` installed, from the repository root:

```sh
mkdir -p results/seed-archive-replay
tar -xzf gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02/raw-records.tar.gz -C results/seed-archive-replay
PYTHONDONTWRITEBYTECODE=1 results/qa-venv/bin/python gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02/sources/gpu/analyze_selective_seed.py results/seed-archive-replay --output results/seed-archive-replay/summary.json
PYTHONDONTWRITEBYTECODE=1 results/qa-venv/bin/python gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02/sources/gpu/analyze_seed_pairs.py results/seed-archive-replay/paired-confirmation --output results/seed-archive-replay/paired-summary.json
```

Read-only direct shared-library ELF extraction requires no GPU or ROCm runtime:

```sh
python3 gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02/sources/gpu/extract_hip_code_objects.py gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02/candidate-libchen_gpu.so results/seed-candidate-elf-replay
```

The ELF reader extracts every concatenated HIP bundle. The second bundle contains the rules/selection kernels; the first contains `chen_gpu.hip`. Recorded resources/static instructions are not occupancy or executed instruction counts. Instrumented API waits overlap kernel work. All reported effect sizes and bootstrap intervals are scoped to this machine-local process sample; they establish neither strength, rating, population coverage nor hardware-global optimality. The initial suite was inspected before declaring the follow-up, and time dependence/selection effects remain uncontrolled.
