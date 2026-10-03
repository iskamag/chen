# Bounded CPU Coz evidence

All profiler and pilot runs terminated normally within their declared bounds.
Root explicitly released CPU5 after its match retired, then CPU5 was released
immediately after confirmation. No GPU runtime work belongs to this milestone.

- `summary.json` contains both causal marker curves, scope and terminal records.
- `pilot/`, `scout/`, `confirmation/` retain actual argv/declarations,
  stdout/stderr, process retirement, raw JSONL profiles and analysis.
- `confirmation-selection.json` declares the adaptive line/iteration choice
  before confirmation, based on scout sampling rather than an observed gain.
- `source-binding.json` records relevant source/build/runtime pins and distinctly
  attributes root's observed up-to-date validation versus original compilation.
- `analyze.py` requires actual zero-speedup controls and positive sample exposure.
  It does not silently use a nonzero intervention as the baseline.
- `analysis-validation.json` records byte-identical replay and 14 point-formula
  comparisons against the pinned official CLI. It also records final absence of
  all three run process groups.
- `run_bounded.py` is the exact executed runner. It expects its frozen folder's
  `bin/`, `tools/` and binding and launches a fresh phase subdirectory. Absolute
  path metadata reflects the original workspace; an archive relocation requires
  adjusting that runner's original-build path lookup or replaying the recorded
  argv with new paths. It is not a standalone portable benchmark launcher.
- `runtime-records.tar.gz` contains the frozen executable, runtime/CLI, dependency
  source closure, build/debug metadata, scripts, documentation and all records.
  `runtime-records-index.json` pins every raw member and the archive.
- `licenses/` preserves Coz and Rust marker license notices.
- `manifest.json` inventories every other compact payload and excludes itself.

This is an exploratory virtual-speedup observation, not an implemented speedup.
The finer node marker suggests a piece lookup hypothesis. Complete-search counts
have substantial boundary quantization and disagree at several levels; all data
is retained rather than reporting only the favorable curve. Only one nonzero
experiment exists per speedup level. There is no confidence, chess strength,
neural or GPU claim. See `docs/COZ_PROFILE.md` for interpretation and analysis.
