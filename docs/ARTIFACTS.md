# Research artifacts

Git contains the engine, browser app, experimental source, documentation,
compact result summaries, and frozen source used by oracle tests.

Generated datasets, compressed evidence archives, compiled engines and GPU
libraries, Lean build products, disassembly dumps, and large raw traces are
ignored. These payloads were removed from all local branch histories during
public repository preparation on 2026-10-04. Local copies were retained.

Historical reports and manifests retain their original hashes, artifact paths,
and source revision identifiers. Some of those files are intentionally absent
from a clone. Those reports describe past runs; they do not promise a complete
replay environment. Corpus admission and replay tools require the original
external evidence package supplied by the caller. CPU engine and browser builds
do not need it.

Keep new runs under `results/` or another ignored output directory. Commit small
summaries with source and environment provenance. Store datasets, model weights,
executables, and full profiling output outside Git. Source revision identifiers
predating the cleanup refer to the original history; the rewrite mapping is
recorded in [history-commit-map.txt](history-commit-map.txt).
