# Closed CPU configured-strength evaluation

See [the result report](../../../harness/cpu-rating-current-2026-10-03.md).
The independent audit admits all 64 games / 32 complete authored pairs. No
missing, capped or failed game was assigned a score. PVS-off and PVS-on remain
separate. These configured-setting proxies are not Lichess ratings; no
confidence interval or independent-opening assumption is asserted.

`frozen-run.tar.gz` contains the complete frozen stage under `stage/`, plus
Stockfish bundled source/licenses and the committed read-only auditor/tests.
`archive-inventory.json` lists every regular member's size, hash and mode;
`archive-review.json` records a cold full member check. `external-stockfish.json`
binds the existing official Stockfish binary instead of copying its 99 MiB.
The original binary, official download archive and run stage remain retained.

Historical declaration and launch paths are preserved. For an unchanged-path
read-only replay, retain the stage at its declared absolute path and the
externally pinned Stockfish binary, then run:

```sh
PYTHONDONTWRITEBYTECODE=1 \
PYTHONPATH=results/cpu-rating-20261003-_584k376/deps \
python scripts/audit_cpu_pool.py \
  --declaration results/cpu-rating-20261003-_584k376/declaration.json \
  --out /tmp/chen-cpu-pool-readonly-replay.json
```

The output path must be absent. Never rerun `supervise.py` or the match collector
from this archive. Supervisor terminal records and the independent `/proc`
retirement check are retained; all owned workers and the supervisor exited.
No new engine search or hardware experiment was performed for archival review.
