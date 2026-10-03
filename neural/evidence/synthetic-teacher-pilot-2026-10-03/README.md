# Reachable TRAIN teacher pilot, 2026-10-03

One bounded CPU-only attempt; no training, GPU calls, heldout read or quality claim.
Declaration SHA256 `1a8e22068f8fab5650496f30d192ee06a02ff5ee87ece0859d5a4b07127130d9`.
Full plan fixed 32 distinct TRAIN families, 16 roots per mover, before any query.

- `plan.json`, `declaration.json`, `metadata.json`, `source-freeze.json`: exact input support/history, 40 planned jobs, options/binary/embedded-net/source pins. The earlier declaration's misleading authorization field was corrected before search, with its original bytes retained separately.
- `source/`: runtime closure plus focused fabricated fixture source. `stockfish-source/`: original pinned GPL sources used for the packet/score/network protocol.
- `query-records.tar.gz`: all 40 raw query JSON records (7,792,321 uncompressed bytes), including every UCI command/emission, score type and ordinary interval or missing reason. `query-records-manifest.json` binds names, sizes and hashes.
- `report.json`, `setup.json`, `run.*.log`, `runtime.json`: terminal lifecycle, timing/work telemetry and actual runtime/binary provenance. Default embedded net was exported/hash-verified before search; the pinned binary reproduces it, so its large bytes are not duplicated.
- `mechanical-replay.json`, `replay-label-records.py`: all 40 saved transcripts replayed with the same frozen parser/frame logic. No engine opened. This is not an independent parser audit.
- `summary.json`: 21/32 base roots produce 410 distinct ordinary action rows; 3/8 calibration packets provide 65 repeated-root rows. All 16 missing packets retain decisive-score evidence. Selected ordinary-packet choice changes 1/3 comparable pairs; final UCI choice changes 0/8. Ordinary pair ordering changes 60/729 including 58 strict reversals.

Four focused protocol/invariance tests passed before search (0.931 seconds test-body time).
They cover balanced family-disjoint deterministic preselection, TRAIN exclusivity,
complete legal support, parent-mover normalization, bounded final output, corrupted
or decisive emissions and calibrated tie/reversal accounting. No optimizer steps.
The only subsequent collector change was the declaration metadata field correction.

Teacher-generation interval 10:08:19.348153–10:08:37.368685 UTC (18.02 seconds) excludes
cold plan admission/net export. CPU 4 worker/SF overlapped a separate CPU 0 search
match; shared package power/cache are uncontrolled. Watchdog group 30083,
collector 30093 and Stockfish 30101 were independently confirmed absent at terminal.

Selected ordinary packets are finite selective estimates at one common reported
depth, not uniform-child-horizon minimax values. Parent material normalization
and parent mover perspective are explicit. Bounds/mate distances never become
ordinary point labels. Base and calibration packets are separate budget variants;
calibration rows must not inflate unique input coverage or enter model inputs.
