# Frozen whole-model INT8 diagnostic, gfx1030

Actual completed TRAIN-only inference, with all eight anchor and 32 transition
checkpoints from the sealed corrected semantic study. No learning, calibration,
teacher quality, heldout reopening, promotion, rating or engine timing claim.

`control` retains the declaration, owned watchdog and terminal logs. `attempt-1`
contains every individual record, runtime admission and complete result.
`root-review.json` maps all 26 frozen dependency paths to archived payloads,
binds externally retained checkpoints/native evidence, and records process/group
retirement. The failed first diagnostic is retained in the separate sibling
`int8-model-missing-chess-2026-10-02` archive.

The exact FP32 value-only control passes. INT8 numerical changes are small on
these inputs, but fused INT8 loses all grouped eager timing comparisons against
value-only FP32. Timing takes priority over storage reduction: no default changes.
The next timing experiment must measure captured/native execution separately.

CPU-only reconstruction from the neural worktree:

```sh
PYTHONDONTWRITEBYTECODE=1 python3 neural/analyze_int8_inference.py neural/evidence/int8-whole-model-gfx1030-2026-10-03 --output /tmp/chen-int8-reconstruction.json
```

`analysis.json` reconstructs aggregate numerical/timing statistics from native-run
records; it does not independently replay every GPU tensor. `manifest.json`
binds the exact outer payload inventory. The original trained checkpoint payload
is already retained in the sealed semantic-study archive and is not duplicated.
