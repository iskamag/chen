# PVS-only direct one-second diagnostic

PVS scored **6 wins, 6 draws and 4 losses** against the default-off CPU baseline,
over 16 games from eight authored openings with colors exchanged. All eight pairs
completed: no technical failures, illegal moves, capped games or evaluation
adjudication. Independent replay verified all 1,773 legal plies.

The score is 9/16 (56.25%). Its direct relative logistic point estimate is +43.66
under this workload. These authored openings are not asserted independent samples:
there is no confidence interval, external Elo anchor or Lichess rating claim.
This modest diagnostic does not select a new default. PVS and aspiration remain
off by default; aspiration was off for both sides throughout this match.

| Opening | PVS as White | PVS as Black | Pair points /2 |
| --- | --- | --- | ---: |
| Italian | Draw | Win | 1.5 |
| Ruy Lopez | Draw | Win | 1.5 |
| Scotch | Draw | Draw | 1.0 |
| Four Knights | Draw | Loss | 0.5 |
| Petrov | Win | Loss | 1.0 |
| Vienna | Win | Loss | 1.0 |
| Open Sicilian | Loss | Win | 1.0 |
| Alapin Sicilian | Win | Draw | 1.5 |

## Matched controls and lifetime

Both engines ran the **same optimized executable**, SHA256
`e48ef713dbe4ccd26532c6a5021a1ea460201240ad0a86dc3abcfbb68f671bb9`,
from the isolated CPU milestone 8fcd891 (integrated into main as 16574cd).
The only configuration difference was `PVS=false` versus `PVS=true`.
Backend=alphabeta, Eval=tapered, Level 20, Hash 16 MiB, TT/quiescence/move-ordering
on and AspirationWindow 0 were identical. No neural evaluator or GPU search ran
inside this comparison. Binary, relevant source closure, compiler record,
Python 3.14.7/python-chess 1.11.2 provider, collector, auditor and exact options are
frozen in the evidence archive.

The opening file is the unchanged prior eight-authored-opening freeze, hash
`58df68de6c75adad5782022060535ee445993ccc503037bcd8871083e9f5ac22`.
The repository's ordinary `benches/openings.txt` currently has only two positions;
it was not silently expanded or used for this 16-game comparison.

The sequential collector and both engine processes had CPU 0 affinity. Each move
requested 1.0 seconds; exact Python-chess outcomes with draw claims ended games.
A 512-ply cap would be unscored, excluding an incomplete pair from the relative
summary. The external watchdog was 2400 seconds plus 5 seconds kill grace, owning
one new process group. Actual execution ran 1706.856 seconds, from
2026-10-03T10:07:16.023937 UTC to 10:35:42.880377 UTC, exit 0. Collector 29637,
engines 29640/29642, timeout/group 29636 and supervisor 29622 were confirmed absent.
No cleanup signals were needed. Other agents' processes were not retired by this
task.

## Timing and interpretation

Every move retains input FEN, legal move, UCI depth/nodes/time and transport wall
duration. Maximum observed wall duration was 1004.245 ms for PVS and 1002.826 ms for
baseline; medians were 1000.949 ms and 1000.946 ms. Early completion was allowed.
UCI nodes/time are the **last completed iteration's info**, not the total charged
work including an interrupted final iteration. Wall duration covers transport
and scheduling. Different policies visit different positions, so their node and
depth averages are descriptive, not an isolated throughput comparison.

Concurrent activity shared host package power/cache:

- CPU 4 teacher: instrumented 10:08:19.348153–10:08:37.368685 UTC (18.0205 s).
- GPU correctness: approximate filesystem envelope 10:16:14.149–10:16:16.028 UTC
  (1.878 s), not an instrumented kernel interval.
- Sequential GPU/CPU 5 decision diagnostic: approximate filesystem envelope
  10:31:04.092–10:31:20.680 UTC (16.5866 s).
- CPU 4 teacher: instrumented 10:33:09.914897–10:33:13.916769 UTC (4.0019 s), GPU hidden.

Those jobs reported terminal retirement; their records identify actual intervals
and distinguish filesystem reconstruction from instrumented timestamps. No
isolation or causal performance effect is inferred. The earlier fixed-depth
[CPU experiment](../CLASSICAL_SEARCH.md) separately measures useful-work savings.
Neither this comparison nor the prior Stockfish-configured reference permits
adding a calibrated rating to the PVS point estimate.

## Evidence and replay

[The manifest](../research/evidence/pvs-one-second-direct-2026-10-03/manifest.json)
binds the full frozen source/binary/provider archive, raw PGN/JSONL move traces,
declaration, options, terminal/process retirement, concurrent-job reports and
independent audit. The auditor's direct mode rejects any option difference beyond
PVS, checks the same engine hash and keeps caps unscored. Four synthetic protocol
fixtures cover complete/capped pairs and invalid configuration/scored caps. The
older Stockfish-reference declaration remains a separate mode; its old reports
and archives were not rewritten. The final audit replay matched byte for byte.

The archive preserves the original `/tmp/chen-pvs-direct-1s-20261003-lhqwh0ny`
paths. After restoring that frozen directory, a read-only match replay is:

```sh
stage=/tmp/chen-pvs-direct-1s-20261003-lhqwh0ny
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH="$stage/deps" python3 \
  "$stage/scripts/audit_timed_match.py" --run "$stage/run" \
  --declaration "$stage/declaration.json" --out /tmp/pvs-audit-replay.json
```

A new timed match requires a fresh declared run; this milestone does not restart
one or promote a default.
