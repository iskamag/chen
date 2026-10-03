# Causal dispatch investigation — 2026-10-03

Question: does replacing ordinary HIP dispatch with a graph materially reduce
search latency, after holding GPU work and synchronization constant?

This is a **real execution intervention**, inspired by [Coz causal profiling](https://arxiv.org/abs/1608.03676).
It is not stock Coz's CPU-thread virtual speedup and does not predict line-level
GPU optimization potential. Coz is absent from the host/container PATH; ordinary
CPU sampling would not automatically measure GPU queue critical paths.

## Matched controls

`gpu/experiments/device_epoch.inc` mode3 queues exactly the same begin,
clear, choose, children and completion kernels as mode0's graph. Dimensions,
stream order, device gates, request upload, result download, event handling,
per-advance publication and final fence are identical. Both group four advances.
Only the dispatch mechanism changes. Graph construction is excluded from warmed
search timing. A third production arm provides context; its host bookkeeping,
packet size, clearing and publication differ, so it cannot isolate graph cost.

`causal_dispatch_probe.py` uses the frozen packed-cache baseline, one middlegame
root, two fixed-work configurations and eight deterministically randomized
samples per arm, following one warmup each. GPU execution is exclusive, host
affinity is CPU2, and an external 60-second timeout bounds the run. Reports are
preallocated; trajectory normalization/comparison and final production PV
readback occur outside wall timing. Whole trajectories of score, full PV and
non-timing stats match between graph/queued. Every production advance's stats
and final score/PV match too. No search algorithm or node-budget shortcut is used.

## Result and decision

Strict gfx1030 compilation and all 48 timed searches pass. The
[raw rows, declaration and source binding](../gpu/evidence/causal-dispatch-gfx1030-2026-10-03/)
retain the new result without another duplicate source/binary archive.

| Work configuration | Queued / graph aggregate wall ratio | Production / graph context |
| --- | ---: | ---: |
| Batch32, cap50,000 | 1.00199 | 1.03185 |
| Batch1, cap5,000 | 1.00453 | 1.07365 |

Graph dispatch saves about 0.2%/0.45% of observed warmed wall time in this matched
control. Individual pairs vary across 1.0; eight samples, one root and uncontrolled
clocks/package effects do not support a strong statistical improvement claim.
Do not invest another graph-launch optimization cycle on this evidence.
The production context shows a larger aggregate difference, but does not assign
it to synchronization, clearing or bookkeeping individually.

The earlier profiler's large serial selection share remains a hypothesis for
the next kernel intervention, not causal proof of achievable speedup. Likewise,
whole-model eager INT8 losses do not establish that all GPU-native/captured
INT8 implementations lose. Keep graph/native neural integration as a separate
experiment with inference progress and identical outputs; this classical search
result cannot decide its value. Full GPU execution remains the target.

## Avoiding redundant work

- Name the decision, progress metric and smallest intervention before execution.
- Use one live implementation per contract and explicit file ownership. Shared
  rules/symbolic helpers remain shared; archived sources are historical snapshots.
- Once a source-bound result is reviewed, reuse it. Reopen it only for a changed
  dependency, discovered defect or new acceptance question.
- Verify changed boundaries and required checks. Repeat broad suites or archive
  replays only when an unresolved concern requires them.
- Preserve raw relevant observations, source commit/hashes and runtime context;
  default to a small record instead of copying entire source trees repeatedly.
- Keep GPU trials sequential. Parallel agents may review code, papers and proofs;
  independent GPU timing jobs must not overlap.
- Stop tuning a mechanism when a matched intervention shows negligible impact.
  Choose the next experiment from end-to-end progress, rather than isolated wins.

Subagents last stopped at the service usage limit; none is currently running.
Root integrates the preserved work directly. Restarted ownership should remain:
coherence owns contracts/review, RDNA owns search scheduling/hardware, fusion
owns selection/theory; neural implementations stay in `research/neural`.
