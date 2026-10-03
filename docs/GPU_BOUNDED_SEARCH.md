# Experimental bounded GPU alpha-beta reference

`gpu/experiments/bounded_search.h/.hip` provides independent, synchronous,
nonrecursive device-owned negamax jobs using the existing exact HIP rules,
transitions, draw history and Material/Tapered evaluator. It is a correctness
foundation for the [algorithmic redesign](research/ALGORITHMIC_REDESIGN.md),
not a production UCI backend or a demonstrated performance/strength gain.

Each job declares its root, previous history (excluding that root), depth,
alpha/beta window, evaluator, mate-distance origin, node budget and quiescence
cap. It returns Exact/Lower/Upper against the original window, or Aborted with
zero score/PV/completed depth. Work counters survive an abort. Completed bounds
refer to the declared search policy, not the game-theoretic value of chess.
Resource-limited unfinished work never silently becomes a score bound.

Tactical quiescence searches captures/promotions and every legal evasion while
checked; stand-pat is disabled in check. Unresolved continuation at the q-ply
cap aborts unless a sound window cutoff already completed. Mate/stalemate takes
precedence over draw counters. `quiescence=0` supplies a static-horizon control.
Canonical move order is intentional. There is no capture ordering, PVS, TT,
selective pruning, iteration publication, deadline or cancellation interface.
The future wrapper must retain its previous completed root result on abort.

The first implementation launches one 64-thread block for at most 64 independent
jobs, with per-call allocations and host publication. It does not establish
scalable GPU occupancy or useful work allocation. Stack 64, depth 16, q-ply 32 and
node budgets up to 1,000,000 are explicit reference limits. Timing optimization
follows decision-quality comparisons; full GPU execution remains the target.

## Executed checks

Strict `-Wall -Wextra -Werror` compilation and an actual RX 6800/gfx1030 run pass:

- 33 fixtures, 131 checked result packets: 72 Exact, 20 Lower, 16 Upper and 23
  deliberate Aborted results across singleton, mixed, 64-job and reverse reuse.
- 19 exhaustive-reference comparisons backed by 11 distinct input configurations.
  Narrow-window inequalities are checked against cutoff-disabled reference
  values; Exact root moves belong to the optimal set. PVs are independently legal.
- Captures/recaptures, check evasions, EP/pins, castling, promotions, terminal
  mate/draw precedence, mate distance, budget boundaries and history repetition.
- Nine malformed host requests leave sentinel outputs unchanged.

Traversal/rules/history are independently computed with python-chess. Tapered
and hash arithmetic reuse the engine formulas; this is not an independent
evaluator/hash proof. Deep stack execution, deadlines, integrated UCI play,
performance and strength remain untested. Lower/Upper PVs are legal traces,
not certified optimal lines.

The first passing attempt is preserved separately. Review found its equal-eval
repetition fixture could hide a missed draw; the fresh reviewed attempt removes
Black's queen so the descendant draw is the preferred move over losing static
alternatives. It retains the same kernel, 33 fixtures and runtime scope. The
reviewed pass is the current witness. Runtime completion durations are logs,
not inference or search benchmarks. Both jobs retired; no GPU task remains.

[Evidence archive](research/evidence/bounded-gpu-search-2026-10-03/manifest.json)
pins source snapshots, actual shared libraries, prepared cases, raw results,
strict-build logs and runtime journals. The reviewed witness additionally records
actual device and loaded HIP/HSA/DRM providers. Root reconstructs every packet,
bound comparison and source hash without a redundant GPU replay.

## Build and replay

On the admitted gfx1030 ROCm machine, install python-chess for the independent
probe, then run from the repository root:

```sh
bash gpu/experiments/bounded_search_build.sh /tmp/libchen_bounded_search.so
PYTHONDONTWRITEBYTECODE=1 python gpu/experiments/bounded_search_probe.py \
  --prepare --cases /tmp/chen-bounded-cases.json
timeout --kill-after=3s 120s env PYTHONDONTWRITEBYTECODE=1 \
  python gpu/experiments/bounded_search_probe.py \
  --cases /tmp/chen-bounded-cases.json --library /tmp/libchen_bounded_search.so \
  --output /tmp/chen-bounded-results.json
```

`HIPCC` overrides the default `/opt/rocm/bin/hipcc`. Architecture overrides are
rejected. These commands rebuild current source; the evidence libraries bind
the actual historical runs. Use the archive snapshots for that exact closure.
