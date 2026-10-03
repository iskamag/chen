# CPU PVS and aspiration controls

The CPU alpha-beta backend now has independent PVS and aspiration controls.
Both remain disabled by default, preserving the current reference policy.
Evaluation, quiescence, move ordering and the history-aware TT key are unchanged.
This CPU experiment informs search work allocation; it does not replace the
full-GPU deployment target or establish playing strength.

PVS searches the first move normally, probes later moves with the integer window
`[alpha, alpha+1]`, and repeats an improving, non-cutting probe with the original
full window. Qsearch itself is unchanged. The mechanism follows the documented
[PVS policy](https://chessprogramming.org/Principal_Variation_Search).

Aspiration starts depth 2 onward around the **previous completed score**, doubling
the half-window after fail-low or fail-high until the window succeeds or reaches
the full range. Both sides widen; this is a conservative initial implementation,
not the more elaborate one-sided widening described in
[Aspiration Windows](https://chessprogramming.org/Aspiration_Windows).
Failed windows and interrupted retries never publish a new depth, score or PV.
Stop/node/time checks and evaluator make/unmake hooks remain active in probes
and retries. TT tags continue to use the original window.

Controls:

```text
UCI:   setoption name PVS value true
       setoption name AspirationWindow value 25
Bench: chen bench --depth 5 --pvs --aspiration 25
```

`SearchConfig.pvs` defaults to `false`; `aspiration_window` defaults to 0 (off).
The API half-window is `u16` and is bounded internally; CLI/UCI admit 0..32000 cp.
`Searcher::stats()` reports probe, full-research and aspiration-retry counts.
Bench JSON reports the selected controls; other backends do not implement them.

## Verification

Three search unit tests, 11 search integration tests and four UCI tests passed.
The changed boundary includes a no-cutoff Material minimax oracle with full,
fail-high, fail-low and boundary-touching windows; original-window TT tags and
optimal root moves are checked. Tapered controls compare TT on/off, qsearch on/off
and all four PVS/aspiration combinations over opening, recapture, en passant and
promotion positions. Incremental evaluator hooks remain balanced on abort.
A node limit exactly after a failed aspiration attempt forces the next retry to
abort: the last completed depth/score/PV and board/history are preserved.

Default-feature constructor compilation, UCI controls, formatting and
`cargo check --features hip` passed. The HIP check is compile-only and performs no
GPU work. No selective pruning, TT history redesign or evaluator change was added.

## Fixed-depth observations

One frozen optimized binary ran the existing six-position suite at depths 4/5.
Four Latin-order rounds compare baseline, PVS, 25 cp aspiration and their combination
with a fresh searcher/16 MiB TT and Tapered evaluator for each search. CPU affinity
was 5; the GPU peer had released its earlier correctness window. Each search had a
2M-node safety bound and each depth's process had a 60-second external watchdog.
All 192 rows completed without reaching the safety bound. Requested depth and score
match within each group; mate positions legitimately terminate at depth 1.
PV legality and complete board restoration are asserted. Node counts are identical
across repeated policy/position runs.

Ratios below are candidate/baseline totals across this suite and four rounds.
Elapsed is `SearchResult.elapsed`, a search-only wall measurement excluding fresh
searcher/TT allocation, process startup and later PV validation:

| Policy | Depth 4 nodes | Depth 4 elapsed | Depth 5 nodes | Depth 5 elapsed |
| --- | ---: | ---: | ---: | ---: |
| PVS | 0.6986 | 0.7192 | 0.8687 | 0.8768 |
| Aspiration 25 | 0.7921 | 0.8110 | 0.8339 | 0.8391 |
| PVS + aspiration25 | 0.7508 | 0.7697 | 0.8290 | 0.8320 |

The improvement is substantial on these fixed-depth controls, with clear local
regressions: aspiration 25 raises start-position depth 4 nodes 36.3% and elapsed 39.6%;
the combination raises elapsed 49.9%. PVS alone raises mate-case nodes 13.8% and
barely changes start-position depth 4 work. Kiwipete dominates aggregate work, so
these totals are not an average over representative game positions. Some very
small promotion/mate timings are noisy; clocks and other host load were not fixed.
Per-position paired observations and round-total ratios are retained without a
population-level latency claim.

PVS-only is the next sensible **equal-time playing-strength candidate**, with
aspiration retained as a separate control. No default adoption or Elo claim is
made here. Root owns the later frozen-binary paired match.

## Replay and integration

```sh
cargo test --lib search::tests -- --test-threads=1
cargo test --test search --test uci -- --test-threads=1
cargo check --features hip
cargo build --release --example classical_ablation --bin chen
timeout --kill-after=3s 60s taskset -c 5 \
  target/release/examples/classical_ablation 4 4 benches/positions.txt > depth4.csv
timeout --kill-after=3s 60s taskset -c 5 \
  target/release/examples/classical_ablation 5 4 benches/positions.txt > depth5.csv
python3 scripts/analyze_classical_ablation.py . --output summary.json
```

Source/compiler/binary pins, exact runtime argv, raw CSV/stderr, per-position
ratios and replayable analysis are in
[the evidence manifest](research/evidence/classical-pvs-aspiration-cpu-2026-10-03/manifest.json).
The verified binaries remain in the isolated `chen-classical/target/release`
directory with explicit artifact hashes. A rebuild is a new binary to pin before
new measurements. Integrate by cherry-picking the `research/classical-search`
milestone; no changes from other worktrees are staged by this task.
