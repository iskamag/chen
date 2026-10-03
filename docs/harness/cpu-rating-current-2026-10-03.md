# Fresh CPU one-second reference — 2026-10-03

The current CPU build completed 64 games in four declared lanes. The default
PVS-off engine scored **5 wins, 2 draws and 9 losses** against Stockfish 19
configured1800. PVS-on scored **21 wins, 11 draws and 16 losses** across the
configured1600/1800/2000 pool. The descriptive configured-setting proxies are
**1711** for default-off and **1844** for the separate PVS-on pool. Defaults remain
unchanged; this evaluation does not establish a Lichess rating or an architecture
winner. GPU search and neural training remain paused.

## Closed results

Every row contains all eight authored openings, each played with both colors.
All 32 pairs completed normally: no cap, technical failure, missing game or
unpaired finished game contributes to these scores.

| CPU variant | Stockfish configured setting | W / D / L | Points / games | Single-setting proxy |
| --- | ---: | --- | ---: | ---: |
| Default PVS off | 1800 | 5 / 2 / 9 | 6 / 16 | 1711 |
| PVS on | 1600 | 11 / 3 / 2 | 12.5 / 16 | 1821 |
| PVS on | 1800 | 6 / 4 / 6 | 8 / 16 | 1800 |
| PVS on | 2000 | 4 / 4 / 8 | 6 / 16 | 1911 |

The PVS-on pooled estimate is 1844.255 from 26.5/48 points, solving
`sum(points) = sum(n_setting / (1 + 10^((setting - proxy)/400)))`.
The default-off estimate is 1711.261 from 6/16 points and is excluded from that
pool. Individual proxies are shown to expose variation across anchors, rather
than hiding it behind the pooled point. These are Stockfish configured-setting
descriptions, not calibrated playing ratings. The authored openings are reused,
correlated controls; no independence, confidence interval or significance is
claimed. The small earlier direct PVS match is separate evidence and its
relative estimate is not added to these proxies.

## Frozen controls and source

The release binary was built with `cargo build --release`, no optional features,
from clean source `28c6f86caa57e360798afc38e9f50a33aa1729ed`. Its SHA256 is
`d979d1f2841f9f707ad2fd7bd619bc7068defebd7026136b09115aa477e0a37f`.
All lanes use this same binary: `Backend=alphabeta`, `Eval=tapered`, `Level=20`,
`Hash=16`, TT/quiescence/move ordering enabled, `AspirationWindow=0`.
Only PVS and the opponent's configured setting vary as declared.

Stockfish 19 is the same pinned official universal binary, SHA256
`0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19`.
It uses `Threads=1`, `Hash=16`, `NumaPolicy=none`, `UCI_LimitStrength=true`,
the table's `UCI_Elo`, and `SyzygyProbeLimit=0`, with its bundled default NNUE
and no EvalFile override. No tablebase adjudication is used.

Each game requests one second per move, with a 512-ply unscored safety cap.
The collector uses python-chess1.11.2 legal replay and rule outcomes with draw
claims enabled. The eight openings are Italian, Ruy Lopez, Scotch, Four Knights,
Petrov, Vienna, Open Sicilian and Alapin Sicilian, unchanged from the previous
eight-opening control. Their exact file SHA256 is
`58df68de6c75adad5782022060535ee445993ccc503037bcd8871083e9f5ac22`.

Four sequential lanes ran concurrently under declared `taskset` affinities
CPU1/2/3/4 for off1800/on1600/on1800/on2000. Engine children inherit each lane's
affinity. Saved launch commands establish that declaration; the saved games
alone do not independently certify affinity. Shared cache, power and clocks
were uncontrolled. GPU visibility was disabled, and no GPU/neural experiment
ran as part of this evaluation.

## Timing and lifetime

The supervisor started at **12:10:24 UTC**. The last lane closed at
**12:35:40 UTC**; total supervisor elapsed time was **1516.566 seconds**
(25m16.6s), within the predeclared 9800-second watchdog and five-second grace.
All four workers exited0 without timeout. The supervisor retired its owned
groups and an independent `/proc` check at12:36:04 UTC found supervisor7715
absent and no members of worker groups7721/7722/7723/7724. No additional engine
was launched to finish or replay this evaluation.

| Lane | Chen median / maximum move wall ms | Stockfish median / maximum move wall ms |
| --- | ---: | ---: |
| off1800 | 1000.996 / 1011.144 | 1000.648 / 1006.594 |
| on1600 | 1000.955 / 1019.148 | 1000.617 / 1006.625 |
| on1800 | 1000.946 / 1017.432 | 1000.579 / 1008.402 |
| on2000 | 1000.975 / 1017.098 | 1000.647 / 1008.034 |

Wall times include UCI transport and scheduling; one second is the requested
budget, not an asserted hard wall deadline. None exceeded1.1 seconds. Early
finishes and every individual timing remain in the raw records. UCI telemetry
describes reported search iterations and does not certify all interrupted work.

## Independent audit and retained evidence

The committed [pool auditor](../../scripts/audit_cpu_pool.py), `c5d25e2`, checked
the exact 16-game inventory per lane, 5810 legal plies, PGN/JSONL agreement,
trace FENs, colors, endpoints, options, source/binary/provider hashes, summary
counts, pair selection and timing. Its 11 focused positive/negative fixtures
were executed before the run closed. The audit reports `COMPLETE`, all four
collectors complete, and no excluded pair. No engine search is needed to audit.

The [evidence manifest](../research/evidence/cpu-rating-current-2026-10-03/manifest.json)
binds the exact declaration, audit, descriptive analysis, terminal/retirement
records, build provenance and a compact frozen archive. Its 167 regular members
retain all raw games, options, timing traces, logs, launch controls, executed
CPU binary, Rust sources/Cargo lock, match/provider sources, python-chess GPL license, and
Stockfish GPL license and bundled corresponding source. A cold byte review
checks every member's name, regular-file type, size and SHA256.

The existing 103153472-byte Stockfish binary is bound by path/hash, rather than
duplicated. Its official download pin, bundled default-network scope and
restoration requirements are in the external dependency record. The original
stage remains at `results/cpu-rating-20261003-_584k376`. Its declaration and
historical absolute provenance paths are unchanged; do not rerun its supervisor.
