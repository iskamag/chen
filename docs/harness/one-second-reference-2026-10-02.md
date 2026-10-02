# One-second classical reference

Chen scored **13 wins, 3 draws and no losses** in 16 games against Stockfish 19
with `UCI_LimitStrength=true`, `UCI_Elo=1320`. Both engines received one second
per move. Eight fixed authored openings were played with colors exchanged;
there were no technical failures, incomplete pairs or adjudicated evaluations.

The score is 90.625%, giving a relative logistic point estimate of +394.11.
Adding the opponent's configured strength produces a **1714 proxy estimate**.
This is an estimate against that setting, not a calibrated Lichess, FIDE or CCRL
rating. The small, correlated opening sample supplies no confidence interval.
[Stockfish describes UCI_Elo as a weakened-play setting](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#how-do-skill-level-and-uci-elo-work);
its name alone does not calibrate this different time control and hardware.

Chen used the classical alpha-beta/Tapered backend, level 20, Hash 16, TT,
quiescence and move ordering. Stockfish used one thread, Hash16, NumaPolicy none
and no tablebases. Engines ran sequentially on CPU0 with a 512-ply game cap;
separate profiling used CPU5 during part of the match. Physical cores were
separate, while shared cache/power effects were not controlled. This match does
not measure the GPU search or neural models.

Independent replay verifies all **1,085 legal plies**, paired identities,
results and draw/checkmate endpoints against PGN and JSONL. Every move's initial
exact FEN, engine identity, move, wall time and reported UCI work is retained.
Chen's 546 measured moves have median 1000.967 ms and maximum 1004.750 ms;
Stockfish's 539 moves have median 1000.495 ms and maximum 1006.082 ms. Wall time
includes protocol transport and scheduling; an engine may finish early.
Corrupted result and misattributed timing-witness controls are rejected.

[The frozen evidence](evidence/one-second-reference-2026-10-02/manifest.json)
contains the Chen executable, exact configuration and opening hashes, raw games,
timing traces, collector/statistics source, independent auditor/report and 23
passing harness tests. Executable provenance is the observed binary hash; this
run does not reconstruct the exact source/compiler build of that earlier
executable. The Stockfish binary matches the previously pinned official release.

This supplies a useful provisional baseline beyond the user's 750 expectation.
A larger opponent pool, sampled openings and separate GPU equal-time matches
are required to establish a rated engine and pursue 2500 Lichess.
