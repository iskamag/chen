# JPEG bots versus Stockfish configured 1800, half a second per move

Both players receive `go movetime 500`. Stockfish 19 uses
`UCI_LimitStrength=true`, `UCI_Elo=1800`, `Threads=1`, `Hash=16`,
`NumaPolicy=none`, no tablebases and no pondering. The JPEG policies use their
previously frozen executables: pure v1 and JPEG-assisted Chen PVS v0. No engine
code or options were tuned between these games.

Two authored openings (`open-game`, `queen-gambit`) are each played with colors
reversed. The pure lane is pinned to CPU 2 and hybrid to CPU 4. This shared-host
development screen is not an isolated rating experiment. Books are disabled;
only rule-based outcomes with draw claims end games. A 512-ply cap would remain
inconclusive.

| Candidate | Wins | Draws | Losses | Points |
| --- | ---: | ---: | ---: | ---: |
| Pure JPEG relation minimax v1 | 0 | 0 | 4 | 0 / 4 |
| JPEG-assisted Chen PVS v0 | 2 | 0 | 2 | 2 / 4 |

Pure loses both colors in both openings. The hybrid wins White in `open-game`,
loses Black there, loses White in `queen-gambit`, and wins Black there. All eight
games end in checkmate. The hybrid's 50% observed score is compatible with a
competitive matchup on this tiny sample; it is not a measured 1800 rating or a
statistical proof of parity. These results also do not isolate the effect of
halving the time budget: the earlier Stockfish screens used different opponent
settings and opening inventories.

Separate PGN/JSON/FEN/rule audits pass all **8 games / 674 legal moves**, including
complete color-pair coverage and unchanged engine executable hashes. There are
no illegal moves, crashes, caps or missing games. Requested budget and transport
allowance are recorded separately:

| Player | Median move wall time | Maximum |
| --- | ---: | ---: |
| Pure JPEG | 465.6 ms | 469.4 ms |
| Hybrid JPEG | 465.7 ms | 469.3 ms |
| Stockfish in pure lane | 500.5 ms | 505.6 ms |
| Stockfish in hybrid lane | 500.5 ms | 505.3 ms |

The candidates retain their 35 ms internal deadline margin. No measured move
exceeds the requested budget plus the declared 100 ms transport allowance.

## Video and architecture boundary

[Watch the hybrid's first game, 86 seconds](../../results/jpeg-halfsecond-sf1800-20261015/jpeg-vs-stockfish1800.mp4).
It is an annotated replay of the actual first game, which the hybrid wins.
The video identifies the configured-1800 opponent, actual move times, 500 ms
requested budget, conventional PVS assistance and 1 s/ply replay speed. Analysis
is full-strength Stockfish run after the game and never assists either player.
All four hybrid PGNs remain available alongside the film, including its losses.

“Pure JPEG” means codec-derived alternatives and shared reply/terminal proofs
inside iterative minimax. Nominal depth four means four half-moves; that was a
one-second starting-position observation, not a guaranteed half-second depth.
The conservative root safety gate can exclude unresolved moves, so minimax is
full-width only after that gate. The hybrid delegates conventional PVS after
the same proof prepass. The separate pyramid and persistent causal experiments
are not integrated, and neither bot reasons over reusable compressed classes
of arbitrary continuations. See [JPEG_PLAY.md](JPEG_PLAY.md).

## Reproduction and evidence

All raw games, configurations, protocols, audits, video and frozen previous
auditor/renderer sources are retained in ignored
`results/jpeg-halfsecond-sf1800-20261015/`. The compact committed evidence is
[`evidence/jpeg-halfsecond-1800-20261015.json`](evidence/jpeg-halfsecond-1800-20261015.json).

The auditor defaults to the older one-second protocol. This run explicitly uses:

```sh
results/qa-venv/bin/python scripts/jpeg_strength_audit.py \
  results/jpeg-halfsecond-sf1800-20261015/hybrid --seconds-per-move 0.5
```

A negative check rejects this run when the expected budget remains one second;
the explicit half-second check passes. The old one-second run also passes the
updated auditor. The video has 2,064 H.264 frames at 24 FPS plus AAC audio,
decodes to EOF, and its preview was visually checked. No GPU execution is used.
