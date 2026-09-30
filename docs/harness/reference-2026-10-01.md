# Offline Stockfish reference, 2026-10-01

Chen recorded **61 losses, three draws and zero wins** against Stockfish 19 in
64 games at a requested **2,000 nodes per move** for each engine. Candidate
score was 1.5/64 = 2.34375%. All 32 openings completed both color assignments;
the pentanomial histogram was `[29, 3, 0, 0, 0]`. No technical failures or
incomplete pairs were recorded or found during independent replay.

This is a fixed-workload diagnostic of the classical alpha-beta/Tapered engine.
It supplies no calibrated Elo or Lichess rating and does not measure the neural
or HIP resident backends. The Fastchess/SPRT acceptance gate remains unfinished.

## Workload and opponent

The candidate used `Backend=alphabeta`, `Eval=tapered`, `Level=20`, `Hash=16`.
Stockfish used one thread, `Hash=16`, `Skill Level=20` and
`UCI_LimitStrength=false`. The harness ran serially, once per opening with colors
swapped, with a 600-ply cap. Both engines received `go nodes 2000`; this is a
requested node limit, not an equal time budget or a claim that internal node
definitions match. The archived data contains no per-move UCI node-counter or
clock trace, so replay verifies game evidence rather than exact budget accounting.

The 32 starts are project-authored standard legal six-ply sequences with related
opening families. Their FENs and source moves are archived and independently
validated. They are a deterministic, correlated diagnostic suite rather than
independently sampled openings. Engines started from each supplied FEN; PGN
replay begins there, without importing earlier opening repetition history.
PGN setup serialization can discard an ineffective raw EP target; the verifier
checks identical legally effective initial rule state, including rights/clocks,
and checks final raw-EP FENs exactly against JSONL.

The fetched official releases are
[Stockfish 19](https://github.com/official-stockfish/Stockfish/releases/tag/sf_19)
and [Fastchess 1.8.2-alpha](https://github.com/Disservin/fastchess/releases/tag/v1.8.2-alpha).
Fastchess was fetched/version-checked but did not run these games; the local
`scripts/match.py` collector did. `scripts/fetch_match_tools.py` verifies pinned
archive digests before extraction or execution and records executable digests.
The archived [tool manifest](evidence/reference-tools-manifest.json) retains
URLs, release names, licenses and actual version responses.

| Artifact | SHA-256 |
| --- | --- |
| Chen executable used | `6a908f52a0fd6d6bbb0cb9a0abf4a429b771593a0c4b3128b4dfea0958b2be68` |
| Stockfish executable used | `0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19` |
| Stockfish release archive | `9defc0d4e55d49c65a6d042f3e571a39fcea499ade6dbe741b53b8c65e03611f` |
| Fastchess release archive | `1003f920bebe841acdab5e6d7871e93171a4586857b3ec91c21ef5777f26ff96` |

These current local files were hashed again and matched the recorded match/tool
metadata. The tracked evidence package contains game data and hashes, rather
than redistributing engine binaries. Match metadata does not pin the exact Chen
build source/compiler snapshot; an audit-time checkout ID is explicitly separate
from executable provenance. Current harness/fetcher source snapshots record the
post-run safeguards and do not masquerade as the original run's exact source.

## Independent replay and statistics correction

An independent verifier, without importing the match harness or its statistics,
replayed **4,015 legal plies** with `chess==1.11.2`. It checked all opening moves,
valid positions, PGN parser errors, color/opening schedules, recorded headers,
results, plies, candidate scores, final FENs and completed pairs. It checked
`board.outcome(claim_draw=True)` before every move and at the endpoint. All 61
losses ended in checkmate; all three draws ended when a threefold claim became
available. The result policy includes available claims, including a claim made
by announcing a next move; it does not imply every draw endpoint has already
occurred three times. Corrupted result, illegal SAN and incomplete-pair probes
verify that the replay rejects inconsistent evidence.

The original [summary](evidence/reference-summary-original.json) and
[run log](evidence/reference-original-run.log) retain the old diagnostic paired
bootstrap interval `[0, 0.0546875]` and logistic relative-Elo calculation exactly
as emitted. **Neither is accepted as inferential evidence here.** Opening-pair
independence was not justified, and pair count alone cannot justify an interval.
The independent replay report explicitly sets accepted bootstrap/Elo/Lichess
inference to `null` and SPRT completion to `false`.

The current harness defaults `independent_pairs=False`; it emits a bootstrap
interval only with at least 30 complete pairs and the explicit CLI assertion
`--independent-opening-pairs`. That option records the caller's design assertion;
it cannot establish independence. It must remain unset for this authored suite.
The [pinned Fastchess manual](https://github.com/Disservin/fastchess/blob/v1.8.2-alpha/man.md)
documents external paired/SPRT operation. No runner integration or SPRT run was
performed in this stage.

## Evidence and replay

The package preserves [PGN](evidence/reference-games.pgn),
[JSONL](evidence/reference-games.jsonl), [metadata](evidence/reference-metadata.json),
the original summary/log, openings, tool metadata and source snapshots. The
[provenance manifest](evidence/reference-provenance.json) hashes every packaged
file and distinguishes match-recorded identity from audit-time observations.
The [replay report](evidence/reference-replay.json) retains per-game endpoints,
counts and input hashes; its [log](evidence/reference-replay.log) records the
independent audit output. The [negative-control log](evidence/reference-replay-negative-controls.log)
captures rejected corruptions. Existing harness/tool tests are preserved in
[their original log](evidence/reference-harness-tests.log).

```sh
python3 -m venv /tmp/chen-reference-audit
/tmp/chen-reference-audit/bin/pip install chess==1.11.2
/tmp/chen-reference-audit/bin/python docs/harness/evidence/reference-replay.py
```

Replay requires no engine binaries, account access, matches or GPU operations.
