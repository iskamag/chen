# JPEG chess: one-second playable screening, 2026-10-15

**The JPEG-assisted PVS hybrid won the direct selection 4–0 against pure JPEG
minimax. The configured-2500 target remains unproven and was not reached in the
observed screening pairs.** The selected hybrid scored one draw and one loss
against that configuration, and lost both games against full Stockfish 19.

The purpose of this round is to select the strongest observed playable JPEG
variant and screen it against an explicitly configured opponent. These games
do not establish human/Lichess Elo or statistically demonstrate parity with the
declared opponent pool. `Stockfish 19 UCI_Elo=2500` identifies an engine setting,
not a calibrated human rating. No score-based resignation or evaluation
adjudication is used.

The baseline is Chen alphabeta with PVS enabled, book disabled and 16 MiB hash.
Pure JPEG v1 uses iterative minimax over exact codec relations and shared
terminal certificates, without alpha-beta or beam pruning. Completed interior
iterations cover legal replies, but a root safety gate can exclude unqualified
alternatives; this is not unconditionally full-width over all legal root moves.
The hybrid
uses a bounded JPEG certificate prepass followed by ordinary Chen PVS. It is
explicitly a JPEG-assisted conventional search engine. The filmed hybrid v0 is
the exact hybrid executable used in the direct selection games.
The executed hybrid v0 is a pinned first prototype. Later source corrections,
including the root qualification-gate correction, do not retroactively change
the frozen executable that produced these games.

Every match uses one second per move, one computation thread, 16 MiB hash, no
book, no tablebases and a 512-ply safety cap. Caps and technical failures remain
unscored. The arbiter accepts only python-chess rule outcomes, including legal
draw claims. Each opening is played with colors swapped. Selection uses
`open-game` and `queen-gambit`; pool screening uses the same `open-game` as the
baseline. Source-tree `scripts/match.py` records every pre-move FEN, UCI move,
wall duration, available engine depth/nodes/time information, PGN, final FEN,
configuration and binary hashes.

The baseline and direct selection run sequentially within their own match on
CPU 4. An identical-config runtime affinity probe confirmed that Stockfish's
UCI/worker threads and Chen's listener inherit only CPU 4. Other root-managed
screening matches run on separate physical cores; these are strength screens,
not isolated hardware performance benchmarks.

`scripts/jpeg_strength_audit.py` separately replays every PGN and JSON trace.
It verifies legal moves, move-by-move FENs, PGN/trace agreement, engine/color
assignment, exact final rule outcome, candidate score, cap/failure handling,
opening-file and launcher hashes, pair coverage, and actual move timings. This
is an independent post-match replay using python-chess, not a second rules
implementation. A 100 ms transport allowance is reported separately from the
requested one-second thinking budget; overruns never silently become wins or
losses.

| Screening | Games | Observed score | Exact endings | Audit |
|---|---:|---:|---|---|
| Chen PVS vs SF19 configured 2500 | 2 | 0/2 | Checkmate, 60 / 93 plies | PASS, 153 legal moves |
| Filmed hybrid v0 vs full SF19 | 2 | 0/2 | Checkmate, 66 / 57 plies | PASS, 123 legal moves |
| Pure v1 vs full SF19 | 2 | 0/2 | Checkmate, 44 / 39 plies | PASS, 83 legal moves |
| Pure v1 vs SF19 configured 2500 | 2 | 0/2 | Checkmate, 60 / 33 plies | PASS, 93 legal moves |
| Hybrid v0 vs SF19 configured 2500 | 2 | 0.5/2 | Threefold claim, 31 plies; checkmate, 43 plies | PASS, 74 legal moves |
| Pure v1 vs hybrid v0, two opening color pairs | 4 | Pure 0/4; hybrid 4/4 | Checkmate, 124 / 39 / 118 / 51 plies | PASS, 332 legal moves |

The baseline's maximum measured wall duration was 1002.36 ms; its Stockfish
opponent's maximum was 1005.34 ms. The filmed hybrid v0's maximum was 969.67 ms;
full Stockfish's maximum was 1005.91 ms. Neither pair had a technical failure,
inconclusive cap or move exceeding the 100 ms transport allowance.

The pure-v1 screening maximum was 971.80 ms across its full/configured
Stockfish pairs. The hybrid-v0 configured-pool maximum was 970.03 ms. All three
additional pairs passed separate trace audits with no technical failures, caps
or 1.1-second wall overruns. The hybrid's repetition draw is a valid rule result,
not an evaluation-based adjudication. One draw plus one loss does not establish
parity with the configured opponent or demonstrate an improvement over the
baseline: two-game samples are much too small for that claim.
The ordinary baseline uses its default aspiration setting of zero, while the
hybrid prototype configures an aspiration window of 25. These screens do not
isolate the causal strength effect of the JPEG prepass.

The direct selection completed all four games with no failures or caps. The
hybrid won both colors in each opening. This selects it as the **strongest
observed JPEG variant in this small comparison**; it does not rank the hybrid
against untested architectures or establish that it improves on ordinary Chen
PVS. Selection maximum wall durations were 966.71 ms for pure and 966.74 ms for
hybrid. The 332-move independent replay included six castlings and four
promotions. Across the six match runs, all **14 games and 858 legal moves**
passed the complete trace audit. Thirteen games ended in checkmate and one in
an exact threefold-repetition claim; none was adjudicated from an engine score.

Before direct selection, both frozen policies passed six one-second smoke
queries: start position, white/black mate-in-one, pinned en passant, castling
through attack and a promotion position. Every returned move was legal and
both mate-in-one fixtures were actually mated. Maximum observed wall duration
was 966.90 ms. This is a legality/timing gate, not playing-strength evidence.

The selection binary hashes are:

* Pure v1: `d3820f466ed095ee81a4a7cbc23dad2f40850886936c879c8ffc2d913637ef09`.
* Hybrid v0: `f42e48387ef7fb5cec50df5fb95197e5121cb6d68caeeb7856dff6b4a8228682`.

The exact same frozen hybrid executable is used in selection and the film.
The final auditor SHA256 is
`56b9a9c7a7bb282ae1df63bdfbbd28ae10c0968b6dcae968d0e677adadd96bdc`.

Raw baseline/selection artifacts stay under ignored
`results/jpeg-strength-20261015/`. Film and parent-managed pool screening
artifacts stay under ignored `results/jpeg-video-20261015/`. The audits retain
input hashes so a changed PGN, trace or engine cannot silently reuse a pass.

Reproduce the post-match check with the environment containing python-chess:

```sh
results/qa-venv/bin/python scripts/jpeg_strength_audit.py \
  results/jpeg-strength-20261015/pure-v1-vs-hybrid-v0 \
  --out /tmp/jpeg-selection-audit.json
```

The relevant raw audits are `jpeg-strength-20261015/baseline-audit.json`,
`jpeg-strength-20261015/selection-audit.json` and
`jpeg-video-20261015/{full-stockfish,pure-full-stockfish,pure-sf2500,hybrid-sf2500}-audit.json`,
relative to `results/`. Each run retains `metadata.json`, `games.pgn`,
`games.jsonl` and `summary.json`. The generic match summary's logistic estimate
is deliberately not used as a rating claim here; these few correlated opening
pairs support only descriptive screening outcomes.
