# JPEG engine strength target

Playing strength is the primary objective. Kernel speed, tactical-suite accuracy,
and representation compression are supporting measurements, not substitutes for
completed games.

The user-approved operational target is **parity against Stockfish 19 configured
with `UCI_LimitStrength=true`, `UCI_Elo=2500`, at one second per move**, on one CPU
thread with 16 MiB hash. This is a declared engine-opponent benchmark, not a FIDE,
Lichess, or independently calibrated absolute 2500 rating.

Stockfish documents its configured Elo calibration at **120s+1s**, anchored to
CCRL 40/4; fixed `go movetime 1000` is a different condition. The limiter also
changes move selection deliberately. See the [official UCI option documentation](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html).

## Frozen match protocol

Before a confirmatory run, freeze candidate source and binary hashes, evaluation
artifacts, opponent binary hashes, UCI options, opening sample/seed, collector,
CPU allocation, stopping rule and adjudication rules. Select the candidate using
separate development games; do not tune it on the confirmatory openings.

Use 100 independently sampled opening positions, each played with colors
reversed: **100 opening pairs / 200 games per opponent**. Deduplicate openings
and near-identical transposition families before sampling. Repeating the same
few authored openings does not create 100 independent opening pairs.

Both engines receive `go movetime 1000`, one physical CPU core without competing
work on its SMT sibling, `Threads=1` where supported, `Hash=16`, and no pondering.
Disable opening books and tablebases for the main comparison; use identical
externally supplied opening positions. Keep game history intact. Start each game
with `ucinewgame`; retain ordinary within-game hash reuse. Record actual wall
latency and overruns separately from requested time. CPU only.

Save all PGNs, move traces, options and terminal reasons. Use rule-based outcomes
with a predeclared draw-claim policy. Safety-cap games remain inconclusive, not
invented draws. Report crashes, illegal moves, deadlines and incomplete pairs;
do not silently remove failures or select only completed favorable games.

Report W/D/L, points, complete-pair count, pentanomial pair outcomes, and a
pair-level 95% uncertainty interval. Resample whole opening pairs, preserving
color pairing; repeated families require clustering rather than a false
independence assumption. A fixed 200-game sample gives an initial estimate,
not guaranteed precision. Predetermine any extension or use a declared sequential
procedure; repeated inspection of ordinary confidence intervals is not a stopping
rule.

Observed 50% score is a point estimate of parity. An interval spanning 50% means
the data are compatible with parity, not proof of equivalence. A stronger
superiority gate requires the score's predeclared one-sided lower confidence
bound to exceed 50%. If practical equivalence is desired instead, declare its
score margin before collecting games and require the interval inside that margin.

## Opponent pool and interpretation

Use these distinct controls, preserving each opponent's results separately:

| Opponent | Role |
| --- | --- |
| Stockfish 19, configured 2500 | Primary operational target |
| Stockfish 19, configured 2300 and 2700 | Local strength ladder and limiter sensitivity |
| Frozen ordinary Chen PVS, Tapered, TT/quiescence enabled | Direct regression and architecture-contribution control |
| Ethereal 13.00 Standard (non-NNUE), full strength, frozen CPU build | Proposed independently developed engine-family anchor |
| Stockfish 19, full strength | Ceiling stress test and clearly labeled exhibition opponent |

Ethereal 13.00 Standard is a proposed future anchor, not an installed, measured or
calibrated opponent in the current evidence. Its [official release](https://github.com/AndyGrant/Ethereal/releases/tag/v13.00) distinguishes the free Standard implementation from the commercial NNUE version.
Verify the selected build, options and artifact hash before admission. If it produces only one-sided games, select a
separately frozen closer-strength engine for an additional informative anchor;
do not alter the confirmatory pool after seeing results. The full-strength
Stockfish lane need not consume the entire confirmatory budget if it is explicitly
predeclared as an exhibition/stress test rather than part of the rating fit.

The three limited Stockfish settings are correlated configurations of one engine,
not three independent engine families. A multi-opponent fitted rating is relative
to the declared pool and chosen anchor convention. Assigning the configured-2500
opponent the number 2500 does not independently validate that number. Publish
per-opponent scores and uncertainty alongside any fit; do not hide opponent-style
sensitivity in a single rating.

## Current evidence and presentation

Fresh screening reported by the coordinator: ordinary Chen PVS scored **0/2**
against Stockfish 19 configured 2500, and JPEG-assisted hybrid v0 scored **0/2**
against full-strength Stockfish. These tiny screens involve different opponents;
they neither establish ratings nor rank the two candidates. They establish no
attainment of the target.

A candidate using conventional alpha-beta after a compressed-relation prepass is
labeled **JPEG-assisted Chen**. Search over codec-derived legal relations is
labeled **codec search**. Neither is presented as a searchless breakthrough.
Choose the strongest tested candidate from completed development games, not from
kernel throughput or its name.

Video games must be actual logged games with attached PGNs. Display the engine
variant, Stockfish version and strength-limit setting, requested time budget and
playback speed. Full-strength exhibitions are separate from the configured-2500
challenge. State when a game was selected for presentation; a selected victory
or loss does not replace the complete match record.
