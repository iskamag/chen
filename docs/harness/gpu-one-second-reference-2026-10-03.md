# One-second GPU selective reference

Production `hip-selective` with Tapered scored **4 wins, 5 draws and 7 losses**
in 16 games against Stockfish 19 configured with `UCI_LimitStrength=true` and
`UCI_Elo=1320`. Both engines received `go movetime 1000`. Eight authored openings
were played with colors exchanged. The match finished normally; independent
replay verified all **1,774 legal plies**, terminal results and per-move witnesses,
with no technical failures or incomplete pairs.

The score is 40.625%, yielding a relative logistic point estimate of −65.924,
or **1254.076 after adding the configured opponent setting**. This is a small,
correlated-workload proxy, with no asserted opening independence or confidence
interval. It is not calibrated to Lichess, FIDE or CCRL. Stockfish's
[weakened-play setting](https://official-stockfish.github.io/docs/stockfish-wiki/Stockfish-FAQ.html#how-do-skill-level-and-uci-elo-work)
does not itself calibrate this hardware and time control.

The earlier [CPU alpha-beta reference](one-second-reference-2026-10-02.md)
scored 13 wins and 3 draws on the same opening set, giving a configured-opponent
proxy of 1714. These separate diagnostic matches suggest a substantial search
quality gap. They are not a head-to-head comparison or a calibrated 460-point
rating difference. Neither match uses a neural evaluator.

## Configuration and timing

Chen used level20, Hash16, batch32, candidate limit1024 and an arena capacity
of 1,000,000 positions. The experimental retained-prefix selector was excluded;
the native library uses the production packed-cache selector. Stockfish used
Threads1, Hash16, NumaPolicy none and no tablebases. Both engines ran sequentially
on CPU0 in `cpp-rocm`, with a 512-ply game cap and rule-based endpoints. The GPU
was reserved for this match; Coz, wave and precision trials began after it closed.
Source/build preparation on separate cores occurred during the match. Shared
cache, clocks and power were not controlled.

Each engine made 887 measured moves. Chen's median wall time was **1001.391 ms**;
Stockfish's was **1000.620 ms**. Chen's first move took **1367.199 ms**, while all
886 later moves were below 1010 ms. Its first UCI report gave 1000 ms; the cause
of the difference has not been instrumented. Wall timing covers the entire
`engine.play` call, including protocol and scheduling. Forty-six Chen moves
finished below 900 ms; early search termination is permitted.

UCI nodes reached at most 946,645 classified positions. Classified work is not
arena occupancy, so this alone does not prove capacity never limited search.
Selective minimum-frontier depth includes terminal and closed leaves; depth2
does not prove a competing open branch remained that shallow. Selective depth
and node counts are not alpha-beta horizon/work equivalents.

## Provenance and next experiment

The Rust executable was freshly built from main `b3f3709` with Rust1.98.1 and
`cargo build --release --features hip`, before optional Coz instrumentation was
added. Frozen source hashes and executable SHA-256 are retained. The native
library matches the already compiled
[symbolic-producer source binding](../../gpu/evidence/native-symbolic-gfx1030-2026-10-03/source-binding.json),
SHA-256 `de50490a16d4395f6a71693fd33d6938a3c4f6d0b8660cf22535ece0e1bc8ad3`.
The collector records both binaries at startup, and closing hashes agree.

[Frozen evidence](evidence/gpu-one-second-reference-2026-10-03/manifest.json)
retains Chen and its HIP library, original declaration/configuration, raw games,
source/build pins, terminal journal, auditor and report. The 99 MiB Stockfish
binary remains in the local run directory; Git records its hash rather than
duplicating it. Existing harness tests were not repeated for this unchanged
collector; the independent closed-game replay is the new verification boundary.

A concrete next quality experiment is bounded tactical leaf scoring with the
selective scheduler unchanged. CPU alpha-beta has quiescence, while this GPU
policy scores nonterminal leaves immediately with static Tapered. Charge all
tactical work to the same budget, record unfinished closures, and compare
static/tactical selective arms at equal work and one second. This is a hypothesis
requiring implementation and fresh evidence. Keep allocation-policy changes and
neural deployment separate so their effects remain identifiable.
