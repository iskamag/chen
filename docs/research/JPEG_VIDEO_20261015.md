# One-second strength screen and bot-game videos

The primary objective is playing strength at a fixed one-second move budget.
The user selected a declared engine-opponent pool. The operational target and
the larger confirmatory design are in [JPEG_TARGET.md](JPEG_TARGET.md).
The present candidates have **not demonstrated that target**.

Two playable variants now exist:

* **Pure JPEG relation minimax v1:** codec alternatives, shared terminal/reply
  proofs, iterative minimax and the existing untuned static frontier. There is
  no alpha-beta, TT or quiescence in this policy. Its conservative root gate can
  exclude unresolved alternatives when a certified mate-one-safe move exists;
  full-width claims apply only after that gate.
* **JPEG-assisted Chen PVS v0:** exact relational terminal/refutation prepass,
  followed by ordinary Chen PVS, TT, quiescence and Tapered evaluation using the
  remaining move budget. This is an explicitly conventional-search hybrid.

Architecture and build instructions are in [JPEG_PLAY.md](JPEG_PLAY.md).

## Actual completed games

Each opening is played with colors reversed. Both bots receive one second per
move, one CPU core per match lane and 16 MiB hash configuration. Books, pondering
and tablebases are disabled. These are small development screens on a shared
host, not independently sampled rating games or isolated performance trials.

| Candidate | Opponent | Candidate W / D / L |
| --- | --- | ---: |
| Pure JPEG v1 | JPEG-assisted Chen v0 | 0 / 0 / 4 |
| Pure JPEG v1 | Stockfish 19 configured 2500 | 0 / 0 / 2 |
| JPEG-assisted Chen v0 | Stockfish 19 configured 2500 | 0 / 1 / 1 |
| Ordinary Chen PVS | Stockfish 19 configured 2500 | 0 / 0 / 2 |
| Pure JPEG v1 | Full-strength Stockfish 19 | 0 / 0 / 2 |
| JPEG-assisted Chen v0 | Full-strength Stockfish 19 | 0 / 0 / 2 |

The hybrid is the stronger observed JPEG-family candidate in this **four-game
selection**, winning all four head-to-head games. That does not prove a stable
rating difference. Its draw against configured 2500 is also insufficient to
claim parity. Ordinary Chen and the hybrid differ in more than their prepass
(including aspiration settings); these screens do not isolate a causal strength
gain from the JPEG representation.

All **14 games / 858 legal plies** independently agree between PGN, JSON traces,
exact pre-move and final FENs, engine colors, and python-chess rule outcomes.
There are no illegal moves, crashes, caps or missing color partners. Outcomes
are thirteen checkmates and one claimed threefold repetition. No evaluation-based
resign/draw adjudication is used. Every measured move remains within the declared
100 ms transport/scheduling allowance. JPEG candidate maxima are below one
second; the ordinary Chen baseline reaches 1002.36 ms including transport.
The 512-ply safety cap would remain inconclusive rather than become a draw.

Configured Stockfish Elo is not an absolute rating under these conditions.
Its [official calibration](https://official-stockfish.github.io/docs/stockfish-wiki/UCI-Protocol-and-Stockfish-Commands.html#uci_elo)
uses 120s+1s, anchored to CCRL 40/4. No numeric rating or uncertainty interval is
inferred from this small authored-opening sample. A confirmatory 200-game design
is specified, not claimed executed.

## Videos

These are annotated **rendered replays of actual logged games**, not desktop
screen captures. They show actual move wall times, reported depths, all moves,
side to move and terminal result. Playback is about 1.1 seconds per ply at
1280x800 / 24 FPS, with synthesized move clicks. Sidebar analysis is independent
post-game full-strength Stockfish analysis at 20,000 nodes per position. It never
feeds moves to either playing bot.

* Stronger screened variant: [JPEG-assisted Chen vs full Stockfish, 80.5 s](../../results/jpeg-video-20261015/jpeg-assisted-vs-stockfish.mp4).
* Pure experiment: [JPEG relation minimax vs full Stockfish, 56.7 s](../../results/jpeg-video-20261015/jpeg-pure-vs-stockfish.mp4).
* [Both hybrid exhibition PGNs](../../results/jpeg-video-20261015/jpeg-assisted-vs-stockfish.pgn).
* [Both pure exhibition PGNs](../../results/jpeg-video-20261015/jpeg-pure-vs-stockfish.pgn).

Each video uses the first recorded game of its color pair. The hybrid's first
game was declared for filming before launch. Both colors and every loss remain
available; the presentation does not select a favorable win. In the films the
hybrid loses after `33...Qh2#`, and pure loses after `22...Nb4#`. Full-strength
Stockfish is visibly labeled separately from the configured-2500 challenge.

The actual H.264/AAC files decode through EOF, with expected frame counts and
duration. Preview and terminal frames were visually inspected. Video manifests
pin MP4, game logs, PGNs, configuration, analysis provider and rendering source.

## A strength failure worth fixing

The pure film reaches this real position before White's sixth move:

```
r1bqkb1r/pp3ppp/2n2n2/2Ppp3/P7/2N2N2/1PP1PPPP/R1BQKB1R w KQkq e6 0 6
```

It chooses `6.Be3` at completed depth three, followed by the legal game sequence
`6...d4 7.Nxd4 exd4`. The pawn advance forks the bishop and knight, and the
recapture crosses the shallow static frontier. Post-game fixed-node Stockfish
analysis drops from -0.73 to -5.73 around `Be3`; this is a noisy diagnostic,
not an exact move-regret measurement. The game itself supplies a concrete
capture-sequence regression that immediate-mate accuracy does not cover.

The next strength experiment should address stable capture frontiers, tactical
refutations and evaluation quality, then measure complete games. A faster
universal attack map would not by itself resolve this observed decision error.

## Evidence and reproduction

Raw runs, immutable executed binaries, audits, videos and analysis are retained
under ignored `results/jpeg-video-20261015/` and
`results/jpeg-strength-20261015/`. The compact committed screen evidence binds
their hashes. The v0 executable is retained independently from the later v1
source/binary freeze; v0 results are not silently reassigned to v1.

Render a recorded match with:

```sh
python -m pip install -r scripts/requirements-video.txt
python scripts/render_chess_video.py \
  --run results/jpeg-video-20261015/full-stockfish --game 0 \
  --candidate-label 'JPEG-assisted Chen' \
  --stockfish results/match-tools/stockfish/stockfish-linux-x86-64-universal \
  --out results/jpeg-video-20261015/jpeg-assisted-vs-stockfish.mp4
```

The renderer checks the recorded budget, thread/hash options, legal transitions,
terminal result and exact final state. `--preview-only` prepares a still before
encoding; `--architecture` labels the pure policy accurately. ffmpeg with
libx264/AAC, Cairo and DejaVu fonts are required. No GPU execution is involved.
