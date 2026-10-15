# Chen, in your browser

A small local chess board playing the actual native Chen CPU engine. No browser
chess engine, external service or neural model is substituted. Legal moves and
game results come from Chen's rules kernel. Hybrid Chen is the default opponent:
she checks forcing replies before running Chen's tapered evaluation and PVS
within the remaining move budget. Serious Chen remains available. This is a
playable development checkpoint, not a rated release.

From the repository root:

```sh
cargo run --release --manifest-path web/Cargo.toml
```

Open **http://127.0.0.1:8787**. Choose your side, time per move and resource level,
then start a game. Level 20 gets the selected time budget; lower levels additionally
cap depth and nodes using Chen's existing strength policy. Optional node/depth limits
can stop search earlier. Levels are not ratings.
Choose **Drowsy Chen** for the deliberately distracted δ–θ search toy. Its two
sliders control score drift before a walker notices (δ, centipawns), and minimum
fish attention (θ). Higher δ keeps remembered scores stale for longer; higher θ
ignores more nonterminal moves. It shares Chen's exact rules and draw handling,
but is an uncalibrated experiment rather than the serious search. Its telemetry
reports explored ply, fail-meows, zoomies and ignored moves; explored ply is not
completed minimax depth. Backend and dials stay with that tab's saved game. Reload older tabs to pick up
the Drowsy name; games saved under the former toy name migrate without losing
the board or settings. **Catnip Chen** is reserved for the ambitious
[batched-search candidate](../docs/research/CATNIP_SEARCH.md), currently design only.

The move list uses standard algebraic notation, including `O-O` / `O-O-O`,
captures, promotions and check/mate marks. Previous saved histories are
reconstructed on reload. Chen's pounce reaction celebrates her own captures.
The time menu offers 0.1, 0.5, 1 and 5 seconds; old 3-second settings become 1 second.

**Open from the book** is on by default: Chen plays varied named opening lines
from the [public-domain Lichess opening dataset](../docs/OPENING_BOOK.md) for the
first few plies, seeded per game so friends see different starts, then search
takes over. Uncheck it to make her search every move from the first.

Promotion offers all four pieces. Threefold repetition and the fifty-move rule
are automatically claimed. The board is keyboard operable using Tab and Enter.

Each browser tab owns its game, persisted in session storage. The API legally
replays the submitted move history with Chen before searching. Requests enter a
shared queue of at most eight waiting searches. Six independent search workers
let friends play concurrently; each game's engine search uses one thread and its
own position, history and TT. Set `CHEN_WEB_WORKERS=1..6` to reduce concurrency.
HTTP requests use separate short-lived threads, capped at 32 connections. The
selected time budget starts **inside search**, after queueing and replay. Queue
wait and actual search time are displayed separately. Cancel stops queued or active
work; the browser keeps its previous position. Completed results are retained up to two minutes from submission, subject to
the 64-result cap.

The server binds to loopback. Set `CHEN_WEB_PORT` to choose another port. Ctrl+C
stops it. This Linux checkpoint uses `/dev/urandom` for opaque job identifiers.
The public checkpoint is live at `https://iskamag.com/games/play-with-chen/`; a proxy
strips that prefix and forwards to this machine. See [deployment.md](deployment.md).

This crate builds against the current checkout. Rebuild after changing the engine;
its displayed base revision and tracked-engine dirty suffix are captured at compile
time. Each serious request uses a **fresh transposition table** rather than the persistent
UCI searcher used in matches. It therefore does not inherit the reported match
strength, despite using the same search implementation. No GPU work is started.

After building, run the bounded API/queue checks:

```sh
python3 web/smoke.py web/target/release/chen-web
node web/client-race-test.js
```
