# Chen, in your browser

A small local chess board playing the actual native Chen CPU engine. No browser
chess engine, external service or neural model is substituted. Legal moves and
game results come from Chen's rules kernel. Serious Chen defaults to tapered
evaluation and PVS. This is a playable development checkpoint, not a rated release.

From the repository root:

```sh
cargo run --release --manifest-path web/Cargo.toml
```

Open **http://127.0.0.1:8787**. Choose your side, time per move and resource level,
then start a game. Level 20 gets the selected time budget; lower levels additionally
cap depth and nodes using Chen's existing strength policy. Optional node/depth limits
can stop search earlier. Levels are not ratings.
Choose **Catnip Chen** for the deliberately distracted δ–θ search toy. Its two
sliders control score drift before a walker notices (δ, centipawns), and minimum
fish attention (θ). Higher δ keeps remembered scores stale for longer; higher θ
ignores more nonterminal moves. It shares Chen's exact rules and draw handling,
but is an uncalibrated experiment rather than the serious search. Its telemetry
reports explored ply, fail-meows, zoomies and ignored moves; explored ply is not
completed minimax depth. Backend and dials stay with that tab's saved game.

Promotion offers all four pieces. Threefold repetition and the fifty-move rule
are automatically claimed. The board is keyboard operable using Tab and Enter.

Each browser tab owns its game, persisted in session storage. The API legally
replays the submitted move history with Chen before searching. Requests enter a
queue of at most eight waiting searches, with one native CPU search active. The
selected time budget starts **inside search**, after queueing and replay. Queue
wait and actual search time are displayed separately. Cancel stops queued or active
work; the browser keeps its previous position. Results expire after two minutes.

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
