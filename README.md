<p align="center"><img src="assets/chen.svg" width="720" alt="Chen, a two-tailed chess catgirl" /></p>

# Chen

A chess engine written from scratch in Rust, with a playable browser app and
experimental GPU search backends. The default CPU engine has no runtime dependencies.
It supports UCI, legal move generation, iterative alpha-beta search, quiescence,
transposition tables, and material or tapered evaluation. An optional opening
book built from the public-domain Lichess `chess-openings` dataset is fetched at
build time and can be enabled per game; see
[the opening book notes](docs/OPENING_BOOK.md).

The browser includes serious Chen and **Drowsy Chen**, a deliberately distracted
rollout experiment. Resource levels are not calibrated ratings.

## Run

Install Rust with Cargo (Rust edition 2024), then:

```sh
cargo build --release --locked
./target/release/chen
```

Load `target/release/chen` as a UCI engine in a chess GUI, or use stdin/stdout:

```text
uci
isready
position startpos moves e2e4 e7e5
go depth 4
```

Wait for `bestmove`. Send `stop` to interrupt a search and `quit` to exit.

For the browser app on Linux:

```sh
cargo run --release --locked --manifest-path web/Cargo.toml
```

Open **http://127.0.0.1:8787**. See [browser instructions](web/README.md) for
controls and server settings.

## Check

```sh
cargo test --release --locked
cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
./target/release/chen perft 5 # 4,865,609
```

A CPU-only [counterfactual codec experiment](docs/research/CHESS_CODEC.md)
composes sparse alternatives together. Its searchless policy is selectable with
`setoption name Backend value codec-policy`; see [how it works](docs/research/CODEC_POLICY.md).

Optional HIP experiments target AMD gfx1030 and require ROCm;
see [GPU setup](docs/GPU_INTEGRATION.md).
Optional [neural experiments](neural/README.md) live in `neural/` and focus on
compact recurrent tactical models. They are not required to build or play.
[Research status](docs/research/STATUS.md) records historical experiments and
their limitations. [Artifact policy](docs/ARTIFACTS.md) explains which evidence
is included in Git. MIT licensed; see [LICENSE](LICENSE) and
[third-party notices](THIRD_PARTY_NOTICES.md) for archived oracle source.
