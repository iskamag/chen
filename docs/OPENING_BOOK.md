# Opening book

Chen can open from a real opening book. The book is the CC0 Lichess
[`chess-openings`](https://github.com/lichess-org/chess-openings) dataset,
fetched at build time and replayed through Chen's own rules. It is **off by
default**; nothing about the default engine, benchmark or match behaviour
changes unless a book is explicitly enabled.

## What it is

- **Source.** The five ECO volumes (`a.tsv` … `e.tsv`) of the aggregated
  `chess-openings` dataset, pinned to a commit and fetched by the root
  `build.rs` into `OUT_DIR`. It is public-domain (CC0) factual data.
- **Format.** A position graph keyed by Chen's Zobrist hash. Each authored SAN
  line is replayed move by move; the move counts weight the candidate edges, so
  common lines dominate and transpositions share a node automatically.
- **Selection.** A probe draws from the weighted candidates using a seeded mix
  of the seed and the position hash. Seed `0` returns the most common move
  (fully deterministic); any other seed varies per game while staying
  reproducible.
- **Validation.** Every authored line is parsed and applied with Chen's legal
  move generator. A line that does not parse, or a SAN token that is illegal,
  ambiguous or carries a wrong check marker, is rejected at load. If the
  dataset is missing or corrupt the book is simply unavailable — never wrong.

## Build-time fetch

`build.rs` downloads the pinned volumes with `curl` (falling back to `wget`)
and writes them into `OUT_DIR`. The fetch is **best effort**: if the network is
unavailable the volumes are left empty, the engine still builds, and the book
reports itself as unavailable. `curl`/`wget` honor the standard
`https_proxy`/`http_proxy` environment, so no proxy is configured in-tree. Set
`CHEN_OPENINGS_URL` to fetch from a mirror instead.

## Enabling it

UCI:

```text
setoption name Book value true
setoption name BookSeed value 12345   # 0 = most common move, else varied
setoption name BookPlies value 12     # half-moves of book play
```

Bench:

```sh
chen bench --book --book-seed 12345 --book-plies 12 --suite benches/reference-openings.txt
```

The browser exposes **Open from the book**, on by default, seeded randomly per
game so friends see different openings. Each book move is reported as
`info depth 0 book score cp 0 ...` in UCI and as `book`/`opening` telemetry in
the browser and bench JSON.

## Scope

The book chooses only legal, named opening moves; it never asserts a score. It
does not affect search correctness, the transposition table, draw handling or
the default policy. Book moves are drawn from authored lines, so this is a
play-variety and opening-consistency feature, not a strength claim on its own.
A descriptive 0.5 s / 1.0 s match against Stockfish 19 configured 1800 is
recorded in [the book evaluation](harness/book-stockfish-2026-10-07.md).
