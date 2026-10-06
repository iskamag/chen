# Third-party source in research evidence

Chen's original code and artwork use the MIT license in `LICENSE`.

Archived Python chess oracle source is by Niklas Fiekas and python-chess
contributors, distributed under GNU GPL version 3 or later. Its license is
included at [licenses/python-chess-GPL-3.0.txt](licenses/python-chess-GPL-3.0.txt).
Upstream: https://github.com/niklasf/python-chess (chess module version 1.11.2).
These historical copies are development evidence, outside the Rust engine and
browser build:

- `gpu/evidence/semantic-resident-gfx1030-2026-10-02/source/chess/`
- `docs/research/evidence/synthetic-mechanics-2026-10-03/chess-oracle-source.py`
- `docs/research/evidence/synthetic-mechanics-balanced-2026-10-03/chess-oracle-source.py`

The archived Coz runtime and Rust marker license notices are retained in
`docs/evidence/coz-cpu-2026-10-03/licenses/`.

## Opening book data

The optional opening book is built from the Lichess
[`chess-openings`](https://github.com/lichess-org/chess-openings) dataset (ECO
volumes `a.tsv` … `e.tsv`). It is a collection of facts released under the
**CC0 1.0 Public Domain Dedication**. `build.rs` fetches the pinned revision at
build time; the data is not committed. See [the opening book notes](docs/OPENING_BOOK.md).
