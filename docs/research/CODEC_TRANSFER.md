# Linear transfer scans and shared codec geometry

The original exact counterfactual codec recomputed each target's near-to-far
ray fold independently. The same prefixes therefore repeated quadratically
within each board line. The refined implementation visits each line twice,
maintaining a nearest-first `Boundary` in each direction. It answers the target
**before** prepending that target's blocker/attacker symbol. This preserves
attacks on occupied targets and prevents a farther slider from leaking through
a nearer blocker. Each operation still acts on 64 alternatives simultaneously;
there is no per-lane board decode, successor construction, orthodox move-generator
call, or search tree inside the relation builder.

All four line families together contain 256 square memberships. Two directions
and two colors now require **1,024 boundary compositions per alternative word**,
versus 2,912 originally. Additional alternatives scale by words, not another
per-target ray walk. The public relation API and algebra are unchanged.

Further changes remove repeated representation work:

- Constant line geometry and membership tables are shared across roots. A root
  keeps 32 symbol bytes, 46 occupancy bytes, and rule metadata. Encoding iterates
  occupied piece bitboards instead of probing every square against piece types.
- Pawn, knight, and king attacks propagate actual nonzero source planes through
  the rules kernel's precomputed attack tables. A union of base and edited source
  squares avoids repeatedly scanning empty source planes.
- Scratch planes and retained attack answers use fixed-size arrays within each
  alternative word. This removes dynamic word-stride indexing from both transfer
  construction and public attack queries. Retained output remains 1,024 bytes per
  alternative word; this is **not** output compression against the scalar
  control's two ordinary attack bitboards per candidate.

## Validation

All six `tests/codec.rs` release tests pass. Existing tests compare complete
candidate attack relations against materialized children, legality against the
orthodox kernel, special moves and castling transit obligations, multiword
promoted positions, seeded legal walks, and known perft counts. New focused tests
check target exclusion/nearest blockers and the exact linear composition bound.

Also passed:

```sh
cargo clippy --release --lib --tests --examples -- -D warnings
cargo fmt --check
git diff --check
```

The tests use the orthodox kernel as an oracle; the relation implementation does
not use its move generator or construct child `Position` values.

## Bounded CPU measurement

AMD Ryzen 5 3600, rustc 1.99.0, ordinary release profile, CPU 4 affinity, shared
host, CPU only. Original (`46de06d`) and refined codec implementations were built
with the same finalized `codec_probe` from the piece-union control experiment.
Four old/new runs alternated process order. Each run uses 512 deterministic
positions, 20 repeats, seed `8ac6c0de20261015`, and 15,865 total candidates.
Before timing, every child attack mask agrees among the codec, per-square
`attackers_to` queries, and optimized scalar source-piece unions.

The codec measurement includes root encoding, candidate proposals, the retained
relation, and consuming all candidate/color/square answers. The strong scalar
control receives candidate packets free, constructs children, computes their
complete attack unions, retains both attack bitboards per candidate, and consumes
the same Boolean answers. This is the relevant control; repeatedly asking for
all attackers at every square is much slower and overstates the codec's benefit.

Elapsed microseconds per root:

| Run | Original codec | Refined codec | Scalar piece union in refined run |
| --- | ---: | ---: | ---: |
| 0 | 14.472 | 3.531 | 3.530 |
| 1 | 13.448 | 3.526 | 3.269 |
| 2 | 13.267 | 3.292 | 3.268 |
| 3 | 13.504 | 3.330 | 3.313 |
| Median | 13.476 | 3.428 | 3.291 |

The refined codec is **3.93x faster than the original codec** on this workload.
It remains about **4% slower** than the median strong scalar control, with
individual refined-run ratios ranging from parity to 1.079. Host noise and fixed
within-process method ordering limit precise small comparisons. This establishes
a substantial mechanism improvement and approximate scalar parity, not an
advantage over the optimized scalar control or a playing-strength breakthrough.
All runs had `attack_checksum=20793680`, `legal_checksum=283620`, and unchanged
retained attack bytes (524,288 across the corpus).

After integrating the finalized piece-union probe, reproduce with:

```sh
cargo build --release --example codec_probe
taskset -c 4 target/release/examples/codec_probe 512 20
```

The ordinary legal-moves-only control performs a smaller workload and remains
faster. No GPU execution, self-play strength test, or deployment is claimed.

[Compact raw observations and source/binary pins](evidence/codec-transfer-20261015.json)
retain the eight executed probe outputs and exact comparison inputs.
