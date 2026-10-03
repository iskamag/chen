# CPU tactical generation — 2026-10-03

Quiescence now generates captures and promotions directly outside check. It
previously generated and legality-tested every move, then discarded quiet moves.
Checked nodes still generate all legal evasions. Empty tactical lists use an
early-exit legality probe before stand-pat or draw handling, preserving stalemate.
The probe still constructs all pseudo moves; this is not a staged move picker.

The shared generator preserves full-generation order, quiet underpromotions,
en passant and castling in the full generator. No pruning, evaluation or search
budget changed. `legal_moves()` remains the full legal reference.

## Fixed-work comparison

Both binaries use Rust 1.98.1 release settings, Thin LTO, one codegen unit and
`RUSTFLAGS='-C target-cpu=znver2'`. Baseline source is `22c1895`, whose engine
sources match the evaluated `28c6f86`; candidate changes are this milestone.
Each arm uses fresh PVS/Tapered/16 MiB searchers, a depth cap of 64 and a
200,000-node cap per fixture. Four balanced rounds run sequentially on CPU5.
Shared clocks, power and caches remain uncontrolled.

| Fixture | Baseline median ms | Candidate median ms | Ratio |
| --- | ---: | ---: | ---: |
| Start | 300.968 | 135.395 | 2.223x |
| Kiwipete | 443.184 | 149.902 | 2.956x |
| Middlegame | 366.692 | 137.160 | 2.673x |
| Rook ending | 230.721 | 119.911 | 1.924x |
| Promotion | 236.622 | 115.884 | 2.042x |
| Forced mate | 0.0156 | 0.0222 | 0.706x |

Median round-total speedup is **2.388x**. The tiny early-mate fixture regresses;
it visits only 33 nodes and cannot support a useful timing conclusion. All 48
rows match node counts, quiescence counts, completed depth, scores, best moves
and complete PVs across arms. This is an equal-work throughput result, not a
strength or population-wide speedup claim.

One additional one-second candidate pass reports 1.18–1.66 million nodes/s over
the five non-mate fixtures. Completed depths are start7, Kiwipete6, middlegame7,
rook-ending10 and promotion10. These timed observations are not fixed-work
comparisons and establish no new Elo estimate.

## Verification and records

43 native-targeted release tests passed across library, core, search and UCI
suites, followed by all 61 default-feature portable release tests. New checks
compare tactical generation and terminal probes to full legal generation along
deterministic legal walks and special-rule fixtures. Direct quiescence checks
cover stalemate, checkmate at halfmove100 and quiet-only check evasions.
Clippy passes all targets with warnings denied; an equivalent existing test
assertion was simplified for its `int_plus_one` lint.

[Raw comparison](evidence/cpu-tactical-generation-2026-10-03/comparison.json)
retains the complete benchmark rows, binary identities and timings.
[Timed pass](evidence/cpu-tactical-generation-2026-10-03/one-second.jsonl)
retains the additional one-second observations. The checked binary predates the
test-only assertion simplification; production code is unchanged by that edit.

Reproduce the candidate, keeping the baseline executable separate:

```sh
CARGO_TARGET_DIR=results/tactical-generation-probe \
RUSTFLAGS='-C target-cpu=znver2' cargo build --release --locked
taskset -c 5 results/tactical-generation-probe/release/chen bench \
  --depth 64 --nodes 200000 --pvs --eval tapered --hash 16 \
  --suite benches/positions.txt
```
