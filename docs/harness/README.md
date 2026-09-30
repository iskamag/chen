# Offline evaluation and the Lichess target

Most development evidence is offline. The user's target is 2500 **on Lichess**,
plus adjustable play strength. Choose a specific pool/time control before the
final rating experiment. Lichess uses Glicko-2 and distinct populations; an
engine-match logistic Elo estimate is not a Lichess rating.
[Lichess's rating documentation](https://lichess.org/page/rating-systems).

## Local harness

`match.py` uses an optional `chess==1.11.2` QA environment. Engines are configured
from JSON commands/options/environment; the supplied config compares alpha-beta
without quiescence with CPU-frontier search. Change this file explicitly for
each experiment. `command[0]` is hashed as a launcher and must identify an existing file.
An optional `artifacts` mapping records additional binary/library/checkpoint paths.
Launcher-only provenance is explicitly labeled. For `scripts/rocm_run.sh uci`,
set `capture_rocm_artifacts: true` in that engine entry: the wrapper saves the
actual copied Rust executable, compiled HIP library, copied GPU sources,
compiler version, container image ID and SHA-256 manifest beneath the run.
Relevant environment overrides must be declared in the engine's `env` mapping;
undeclared inherited environment remains a reproducibility limit.

```sh
python3 -m venv /tmp/chen-match
/tmp/chen-match/bin/pip install chess==1.11.2
cargo build --release --features hip
/tmp/chen-match/bin/python scripts/match.py \
  --config docs/harness/smoke-config.json \
  --openings benches/openings.txt --depth 2 --max-plies 300 \
  --out results/my-match
/tmp/chen-match/bin/python -m unittest discover -s tests/harness -v
```

Select exactly one `--depth`, `--nodes` or `--movetime` (seconds per move).
Each opening runs with engine colors swapped. Concurrency is one, which avoids
competing GPU workers. Full PGN, game JSONL, config, binary hashes, opening hash,
limits and environment are saved. Outcomes are independently adjudicated by the
rules library, including available draw claims. A ply cap records an unfinished
game and excludes the entire pair; it is never silently converted to a draw.
An illegal move, timeout, crash or failed search records a technical failure and
fails the run. These remain implementation defects to fix before strength tests.

Complete pairs produce a pentanomial histogram and descriptive score. A logistic
relative-Elo point estimate is omitted at score 0 or 1. At least 30 pairs are
required even to emit a diagnostic paired bootstrap interval; it assumes
independently sampled opening pairs and is not a sequential acceptance test.
Tiny mate fixtures exercise result/PGN handling only. Neither small samples nor
deterministic opening repetitions justify an absolute rating inference.

For patch acceptance, use a version-pinned external runner with predeclared
SPRT hypotheses, clocks, openings, pair model, concurrency and stopping rules.
[Fastchess's primary manual](https://github.com/Disservin/fastchess/blob/master/man.md)
describes its paired-opening and SPRT options. Keep this local harness as a
protocol/legality oracle and result collector; a full statistical acceptance
runner remains an explicit next gate.

## Levels

UCI `Level` accepts 1..20. Level 20 preserves requested budgets. Levels below 20
cap depth and total node budget monotonically; stricter client limits remain in
force. The `StrengthPolicy` trait separates budget control from future learned
skill policies. Level numbers are not ratings. Test monotonic *playing strength*
and human-style behavior on held-out opening pairs before calibrating labels.

## Rating gate

Offline relative strength drives iteration. A calibrated offline comparison to
Lichess-rated reference opponents may estimate target progress with explicit
uncertainty and domain assumptions; it does not certify the account's rating.
The final claim requires observed Lichess performance in the selected pool,
reported sample size/provisional status/uncertainty, and a declared criterion.
No account access, matchmaking or online deployment is performed by this stage.
