# Pinned paired match runner

`scripts/fastchess_match.py` executes the official Fastchess **v1.8.2-alpha**
Linux x86-64 release fetched by `fetch_match_tools.py`. It rechecks the archive,
manifest and fixed binary SHA before execution. Statistical calculations and
boundary decisions belong to that version of Fastchess. Chen independently
replays its games and checks its saved scoreboard.

```sh
python3 scripts/fetch_match_tools.py
python3 -m venv results/qa-venv
results/qa-venv/bin/pip install chess==1.11.2
results/qa-venv/bin/python scripts/fastchess_match.py \
  --config docs/harness/smoke-config.json \
  --openings benches/openings.txt --nodes 2000 \
  --max-games 4 --wall-seconds 600 --out results/runner-smoke
```

Select one budget: `--nodes`, `--depth`, or `--tc` (the runner's clock syntax,
for example `10+0.1`). The game cap must be positive and even. Each round plays
one opening twice with colors swapped, at concurrency one. Input `name|FEN`
rows become six-field FEN rows in an EPD book, preserving clocks and fullmove
numbers. Diagnostic mode may cycle a finite opening suite; it emits no
statistical acceptance decision. Node and depth limits are requested UCI limits;
different engines can implement their accounting differently.

Engine entries use the existing `base`/`candidate` config format. Python adapters
preserve argv boundaries, working directories and per-engine environment without
a shell. A null environment value removes an inherited variable. Declared
artifacts receive hashes. Declare actual binary, HIP plugin, model and sources
for GPU experiments; launcher hashing establishes only the launcher identity.
Engine configuration is frozen before launch. Engines must remain in the local
owned process group. Known Docker/SSH/ROCm wrapper commands are rejected because
terminating their host CLI cannot prove remote engine retirement. For container
matches, run the entire harness and pinned runner inside the engine's container,
using its local executable and library/model paths. Custom launchers must retain
the same local ownership contract. Remote wrapper capture remains available in
the separate local protocol harness, with its separately scoped lifetime limits.

For a statistical experiment, supply `--sprt declaration.json` with exactly:

```json
{
  "elo0": 0,
  "elo1": 5,
  "alpha": 0.05,
  "beta": 0.05,
  "model": "normalized",
  "sampling_design": "Describe the independently sampled, held-out opening pairs and their provenance here."
}
```

Those numbers illustrate the schema; choose and freeze them for the actual
experiment. Supported paired models are `normalized` and `logistic`. Candidate
is the first engine, so positive hypotheses favor candidate over base.
The driver requires distinct opening identities and enough opening rows for
the entire cap, preventing suite cycling in SPRT mode. The declared sampling
design remains the experimenter's responsibility: deduplication does not prove
independence, and project-authored mate fixtures or correlated opening lines
cannot substantiate a strength claim. Do not select openings, limits or stopping
rules after inspecting their match results.

Every admitted game must have a normal termination, legal replay, exact result,
matching header/body result tokens, ply count, expected initial FEN and expected
pair/round/color identity.
Available rule draw claims are admitted. Scoreboard WDL and all six native paired
bins must match the replay. Adjudication, time forfeit, illegal move, abandonment,
interruption, parse failure or missing pair fails the run. No score adjudication,
recovery, tablebase adjudication or move cap is enabled. The whole-run watchdog
retires the owned runner/engine process group and reports technical failure.
Cleanup also runs after normal exit, interruption or an exception while waiting;
handled SIGINT interrupts retain a failure summary. Abrupt process death is
outside this local cleanup guarantee.

After those audits, only a normal exit with Fastchess's explicit H0/H1 completion
message **before the declared cap** yields `external_sprt_H0/H1`. Reaching the
game cap yields `inconclusive_game_cap`, including a genuine boundary reached on
the final pair. This deliberately conservative classification avoids interpreting
finite game completion as statistical acceptance. Rounded printed LLR values
remain diagnostic; Chen does not reconstruct a boundary from them. An early stop
without a consistent completion message fails the experiment.

Saved evidence includes original inputs, transformed openings, exact runner argv,
tool provenance, launch adapters, harness source snapshots, stdout/stderr, engine
communication log, native runner state, PGN, independently replayed game records,
summary and hashes of every payload. Logs and incomplete artifacts survive failed
runs. A descriptive pentanomial summary emits no bootstrap interval. This is an
offline relative experiment; the eventual 2500 Lichess target needs separate
rating evidence in a declared pool.

## Verification checkpoint, 2026-10-01

The [saved driver verification](evidence/fastchess-driver-2026-10-01/verification.json)
includes actual pinned-runner node/clock fixtures, a depth-2 current-guard capped
SPRT fixture, illegal-move and whole-run watchdog failures, and a SIGINT retirement
control. Ordinary fixtures contain both mover colors and a nonzero halfmove
clock; all four games per successful fixture independently replay and match the
native scoreboard. Final configuration guards also recheck the earlier node/
clock records. Recorded source hashes distinguish execution snapshots from the
final audit source. All 20 harness tests pass, including header/body result
corruption, configuration corruption, literal argv/environment/PATH, technical
forfeits, capped stopping and a live orphan-child retirement control.

These authored mate fixtures establish protocol and result handling. No actual
SPRT boundary crossing, independent strength sample, long GPU match, remote
engine retirement or Lichess rating is claimed. Real declared acceptance matches
remain a separate gate.
