# Reachable mechanics pilot

`scripts/generate_synthetic.py` generates replayable standard-start TRAIN
families with complete legal action effects and sparse attack-change targets.
This implements the first cheap tier of the
[data program](research/SYNTHETIC_DATA_PROGRAM.md); it does not label strategic
decisions, train a model or execute generation on the GPU.

The generation policy alternates uniform play and noisy one-ply material play.
The latter evaluates complete legal alternatives using only current action
effects, material, castling and check bonuses. It is deliberately a declared
weak policy, not stronger-teacher play. Phase anchors and event-priority samples
are separated by at least eight plies, with at most eight observations per game.
Games stop at the engine's current-state draw/mate rules or at192 plies; the
pilot gives no outcome labels. Event selection probabilities are unknown.

## Executed pilots

| Run | Families | Roots | Complete actions | Changed attack sources | Movers W/B | Generation wall |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| [Initial diagnostic](research/evidence/synthetic-mechanics-2026-10-03/report.json) |256|1940|35,533|95,790|1362/578|25.827s|
| [Corrected sampler control](research/evidence/synthetic-mechanics-balanced-2026-10-03/report.json) |16|120|2437|6580|60/60|5.217s|

The first diagnostic exposed an even-ply sampling bias. The live sampler now
alternates phase-anchor mover parity and fills event slots toward balanced
movers. The second run uses a different seed and demonstrates the correction;
it is not a large natural-distribution or strength study. Both raw runs are
retained. Their observed roots and planned legal children have
[zero overlap](research/evidence/synthetic-mechanics-balanced-2026-10-03/cross-pilot-audit.json).

The larger pilot contains1193 capture,533 check,331 pin,160 promotion,65 castling
and12 EP observation roots. These are nonexclusive **available legal-action or
current-state motifs**, not played-event counts or teacher-certified tactics.
It contains447 high-material,653 middle-material and840 low-material roots.
Rare repetition/high-clock strata did not appear and need guided reachable
generation. Quiet defenses and useful sacrifices still need stronger labels.

The larger compressed family shard is3,710,185 bytes, about1912 bytes/root.
Its measured throughput is75.1 roots/s or1375.8 legal effects/s. Reported wall
includes shard construction and writing the supplied exclusion index; it
excludes rebuilding that index from old admissions. The16-family control has
the same large fixed index-writing cost. These two timings do not establish a
scaling law or a production label budget. Strategic-teacher, tactical-trace,
GPU generation and end-to-end training costs remain unmeasured here.

## Observation and supervision boundary

Each family retains the full UCI prefix. Each root stores its FEN, complete
sorted schema1 action IDs, mover-oriented source pseudoattack masks, separate
pins/checkers, native-rule repetition hashes and retained reversible history.
Both source indices and target bits use black rank reflection; pinned pieces
retain pseudoattacks, sliders include the first friendly blocker, pawn masks
exclude pushes and EP. Root observations refer only to that exact current state.

The four history counters are halfmove clock, current-position repetition count,
retained window length including current, and ply from the **standard initial
continuation root**. This last counter equals source-game ply. A consumer that
resets an independent search root must explicitly reset/remap it, not claim an
identical native root context. The source family is not an arena node/epoch ID.

Action records retain exact parent-frame changed-square edits (including rook,
EP victim and promotion), complete sparse XOR changes of source attack masks,
child state/hash, typed terminal sidecar and simple parent-frame material.
Child boards, attack changes and terminal outputs are supervision. An inference
adapter may use causal current legal action effects; it must not silently feed
the entire future child/topology target to a parent predictor. These material
scores are not stronger-teacher estimates, calibrated CP or mate scores.

Supplied old input admissions, including rejected planned families, plus seven
spent reports exclude777,264 feature identities before sampling. Complete
admission action-root support and legacy two-ply root descendants are included.
The exclusion ledger and its source SHA bindings are retained. Families are
rejected whole on root/planned-child collision; early shared opening prefixes
do not merge games. This pilot has TRAIN only; no closed-test scores or labels
were read. Coverage is the listed files, not an assertion of a universal registry.

## Verification and replay

`scripts/audit_synthetic.py` independently walks scalar attack rays, replays
every standard-start legal prefix, checks every legal action/edit/attack delta,
history identity, terminal authority, canonical ordering, counts and supplied
exclusions. Both complete shards pass. Legality comes from shared python-chess;
the native hash formula is also shared, so this is not a second independent
chess implementation. Four focused tests additionally cover both mover frames,
castling/EP/underpromotion, deterministic replay, sampling parity, whole-family
rejection, overwrite protection and a rehashed corrupted-mask rejection.

With python-chess1.11.2 on `PYTHONPATH`:

```sh
python scripts/generate_synthetic.py --output /tmp/chen-mechanics-new \
  --families 256 --seed 20261005 --exclude-admission PATH_TO_INPUT_ADMISSION
python scripts/audit_synthetic.py /tmp/chen-mechanics-new
python -m unittest discover -s scripts/tests -p test_generate_synthetic.py -v
```

Generation refuses an existing output directory and exits2 if its bounded
attempt queue cannot reach the requested family count. Declaration, raw shards,
reports, scalar-ray audits, frozen generator/auditor sources and oracle source
are retained in the linked evidence. The python-chess oracle copy is GPLv3+
with its accompanying license; Chen's generator remains repository code.

Next: native sparse-edge ingress, a bounded tactical trace emitter, stronger
complete-root labels and convergence/HPO runs. The CPU pilot establishes usable
mechanics supervision; it does not establish that this data improves decisions.
