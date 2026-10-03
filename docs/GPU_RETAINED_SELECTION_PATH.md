# Retained canonical selection prefix

The named research variant `retained_canonical_prefix_v1` retains the current
root-to-leaf path during one classical choose call. The production default
remains the packed best-child cache with root restart.

Scores and open counts are fixed while reservations increase. Each eligible
canonical child on a retained prefix stays canonical under those reservations,
as in the existing finite `cached_winner_stable` theorem. Fully reserved suffixes
are popped before rediscovery; rejected candidates still reserve their entire
path. The candidate writes paths into the first sparse-backup bucket, which the
subsequent backup rebuilds. No allocation, ABI, kernel launch or memory bound
changes. The external uncached selector delegates to the original implementation.
The prefix induction and HIP refinement are source reasoning and tests here,
not new machine-checked Lean claims.

[The staging script](../gpu/experiments/selective_path_stage.py) freezes baseline
`5a2c82b`, follows its actual local include closure and replaces only classical
choose. [The candidate](../gpu/experiments/selective_retained_path.inc) remains
separate from production. Build each staged arm with its existing `build.sh`;
`CHEN_GPU_LIBRARY` selects the resulting library without a new scheduler API.

## Actual gfx1030 result

[Compact evidence](../gpu/evidence/selective-retained-path-gfx1030-2026-10-03/)
binds executed libraries, exact sources, raw observations and failed setup runs.
Strict compilation passes. Forty-eight synthetic selector cases preserve ordered
admissions, offsets and every masked reservation count without changing nodes.
Thirty-three native cases preserve all 1,336 score/full-PV/counter snapshots,
including legal PV replay, depth64, batches1/32/256, capacity and history.

Each timing configuration has twelve balanced fresh-process pairs, six positions,
one warmup per arm, exclusive GPU ownership and CPU5 affinity. Every non-timing
benchmark field agrees. These are geometric paired ratios:

| Fixed work | Reference/candidate | Time reduction |
| --- | ---: | ---: |
| Depth64, cap50,000, batch32 | 1.05262 | 5.00% |
| Depth4, cap2,000, batch8 | 0.99211 | −0.80% |

All large aggregate pairs favor the candidate (1.0495–1.0567); all six positions
improve there, by 1.83–7.55%. Small pairs are noisy (0.9363–1.0427), with several
small regressions. Clocks and shared package effects are uncontrolled. There is
no significance, uncapped one-second strength or universal latency claim.
Keep the candidate available for larger batches; do not adopt it universally
from this mixed result. A future batching/admission policy needs its own timing.

The initial integration command referenced an absent staged fixture and failed
before GPU work. The first small harness still had hardcoded large-work argv;
its guard rejected the first warmup, with no measured pairs. The corrected
[generic pair runner](../gpu/selective_variant_pairs.py) derives actual argv and
declarations from one workload configuration, avoiding this duplicated control.
It reuses existing parsing/process-retirement helpers and successfully completes
the small comparison. Future selector variants should use this runner.
