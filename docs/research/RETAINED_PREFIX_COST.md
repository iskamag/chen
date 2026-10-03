# Retained selection: finite correctness and a usable cost model

The retained-prefix experiment removes repeated **cached traversal**, not
repeated full child reductions. It leaves the number of ancestor reservation
writes unchanged. This distinction favors measuring path reuse and ancestor
updates before increasing neural batch sizes or replacing the selector.

This review binds `retained_canonical_prefix_v1` in commit
`b3f370925f94e06801294c2369b3034f78fde7bc`, specifically
[the isolated selector](../../gpu/experiments/selective_retained_path.inc), SHA-256
`be32f65630518a7caa542c8f7f7b0aba2124c24cdbcda3422a1419dadbfd9719`.
Its executed baseline is `5a2c82b406b24a84a9f75266b477ef3e6c242434`.
The existing [source binding](../../gpu/evidence/selective-retained-path-gfx1030-2026-10-03/source-binding.json)
pins the compiled closures and libraries. This review adds no hardware runs and
does not re-audit that archive or the previous 22 Lean certificates.

## What is now machine checked

[RetainedPrefix.lean](../../proofs/RetainedPrefix.lean) imports the previously
checked `cached_winner_stable` and canonical tie theorem. Twenty new Lean/Std
theorems cover finite path certificates, stack/call accounting and distinct
reservation updates. The [new verification record](evidence/retained-prefix-lean-2026-10-03/verification/verification.json)
records actual compiler invocations, source/toolchain hashes and `#print axioms`
for these new theorems only. It compiles with warnings as errors, without
Mathlib, user axioms or admissions. The actual trusted foundations are
`propext` and `Quot.sound`, recorded per theorem; this is not a claim of an
axiom-free proof.

`childOpen` restricts every winning child to indices below a finite arena bound
and to the parent's child relation. Canonical choice maximizes a fixed integer
score, breaking ties toward the lowest child index. For the HIP mapping, that
score is the negative child score; contiguous canonical child order gives the
same tie rule.

`retained_restart_equivalence` proves that a trimmed old canonical path equals
the same-length prefix of a newly certified root-restart path, provided:

- scores, open counts, child membership and tie order stay fixed;
- reservations increase pointwise;
- the retained endpoint remains eligible and eligibility propagates upward;
- the reference path has the same root and reaches at least that endpoint.

The proof first trims the old certificate, propagates endpoint eligibility to
its ancestors, reuses cached-winner stability on every retained edge, and uses
canonical uniqueness to identify the two prefixes. It is an equivalence of
finite certificates. Reference-path existence, depth/termination, upward
eligibility and actual array representation remain premises.

`child_available_parent_available` proves the scalar sum step underlying upward
eligibility: if one child has remaining capacity and the other children have
nonnegative aggregate capacity, their parent does too. It does not establish
the implementation's parent/child sum invariant. Similarly,
`strict_ply_path_unique` proves path entries are distinct when their node ranks
equal their path indices. The candidate validates parent and ply on each push,
but the Lean model does not refine these C++ checks or sticky failures.

No theorem here proves full HIP control-flow refinement, signed-score sentinel
bounds, packed-word arithmetic, memory visibility, scratch lifetime, neural
numerics, global chess optimality or minimum GPU latency. The existing
[mechanical-policy research](FINITE_MODEL_STRATEGIES.md) separately states the
limited score-information obstruction and the assumptions of oracle-search
theory. Those information claims do not lower-bound this cache-hit workload.

## Source correspondence and work accounting

During one choose call, let attempt `i` have `r_i` retained edges and `d_i`
newly discovered edges. Its selected or rejected leaf has depth
`L_i = r_i + d_i`. Let `k` count **examined** candidates, including rejected
ones, rather than only admitted leaves. In the current bounds `k ≤ 1024` and
leaf depth is below the configured maximum ply, at most 64.

| Quantity | Root restart | Retained path |
| --- | ---: | ---: |
| Cached child-helper calls | `Σ L_i = C + R` | `C = Σ d_i` |
| Calls avoided | — | `R = Σ r_i` |
| Ancestor reservation updates | `A = Σ(L_i + 1)` | same `A` |
| Path pushes excluding root | — | `C` |
| Successful suffix pops | — | `P ≤ C` |
| Tail eligibility checks | — | `Q ≤ P + k` |

These are conditional execution correspondences, not counters extracted from
the timed executable. The Lean accounting takes the visit decomposition and
legal stack trace as inputs. `exact_call_accounting` and
`unchanged_reservation_accounting` certify the sums; `stack_accounting` bounds
pops; `stack_bookkeeping_bound` gives `pushes + checks ≤ 2C + k` when each pop
check succeeds or ends that attempt's trimming. `restart_calls_depth_bound`
gives the finite `Dk` helper-call bound when `L_i ≤ D`.

Every skipped prefix edge still has an eligible canonical child. In the
reference packed cache, the earlier discovery installed that child and its
cache remains valid. Revisiting the edge therefore succeeds through the cache
hit branch. The same exhausted suffix requires rediscovery and score scanning
in either algorithm. The retained variant primarily saves fixed traversal,
validation and dependent metadata loads, rather than `R` complete child scans.
This correspondence assumes the prior helper populated the packed cache;
it is source reasoning, not a new packed-cache refinement theorem.

The candidate also changes reservation traversal from following
`nodes[parent].parent` to loading contiguous path entries. The write count is
unchanged; the dependency chain and locality may improve. Conversely, the
candidate writes that path and performs trimming loads. A weighted work model
therefore requires measured coefficients. `weighted_work_improves` proves
only the conditional inequality: saved hit work exceeds new stack overhead.
It cannot determine those coefficients, occupancy effects or wall-clock time.

For a first attempt there is no retained edge. A batch with one attempt saves
no child-helper calls. With repeated long shared prefixes, `R` can grow as
`kD` while `C` stays small, but `A` still grows as `kD`. Candidate rejections can
raise `k` far above the admitted batch size, so using admission count alone to
estimate selector cost is misleading.

The path borrows `work->sparse.nodes[0]`, a 256-index bucket of existing scratch.
The source checks that the maximum path fits and validates every pushed parent
and ply. Backup rebuilds this scratch after choose and expansion. That phase
ordering, clearing the path between advances and preserving the packed cache
fields are essential implementation obligations outside this Lean model.

## What the existing timing supports

The previously recorded [hardware summary](../../gpu/evidence/selective-retained-path-gfx1030-2026-10-03/summary.json)
contains 12 balanced pairs per workload, six positions, exact matching
non-timing fields and the same compiled arms. This review consumes those
observations without replaying them.

| Workload | Geometric reference/candidate ratio | Time reduction |
| --- | ---: | ---: |
| Depth64, cap50,000, batch32 | 1.0526203 | 4.9990% |
| Depth4, cap2,000, batch8 | 0.9921086 | −0.7954% |

All large aggregate pairs favor the candidate; all six large-position effects
range from 1.83% to 7.55%. Small pair ratios range from 0.9363 to 1.0427 and
include small per-position regressions. The work model is consistent with
these observations, but does not establish their cause: depth, batch and cap
all differ across the two configurations. Clocks and shared package effects
were uncontrolled. There is no universal speedup, significance, neural or
strength result, and no default adoption.

## A decision for the full GPU and neural pipeline

Keep this candidate as a selectable larger-work arm. The useful next seam is
the still-serial reservation sweep over an already materialized path, rather
than another claim that canonical child search needs a full scan. Unique
strict-ply indices permit disjoint per-ancestor updates:
`distinct_reservations_commute` proves the natural-number updates commute.
This provides an arithmetic basis for an experimental wave-distributed
ancestor sweep. A HIP candidate would still need unique addresses, low-count
bounds, preservation of upper cache/mark bits and a uniform synchronization
before the next canonical selection. It must also preserve the borrowed
scratch lifetime. The theorem does not prove race freedom or that the extra
synchronization is faster. Sequential candidate decisions and trimming remain
dependencies even if that sweep is parallelized.

For a neural integration, retaining a prefix is valid only while all relevant
scores and open metadata are fixed inside choose. Evaluate admitted sibling
groups and back up their new scores after that choose; invalidate the path
before the next advance. Updating scores asynchronously inside the same choose
would invalidate the proof's key premise. The current experiment delegates
the external neural selector to the original uncached implementation, so its
classical timing is not a neural result.

Changing the batch size is a separate policy experiment: it changes how long
scores remain stale and potentially changes the searched tree. Neural kernel
shape/precision may also alter rounded scores and ties. An equivalence for
one fixed batch cannot establish equivalence across batch sizes. Each admitted
leaf contributes an indivisible full legal-sibling group; the neural row count
is `Σ legal_child_count`, not simply the selected leaf count. Scheduling should
compare the critical path through choose, row production, neural evaluation,
backup and publication, with real occupancy and synchronization costs.

An actionable next study would first record `k, C, R, P, Q, A` in a separate
instrumented arm, then compare four uninstrumented arms at the same scores,
batch and caps: restart/retained path crossed with serial/distributed ancestor
updates. Keep row batching and neural precision identical in that equivalence
study. Any subsequent batch-size change needs its own quality and latency
evaluation. This document declares no new hardware study or production change.

## Reproduce the new proofs

From the repository root, with the pinned Lean 4.27.0 toolchain:

```sh
python3 proofs/verify_retained_prefix.py --output /tmp/chen-retained-prefix-proof-replay
```

Use a fresh output directory. The script freezes both proof sources, compiles
the imported module once, compiles the new module, and audits only its 20
theorems. Compiler output, exact commands and trusted axioms are recorded in
that directory. The evidence includes two failed development sources and their
recorded causes; both preceded the successful frozen source. A separately
compiled cold replay passed and matched the new axiom-audit output byte for byte.
