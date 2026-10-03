# Certificates for parallel bounded root-child search

[RootCertificates.lean](../../proofs/RootCertificates.lean) imports the existing
[SearchBounds](../../proofs/SearchBounds.lean) model. Its 23 new theorems compiled
with Lean 4.27.0 and warnings treated as errors. The actual per-theorem axioms
audit uses only `propext` and `Quot.sound`; no `sorryAx`, admissions, or new
axioms occur. [Frozen evidence](evidence/parallel-root-certificates-2026-10-03/verification.json)
contains commands, toolchain hashes, copied sources, and the actual audit output.
The dependency was recompiled to import it; previous proof archives were not
re-audited.

This is an abstract contract for the proposed root-child pool, not a proof of
its HIP implementation. The runtime design was coordinated with the GPU worker
agent before its new pool API/source was frozen. These proofs neither establish
strength nor choose a latency-optimal scheduler.

## Packet identity and score frame

`Fin n` contains **all** legal root actions in their canonical order. That
mapping and its completeness are assumptions. A candidate ID makes the action
set nonempty; terminal roots with no legal action need their separate terminal
publication path.

Each child packet carries its ID, a typed `Result`, and provenance: exact state,
retained history, evaluator, main horizon, leaf/qsearch policy, mate-distance
frame, and job epoch. Identity tokens represent equality of complete objects;
the proof does not turn a possibly colliding hash into equality. Evaluator
identity must include weights, precision/rounding policy, and any behavior that
changes scores. Batch-sensitive neural outputs cannot silently reuse a fixed
value certificate.

`Admitted` requires matching provenance/ID and a sound child-frame result for
the same fixed action value. `admitted_parent_sound` uses the imported negation
theorem: child Exact becomes parent Exact, child Lower becomes parent Upper,
and child Upper becomes parent Lower. The absolute mate-distance frame must
already match; negation does not establish a mate rebasing theorem. Bounds may
come from different valid windows or resource budgets, but not different
horizons, histories, evaluator versions, or leaf semantics.

Worker soundness is a premise. In particular, finite qsearch truncation must
have declared leaf semantics or yield Aborted, rather than pretending to finish
a different tactical closure. A paused DFS cursor supplies no completed bound.

## Exact publication and canonical ties

`Certificate` requires a completed Exact incumbent and a completed upper
endpoint for **every other legal child**. Each upper endpoint must be at most
the incumbent score. An earlier canonical ID requires it to be strictly less.
Thus a later equal-score child may be rejected with an equal upper bound; an
earlier equal-score possibility cannot be rejected that way.

`certificate_canonical_winner` proves maximum value and earliest-ID ownership
of ties. `certificate_all_children_maximum` additionally identifies the exact
root score. This is sufficient for root Exact publication, assuming the packet
admission and complete action mapping. A Lower/Upper search PV is not thereby
an exact or canonical line.

`certificate_retained_under_tightening` preserves the certificate when the
incumbent Exact is retained and every previously known upper endpoint is
replaced by an endpoint no larger. `same_family_tightening_preserves_winner`
requires admission of both sets for the **same** provenance family and value
function. The endpoint relation alone grants no worker soundness or identity
check. A larger exact incumbent score can be useful in a scheduler, but is not
the retention theorem's fixed-incumbent case.

Aborted has no parent bound and an aborted competitor packet cannot satisfy a
certificate. The imported `interrupted_publication_preserves` rule retains a
previous completed result on abort. To use it on device, the pool must retain
the whole prior immutable packet and match its epoch/provenance; it must not
overwrite that packet with the paused job's Aborted result.

## Proposed challenger window

The coordinated scheduler proposal gives a challenger child the window
`[-32000, 1 - incumbentScore]`. A completed fail-high lower endpoint at least
`1 - incumbentScore` negates into a parent upper endpoint at most
`incumbentScore - 1`. `integer_challenger_cutoff_strict` proves strict rejection
for integer scores. This stronger policy rejects ties for every challenger, so
potential tied winners must finish Exact and the reducer can choose their
earliest ID. It can spend more work than the sufficient later-ID equal-bound
rejection rule.

Window admission is explicit: `-32000 < 1 - score <= 32000`, equivalently
`-31999 <= score <= 32000`. Scores in `[-30000,30000]` satisfy it. An incumbent
of `-32000` does **not**: its proposed beta would be `32001`. The runtime must
prove its actual score-range invariant or choose another admitted window and
handle that boundary separately. No theorem here validates clamping, terminal
special cases, or overflowing machine arithmetic. The integer model needs a
separate representability/refinement argument for HIP `int` operations.

## What an unresolved competitor forces

The adversarial model has a finite integer interval for every child and permits
every vector in their Cartesian product. Given a feasible value vector, if a
competitor's upper endpoint exceeds the exact incumbent score, there is another
feasible vector with the incumbent unchanged where it loses. An earlier ID
whose endpoint can tie the incumbent also provides a counterexample to
canonical ownership. These are constructive `spike` witnesses, checked by
`unresolved_competitor_obstruction` and `earlier_tie_obstruction`.

`unresolved_not_universally_safe` therefore forbids universally sound incumbent
publication from those intervals alone. It is an information obstruction, not
a compulsory query of that specific child: another oracle observation could
reveal shared structure or improve the incumbent. Chess children are related;
the independent-interval oracle is an explicit conservative model, not a
proved property of their minimax values. No minimal-query theorem is claimed.

For the pool, this supports a concrete decision rule: retain same-horizon
completed bounds; exclude already dominated children from further certification
work; schedule unresolved children or improve the incumbent; publish Exact only
after complete competitor coverage. Lower-depth scores and deterministic neural
estimates may prioritize work but cannot fill missing certified endpoints.

Scheduling latency still depends on search work revealed by each window,
resident worker imbalance, epoch/launch/fence costs, evaluator batch size, and
publication overhead. No fixed cost or oracle-resolution model is proved here,
so the information obstruction gives no globally optimal batching or performance
claim. Those costs must be measured with the same task semantics and retained
packet policy before choosing scheduling priorities.

## Replay and remaining refinement

From the repository root, with Lean 4.27.0 installed, choose a fresh output:

```sh
python3 proofs/verify_root_certificates.py --output /tmp/chen-root-certificate-replay
```

The verifier copies the two modules, compiles their `.olean` files, generates
`#print axioms` for all new theorems, and rejects any actual axiom outside
`propext`/`Quot.sound`. It records the executable and `Std.olean` hashes. It does
not run an engine, access GPU hardware, or replay older evidence archives.

HIP refinement remains open: exact legal-ID coverage, history/evaluator identity
admission, finite-horizon/qsearch worker soundness, score representation, correct
packet negation, stack/cursor resume semantics, cooperative scheduling, memory
ordering, and epoch-safe publication/cancellation. The proof supplies conditions
to check in that implementation, rather than claiming they are already true.
