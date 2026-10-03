# Completed search-bound composition

[SearchBounds.lean](../../proofs/SearchBounds.lean) models the new bounded-search
packet contract, separately from its HIP implementation. Eight theorems compile
under Lean4.27.0 with warnings treated as errors:

```sh
lean -DwarningAsError=true proofs/SearchBounds.lean
```

The compilation also prints each theorem's axioms. The observed dependencies
are only `propext` and `Quot.sound`; there are no admitted proofs.

The model's `Sound result value` is a premise supplied by a completed search,
not proved by this file. All comparisons must concern the same exact state,
history, evaluator, horizon and leaf policy. Child scores must be negated into
the parent mover frame before root composition.

The certificates establish:

- Negation preserves soundness while swapping Lower and Upper.
- A sound extracted lower bound can justify a beta cutoff; a sound upper bound
  can reject an action below alpha.
- Aborted results expose neither bound.
- A single child lower bound gives a root lower bound, while a root upper bound
  requires coverage of **all** child values.
- An exact candidate dominates the root only when all competitors are bounded
  beneath it.
- Interrupted publication retains the previous completed result. Actual
  publication also requires matching state/job epochs.

This identifies useful pruning and publication obligations before building a
parallel dependency scheduler. It does not establish quiescence's game-theoretic
soundness, rules/evaluation parity, kernel refinement, optimal scheduling,
playing strength or minimum elapsed time. The
[algorithmic redesign](ALGORITHMIC_REDESIGN.md) defines the implementation and
comparison program.
