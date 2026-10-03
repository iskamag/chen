# Native current-state symbolic residual input

The HIP rules arena now exports observed board features and 12 symbolic channels
directly into Torch-owned GPU allocations. [ABI1](../gpu/chen_symbolic.h) and
[the implementation](../gpu/chen_symbolic.inc) retain exact position/history
identity; no teacher, future score or PV enters these observations. Private
attack helpers are shared through `chen_position_priors.inc`.

Channels contain incoming/outgoing pseudoattack counts for each side, pins for
each side, current king checkers, material phase and four normalized history
counters. Black positions use current-mover rank/color orientation. Raw history
and position hash/node ID/root epoch remain separate int64 identity tensors.
The header defines clipping, normalization and the reversible-history window.

Export preflights state/hash and the entire ancestor chain before writing.
Outputs require complete independent contiguous allocations, matching HIP
runtime/device, disjoint addresses and current epoch. The caller serializes the
arena and completes Torch producers before native entry; the native private
stream finishes before return. Torch retains an owned snapshot across subsequent
arena resets or destruction. This synchronous bridge performs two scalar status
readbacks; it is not a captured search epoch. Device allocations and dense
attack-matrix construction are absent from the native export call.

The neural worktree supplies `NativeSymbolicProducer` and `CompactSymbolicState`.
Existing symbolic and multihead transformer residual projections accept this
typed compact state without reducing a dense attack graph. Model arithmetic is
unchanged; initial zero projections preserve the existing baseline pathway.

[Actual gfx1030 test records](../gpu/evidence/native-symbolic-gfx1030-2026-10-03/)
bind the strict compiled source and library to six passing GPU tests (4.773 s).
Independent python-chess/feature oracles check 940 rows, including all children
of 32 sampled roots, both movers, special rules, pins, checkers, repetition and
deep ancestors plus retained root history. Feature/channel FP32 comparisons use
absolute tolerance 1e-7; history and identities are exact. Failure tests cover
stale epoch, insufficient capacity, bad range, aliases and unchanged result
publication. GPU consumer checks exercise value/residual gradients and prohibit
board downloads or host scalar extraction inside the tested hot path.

These checks establish export and consumer mechanics. Learned quality, timing
improvement and neural search deployment remain open. The next integration
should reuse admitted buffers and expose an asynchronous native graph phase;
capturing this synchronous C bridge directly would retain its host barriers.
