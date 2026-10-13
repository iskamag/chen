# Tactical operator mechanics prototype

`neural/tactical_operator.py` implements one sparse recurrent processor over
piece, square and complete legal-action tokens. It is untrained, has no inference-latency measurement or deployment admission,
and has not played a game. CPU fixtures verify arithmetic/interfaces; the
[exact GPU producer](GPU_TACTICAL.md) now supplies resident
observations with an actual width8/steps2 forward/backward check. Neither scope
establishes chess strength or algorithmic generalization.

## Supplied evidence and learned computation

`TacticalBatch` contains FP32 piece `[B,P,Cp]` (`1<=P<=32`), square `[B,64,Cs]`
and legal-action `[B,K,Ca]` features. Feature widths and their schema are
explicit configuration. The default widths match the versioned `chen_tactical_exact_v1` CPU and native
producer schema; callers must select that schema explicitly in configuration. Node IDs concatenate the P piece slots, 64 canonical squares and K
action slots. `piece_squares[B,P]` binds object slots to canonical squares.
Piece identities are packet-local; persistent event memory is not implemented.

The caller supplies directed `edge_index[B,E,2]` (source, receiver), relation
IDs, evidence features, edge masks and an overflow flag. The model constructs
no topology, adds no reverse edges, and never builds an N-by-N attention map.
Relations distinguish occupancy/reverse occupancy, occupancy-blocked
pseudoattacks to squares, four own/enemy piece attack/support directions,
pinned-piece-to-king, pinner-to-pinned-piece, checker-to-current-mover-king and
action source/destination/victim/effect. Empty-square pseudoattack targets
remain expressible. Pin/check evidence is separate from pseudoattacks: a pin
must not silently remove a pinned piece's pseudoattacks.

Action IDs use `neural.features` canonical mover orientation and promotion
coding. Complete canonical IDs must be sorted and unique in valid slots.
En-passant victim and castling rook effects require explicit caller edges;
destination alone is insufficient. An E-capacity overflow rejects the packet
at cold admission instead of silently discarding edges. Incoming/outgoing
attack counts alone cannot produce this topology. A future native emitter
needs outgoing attack bitboards (or bounded sparse targets), object mapping,
legal action effects, and exact state/hash/history/epoch/stream binding.

`admit_contents` is a CPU structural check, optionally comparing the complete
action list to an independent oracle. It verifies finite active payload,
indices, endpoint categories, object squares, canonical source/destination
agreement and overflow. It cannot certify that attacks/pins/features or priors
are truthful. Strings identifying schemas are not evidence of causality.
The production producer must validate chess semantics and provenance.

One cell normalizes node latents and projects source and receiver nodes once,
then gathers those projections per edge. Exact edge features and learned
relation embeddings condition GELU messages. Sparse scatter operations compute
incoming mean, maximum and minimum; no incoming edges give exactly zero.
A gated learned MLP updates the latent, followed by normalization and masking.
The same cell weights execute 1, 2, 4 or 8 iterations. This supplies an
algorithm-like computation scaffold; it does not implement or guarantee SEE,
quiescence, minimax, exchanges or tactical correctness. Algorithm trace losses
and supervised training are future work.

The value readout pools piece, square and action latents and produces a CP
residual. Each action readout combines its token with the same pooled context.
Both heads initialize to zero, preserving supplied priors initially. Priors
must be causally calculated classical evaluations from the current state and
known legal action effects, in declared CP units. Teacher labels, later PVs,
future completed search results and hidden successor targets are forbidden
as forward priors or features. Supplying a truthful prior remains a producer
obligation; the model does not recognize its origin from a tensor.

Value and action scores use the CURRENT node's mover perspective. When the
batch represents a parent, higher action CP is better for that parent mover;
the action prior must already be in that perspective. Illegal/padded slots
score negative infinity and receive zero probability. Ties select the first
canonical action. No legal actions returns `-1` and zero probabilities. The
raw softmax is an arithmetic readout, not a calibrated policy. Exact terminal
value overrides (mate, stalemate, repetition and draw rules) remain simulator
responsibility, even when the learned value residual is finite.

## Comparable controls

- `steps=1/2/4/8` runs the same cell and checkpoint; extra iterations do not add
  parameters. `return_trace=True` is an optional mechanics/training diagnostic
  and should be excluded from inference timing.
- `typed=False` removes learned relation embedding contributions while keeping
  identical weights, exact feature payload, connectivity, priors and steps.
  It is a relation-label ablation, not removal of all typed symbolic evidence.
- `exact_child_control` runs the SAME model on independently supplied exact
  successor graphs with their complete child legal sets. A supplied mapping
  joins each parent legal-action slot to one child row. Child-mover values are
  negated into the parent-mover frame, without a parent-latent shortcut.
  `admit_child_mapping` verifies a CPU one-to-one row mapping, not actual
  successor identity; that requires the exact simulator. This reference may
  cost more than parent action processing and must be timed at equal budgets.

`config()` reports schema, relation vocabulary, arithmetic frame and untrained
status. `parameter_report()` separates total/processor parameters and explicitly
leaves measured runtime empty. The default width32 configuration has 18,850
parameters, of which 17,248 belong to the shared processor; all step counts
use the same parameter set. That default is a mechanics fixture, not a target
model size. Width has no arbitrary256 ceiling; `TacticalConfig.estimate()`
describes large settings without allocation, and constructor `max_parameters`
is an optional caller budget checked before weights allocate. See the
[scaling design](SCALING_STUDY.md) for separate size and compute experiments.
Approximate cell work is
`O(N D^2 + E Ce D + E D)` per iteration, with `N=P+64+K`; input/readout work
is separate. Source/receiver projections cost N D^2, not E D^2. This is an
operation model, not measured latency. Sparse edge-expanded tensors still have
storage O(E D), and scatter max/min gradients were checked on actual gfx1030 at width8/steps2; measured inference cost remains
open for useful learned shapes.

## Evidence and next boundary

Focused CPU tests cover an independent sparse reduction oracle, node-projection
equivalence, shared recurrent unrolling, typed/untyped controls, active
gradients, ignored poisoned padding, terminal rows, canonical ties, child-score
negation and mapping, admission failures, and complete python-chess legal sets
including black orientation, EP, castling and promotion. Forward and backward
are checked without host scalar extraction. No optimizer steps are performed.

The [exact CPU replay adapter](TACTICAL_INPUTS.md) now supplies complete sparse
relations, action effects and causal material priors under a versioned schema.
Its compact topology words match the independent native ABI.
`neural/native_tactical.py` now constructs an owned `TacticalBatch` directly
from native arena storage, with exact state/history/action/epoch admission and
terminal sidecars. This synchronous correctness bridge allocates output tensors;
it does not claim captured or allocation-free inference. The
[closed exchange targets](EXCHANGE_TARGETS.md) supply a separately typed
restricted task for the next learning study. First train
this processor and the static exact-child reference, with trace-loss and
recurrence controls. Persistent move-event memory and learned search-aware
refresh are extensions of this same processor after a quality signal, not
already implemented candidates. No held-out data has been consumed here.

The computation direction borrows step-supervised graph execution from
[Neural Execution of Graph Algorithms](https://arxiv.org/abs/1910.10593) and
learned operator composition from
[Neural Bellman-Ford Networks](https://proceedings.neurips.cc/paper/2021/hash/f6a673f09493afcd8b129a0bcf1cd5bc-Abstract.html).
Those papers do not validate this chess-specific prototype.
