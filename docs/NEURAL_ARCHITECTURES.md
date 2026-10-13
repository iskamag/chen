# Chen latent architecture laboratory

This branch tests chess-specific compositions from independently written code.
It has three executable models, differentiable move dynamics, explicit latent
branching, and a GPU-resident training prototype. It does not yet have trained
playing strength, calibrated WDL, a production inference bridge, or evidence of
2500 Elo. Architecture names below identify hypotheses, not novelty claims.

## Prior art and what these experiments change

* [MuZero, Schrittwieser et al.](https://arxiv.org/abs/1911.08265) established
  learned dynamics with policy/value prediction for planning, including chess.
  Chen retains exact chess rules and uses learned dynamics as an auxiliary and
  potential candidate-ranker. The initial implementation is not MuZero search.
* [AlphaGateau, Rigaux and Kashima](https://arxiv.org/abs/2410.23753) already
  applies graph representations and edge-aware attention to chess. Its
  [author implementation](https://github.com/akulen/AlphaGateau) is a concrete
  comparator. Merely adding chess attack edges is not a new architecture.
* [Perceiver, Jaegle et al.](https://proceedings.mlr.press/v139/jaegle21a.html)
  uses learned latent bottlenecks and cross attention. Chen's plan-token
  bottleneck is a proposed chess composition of this established idea.
* [Slot Attention, Locatello et al.](https://arxiv.org/abs/2006.15055) binds
  exchangeable slots to objects. Chen's implemented slots are square indexed;
  they do not implement competitive object binding. Persistent piece slots
  remain a future alternative, with capture identity and promotion specified
  before replacing square latents.
* [SPR, Schwarzer et al.](https://arxiv.org/abs/2007.05929) predicts future latent
  states with an EMA target encoder. Current Chen training uses a detached
  online target for one-ply consistency; EMA targets and longer horizons are
  distinct experiments. [VICReg, Bardes et al.](https://arxiv.org/abs/2105.04906)
  supplies a known variance/covariance route to preventing representational
  collapse. Chen has not yet implemented or validated that regularizer.

These primary sources establish substantial overlap. The proposed research
contribution is to test the interaction of exact occupancy relations, a compact
plan bottleneck, chess-frame move dynamics, and exact-state transposition
consistency under the same data and search budget. No comprehensive literature
review establishes global originality, and no positive result exists yet.

## Immutable input and action schema

Schema `1` features are float32 `[B,18,8,8]`. Square order is a1=0, h8=63.
For Black to move, flip ranks (`square xor 56`) and swap piece colors. Files
remain fixed so kingside/queenside castling retains its meaning. Planes are:

| Planes | Meaning |
| --- | --- |
| 0–5 | Mover's pawn, knight, bishop, rook, queen, king |
| 6–11 | Opponent's same piece order |
| 12–15 | Broadcast mover K/Q, opponent K/Q castling rights |
| 16 | One-hot en-passant target, all zeros if absent |
| 17 | Broadcast `min(halfmove_clock/100,1)` |

Move index is `((canonical_from * 64 + canonical_to) * 5 + promotion)`.
Promotion codes are none, knight, bishop, rook, queen. There are 20,480 labels,
including geometrically impossible labels; the authoritative simulator supplies
a bool legal mask. This intentionally simple mapping is bijective and handles
castling and all promotions without direction-plane exceptions. Schema 1 omits
repetition history, and its clipped clock cannot distinguish clocks above 100.
Exact terminal adjudication therefore stays with the simulator. Learning
history-sensitive values requires an explicitly versioned history schema.

WDL channels are loss/draw/win **for the side to move**. Current smoke training
supervises a material proxy through expected WDL, which cannot identify draw
probability or calibrate game outcomes. Future engine conversion must publish
an empirically calibrated mapping from WDL expectation to centipawns; latent
successor values change sign at each ply.

## Executable hypothesis family

All models share the same input projection, `[B,64,D]` square latents, output
heads, and action-conditioned transition. This makes representation changes
separable from changes to the move decoder or training objective.

| Model | Representation mechanism | Testable hypothesis |
| --- | --- | --- |
| `square` | Residual per-square MLP with mean-square global context | Cheap global mixing is an adequate control for local geometry |
| `relation` | Dense square attention, learned geometry bias, occupancy-blocked pseudoattack bias | Explicit rays reduce tactical/dynamics sample complexity |
| `plan` | `relation` plus learned plan queries cross-attending squares, then feeding back | A small tactical-plan bottleneck improves long-horizon/transposition consistency |

Relation channels cover files, ranks, diagonals, knights, king adjacency and
self. Attack edges include directional pawns and current blockers. They are
pseudoattacks: pinned pieces still attack and friendly occupied endpoints are
defended. This follows the semantics needed for tactical relations rather than
pretending to be legal move generation. Dense 64-square attention is convenient
for GEMMs and creates no sparse gather pipeline. It is not proven fastest on
gfx1030. The fixed ray-between tensor is `[64,64,64]` float32 (1 MiB).

The move transition conditions every square on source latent, target latent,
promotion embedding and source/target flags. A shared residual network can
update nonlocal effects such as castling rights and the rook's castle move.
It then flips ranks to the next player's frame. It does not execute chess
rules or guarantee correct capture, en-passant, promotion, color exchange or
castling. Piece/rule reconstruction supervises those effects; differential
tests against exact successor boards are the production gate.

The policy head uses a low-rank source–destination product plus source and
destination promotion biases, materialized as `[B,20480]`. Thus its expressivity
is intentionally restricted. The controlled follow-up compares this head with
a pair-conditioned MLP under matched time/parameter budgets. Rule heads predict
four castling rights, 65 en-passant classes (64 means absent), and clipped clock.

## Latent exploration that can be audited

`encode(features)` gives a position embedding; `transition(z, actions)` gives
one learned successor; `decode(z, mask)` gives WDL, move logits, pieces and rule
state. `branch(z, actions[B,K])` expands K candidate moves as one resident batch.
`rollout(z, actions[B,H])` returns the root plus H learned successor embeddings.
Each action uses its own current ply's canonical frame. These methods perform
no D2H copy. Legal moves must come from exact boards at every expansion.

Use three experiments to explore the space:

1. **Relation and dynamics ablation.** Fix generated game split, seeds and
   losses; compare all three representations with and without successor
   reconstruction/latent consistency. Report occupied-square reconstruction,
   EP/castling accuracy and rare-move groups separately. Whole-board BCE alone
   rewards predicting empty squares. Match both width and actual parameter/
   GPU-time budgets as separate comparisons.
2. **Transposition-constrained plan bottleneck.** Mine two exact legal paths
   to identical rule states, then penalize endpoint latent disagreement.
   Example: `Nf3 Nf6 Nc3 Nc6` versus `Nc3 Nc6 Nf3 Nf6` from the initial state.
   Match rights, EP, clock and repetition context, not just occupancy. Test
   1/2/4/8-ply drift and token counts 1/4/8/16 on held-out games and material
   classes. A later persistent-piece-slot variant can ask whether captured
   identities help without assuming slot binding is already implemented.
3. **Verified latent search.** Encode a real root once, branch exact legal
   moves in latent space and rank descendants. Replay selected paths through
   the exact simulator and compare learned versus re-encoded values, policies
   and states. Fall back to exact re-encoding beyond a validated drift bound.
   Compare against exact neural leaf evaluation at equal wall time. An Elo
   result belongs to the whole engine/search configuration, not the embedding.

Nearest-neighbor retrieval of encoded exact positions can characterize tactical
clusters. Linear latent interpolation or gradient ascent may leave the manifold
of legal chess positions; decoded plausible boards do not prove legality or
engine quality. Any invented position or move must be replayed by the rules
engine before becoming training experience or strength evidence.

## VRAM and gfx1030 integration boundary

The prototype generates trusted positions on CPU, copies the replay tensors to
the chosen device once, and samples/trains there. This is resident replay;
it is not direct VRAM production from the Rust/HIP rules engine. Dense policy
masks cost 20 KiB per example, much more than the 4.5 KiB feature tensor. Packed
masks or ragged legal candidate decoding should be measured before large replay
allocation. Batch-size sweeps need peak allocated VRAM and warm/steady timing.

The next direct-ingestion implementation must agree on schema version, tensor
shape/dtype/strides/device, lifetime ownership and stream ordering. A HIP producer
can transfer owned allocations through a correctly synchronized DLPack capsule;
an exported pointer alone is insufficient. Producer completion precedes consumer
reads, consumer completion precedes buffer reuse, and the deleter runs exactly
once. Ring entries carry generated state/outcome provenance and terminal flags.
Replay capacity, overwrite and sampling policies remain explicit. Candidate
move indices should be generated on device with features to avoid reconstructing
dense legal masks on the host for every batch.

gfx1030 is the measured target. PyTorch in `cpp-rocm` reports
`2.10.0+rocm7.0`; small float32 forward/transition and gradient comparisons run on
the actual device. This does not validate mixed precision, graph capture,
custom fused kernels, zero-copy ingestion, or end-to-end engine inference.
Start float32; measure fp16/bf16 availability and numerical errors independently.
Promote a kernel or precision mode only with move-mask, value and training checks.

## Promotion gates

* Train on real outcomes or declared teacher labels, with held-out games and
  opening/material groups, checksums, seeds, schema/model/optimizer metadata.
* Compare short/long horizon learned dynamics against exact successors, including
  castling, EP, all promotions and checks; report latent collapse/rank diagnostics.
* Run fixed tactical suites, then paired color-swapped matches against pinned
  opponents with time/node control and uncertainty intervals.
* A 2500 Elo claim needs a specified rating pool and statistically adequate games;
  self-play improvement and proxy-loss decrease cannot establish an absolute Elo.
* Integrate inference/checkpoint loading into the Rust batch evaluator and expose
  model/search configuration in the engine's result provenance before matches.

The current executable/test record and commands are in [neural/README.md](../neural/README.md).
