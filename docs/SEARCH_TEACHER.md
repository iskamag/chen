# Resident finite-depth search teachers

The first controlled run trained on **actual backed-up minimax labels**, but
the resulting neural models underperformed the material control on heldout
positions. The best neural result had 49.6% exact teacher-action agreement and
308 cp ordinary-score MAE; the material control had 54.7% and 191 cp. This stage
establishes a validated resident supervision path and deployable trained
artifacts. It provides no engine match result or Lichess rating evidence.

## Exact producer and label semantics

`neural.teacher.backed_up_tree` constructs full legal breadth-first levels from
an uploaded root and retained prior-position hashes. The captured run builds
through root depth three, statically evaluates the entire deepest frontier
with classical Tapered evaluator 1, then reduces levels 2, 1 and 0 in that order.
Every span, level, evaluation selector and reduction is recorded in the corpus
manifest. Early mate/stalemate or current-position draw branches retain exact
terminal scores. Arena capacity failures are explicit; there is no selective
or budget-truncated teacher fallback.

The independent `gpu/chen_teacher.h` ABI 1 exports exact parent labels for the
same selected child-edge rows as the [resident bridge](VRAM_BRIDGE.md). Its
48-byte sink owns three Torch device pointers: int32 scores, int64 canonical
best actions, and int32 absolute-root parent ply. Its 32-byte result carries
row count, root epoch, timing, actual locked evaluator and label kind. The
exporter rejects unscored/unreduced sources, incomplete child scores, mismatched
stored maxima/best pointers, stale epochs, invalid pointers and incompatible
schema before writing labels. Learned external-leaf evaluators cannot be
exported under the classical teacher label kind.

Scores are exact centipawns from the **current parent's mover** perspective.
Static values are clamped to ±29000. Terminal checkmate is -30000 plus absolute
root ply, propagated by negation; `parent_ply` preserves that frame. The chosen
action is the first maximum in the simulator's legal-generation order,
canonicalized for the current mover. The C boundary certifies a completely
backed-up resident tree; uniform depth is established by the Python construction
schedule rather than inferred from an arbitrary manually built arena.

The experiment exports second-level edges, whose current parents are at root
ply one and have a **two-ply remaining search horizon**. Sibling edges repeat
the same parent score/action. Resident `unique_consecutive` grouping retains one
record per parent, validates sibling label equality and requires every sibling
edge to be present. Consequently, a parent with many legal moves gets one vote.
The deduplicated observation, exact score, action and lineage enter an owned
`SearchReplay` ring without a CPU payload detour.

`SearchReplay` composes the unchanged compact replay with resident typed teacher
fields. Policy targets are replaced with the exact backed-up chosen action;
`search_score_cp` is a separate int32 field. The existing material proxy remains
a named schema field and is unused by `search_loss`. Parent/child hashes, node
IDs, epoch, teacher-root ID and source-game ID remain resident. Context UUID,
root/history digests and the root-ID mapping remain in the corpus manifest.
Sampling gathers owned compact and teacher fields using the same device indices,
so labels, observations and monotonic admission identities stay aligned through
wraparound. Invalid label ingress fails before compact admission.

## Root families and split

The captured dataset has 64 training and 16 heldout game families, each a seeded
uniform legal-move walk from the standard start position. It chooses roots at
plies 12 and 24, retains their exact move prefixes and previous-position hashes,
and rejects terminal/draw-claim roots or previously repeated histories. The
sampling distribution is artificial and has no real game outcomes or player
strength labels.

Whole source-game families stay on one side of the split. A conservative
canonical collision set contains every root, every legal first-ply observed
parent, and every legal second-ply stored successor. Identity matches actual
schema-1 features: own/enemy pieces, black rank flip, current-mover rights, raw
EP and clipped halfmove. Fullmove and repetition history are absent from that
feature schema. A candidate family colliding with accepted positions is rejected
as a whole. The run checked 117,950 training and 30,282 heldout canonical positions
with zero overlap; no candidate rejection was needed for this seed.

All games share early start-position ancestry, which is disclosed. Those shared
early positions are outside the trained/heldout observation sets. The split
controls lineage and exact/canonical input overlap; it does not make the random
games representative of human or engine play. Retained simulator history drives
teacher adjudication, while the model still lacks repetition context.

## Search-specific objective and controls

The existing three-logit head becomes a bounded scalar regressor:
`utility = softmax(logits)[2] - softmax(logits)[0]`.
The target is `tanh(search_score_cp / 800)`. The loss combines chosen-action
cross entropy with ordinary-score utility SmoothL1 at equal weight. Mate-band
records (absolute score ≥29000) remain in policy learning but are excluded from
the scalar regression. The head is trained against search scores, without an
outcome/WDL interpretation. Inference uses the matching scale/epsilon contract:
clamp utility to ±(1-1e-6), apply atanh×800, nearest-even rounding and ±29000
clamping. Saturated utilities cannot certify mate.

The comparison uses one immutable resident training corpus and one heldout
corpus for square, relation and plan. Shared input, positional, norm, utility
and policy weights start identically. Architecture-specific encoder blocks
retain their seeded initialization. Width/depth are 32/1; total parameter
counts are 19,908 / 24,203 / 28,683, so this comparison is not parameter-matched.
Every model takes 256 AdamW steps, lr 0.0003, batch 64, seed 20260930. Sample-stream
SHA-256 is identical across all three models.

Two controls are reported: the same network before training, and current
material value paired with a one-ply material-greedy legal policy. Material
counts exact own/enemy pieces with weights 100/320/330/500/900/0 and the same
800 cp utility transform. The greedy policy maximizes resulting material with
first simulator-order ties. Exact chosen-action accuracy penalizes equivalent
search-tied actions; it is agreement with this teacher, not playing strength.

## Captured negative result

There are 3,872 training and 1,048 heldout unique parents. Ordinary value metrics
exclude 14 training and one heldout mate-band record. The table gives heldout
results after training; MAE uses inverse utility conversion on ordinary scores.

| Model/control | Utility RMSE | Ordinary cp MAE | Exact teacher action accuracy |
| --- | ---: | ---: | ---: |
| Material value / one-ply material policy | **0.281** | **191.0** | **54.7%** |
| Square | 0.386 | 307.9 | 49.6% |
| Relation | 0.397 | 316.9 | 48.4% |
| Plan | 0.392 | 313.6 | 48.2% |

Before training, neural action accuracy was 2.5% / 14.6% / 13.6% and cp MAE
317.3 / 326.7 / 323.5. Training learned policy agreement, while heldout value
improved only modestly. Training action accuracy reached about 69%, leaving
roughly a twenty-percentage-point heldout gap. None of these models beat the
simple material control. A single seed and only sixteen heldout families cannot
support a reliable ranking of the three architectures; sibling positions within
a family are correlated.

Root admission/split took 4.52 s CPU. Direct resident teacher production took
1.70 s training / 0.29 s heldout. The three 256-step training loops took
1.56 / 1.46 / 1.95 s on actual gfx1030. Producer timing includes arena creation,
root uploads, expansion/reduction, export and replay admission, after replay
allocation. Training timing includes sampling, loss, updates and lazy optimizer
state allocation; model creation, initial evaluation and checkpoint writing are
outside that timer. These bounded timings are not a throughput study.
Runtime was Torch 2.10.0+rocm7.0, Torch HIP 7.0.51831. Producer library SHA-256 is
`5e0e0dd026211a6489c20faf90dcf0c3d3e7677daa64f5bd85a6a79765e167c4`;
its immutable teacher-only source snapshot is used independently of later
external-neural-leaf changes.

## Artifacts, checks and reproduction

The [full report](../neural/evidence/search-teacher-gfx1030-2026-09-30.json),
[console log](../neural/evidence/search-teacher-gfx1030-2026-09-30.log),
[GPU tests](../neural/evidence/search-teacher-tests-gfx1030-2026-09-30.log) and
[CPU tests](../neural/evidence/search-teacher-tests-cpu-2026-09-30.log) preserve
configuration, root game histories, exact construction schedules, source/library
hashes, controls and negative results. The [artifact hash manifest](../neural/evidence/search-teacher-source-sha256-2026-09-30.json)
binds those reports to final source and checkpoint bytes. The native producer's separate
[Rust-backed oracle evidence](../gpu/evidence/teacher-gfx1030-2026-09-30.json)
checks Material/Tapered labels, chosen actions and absolute-ply mate scores.

All nine teacher tests passed on actual gfx1030; the separate CPU run passed
seven and skipped the two hardware tests. They verify schema, complete-sibling
deduplication, atomic bad-label rejection, aligned wraparound/snapshots, strict
canonical family splits, common initialization, finite update/mate exclusion,
restricted checkpoint corruption rejection, actual Material minimax scores and
optimal legal actions against an independent Python search, and unbacked/stale
source rejection. Existing bridge tests retain their original ABI binding because
new evaluate/reduce symbols are bound lazily when used.

Toy synthetic checkpoint loading remains unchanged. Search artifacts use the
separate `search_teacher_cp_utility_checkpoint` kind and
`load_teacher_checkpoint` restricted CPU-first loader. It checks the envelope,
teacher horizon/evaluator/utility metadata, exact tensor shape/dtype/layout,
finite weights and every deterministic geometry/orientation buffer before
moving a validated model to a device. Restricted roundtrip tests reject integer
orientation corruption, wrong learned dtype, Boolean evaluator/schema aliases,
incorrect utility exclusions and legacy artifact kinds.

Trained checkpoints are ignored local artifacts under
`results/neural/search-teacher/{square,relation,plan}.pt`, with copied report,
manifest and logs. Exported TorchScript models and matching sidecars are under
`results/neural/deployment/{square,relation,plan}-inference.pt[.json]`; that
directory's `index.json` binds archive, checkpoint, state and report hashes.
The deployment exporter consumes that dedicated loader and
preserves the search-teacher status; it does not change the evidence into an
outcome or rating claim.

```sh
python -m unittest neural.tests.test_teacher -v
CHEN_TEACHER_TEST_GPU=1 CHEN_GPU_LIBRARY=/path/to/teacher/libchen_gpu.so \
  python -m unittest neural.tests.test_teacher -v
python -m neural.teacher --library /path/to/teacher/libchen_gpu.so \
  --train-families 64 --heldout-families 16 --tree-depth 3 --evaluator 1 \
  --steps 256 --batch-size 64 --width 32 --depth 1 --seed 20260930 \
  --producer-manifest /path/to/teacher-producer-evidence.json \
  --output results/neural/search-teacher-gfx1030.json
```

Root/history upload and scalar control/validation reads are explicit. The
producer/learning path has no state/label payload CPU detour; sampling/loss/
gradients are covered by Torch dispatch audits. AdamW's CPU scalar bookkeeping
is permitted control. After learning, a separately marked cold audit downloads
payloads once for exact corpus fingerprints and writes CPU checkpoint weights.
Those diagnostic bytes are declared and never copied back into training. Native
copy traces cover the C teacher export only, rather than every Torch/runtime API.

The next controlled tests should separate value-only from policy-only learning,
measure whether policy gradients suppress score learning, compare a residual
material score feature against the unchanged encoder, and test deeper teachers
on several disjoint game-family seeds. Report ordinary score error, tied-action
regret where available, family-level uncertainty and offline paired-game results
before promoting any checkpoint. The 2500 Lichess objective remains open.
