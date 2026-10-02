# Semantic resident corpus contract — v1

The [corpus package](../../experiments/semantic_corpus/README.md) now implements immutable split packing, a strict reader and complete-root resident batches. Its [closed CPU audit](evidence/semantic-corpus-2026-10-02/audit-all-splits.json) independently reconstructs every table and checks the Rust rules oracle over all 15,501 states and 12,701 legal root children. The splits retain 233/62/55 whole families, 2,800 standalone observations and 350 complete roots, with 7,232,128 payload bytes and full history. The [actual TRAIN gfx1030 admission](../../gpu/evidence/semantic-resident-gfx1030-2026-10-02/attempt2/runtime-report.json) checks all TRAIN batches against CPU decoding. No model training, heldout quality metrics or strength result follows from these mechanical checks.

The contract separates immutable exact simulator state from schema1 neural observations, and separates standalone teacher queries from parent-MultiPV action queries. Whole game families, complete legal root support and frozen score-frame provenance are the units of admission. A record ring is insufficient for those guarantees.

## Existing interfaces and the actual mismatch

The inspected committed neural tree is `2a08ebb4b0cf8f7d2e78e63df8e4a4815628e3e0`. These seven source files match that commit exactly; the linked same-repository frozen preparation copies have the same hashes, so reading this contract does not require a sibling checkout:

| Source | SHA256 |
| --- | --- |
| [semantic_strategy.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/semantic_strategy.py) | `943b5ba6c7947a8107acbafab4f8f463c21e5a2556efcf29e7fd0eca16aa9b84` |
| [strategic_transport.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/strategic_transport.py) | `319f9bb99e712d1164ae7f7744ec715278a6071cc41dace2b4e77c044c5c6a6f` |
| [bridge.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/bridge.py) | `e77c3408d6011b99587b476ae5e84dfe5efee2577d61dd8bd407729774b3a73e` |
| [compact.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/compact.py) | `a52ba420dca331edf999e539cc37ea5b9208630ed0f9d9587fe24501b2c944d3` |
| [features.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/features.py) | `3aecc21de78d1c586543117aca78b3e8920b84b3694377d700228425a5e67cfc` |
| [score_frames.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/score_frames.py) | `efc0e8da2ad65b41523fdcf0fd514fa70e3ddbff066854a88c2ad64726a02f57` |
| [teacher.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/teacher.py) | `d0587c2e2cbd6917b4bfe1906944e0782146a69988a6f2e0983faca9169b1918` |

The readonly worktree's `semantic_data.py` and `semantic_study.py` are untracked older producer sources, so they are not attributed to that commit. For packet-v2 grammar and provenance this design references only the frozen preparation sources: [ordinary_packets.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/ordinary_packets.py), SHA `4e6d7e9461fe0c001622acb39f5be48d461c95e1994526f9ca48f09a27cbd83f`, and [semantic_data.py](evidence/semantic-packet-v2-preparation-2026-10-01/neural/semantic_data.py), SHA `b5bdc1b2df754673504c656c53cd37389a7989ed7c67644fa39d8bac606c7df5`. This is a source reference, not admission of any attempt output.

[Replay ABI1](../../gpu/chen_replay.h) exports selected existing parent→child edges into independent Torch-owned outputs: exact absolute-color bitboards, side/castling/raw EP/clocks, current-parent sorted legal actions, executed action, successor terminal bit, and parent/child hash/node/epoch sidecars. Its parent rows repeat once per child. It does not export a history pool or strong teacher intervals. Python fills both history handles with `-1`, and its `material_proxy` and executed-action imitation are explicitly toy supervision.

[Teacher ABI1](../../gpu/chen_teacher.h) exports a backed-up classical finite-tree **parent** score, chosen action and parent ply. This is neither the SF19 reference-material frame nor a standalone/parent-MultiPV strong label. Do not reinterpret these fields as external teacher outcomes or intervals.

`CompactReplay.append` requires positive current legal counts and can trim rows to ring capacity. It does allow a terminal *next* state, but provides no atomic whole-root grouping, standalone observation table, strong-query frame fields or corpus history ownership. It must not be reused as the authoritative semantic corpus.

`bridge.validate_output_tensors` requires storage offset zero, the exact whole backing allocation, contiguous layout and nonoverlapping outputs. An aligned slab's typed subviews fail this contract. Native export must retain independent complete base allocations; a later Torch copy/freeze into corpus tables is a separate operation. No live arena pointer, zero-copy slab export or existing one-DMA corpus ingress is claimed.

## Immutable relational tables

Use one immutable corpus per split. Let `S` be exact state rows, `O` standalone observations, `R` complete action roots, `A=sum(K_r)` legal root edges, `H` previous-history hashes and `G` game families. All row indices fit signed int32; all counts, byte sizes and offsets are validated before allocation. `R` contains nonterminal roots with `1<=K_r<=256`. A terminal child remains an edge and state row even if it has no legal continuation.

| Table | Required resident fields | Meaning |
| --- | --- | --- |
| States | `boards int64[S,12]`, `fields int32[S,5]`, `hash int64[S]`, `terminal bool[S]`, `family int32[S]` | Absolute white PNBRQK then black PNBRQK; fields are side, rights, raw EP, halfmove, fullmove. No dense-feature inversion. |
| History | `offsets int64[S+1]`, `keys int64[H]` | Corpus-owned CSR spans of previous hashes, excluding each current state; not borrowed arena IDs. |
| Observations | `state int32[O]`, `query int32[O]`, `reported_cp int32[O]`, `query_material int32[O]`, `midpoint/halfwidth float32[O]`, `ordinary bool[O]` | Standalone source queries; positive/current-mover sign is retained. |
| Roots | `state int32[R]`, `edge_offsets int64[R+1]`, `legal_count int32[R]`, `query int32[R]`, `family int32[R]` | One complete current-parent action set, ordered by canonical action ID. |
| Edges | `root int32[A]`, `child_state int32[A]`, `action int16[A]`, `reported_cp int32[A]`, `midpoint/halfwidth float32[A]`, `rank int16[A]`, `ordinary bool[A]` | Complete MultiPV labels in child-mover reference-frame units; raw CP is still parent-mover UCI CP. |
| Source identities | `origin_context int32[S]`, `origin_epoch/node int64[S]`; native parent/child IDs retained alongside edge rows | Context index resolves to a UUID in the manifest. CPU-ingested observations use an explicit CPU-origin tag and absent native IDs. |

No table overwrites data after publication. Owners retain the tensors and history pool for all consumers; a gathered minibatch owns its gathered storage. Hash, clocks and native IDs preserve unsigned bit patterns in signed transport. Halfmove is restricted to `0..65535`, fullmove to `1..2^32-1`, side to 0/1, rights to 0..15 and raw EP to `-1` or a valid raw target square. Fullmove is decoded unsigned, never clamped to signed-int32 range.

A state's identity includes its exact rule fields **and history digest**. A board Zobrist hash, canonical action ID or feature key alone is insufficient for draw/history identity. Raw EP is retained even where legal-EP Zobrist identity ignores it. A child's previous history extends the parent's history with the parent's current hash; it never includes the child itself. CPU source moves, native hash conventions and the history window must be independently matched before claiming simulator parity.

Current native exports supply a terminal bit, not a terminal reason or child legal-action set. Store that bit and the global exact adjudication-policy ID. If a future consumer needs terminal causes or next-state support, add an independently certified sidecar or bounded export extension; do not infer causes from the bit. Mate/stalemate precede current halfmove/material/repetition draws. Prospective next-move draw claims are not added. A drawn terminal can still have geometrically legal moves; it is nevertheless closed under this policy.

The CPU manifest owns full query/family provenance: game and source hashes, position/history digest, split, query kind, node request and actual telemetry, reported depth/seldepth, packet-v2 selected emission ordinal and source line interval, raw transcript SHA, final-bestmove provenance, source/binary/options hashes and score-frame manifest. Query IDs on device index this frozen table. Source transcripts are audit records, never neural input.

## Exact state versus schema1 and eight role targets

Decode neural inputs with `compact.decode_features`: float32 `[B,18,8,8]`, own/enemy piece planes, black rank XOR56, four mover-relative rights planes, raw EP plane and `min(halfmove/100,1)`. Match its float64-division→float32 clock rounding. Fullmove, prior hashes and repetition context stay outside schema1. Equal neural observations may belong to distinct simulator contexts; cache identity and search identity are separate.

`SemanticAnchor.semantic_targets` supplies board class, rights, EP class (64 absent), clipped clock and float role maps `[B,8,8,8]`. Generate these on device from admitted feature tensors, rather than uploading another dense map corpus. Its role order is fixed:

| Token | Target |
| --- | --- |
| 0/1 | Own/enemy king-zone pressure by opposing blocked pseudoattack counts, clipped at 8 and divided by 8 |
| 2/3 | Own/enemy nonking occupied squares attacked by the opponent and lacking own pseudoattack support |
| 4/5 | Own/enemy blocked pseudoattack counts, clipped at 8 and divided by 8 |
| 6/7 | Own/enemy pawn one-step advance onto empty squares, with board-edge exclusions |

These are deterministic diagram priors, not legal attack maps, pin-aware threats, proof of winning plans or external teacher tactics. Child maps use child-mover features. The transition swaps role tokens by `[1,0,3,2,5,4,7,6]`; it does not transport them as pieces. Its exact move algebra handles already-certified legal capture, EP, standard castling and promotion, then changes mover orientation. It does not reconstruct authoritative child clocks, rights or history.

The first bounded comparison should train its anchor on admitted **training standalone observations**, fit one positive material scale on those training standalone labels only, then freeze scale, anchor and all shared decoders before transition fitting. Do not silently enlarge scalar anchor training with MultiPV child labels. Observed child features and frozen child latents may serve detached transition targets; they must not enter `predict(root_features, actions, root_latent)` as successor observations. Multistep supervision remains the declared hybrid: current simulator observations plus predicted plans, never an unannounced intermediate re-encode.

## Teacher frames and scalar admission

Keep query kinds explicit: standalone-root score and parent-root-normalized action score. Native Material/Tapered CP and toy material proxies are separate kinds, excluded from this strong-label table.

Let `c` be admitted ordinary integer SF19 UCI CP and `M` total phase material of the **actual queried root**, using both sides' weights `[1,3,3,5,9,0]`. Define `a(x)=((AS[0]*x+AS[1])*x+AS[2])*x+AS[3]` for normalized material `x`, with the pinned coefficients/source hashes in `score_frames.py`. Then the positive factor is `f=a(clamp(M,17,78)/58)/a(1)`. The source helper `_a(M)` instead accepts total material and performs the clamp/division internally; consequently its equivalent expression is `_a(M)/_a(58)`. Standalone midpoint is `c*f`. Parent-MultiPV child-mover midpoint is `-c*f_parent`: convert with parent material first, then negate. Never substitute a capturing/promoting child's material into the parent's query conversion.

The source reporting interval has halfwidth `0.5*f`, plus its declared numerical guard. Stored float32 midpoint/halfwidth must include casting error and outward rounding as in `sf19_reference_batch`; do not simply cast a double midpoint and width independently. Keep raw CP and query material so a cold auditor can reconstruct the interval. Negating a label changes bounds to `[-upper,-lower]`. These intervals account for reporting quantization, not search error or statistical confidence.

Admit only audited packet-v2 ordinary unbounded non-tablebase source kinds. An incomplete, mixed-depth or bounded emission is ineligible, while a valid earlier ordinary complete emission can still be selected. Malformed grammar/support/PV/rank evidence or any decisive mate/tablebase/reserved-region evidence rejects the query; do not splice ranks across emissions. Record final `bestmove` separately from the selected packet's rank1, which can differ. Common reported depth is an admission attribute, not exact fixed-horizon minimax. Do not compare standalone and MultiPV differences as though normalization erased their different search workloads.

`anchor_losses` requires resident float32 midpoint and halfwidth `[B]` plus bool ordinary mask `[B]`; supply the real interval instead of its synthetic default zero width. Use the same interval-distance loss for transition CP. A terminal child remains in complete action support, while its ordinary scalar-loss mask is false; simulator terminal adjudication overrides learned value at decision time. Do not convert native mate-distance scores into this reference CP frame. Terminal semantic targets can remain available according to the declared loss mask. For the first comparison, roots with terminal candidates must be flagged and their terminal-aware decision policy predeclared separately from ordinary CP calibration/ranking summaries.

Every sibling has the same positive parent normalization factor. It preserves raw ranking and ties. Keep both MultiPV rank and sorted canonical action order; these are not the same order. Ranking metrics choose the lowest **child-mover** CP, use a declared deterministic tie rule/optimal set, and preserve whole-root support.

## Ownership and transfer scope

The implemented CPU-origin path uses no new native ABI:

1. Separately pinned serialization seals bind each split's teacher archive, packer source and allocation limits. Cold admission reconstructs exact states/history, full PVs and canonical label joins without model metrics; an independent decoder checks every byte against those witnesses and the Rust rules implementation.
2. The loader validates the immutable pack before importing Torch. It creates private owned CPU storage, appends validated structural schedule and history-gather indices, and uploads that combined blob once through `Tensor.to(device)`. Publication waits for upload completion; the owner retains all typed read views.
3. Prepared root and observation batches gather from resident indices into their own storage. They retain complete action support and decode features on device; batch preparation creates no new host index tensors. Consumers using another stream must perform the documented wait/record-stream handoff.

Actual TRAIN QA observed one instrumented packed upload call for 8,006,592 bytes: 4,822,336 payload bytes plus 3,184,256 schedule/history-index bytes. Every field in 74 root batches and 59 observation batches matched CPU, including 8,477 children and 1,864 observations. Frozen feature decoding checked 10,574 rows. Peak allocated Torch memory was 11,650,560 bytes for this bounded mechanics run. Physical DMA counts, sustained training throughput and model activation/optimizer memory were not measured. A first checker attempt failed before calling the GPU loader because allocator statistics were reset before device initialization; both attempts are archived.

Native GPU generation remains a separate path: `RuleArena.reset/expand/export` ingests per-root CPU packets/history and writes independent complete base allocations. A future corpus freeze must join those outputs by canonical action, preserve native context/epoch/node identities and complete any D2D copies before publication. This corpus is explicitly CPU-origin; it does not claim native generation, zero-copy export or elimination of per-root reset traffic. Aligned corpus views cannot be passed as current native export sinks.

Dense schema1 float32 features cost 4,608 bytes per state; eight float32 role maps add 2,048 bytes. Prefer compact state tables and per-batch decode/target generation initially. Report base payload, history, label tables, index schedules, gathered batches, targets, activations, optimizer and allocator reserved/transient memory separately; existing arena creation also reserves its fixed legal-move buffers. Set a sealed byte cap before allocation and refuse overflow without trimming roots or successors.

## Whole-root batches, sealing and test isolation

`SemanticTransition.branch` accepts roots `[B,18,8,8]` and int64 actions `[B,K]`, with **no padding** and `K>0`. Store edges in CSR form, bucket roots by exact legal count and gather every candidate for each selected root. The int16 corpus IDs become int64 forward inputs. Terminal successors stay inside these complete parent groups; never invoke branch on a terminal state merely to satisfy a positive-count ring. Do not select a few convenient actions or let `-1` padding reach move algebra.

Build bucket permutations and batches from structural counts only. Seal the RNG algorithm/seed, root and standalone row schedules, epoch counts, replacement policy, partial-tail handling and loss coefficients before training. All compared routing modes receive the same root groups, action order, standalone batches and initialization snapshot. Store resident schedule arrays once; minibatch gather/decode/loss/backward does not extract tensor payloads on CPU. Use deterministic root/group weights so a large legal set does not silently give its family more loss weight. Exact-K grouping changes batch geometry; record it rather than promising shape-independent GEMM equality. Preserve the blind control's once-per-root residual computation and exact same-material ties.

For two/four-ply hybrid tasks, admit exact contiguous simulator sequences as separate trajectory groups. The sparse eight-observation schedule is not automatically a four-ply trajectory. Any additional predecessor/intermediate/successor model inputs must enter family/canonical collision and spent-input admission **before** the new declaration is sealed. A family and every derived root/child/trajectory state stay in one split. Feature-equivalence collision checks include mover canonical pieces, rights, raw EP and clipped clock, matching `teacher.canonical_position_key`; fullmove/history are independently preserved despite their absence from that key. Whole-game separation does not establish independent players or openings.

Create train, validation and test immutable packs and manifests separately. Training receives only train data; validation runs with frozen weights and can inform only the predeclared selection procedure. The final test evaluator receives the sealed selected checkpoint/configuration and test pack after selection is frozen; it cannot update anchor scale, losses, optimizer, masks, buckets or model choice. No late family replacement after seeing labels or metrics. Any exposed test/diagnostic inputs become spent for later declarations, with predecessor/successor feature keys included. A shared dictionary containing all splits is not the intended ownership boundary.

Root latent cache identity includes canonical feature bytes/schema, fitted scale, anchor/transition weight hash, dtype/device and architecture. Simulator context/epoch/history remain separate support/terminal identities. Cached child targets are detached frozen-anchor outputs; cache invalidation follows weight changes. Record actual batch IDs and optimizer steps, not merely a seed, so repeated deterministic schedules are comparable.

## Implemented mechanics and pending learning gates

The immutable packer, strict reader and exact-K adapter are implemented with 44 passing CPU tests, including rehashed corruption, interval frames, split gates, byte caps and ownership controls. Independent actual-data audits pass all three splits; actual GPU mechanics admit TRAIN only. The archived [source and payload manifest](evidence/semantic-corpus-2026-10-02/manifest.json) and [GPU runtime manifest](../../gpu/evidence/semantic-resident-gfx1030-2026-10-02/manifest.json) bind those results. Native export→D2D corpus construction and the learning configuration remain separate pending work. A GPU data admission does not seal optimizer, loss, selection or test policy.

Required admission gates are:

- Exact integer state/history/hash parity across both mover orientations, EP, castling, promotion, clipped clocks/full unsigned fullmove and same-feature/different-history cases; stale context/epoch and incomplete publication rejected.
- Native and audited CPU legal support bijection for every root; action labels reorder correctly from rank to canonical order; no missing/duplicate/padded candidate. Terminal children retained without demanding their own positive support.
- Standalone versus parent-material capture/promotion conversion, interval sign reversal, outward float32 rounding and wrong-kind rejection; never fit scalar targets through an untyped toy proxy or native parent-teacher field.
- Device schema1 decode and all eight deterministic diagram targets match frozen CPU mechanics, while successor observations stay out of transition forward inputs. Frozen anchor/scale and loss-only detached successor targets are checked.
- Cross-split/spent feature and family closure, train-only calibration, exact-K schedule/permutation replay, identical compared-mode batch IDs and active-parameter accounting; test access occurs only after checkpoint/configuration sealing.
- Actual ownership/stream-failure tests, byte-cap refusal and scoped transfer measurements for the implemented packing path. Existing bridge evidence does not validate the new corpus, histories or packed loader.

These gates establish data/implementation comparability, not stronger play or autonomous neural chess rules. Selective search strength, calibrated decision quality, warm throughput and long-run reliability require separate declared evidence.
