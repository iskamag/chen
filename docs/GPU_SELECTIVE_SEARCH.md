# Selective resident minimax foundation

This experimental arena retains a mixed-depth chess tree and refines its most promising open leaves. Rules, frontier selection, complete sibling expansion, static evaluation and ancestor backup execute on the GPU. Rust controls bounded steps and publication. The existing full-width arena remains the fixed-horizon oracle. [The baseline hardware report](../gpu/evidence/selective-validation-gfx1030-2026-10-01/verification.json) records stepwise classical parity, neural closed-root comparisons and a bounded latency/depth diagnostic; implementing a deeper mixed tree does not establish stronger play.

The design draws on the root-to-leaf refinement mechanism in [Korf and Chickering, Best-first minimax search](https://www.maxchickering.com/publications/aij96.pdf). It uses full propagation to the root, whose interaction with ties is discussed by [Cohen-Solal and Cazenave](https://arxiv.org/html/2505.04525v1). Our batching, fixed maximum ply and deterministic reservation scheme are implementation choices, so this is not claimed to reproduce either paper's algorithm or performance. The [completion analysis](https://arxiv.org/html/2603.24572v1) motivates making closed states explicit; its unbounded winning-strategy theorem is not claimed for this bounded scheduler.

## One bounded step

Reset validates and uploads the root and previous hash history without charging or evaluating it. The first advance admits root classification only when the absolute node budget is at least one. A terminal root can immediately publish its exact score. A nonterminal root must expand and score **every** legal root move before publishing a selected result. At the initial position, node limit 20 therefore leaves the caller's legal fallback; 21 completes the first 20-child expansion. Root static evaluation is unnecessary.

After that bootstrap, each step uses this sequence:

1. Clear temporary reservation counts. From the root, descend through the available child maximizing `-child.score`, with the first legal child winning ties. Availability means its open-leaf count exceeds its reservations.
2. Reserve the candidate and every ancestor, even when the candidate's complete sibling set does not fit. Examine at most `candidate_limit` distinct leaves and admit at most `batch_leaves` expansions.
3. Admit a leaf only if its entire legal child set fits both remaining classified-node budget and retained arena capacity. Commit admitted leaves in selection order; all newly created children occupy one contiguous device block.
4. Classify and score that entire admitted block, preserving exact terminal adjudication. Classical evaluators or the existing checked external neural callback consume these positions without downloading boards.
5. Initialize new closure metadata, then back up marked ancestors in reverse allocation-index order on the GPU. Every child index exceeds its parent, so this ordered reference kernel has no parent/child write race. Static-score changes, best-child links, open counts and minimum/maximum leaf depths propagate together.

Terminal and maximum-ply leaves remain in value backup but are never selected. When the best valued child is closed, selection considers eligible alternatives rather than repeatedly selecting the closed branch. Reservations disappear at the next step. Selection within a batch uses the pre-step values; changing batch size can change the explored tree and is an experiment parameter.

If at least one complete sibling set is admitted, the step reports progress unless the tree closes. With no admission, exhausting the candidate bound while an unreserved open leaf remains reports candidate-scan stop: this cannot prove that no cheaper candidate fits. Otherwise node-budget blockage takes precedence over arena capacity. Zero remaining budget or capacity can stop immediately, since an expandable nonterminal has at least one child.

## Scores and depth

Every published score is exact negamax over the **retained finite tree**, whose nonterminal leaves use the chosen heuristic. It generally differs from fixed-depth minimax. Only `HORIZON_COMPLETE`, meaning every retained leaf is terminal or at configured maximum ply, guarantees equality to the full-width target-horizon oracle with the same rules, draw policy and identical per-node evaluator scores. Neural floating-point results can depend on batch shape: model identity alone does not prove score equality, which requires measured parity for the compared executions.

Minimum and maximum frontier depths include all retained leaves: terminals, horizon leaves and open heuristic leaves. They are actual visited plies, not a promise that the maximum ply was reached uniformly. The Rust scheduler must expose selective depth separately and avoid claiming a uniform horizon from that maximum.

Mate scores retain the existing absolute-ply frame: a mated leaf is `-30000 + ply`; nonterminal static values are clamped to `[-29000,29000]`. A backed-up value outside that static interval proves a forced mate outcome under the engine's adjudication policy. While other branches remain open, the reported finite-tree mate distance is not asserted to be globally shortest. PV links follow deterministic best legal children and terminate at a retained leaf.

No transposition merging is performed. Draw adjudication depends on the ancestor chain, prior root history and halfmove clock; equal board hashes do not establish interchangeable histories. Legal generation precedes mate/stalemate and then current-position draw adjudication, matching the full-width arena. This does not add prospective next-move draw claims.

## ABI and ownership

[chen_selective.h](../gpu/chen_selective.h) defines independently versioned ABI1 with Config32 and Stats96. The implementation is [chen_selective.inc](../gpu/chen_selective.inc), included by the rules translation unit to reuse its exact device rules and evaluator code. `ChenSelectiveContext` owns a private rules arena; it exposes neither that handle nor device addresses. The full-width API and its prohibition on expanding statically scored leaves are unchanged.

Additional arena allocations are bounded at creation: 16 bytes per node for open/depth metadata, a 4,112-byte selection work buffer and a 96-byte device statistics record. Reservations reuse the inner arena's existing scratch flags. These extra private allocations are registered in the shared replay/teacher sink alias validator as defense in depth. The selective arena allocates no new storage during search; an external evaluator retains its own allocation policy.

All calls are synchronous and caller-serialized; each device operation selects its device, permitting exclusive transfer to the engine worker thread. The external callback uses the existing five-runtime-token guard and borrowed contiguous position/score contract. Evaluator identity, callback, owner and maximum batch are fixed until reset. No CPU replacement is performed on a backend failure.

Node limits count newly classified positions, including the root once; expanding an already evaluated leaf does not charge it again. Newly admitted children are classified/scored before the step completes, including at an exact node boundary. Cancellation is checked between calls. The largest atomic work unit is a configured expansion batch and its evaluator callbacks, so a wall deadline is not an interrupt inside a GPU operation.

Clean budget, capacity and candidate-scan stops preserve the last completed root/PV. Failed reset/advance invalidates that context and prevents stale `finish` publication. A caller can retain an earlier snapshot received through a report callback, but the current Rust search backend returns an error after advance failure and UCI emits `bestmove 0000`; it does not automatically return that earlier snapshot. Actual HIP/native faults may require process recreation; reset does not promise runtime recovery. Destruction checks device/stream retirement before freeing extension allocations; an unproven retirement fails the owned engine process with exit 70 rather than releasing potentially referenced storage.

Telemetry accumulates classified/evaluated nodes, completed batches, expanded leaves, open leaves, actual frontier/PV depths and event spans. Event spans include enqueue gaps and external callback dispatch; they are not pure active-kernel time. Scalar summaries/statistics and the final legal PV cross the host boundary. This design does not claim zero host transfer traffic.

## Verification boundary

The isolated `gfx1030`-target build completed with HIP 7.2.53211 / ROCm 7.2.4, exit 0 and no compiler output under `-Wall -Wextra -Werror`. [The build record](../gpu/evidence/selective-build-gfx1030-2026-10-01/report.json) and [hashed source/log archive](../gpu/evidence/selective-build-gfx1030-2026-10-01/manifest.json) bind that compiled library. This is compilation evidence only; no GPU runtime call was made during that check.

The first GPU kernels deliberately use ordered selection and ancestor backup as a reference implementation. They can later be replaced with parallel selectors or ply-wave backup behind the same ABI, keeping admission, tie and score semantics comparable. Tests should compare each completed batch with the portable Rust arena, compare closed target horizons with full-width minimax, exercise terminal/history/EP/promotion/mate cases and invalidation, and demonstrate greater selective depth under the same node/capacity budget. Equal wall-budget and playing-strength experiments remain separate evidence requirements.

## Proposed classical fusion

This is a proposed optimization, awaiting implementation authorization after the verified baseline is committed. It preserves ABI1, selection order, complete-sibling admission, integer evaluator scores and all phase/error rules. External neural evaluation keeps the baseline path.

One classical advance would enqueue admitted-root classification when needed, reservation clearing, selection, fused child expansion/classification/evaluation, and ordered backup plus a completion snapshot on the same private stream. Child launch dimensions use the configured bounded maximum, and each thread checks the device selection count rather than requiring a host copy of that count. A child locates its selected parent by binary search over the existing device prefix offsets. Its history walk references its own freshly written node and previously retained ancestors; siblings do not depend on one another. The first child installs its parent's expansion metadata, which backup reads only after that kernel completes.

Every dependent kernel observes a sticky error; every launch is checked immediately. A private completion packet containing Stats96 and error status would be copied once into an owned pinned host buffer, followed by one checked stream fence. Host counters change only after successful completion. Device failures can leave partial unpublished state, invalidate the context and require the existing caller error behavior. No per-step allocation or silent fallback is introduced. One event interval covers the advance; its interpretation remains elapsed stream span rather than pure active-kernel time.

The falsifiable hypothesis is that eliminating intermediate host fences and scalar reads reduces classical advance wall time for an identical tree. Verification must compare every completed step's scores, PV, admissions, stop reasons and integer statistics against both the pinned baseline library and portable oracle, then measure paired workloads with identical parameters and work counts. The existing `finish`/PV publication cost remains outside this fusion, so an advance improvement alone would not establish an overall engine speedup.
