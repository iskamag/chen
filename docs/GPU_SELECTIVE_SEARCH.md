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
5. Initialize new closure metadata, then propagate scores, best-child links, open counts and minimum/maximum leaf depths in dependency order. Classical evaluation uses the sparse admitted-ancestor buckets below, processing greater plies first. External evaluation retains the reverse allocation-index reference pass. Every child has greater ply and allocation index than its parent; both orders avoid parent/child write races and preserve canonical child ties.

Terminal and maximum-ply leaves remain in value backup but are never selected. When the best valued child is closed, selection considers eligible alternatives rather than repeatedly selecting the closed branch. Reservations disappear at the next step. Selection within a batch uses the pre-step values; changing batch size can change the explored tree and is an experiment parameter.

If at least one complete sibling set is admitted, the step reports progress unless the tree closes. With no admission, exhausting the candidate bound while an unreserved open leaf remains reports candidate-scan stop: this cannot prove that no cheaper candidate fits. Otherwise node-budget blockage takes precedence over arena capacity. Zero remaining budget or capacity can stop immediately, since an expandable nonterminal has at least one child.

## Scores and depth

Every published score is exact negamax over the **retained finite tree**, whose nonterminal leaves use the chosen heuristic. It generally differs from fixed-depth minimax. Only `HORIZON_COMPLETE`, meaning every retained leaf is terminal or at configured maximum ply, guarantees equality to the full-width target-horizon oracle with the same rules, draw policy and identical per-node evaluator scores. Neural floating-point results can depend on batch shape: model identity alone does not prove score equality, which requires measured parity for the compared executions.

Minimum and maximum frontier depths include all retained leaves: terminals, horizon leaves and open heuristic leaves. They are actual visited plies, not a promise that the maximum ply was reached uniformly. The Rust scheduler must expose selective depth separately and avoid claiming a uniform horizon from that maximum.

Mate scores retain the existing absolute-ply frame: a mated leaf is `-30000 + ply`; nonterminal static values are clamped to `[-29000,29000]`. A backed-up value outside that static interval proves a forced mate outcome under the engine's adjudication policy. While other branches remain open, the reported finite-tree mate distance is not asserted to be globally shortest. PV links follow deterministic best legal children and terminate at a retained leaf.

No transposition merging is performed. Draw adjudication depends on the ancestor chain, prior root history and halfmove clock; equal board hashes do not establish interchangeable histories. Legal generation precedes mate/stalemate and then current-position draw adjudication, matching the full-width arena. This does not add prospective next-move draw claims.

## ABI and ownership

[chen_selective.h](../gpu/chen_selective.h) defines independently versioned ABI1 with Config32 and Stats96. The implementation is [chen_selective.inc](../gpu/chen_selective.inc), included by the rules translation unit to reuse its exact device rules and evaluator code. `ChenSelectiveContext` owns a private rules arena; it exposes neither that handle nor device addresses. The full-width API and its prohibition on expanding statically scored leaves are unchanged.

The verified baseline bounds additional arena allocations at creation: 16 bytes per node for open/depth metadata, a 4,112-byte selection work buffer and a 96-byte device statistics record. The hardware-validated classical fusion below replaces that record with a 104-byte device completion packet and adds a 104-byte owned pinned host packet. Reservations reuse the inner arena's existing scratch flags. These extra private allocations are registered in the shared replay/teacher sink alias validator as defense in depth. The selective arena allocates no new storage during search; an external evaluator retains its own allocation policy.

All calls are synchronous and caller-serialized; each device operation selects its device, permitting exclusive transfer to the engine worker thread. The external callback uses the existing five-runtime-token guard and borrowed contiguous position/score contract. Evaluator identity, callback, owner and maximum batch are fixed until reset. No CPU replacement is performed on a backend failure.

Node limits count newly classified positions, including the root once; expanding an already evaluated leaf does not charge it again. Newly admitted children are classified/scored before the step completes, including at an exact node boundary. Cancellation is checked between calls. The largest atomic work unit is a configured expansion batch and its evaluator callbacks, so a wall deadline is not an interrupt inside a GPU operation.

Clean budget, capacity and candidate-scan stops preserve the last completed root/PV. Failed reset/advance invalidates that context and prevents stale `finish` publication. A caller can retain an earlier snapshot received through a report callback, but the current Rust search backend returns an error after advance failure and UCI emits `bestmove 0000`; it does not automatically return that earlier snapshot. Actual HIP/native faults may require process recreation; reset does not promise runtime recovery. Destruction checks device/stream retirement before freeing extension allocations; an unproven retirement fails the owned engine process with exit 70 rather than releasing potentially referenced storage.

Telemetry accumulates classified/evaluated nodes, completed batches, expanded leaves, open leaves, actual frontier/PV depths and event spans. Event spans include enqueue gaps and external callback dispatch; they are not pure active-kernel time. Scalar summaries/statistics and the final legal PV cross the host boundary. This design does not claim zero host transfer traffic.

## Verification boundary

The isolated `gfx1030`-target build completed with HIP 7.2.53211 / ROCm 7.2.4, exit 0 and no compiler output under `-Wall -Wextra -Werror`. [The build record](../gpu/evidence/selective-build-gfx1030-2026-10-01/report.json) and [hashed source/log archive](../gpu/evidence/selective-build-gfx1030-2026-10-01/manifest.json) bind that compiled library. This is compilation evidence only; no GPU runtime call was made during that check.

The first GPU kernels deliberately use ordered selection and ancestor backup as a reference implementation. They can later be replaced with parallel selectors or ply-wave backup behind the same ABI, keeping admission, tie and score semantics comparable. Tests should compare each completed batch with the portable Rust arena, compare closed target horizons with full-width minimax, exercise terminal/history/EP/promotion/mate cases and invalidation, and demonstrate greater selective depth under the same node/capacity budget. Equal wall-budget and playing-strength experiments remain separate evidence requirements.

## Classical fusion

The verified baseline is committed at `5a82052`. The classical-only fusion now has [actual gfx1030 parity and timing evidence](../gpu/evidence/selective-fusion-gfx1030-2026-10-02/report.json), with [hashed sources, binaries, runtime identity and raw process records](../gpu/evidence/selective-fusion-gfx1030-2026-10-02/manifest.json). Both baseline and fused libraries passed the same two portable-oracle hardware tests: 48 root/evaluator/batch configurations and 45 capacity/node-cap pairs per arm, plus history, invalidation, reset and thread-transfer cases. This establishes the tested ABI1, selection, admission, score and publication behavior. External neural evaluation keeps the baseline path and was not exercised in this fusion study.

One classical advance enqueues admitted-root classification when needed, reservation clearing, selection, fused child expansion/classification/evaluation, and ordered backup plus a completion snapshot on the same private stream. Child launch dimensions use the configured bounded maximum, and each thread checks the device selection count rather than requiring a host copy of that count. A child locates its selected parent by binary search over the existing device prefix offsets. Its history walk references its own freshly written node and previously retained ancestors; siblings do not depend on one another. The first child installs its parent's expansion metadata, which backup reads only after that kernel completes. Sibling threads read only immutable parent fields, avoiding a whole-node read concurrent with that update.

Every dependent kernel observes a sticky error; every launch is checked immediately. A private completion packet containing Stats96 and error status is copied once into an owned pinned host buffer, followed by one checked stream fence. Host counters change only after successful completion. Device failures can leave partial unpublished state, invalidate the context and require the existing caller error behavior. No per-step allocation or silent fallback is introduced. One event interval covers the advance; its interpretation remains elapsed stream span rather than pure active-kernel time.

The declared comparison ran six alternating cold-process benchmarks, three per library, on the same six-position suite with maximum ply 4, 2,000 classified-node budget, batch 8 and capacity 250,000. All 36 raw rows agreed on every field except timing/rate telemetry, including score, PV, admissions, stop reason and integer counters. The median sum of six search times fell from **164.763101 ms to 144.956140 ms**, a **12.02% reduction** (baseline/fused ratio 1.13664). Search timing includes the existing `finish`/PV publication cost. This small, bounded measurement supports a latency improvement for that workload; it does not establish warm throughput, equal-time playing strength, rating, statistical significance or long-run reliability.

The study used the preserved exact-hash verified baseline library `0cf4171f…` and a newly compiled fused library `04447c45…`, with identical Rust engine/test executables. Two clean fresh baseline compilations from identical frozen source bytes produced different binary hashes from the original artifact; the cause was not established, and no reproducible-build claim is made. Both fresh builds are retained in the evidence archive, while the declared comparison's baseline hash gate remained unchanged. All direct child processes and their owned process groups retired successfully; there was no timeout or replay.

## Classical sparse ancestor backup

The experimental variant `sparse_ply_single_thread_admitted_ancestors_v1` replaces the classical completion kernel's full `old_used` scan with unique ancestors of **admitted expanded parents**. Rejected candidates do not change scores or closure metadata and need no backup. [Actual gfx1030 evidence](../gpu/evidence/selective-sparse-ply-gfx1030-2026-10-02/summary.json) compares preserved fused library `04447c45…` with sparse library `c33fc8f9…`, using the same frozen Rust executables. [The manifest](../gpu/evidence/selective-sparse-ply-gfx1030-2026-10-02/manifest.json) binds exact compiled sources, executed binaries, raw records and profiler/code-object evidence. The external neural backup implementation remains the reverse-index reference and was not exercised in this study.

After selection and child initialization, completion walks each admitted parent's ancestor chain. A high scratch bit marks already enumerated nodes; reaching one ends that walk because its ancestors were already enumerated. Reservation counts occupy the lower bits, bounded at 1,024. Selection never reads the marker: the existing next-advance reservation clear removes it before any selection. Each ancestor goes into its ply bucket. One thread reduces buckets in descending ply, retaining the original canonical child loop and strict `score > best` tie comparison. This variant adds neither arena sorting nor separate kernel launches per ply. Parent/index/ply bounds, bucket occupancy and scored children are checked through the existing sticky device error.

The ABI maximum remains 64 plies and 256 admitted parents per step. At any ply, each admitted parent contributes at most one distinct ancestor, so 64 × 256 indices and 64 counts bound the scratch. The added storage is **131,328 bytes**; the existing work allocation grows from 4,112 to **135,440 bytes**, including for an external-evaluator context. Creation's `sizeof(*work)` allocation and private alias range cover the extension; the same checked destructor frees it after stream retirement. There is no new search-time allocation or ownership path. Reservation clearing remains a parallel full-arena operation; selection remains serial.

Both libraries passed the original portable-oracle hardware tests (48 root/evaluator/batch configurations and 45 capacity/node-cap pairs per arm, plus history, invalidation and host-thread transfer). A native comparison additionally checked **1,336 completed advance snapshots across 33 cases**, including configured depth 64, batches 1/32/256, both classical evaluators, budget/capacity stops, EP/promotion/terminal positions and context reuse through resets. Scores, PVs and every public non-timing statistic agreed after each call; an independent rules package checked the reported PVs. A separate mechanism fixture compared every node score/best link/flag and closure record on seven synthetic trees, including actual ancestor depth 64 and batch 256, with eight scratch guard words. Those synthetic trees establish maximum mechanism bounds; they are not legal chess searches.

The fixed-work comparison used three alternating cold-process pairs per workload, six authored FENs and capacity 250,000. All **72 raw rows** agree on every non-timing field. Median sums of six search times were:

| Workload | Fused baseline | Sparse variant | Reduction | Baseline/sparse |
| --- | ---: | ---: | ---: | ---: |
| Maximum ply 4, nodes 2,000, batch 8 | 143.180938 ms | 135.234465 ms | 5.55% | 1.059× |
| Maximum ply 64, nodes 50,000, batch 32 | 1,593.305559 ms | 818.045340 ms | 48.66% | 1.948× |

The improvement is not uniform: the small-workload mate position's median rose from **3.252472 to 3.342331 ms** (approximately 2.76% slower), while the aggregate improved. Three samples cannot distinguish a small change from timing noise; the per-position raw timings are preserved.

Twelve separate alternating one-second UCI observations produced 2,628 legal reported PVs. The sparse opening and Kiwipete searches reached the 250,000-position capacity at approximately 743 and 623 ms; the baseline completed approximately 123,000 and 142,500 classified nodes near its deadline. In the rook ending, sparse completed approximately 231,000 nodes versus approximately 90,000. These observations perform unequal work and return different finite-tree scores/PVs; they establish neither equal-work latency nor improved playing strength.

Six separate ROCm traces produced another 1,288 legal reported PVs. The completion dispatch share fell from 77.21–78.72% in the fresh baseline traces to **15.20–18.61%** in sparse traces; selection now accounts for **71.04–75.43%**. Each worker still had `3 × published advances + 3` stream fences and no allocations/frees. Host API waits overlap kernel execution and must not be added to device duration. The next measured target is repeated best-open-child scanning during selection; tiny sliding-attack benchmarks alone do not explain the remaining critical path.

The tuning used local `../bitmatrices` scalar/branchless live-set sources and AMD's [RDNA performance guidance](https://gpuopen.com/learn/rdna-performance-guide/) and [machine-readable RDNA2 ISA](https://gpuopen.com/download/machine-readable-isa/latest/). The actual timed libraries' second concatenated HIP bundle contains the chess rules; extracting only the first would inspect a different translation unit. The completion code objects use wave32 and 23 VGPRs in both arms, with SGPR metadata changing from 33 to 42 and zero private scratch/LDS. Static completion instruction counts increase from 370 to 553 while repeated arena scanning disappears; static counts are not executed instruction counts or occupancy measurements. Input library hashes remain unchanged through extraction using an explicit output copy.

The archive preserves the initial fixture constructor compile failure and the initial history fixture that set halfmove zero and therefore did not reach historical draw adjudication. The corrected history case sets halfmove two and explicitly requires a closed zero-score terminal root. All executed follow-up processes retired with exit 0. These small machine-local suites do not establish statistical significance, rating, neural execution or long-run fault recovery.

## Classical best-open-child cache

The next measured variant, `packed_best_open_child_cache_v1`, caches each expanded parent's canonical best eligible child **during one classical choose call**. Scores, child order and open metadata do not change during that call; reservations only increase. If the cached child remains eligible, every remaining competitor was previously no better, so the same winner remains canonical. Exhausting that child's open subtree triggers the original complete child scan. Cached ties therefore preserve the first legal child, and rejected candidates still consume the same reservation counts. No score, PV link or `nodes.best` field is changed by selection.

The existing scratch word packs a reservation count in bits 0–15, a canonical child offset in bits 16–23, cache-valid in bit 30 and sparse-backup marker in bit 31. Typed count helpers mask every classical reservation comparison and preserve upper fields while incrementing counts. Static bounds prove the configured maxima fit their fields; the existing next-advance scratch clear invalidates caches. The external selector uses a separate uncached template instantiation with the original plain-count semantics. This variant adds no allocation, per-node memory, clearing pass or kernel launch. Its sparse backup continues to mask the count separately from the marker.

[Actual cache evidence](../gpu/evidence/selective-cache-gfx1030-2026-10-02/summary.json) compares preserved sparse library `c33fc8f9…` against cache library `a47b8c4e…`, with [exact source/binary/raw-record bindings](../gpu/evidence/selective-cache-gfx1030-2026-10-02/manifest.json). Both original portable-oracle hardware tests pass per arm; another 33 native cases agree across 1,336 advance snapshots. Forty-eight mechanism cases compare admitted parent indices in order, child-block offsets, every masked reservation count and immutable node bytes between cached and uncached selectors. They exercise maximum ancestor depth 64, batch 256, candidate bound 1,024, tied/offset-255 winners, budget/capacity rejection and all-rejected candidates. A separate device assertion increments a packed count to 1,024 while preserving cache and sparse bits. The fixture's initial host call to a device-only helper failed compilation; its exact source and diagnostics are preserved alongside the corrected mask check.

All 72 alternating cold-process fixed-work rows agree on every non-timing field. Three pairs per workload give:

| Workload | Sparse baseline | Packed cache | Reduction | Baseline/cache |
| --- | ---: | ---: | ---: | ---: |
| Maximum ply 4, nodes 2,000, batch 8 | 137.194866 ms | 100.003288 ms | 27.11% | 1.372× |
| Maximum ply 64, nodes 50,000, batch 32 | 817.526710 ms | 405.193452 ms | 50.44% | 2.018× |

The mate position again shows a small observed slowdown: 3.507662→3.560402 ms in the small workload (1.50%) and 3.329761→3.423511 ms in the larger workload (2.82%). These three-sample medians do not establish significance or a uniform improvement.

Twelve separate one-second observations and six isolated traces produce **5,484 independently legal reported PVs**. Opening and Kiwipete trace arms have identical complete published trajectories to capacity (267 and 193 advances respectively): their selection dispatch sums fall approximately 507→118 ms and 404→81 ms. Clean one-second cache observations reach the opening/Kiwipete cap in approximately 344–352/298–302 ms. Rook cache reaches the same capacity in approximately 536–551 ms, while the sparse baseline completes approximately 230,000 nodes near its deadline. These capacity-limited observations establish no playing-strength improvement.

Selection now takes 32.23–46.53% of cache dispatch duration, with completion taking 33.68–43.59%; both remain serial critical-path targets. Every worker still has `3 × advances + 3` stream fences and no search allocations/frees. Exact timed-library ISA extraction reports choose metadata changing from VGPR16/SGPR47 to VGPR11/SGPR57, wave32 and scratch0/LDS0; completion stays VGPR23/SGPR42. Static resources are not an occupancy measurement. The independent archive replay checks all raw-member hashes, legal PVs, trace trajectory equality and code-object hashes without GPU execution. External neural execution and long-run fault recovery remain outside this study.

## Isolated best-open seed experiment

`gpu/experiments/selective_best_open_seed.inc` is a separate candidate based on the frozen packed-cache closure; production still uses the committed cache. [Actual seed evidence](../gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02/summary.json) compares cache library `a47b8c4e…` with seed library `f71eabee…`. [The manifest](../gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02/manifest.json) binds both source closures, private metadata sizes, executed drivers, binaries and raw results. The [finite-model proofs](research/FINITE_MODEL_STRATEGIES.md) justify reusing an eligible canonical winner under fixed scores and increasing reservations; they do not establish HIP refinement or optimal latency.

The candidate appends a private `uint64_t best_open` to `SelectMeta`, increasing it from 16 to 24 bytes, or 2 MB extra at 250,000 nodes. Classical sparse backup computes the best eligible child alongside its existing value/open reduction, preserving canonical ties and `Node.best`. A newly invalid packed choose cache tries this seed before scanning; exhausted winners retain the existing scan. Leaves/root initialization use `NONE`, and each affected ancestor refreshes its seed after initialized descendants. Allocation and alias registration both use `capacity * sizeof(*meta)` and destruction retires/frees the same owned allocation. External selection still instantiates the uncached implementation and does not consume seeds; its private metadata allocation also grows, so external latency is not covered by the classical result.

Both original portable-oracle tests pass per arm. Another 33 native cases agree across 1,336 steps, including reset reuse, depth 64, batches 1/32/256, capacity/node limits and repeated-root history. Forty-eight device selector traces agree on admitted-parent ordering, offsets, all masked reservation counts and unchanged node/metadata bytes. Fourteen synthetic backup cases compare value/open/min/max fields and independently check each canonical best-open seed, including mixed open/closed children, maximum depth 64/batch256 and eight guard words. The initial orchestration referred to a missing followup script and stopped before native work; its source/declaration/stdout/stderr are preserved, then the corrected run passed.

Each workload uses three samples per arm in alternating fresh processes on CPU5 with exclusive GPU ownership. This is a modest observed effect, with no significance or strength claim:

| Identical-work workload, six positions | Packed cache median aggregate | Seed median aggregate | Reduction |
| --- | ---: | ---: | ---: |
| Maximum ply 4, nodes 2,000, batch 8 | 99.703657 ms | 97.466158 ms | 2.24% |
| Maximum ply 64, nodes 50,000, batch 32 | 401.543681 ms | 384.256264 ms | 4.31% |

Per-position medians temper the aggregate: original opening improves only 0.21%, while promotion is 0.24% slower. The larger workload improves 3.41–5.84% on the five nonmate positions; mate is effectively unchanged at 3.326551→3.326602 ms. These small differences and three repeats do not support default adoption on their own. The previous cache study's small mate regressions remain recorded above.

Twelve plain one-second observations and six separately instrumented runs complete with 5,628 independently legal reported PVs. All three profile pairs have identical complete score/PV/counter trajectories: opening 267, kiwipete 193 and rook 478 advances, reaching 249,999/250,000 nodes before the deadline. The cap limits conclusions about uncapped one-second throughput. Plain seed times are 322–323 ms opening, 265–270 ms kiwi and 501–508 ms rook; instrumentation is analyzed separately.

| Equal completed profile work | Choose sum: cache→seed | Completion sum: cache→seed |
| --- | ---: | ---: |
| Opening, 267 advances | 120.491→105.679 ms | 115.069→105.297 ms |
| Kiwipete, 193 advances | 79.720→68.328 ms | 106.265→96.422 ms |
| Rook ending, 478 advances | 204.387→182.522 ms | 151.087→140.351 ms |

Completion also becomes shorter in these particular traces despite doing an extra reduction; these observations do not prove metadata traffic is free or isolate its cause. Candidate choose/completion still account for 30.53–44.55%/34.25–43.08% of dispatch duration. Every worker retains `3 × advances + 3` fences and zero search allocation/free calls. Host waits overlap device work.

Exact timed-library ISA extraction leaves choose at VGPR11/SGPR57 and increases completion from VGPR23 to 26, with SGPR42; both remain wave32, scratch0/LDS0. These are static resource declarations, not occupancy measurements. The candidate remains an isolated, source-reviewed comparison arm; the separate timing confirmation below strengthens the measured result without changing the production default. Full-GPU bounded epoch control and parallel reduction are separate experiments.

A separately declared [12-pair timing confirmation](../gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02/paired-summary.json) reuses the same frozen libraries at 50,000 nodes/batch 32. Six pairs run cache→seed and six seed→cache; two warmup processes are recorded and excluded. All 144 measured rows plus 12 warmup rows have exact non-timing parity, and all 26 native workers retire. No correctness tests are rerun. Every aggregate pair favors the seed, with ratios 1.038–1.069; the geometric mean paired speed ratio is 1.04675, corresponding to 4.47% lower search time.

| Large-suite position | Paired geometric mean reduction | Exploratory bootstrap interval |
| --- | ---: | ---: |
| Opening | 4.89% | 4.45–5.31% |
| Kiwipete | 5.32% | 4.46–6.12% |
| Rook ending | 3.97% | 3.34–4.66% |
| Middlegame | 5.77% | 4.80–6.91% |
| Promotion | 3.32% | 2.72–3.95% |
| Mate | −1.93% | −7.35–2.72% |
| Aggregate | 4.47% | 4.09–4.91% |

These are percentile intervals from 10,000 deterministic bootstrap resamples of whole paired runs, stratified to retain the six/six order balance. They describe this small machine-local process sample under the resampling assumption. Time dependence, shared power/cache, selection of the follow-up after inspecting the first study, and repeatability elsewhere remain uncontrolled; no hypothesis test or population-coverage claim is made. The tiny mate case remains noisy and does not improve consistently. The larger-suite confirmation supports a modest local gain, while default adoption, the original small suite and uncapped one-second strength remain separate decisions.
