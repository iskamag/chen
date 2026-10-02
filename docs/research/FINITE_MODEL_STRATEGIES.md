# Mechanical certificates and strategies for latency

This milestone proves small abstract mechanical properties and compares explicit cost models. It does not prove correspondence to the HIP program, global chess optimality, neural accuracy, a PAC guarantee for Chen, or optimal GPU latency. The timing objective remains stronger play within an offline one-second budget. Search-policy changes and mechanical accelerations need separate experiments.

## Compiled proof boundary

[CacheAndBackup.lean](../../proofs/CacheAndBackup.lean) imports only `Std`. The [frozen evidence](evidence/mechanical-policy-lean-2026-10-02/manifest.json) contains the exact source, object, compiler outputs, failed development attempts, and actual `#print axioms` output for **22 theorems**. Lean 4.27.0 and Lake 5.0.0 are installed. Compilation treats warnings as errors. There are no user-declared axioms, `sorry`, or `admit` declarations; the actual trusted foundational axioms reported across these theorems are **`propext`, `Classical.choice`, and `Quot.sound`**. The per-theorem dependency list is in [verification.json](evidence/mechanical-policy-lean-2026-10-02/verification.json). Lean's kernel, installed `Std`, and compiler installation remain trusted; this is not a hermetic bootstrap audit.

Reproduce from the repository root without modifying the frozen evidence:

```sh
lean -DwarningAsError=true proofs/CacheAndBackup.lean
python3 proofs/verify_policy_proofs.py --output results/policy-proof-replay
```

The verifier records the actual compiler/toolchain hashes and audits every theorem. It needs no Lake project or downloaded packages. Earlier parse/induction-name errors and an out-of-tree replay module-root failure are preserved as failed attempts, not proof admissions. The verifier now sets its subprocess module root explicitly.

| Certificate | Explicit premises and conclusion | Missing implementation bridge |
| --- | --- | --- |
| `cached_winner_stable`, `canonical_winner_unique` | Fixed scores/open counts; reservations increase; the cached canonical maximum remains eligible. Eligibility shrinks, so that winner stays the unique score maximum with earliest-index tie breaking. | HIP loads, packed masks, array bounds, and preservation of these premises are checked by source review and hardware tests, not formal refinement. |
| `finite_two_unread_obstruction` | Candidate children are exactly indices below finite `n`. Two distinct valid children are unread; the selector depends only on the fixed observed set. Two score arrays agree on observations but have different canonical winners. Universal correctness is impossible. | No adaptive decision-tree model, exact read-count theorem, or comparison lower-bound theorem is compiled. |
| `unaffected_children`, `backup_order_unique` | Ancestor closure; bounded rank; agreement outside the affected set; the same completed reduction equations depending only on greater-rank children. Both resulting records agree everywhere. | Does not prove that either concrete pass terminates or satisfies the equations; initialized new children and canonical local reductions remain premises. |
| `key_*`, `wave_max_*` | Positive biased scores, offsets at most 255, an invalid key 0; integer maximum is associative/commutative with identity0. The packed key prioritizes score then earlier offset. | No full Wave32 shuffle, lane participation, or machine-integer conversion proof. |
| `packed_increment_preserves_fields` | A natural-number low16 count below1024 is incremented. Quotient upper fields stay fixed and the low remainder becomes count + 1. | This is arithmetic, not a C++ bitvector or overflow refinement theorem. |
| `int8_*`, `integer_sum_bound`, `integer_grouping_exact` | Symmetric operands have absolute value at most127; at most 4096 products. Integer sums fit signed32 and integer addition regrouping is exact. | Does not verify thread indexing, padding, actual ISA, FP32 rescaling, or a neural network. |

The finite observation obstruction permits the conclusion that two unknown candidate scores cannot both remain unread in this model. It does **not** prove that all `n` scores must be read. A separate informal comparison-model argument gives the familiar worst-case `n−1` comparisons for a cold maximum: with distinct arbitrary scores and no prior certificate, each nonwinner must lose a comparison. This argument is not among the compiled Lean theorems. A previously computed best-child certificate changes the information model and can legitimately avoid those comparisons.

## Selection strategies under one immutable snapshot

For one choose batch, let `d≤256` be one parent's child count, `k` its successful visits, and `e≤d` the number of preferred children exhausted by increasing reservations. Scores, child order, and open counts stay fixed within the batch. No child reactivates. Across advances, scores and open counts can change, so these bounds require invalidation or maintenance.

| Strategy | Snapshot operation bound | Hardware tradeoff |
| --- | --- | --- |
| Canonical full scan on every visit | `O(k d)` eligibility/score work | Simple serial contiguous accesses, repeated reads. |
| Current packed winner cache | `k` winner eligibility checks plus at most `min(k,d)` full scans: `O(k + d min(k,d))` | No additional allocation; little benefit if each chosen child immediately exhausts. |
| Precomputed canonical descending order with monotone cursor | `O(d log d)` preprocessing, then `O(k+d)` eligibility checks | Sorting/storage and rebuilding can dominate parents visited once. Stable ties must be encoded. |
| Snapshot binary max-heap, deleting exhausted roots | `O(d)` construction, `O(k+e log d)` subsequent work | Extra metadata and irregular writes; changed scores and next-batch reservation reset require explicit rebuilding/repair. |
| Best-open child computed during existing backup, then packed cache | Adds an eligible-winner reduction to the already necessary child pass. First visit needs one seed eligibility check; exhausted winners retain full-scan fallback. | Moves the cold scan into an existing pass, but adds metadata traffic/arithmetic and may increase registers. It does not eliminate later exhaustion scans. |

These are my operation-count analyses under the stated model, not measured latency promises. A lower comparison count can still be slower because occupancy, global-memory traffic, dispatch, and dependency span matter.

The [committed packed-cache study](../../gpu/evidence/selective-cache-gfx1030-2026-10-02/summary.json) measured the six-position fixed-work aggregate dropping from 137.195 to 100.003 ms at 2,000 nodes/batch 8 and 817.527 to 405.193 ms at 50,000 nodes/batch 32. Its one-second profiles put selection at 32.23–46.53% and completion at 33.68–43.59% of dispatch duration. The small mate case was observed 1.50% and 2.82% slower in the two workloads, with only three repeats; there is no significance claim. Thus best-open metadata and parallel child reduction are plausible next experiments, while neither selection nor backup alone explains total time.

### Separate best-open experiment

Freeze the packed-cache library `a47b8c4e…` and its complete archived source closure. Append a private `uint64_t best_open` to `SelectMeta`: 16→24 bytes, an extra 2 MB for 250,000 nodes. This does not change public ABI. Initialize leaf/root seeds to `NONE`. During classical sparse backup, retain the existing value-best reduction and `Node.best` semantics; separately choose the canonical highest-scoring child whose open count is positive. Publish the seed only after complete child initialization and descendant backup.

When the packed choose cache is initially invalid, use this seed only after checking that it belongs to the parent's child block and remains eligible under masked reservations. An exhausted packed winner triggers the existing scan. Clear packed cache fields each advance as before. The external/neural selector remains unchanged and does not consume the new seed. All rejected candidates leave value/open metadata unchanged; admitted ancestors refresh the seed alongside their other metadata. These are design invariants awaiting implementation and actual parity/timing, not an adopted optimization.

A Wave32 alternative maps each child's signed score in `[-30000,30000]` to positive bias `score+30001`, then key `(bias<<8)|(255-offset)`. This fits 24 bits and gives deterministic ties. Integer max can reduce those keys in a tree, with invalid lanes0. Mapping all 256 children across waves still requires a correct second reduction and synchronization; the Lean algebra does not establish that GPU program. Actual emitted instructions, register counts, occupancy, and dispatch traces are required.

The neighboring `../bitmatrices` source was inspected read-only: `hip/live_set_scalar.h` and `hip/live_set_hip_reg_branchless.cu` expose scalar uniform masks and explicit wave communication. Their hashes are in the evidence bindings. No Lean proof files were found there. Those mechanisms motivate stable integer reductions and uniform control, but their source alone does not establish Chen's correctness or performance.

## Statistical optimality requires a different oracle model

[2FFS v2](https://arxiv.org/html/2606.01708v2), sections 2–3 and Appendix B.8, assumes a finite minimax tree, known deterministic fast-error envelopes, and independent sub-Gaussian slow samples centered on the true node minimax value. Its cost is `N_fast+c N_slow`. Theorem 3.1 separates finite-output confidence correctness from stopping. Theorem 3.6's stopping/cost bound requires local regularity; optimized allocations also require uniform regularity. Remark 2.5 explicitly identifies `H*` as an analysis-only recursive oracle complexity, not an information-theoretic lower bound. Unknown neural bias and repeated deterministic NN calls do not meet these sampling premises. Therefore this paper does not establish unrestricted instance optimality or a confidence certificate for current Chen.

A genuine information-theoretic comparator is [Garivier and Kaufmann, 2016, Theorem 1](https://proceedings.mlr.press/v49/garivier16a.pdf): for their independent exponential-family bandit model and uniformly correct fixed-confidence algorithms, expected samples are bounded below by `T*(μ) kl(δ,1−δ)`, with `T*` defined by a KL allocation game against alternative best arms. Track-and-Stop approaches the associated asymptotic sampling limit. These assumptions and the sample-count objective differ from arbitrary biased chess evaluations and finite GPU wall time.

My GPU interpretation: batching introduces a scheduling DAG, shared-memory/occupancy constraints, dispatch overhead, and stale decision information. Converting a query bound into latency optimality needs an explicit hardware cost model and admissible scheduler class. A valid confidence union budget and oracle law must survive any asynchronous sample selection; merely accelerating deterministic evaluations supplies neither. Existing [two-fidelity work](../TWO_FIDELITY.md) remains an isolated reference experiment.

## Full-GPU control experiments after metadata

Current worker traces contain `3×advances+3` stream fences and no search allocation/free calls. API wait duration overlaps device execution and must not be added blindly to kernel durations. A useful model accounts for critical-path device span, host/dispatch gaps, transfer dependencies, and completed work under a deadline.

Compare bounded epoch lengths 1/2/4 before designing an unbounded resident scheduler:

- Fixed-step graph replay could gate inactive work through device-owned state. Measure wasted dispatch and deadline overshoot; do not assume conditional HIP graph nodes. The [conditional-node request](https://github.com/ROCm/hip/issues/3905) was still open when checked on 2026-10-02.
- A single resident workgroup can impose uniform choose→expand/classify→backup barriers, but may underuse the GPU. Sticky errors, capacity checks, counters, and publication must be coherent even when work becomes inactive.
- A multi-workgroup persistent scheduler requires a supported cooperative launch and an actual scheduler-kernel occupancy query. Do not build an arbitrary global spin barrier or infer its safe grid from a different kernel's occupancy.

The [official HIP cooperative-groups documentation](https://rocmdocs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/cooperative_groups.html) requires cooperative launch for grid synchronization and warns that AMD lacks independent thread scheduling. That page describes 7.15.0, while this study uses 7.2.4. The team's archived RX6800 admission reports cooperative-launch support, wave32, and 30 reported multiprocessors; support does not establish a safe resident grid for a new kernel. Check installed headers/runtime and its specific occupancy before executing it.

Every epoch needs bounded work, uniform participation in barriers, no partial-admission publication, and the last fully completed result available to the host. Host deadlines between bounded epochs permit measurable overshoot bounds; device clock conversion/DVFS and cancellation behavior require separate validation. External host neural callbacks cannot run inside a classical persistent kernel without a new device evaluator/control path.

## INT8 arithmetic and fidelity

With symmetric integer operands in `[-127,127]`, each product's magnitude is at most 16,129. At 4096 terms the absolute worst-case sum is 66,064,384, safely within signed32. The compiled sum theorem also applies to partial groups containing at most 4096 such terms, provided the implementation does not duplicate terms. Associativity proves exact integer regrouping; shuffle indexing and ISA lowering remain separate obligations.

For one unclipped round-to-nearest quantized linear dot product, let per-coordinate errors be `e_A=s_A/2` and `e_W=s_W/2`. Expanding `(a+δa)(w+δw)` and applying the triangle inequality gives

`|dot(a,w)−dot(q_A,q_W)| ≤ e_W ||a||₁ + e_A ||w||₁ + K e_A e_W`.

This is an analytic bound here, not a compiled Lean real-arithmetic theorem. Add FP32 scaling/bias/rounding error separately. Clipping invalidates the half-step premise. Extending through nonlinear layers requires their stability bounds and the actual quantization scheme; this per-linear result gives no chess-strength or global neural-error guarantee. The hardware INT8 study owns actual index/ABI/ISA/parity/timing admission.
