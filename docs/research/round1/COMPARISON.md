# Radical chess round 1: joint reassessment

The original prototype was an exact one-transition kernel followed by a weak
static heuristic. It was not a substantial implementation of the proposed
JPEG-like decision architecture. Its bit slicing was real, but that did not
establish an effective explicit SIMD implementation or competent move choice.
The mate-in-one criticism was decisive: the policy lacked the opponent-reply
existence predicate needed to recognize mate.

This round implements three distinct architectural branches in isolated
worktrees, including Astra's shared-reply branch. All three are integrated as
standalone research crates. The results support two narrower mechanisms:
shared reply-existence reasoning and target-conditioned constraints. They do
not establish a stronger complete engine or a general replacement for search.

## What was actually tested

* Causal: 14 named configurations spanning persistent line/source contributions,
  three dependency-discovery methods, counts, OR trees, deferred dense SIMD
  reduction, and demand/proof-support king predicates.
* Bands: three factorizations (2x2 microblocks, 2x8 source bands, cached 4x4
  leaper pyramids), each scalar and explicit AVX2; progressive king constraints
  are a separate query ablation.
* Relation: streamed existential witnesses, the complete retained reply relation,
  and density-ordered witness retirement; three root front ends expose ingress
  cost. Exact nested certificates detect moves allowing an opposing mate in one.

These are three architectural branches with internal ablations, not 23 wholly
independent architectures. Ordinary Chen supplies named controls and some
ingress/validation paths; each report identifies runtime dependencies.

The frozen corpus has 727 DEV and 659 reserved CERT roots, with 16,721 and
14,834 legal actions respectively. It includes seeded reachable trajectories,
special-rule fixtures and constructed sparse tactical quotas. Constructed
positions are locally valid, not claimed reachable. This is deliberately a
challenge distribution, not a representative game or Elo sample. Both colors,
promotions, castling, en passant, pins and multiple-word action sets are covered.
No repetition history is supplied; board-local mate and draw interpretation
remain separate.

Agents ran 38 frozen DEV process measurements in total. After freeze, the root
ran another **30 serial CERT processes**: three alternating portable/native pairs
for each branch, with three front ends in the relation branch. Candidate and
control always share compiler flags. Native means `-C target-cpu=znver2`, with
thin LTO and one codegen unit on a Ryzen 5 3600. Root replay uses CPU 2 after
agents stopped CPU work. Frequency, thermals and unrelated host activity are
not isolated; small differences are not significance claims. Bands workloads
have individual warmup and minimum measurement duration; other harnesses retain
their own declared repetition protocols. Do not compare absolute costs between
different query contracts.

## Speed: final independent CERT replay

Medians of three process samples, native build. All rows have a matching query
contract within the row; ingress exclusions are material.

| Workload | Experimental route | Strong control | Interpretation |
| --- | ---: | ---: | --- |
| All-action reply classification, actions prepared | 1.55 us/root | 4.53 us/root | Shared stream about 2.9x faster |
| Same, independent root front end included | 5.73 us/root | 5.60 us/root | Rough parity; no independent-route win |
| Same, conventional root front end included | 3.51 us/root | 5.64 us/root | Hybrid about 1.6x faster for this task |
| Checking-only reply classification, actions prepared | 1.00 us/root | 0.71 us/root | Sparse demand loses |
| Both full attack masks, prepared pyramid | 278 ns/query | 36 ns/query | Ordinary multi-source word operations win |
| Full attacks + both kings' constraint packet | 321 ns/query | 80 ns/query | Rich output does not rescue the transform |
| Both kings' progressive constraint query, cached state | 26.7 ns/query | 39.8 ns/query | Narrow query win, maintenance excluded |
| Persistent full attacks on actual trajectories | 115.6 ns/event | 59.5 ns/event | Best causal route about 1.94x slower |

The causal scalar control receives already-made child states; the event route
receives sparse packets. Sibling timings additionally include experimental
apply/undo. These are explicit system-boundary comparisons, not equal-input
instruction microbenchmarks. Packet construction is measured separately.

The band constraint win also has an important amortization boundary. Copying
and patching its state costs about 60 ns before querying on CERT. The per-query
saving is about 13 ns, so roughly five repeated queries at the same state would
be needed to amortize that patch cost under this comparison. Fresh encoding,
packet construction and complete tactical decisions are not free.

There is no fair end-to-end speed result for the nested opponent-mate procedure;
only its exactness is promoted. Validation-side timing includes extra checking
and output. No NPS or engine-level speedup is inferred from these kernel costs.

## SIMD reassessment

The original `u64` Boolean planes process up to 64 alternatives per instruction:
that is bit-level parallelism. Its attack kernel had no explicit AVX2 path.
Changing compiler target alone previously changed performance only modestly.

This round tests SIMD as an actual implementation factor:

* The band kernel explicitly evaluates eight topology lines using AVX2 gathers,
  shifts and Boolean operations. Actual measured-binary disassembly contains
  four `vpgatherdd` sites in its hot function; scalar uses the same tables.
  Native scalar bands take 289 ns versus explicit AVX2 333 ns on CERT. The
  pyramid is 278 versus 310 ns. This gather design loses on this CPU.
* The causal deferred OR reduction is compiler-vectorized: portable SSE2 and
  native AVX2 are confirmed in the measured hot function. Dense reduction beats
  eager sparse tree propagation, but the complete update still loses to scalar
  recomputation.
* The relation branch remains `u64` bit sliced. Its native improvement must not
  be attributed specifically to AVX2; no explicit vector intrinsic path exists.

These observations reject both “there is no parallelism” and “AVX2 alone will
fix it.” Output layout, demand density, ingress and dependency costs matter.
Static assembly evidence confirms implementation, not dynamic cache behavior.
No GPU code was executed.

## Mate competence and independent correctness

Actual old-policy choices on CERT find only **22 of 68** available mate-in-one
positions. Ordinary depth-one Chen with quiescence disabled finds **68 of 68**.
Both select an opposing-mate-in-one blunder on 1 of 15 roots where such a blunder
and a safe alternative exist. These are challenge-corpus results, not ratings.

The new relational classifier independently matches all legal/check/mate/
stalemate action sets on CERT, including **all 95 mating actions across those
68 roots**, for all three consumers. It also correctly classifies every one of
14,834 root actions for allowing an immediate opposing mate, with all 79 positive
reply witnesses independently verified as legal and checkmating. Both portable
and native binaries pass. Budget exhaustion produces unknown, never a mate proof.

This is a new certified tactical capability. It has not been wired into the old
static policy, so that policy still has the measured failure. A classifier
without a move-selection procedure is not credited with playing strength.

The other branches also have independent full-answer gates:

* Causal: all 659 roots plus all 14,834 legal children, totaling 15,493 boards
  and 1,983,104 square/color attack predicates, agree with python-chess for both
  builds. Internal validation separately checks each update architecture,
  illegal pseudo-actions and 3,036 shared trajectory events. The full report
  distinguishes individual lazy-king validation from aggregate sibling checks.
* Bands: six variants times 659 roots agree on 506,112 attack predicates and
  15,816 additional checker/pin/evasion masks per build. In-crate independent
  walkers check all 14,834 legal child patches. Exhaustive local tests cover
  65,536 line states and all 28,561 microstates at 16 placements.

The common auditors additionally reject a deliberately wrong attack bit, an
illegal classification entry, missing nested-action coverage and an invalid
positive witness. After integration, all **11 research tests**, formatting and
strict all-target Clippy pass in the primary checkout. No production source or
backend defaults change in this round.

## What the proposal now means experimentally

| Proposal component | Evidence this round | Remaining gap |
| --- | --- | --- |
| Different bases for chess operators | H/V/diagonal transfer channels, leaper bands | Actual staggered 2x8 + 8x2 stores untested |
| Tiny local dictionaries and hierarchy | Exhaustive 2x2 dictionary, maintained 4x4 leaper pyramid | No general tactical macroblock composition |
| Sparse typed edits and propagation stopping | Exact source subscriptions, OR trees, unchanged ancestors, proof dependencies | Maintenance loses on cheap whole-board attack tasks |
| Reason over compressed alternatives | Shared reply masks, existential elimination, witnesses and exhaustive absence | No reusable arbitrary continuation equivalence |
| Progressive coarse-to-fine queries | Exact slider-refinement skipping and first-two-blocker obligations | No complete policy built from these constraints |
| Lossy decision representation | Measured projected answer classes | No learned strategic channels or trained direct policy |
| Modern hardware | Explicit AVX2, automatic SIMD, matched native controls | No GPU run, no SIMD-over-obligations implementation |

The strongest representational result is that **78,125 classified king-line
inputs produce only 14–48 distinct joint constraint answers**, depending on king
file. Conversely, 28,561 combined microstates produce 28,507 distinct descriptors:
a general dictionary almost preserves identity. Selecting the query first creates
far more compression than collecting every fact first.

These are equivalence classes of current named answers. Equal current output
does not imply equal behavior after a future move. A continuation quotient must
preserve the chosen transition/composition operations; that is still unproved
and unimplemented. Likewise, the pyramid's 47.7% unchanged leaper-macro consequence
rate does not imply that 47.7% of complete chess decisions need no update.

## Proposed next round: three different architectures

1. **Compiled residual proof graphs.** Represent king escapes, checker removal,
   interposition and existential reply obligations as shared residual functions.
   Hash-cons functions that remain equivalent under a declared finite operator
   vocabulary, rather than merging equal final attack bits. Compile small
   target-conditioned channel codes and preserve refutation witnesses. Measure
   construction, reduction, useful SIMD lane density and output together. Gate:
   exact mate/refutation sets and an end-to-end win over both scalar and streamed
   controls on broad and checking-only queries. Reject if only repeated identical
   roots create reuse or graph construction consumes the gain.
2. **Lossy channel policy with exact tactical precedence.** Train compact factored
   decision tables, sparse cross-channel terms or a small quantized model to score
   actions directly in topology space. This finally tests the proposal's lossy
   decision side. Mate/stalemate and immediate refutation certificates take
   precedence over heuristic scores. Include a size/latency-matched flat-feature
   learner and the original policy as controls. Gate: zero avoidable mate-one
   omissions, held-out tactical improvements, then equal-time CPU matches with
   uncertainty reported. Teacher agreement alone is insufficient.
3. **Event-driven obligations across alternative worlds.** Subscribe a set of
   current proof obligations to typed edits; merge equal dependency residuals,
   retire witnessed lanes, and update only demanded constraints. Batch independent
   obligations in structure-of-arrays SIMD rather than gather full attack maps.
   Test a persistent staggered-band version against a rebuild-on-demand version.
   Gate: total certified decisions per second, including subscriptions, packet
   ingress, retirement, undo and cache footprint; preserve explicit unknowns.
   Reject if dependency maintenance repeats the causal branch's overhead loss.

The next round should spend effort on decisions and composition-preserving
residuals. Further tuning of universal full-attack expansion has weak evidence
as the main architecture on this CPU. Production Chen remains the competent
baseline while these research procedures acquire actual move-selection tests.

## Evidence and reproduction

Branch reports: [CAUSAL.md](CAUSAL.md), [RELATION.md](RELATION.md),
[BANDS.md](BANDS.md). Frozen corpus declaration:
[corpus-manifest.json](corpus-manifest.json). Compact independent audits, all
serial aggregate samples, binary pins and source hashes:
[cert-summary.json](cert-summary.json).

Full corpora, oracle labels, frozen binaries, raw process output and assembly
remain under ignored `results/radical-round1/`. The root serial replay manifests
live in `root-serial/{causal,relation,bands}/manifest.json`. They record every
command and stdout hash. The harness in `experiments/radical_round1/harness/`
generates corpora/threat labels, audits complete answers, runs frozen matrices,
reduces raw rows and assembles compact evidence. Branch reports give build and
individual CLI protocols. No optional GPU feature is needed for reproduction.
