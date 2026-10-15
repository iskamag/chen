# Round 1: quantified reply relations

The shared-reply kernel establishes exact candidate mate/stalemate predicates and
immediate opponent-mate certificates. On the frozen CPU development workload,
its prepared-action broad-query kernel is about 3x faster than Chen's scalar
reply-existence control. Its independent root generator consumes that gain:
with native compilation the complete independent route is slightly slower than
the scalar control. A conventional root front end plus the shared-reply consumer
is about 1.6x faster for this bounded classification task. Checking-only queries
lose. This is useful evidence about the unit of reasoning, not engine strength,
a general search replacement, or a breakthrough claim.

The standalone crate is `experiments/radical_round1/relation`. Production Chen
source and its default backend are unchanged.

## Mechanism and the original proposal

A root action is a sparse packet plus updated castling rights, en-passant square,
and halfmove clock. Up to 64 root actions become lanes in occupancy and typed
piece planes. Opponent reply templates are enumerated once from the opponent's
base piece sources and empty-board movement topology. Pseudo-legality is a mask
operation over all root lanes: source survival, destination ownership, blocker
interval, pawn direction/push/capture, en passant, and castling rights.

For each shared reply operator, king safety is evaluated directly on those root
planes with sparse reply overrides. A reply removes captured hostile sources,
changes blockers, and moves the defending king when necessary. Castling checks
initial, transit, and final positions separately. No `Position` successor is
constructed per root/reply pair. Root piece identities and occupancy are shared;
reply operators do not rebuild a complete attack map.

For legal root actions m and opponent replies r:

```
R = OR_r LegalReply_r
Mate(m) = Check(m) AND NOT R(m)
Stalemate(m) = NOT Check(m) AND NOT R(m)
```

This is actual existential elimination over a compressed relation. The result
contains legal reply witnesses; absence is established only after exhausting the
finite template basis. It realizes more of decision-making within the
representation than the previous static exposure heuristic. It still enumerates
a finite move-template basis and does not establish an equivalence that collapses
arbitrary continuations. No historical novelty is claimed.

Three kernels share the exact operator semantics:

* `Stream`: evaluate templates lazily and retire each root lane after its first
  legal reply witness. Store witnesses and final verdicts, not the full relation.
* `Full`: retain every tested `(reply, legal_root_mask)` row, with no lane
  retirement. This is a compact complete root/reply relation; tests can inspect
  its rows without expanding boards. Its output is richer than existence alone.
* `Grouped`: first compute pseudo-legal masks and sort by descending population,
  then retire lanes. This tests greedy mask coverage as an operator schedule.
  It is ordering, **not** a decision diagram or semantic state quotient.

The complete and existential forms are materially different consumers of the
relation. The third variant investigates scheduling within the existential form.
An initial eager-mask implementation and a subsequent lazy-mask/attacked-lane
retirement refinement are retained in raw local observations. They are not
compared as paired final binaries.

The kernel uses `u64` Boolean bit-slicing. Native compilation is
`-C target-cpu=znver2`; there are no explicit AVX2 intrinsics in this branch.
Native results must not be attributed to AVX2 alone.

## Bounded opponent-mate certificates

`mate_threats` computes, for each root action:

```
AllowsImmediateMate(m) = EXISTS r:
    Legal(m,r) AND Check(m,r) AND NOT EXISTS s: Legal(m,r,s)
```

It builds one small board view per root action, obtains reply packets with the
independent generator, and applies the shared relation again. It never builds
one `Position` per pair. A positive result returns an actual mating reply;
a complete negative result exhausts the applicable reply set. This is bounded
adversarial reasoning mathematically, not a renamed new minimax theorem. There
is no general score, long-term policy, selective search, or full match result.

An operator-word budget returns `Unknown` for unresolved candidates. A legal
reply witness remains conclusive after budget exhaustion. An exhausted budget
is never converted into mate or stalemate. Budgets bound the relation operator
stage, not wall time or all front-end work.

Candidate/current-root terminal labels are board-local legal-existence facts.
They are computed before any separate draw-clock interpretation. History-based
repetition and full game adjudication are outside this experiment. Carrying a
halfmove clock is not a claim that repetition history has been compressed.

## Verification and independent certification

Five Rust tests cover special rules, full legal sets on twenty seeded legal
walks of at most 120 positions each, budget unknowns, nested certificates, and
complete reply relation rows across multiple words. The latter compares all
reply masks against scalar complete legal reply sets on castling, en-passant,
and many-promoted-queen fixtures. All tests and strict Clippy passed.

Frozen DEV: 727 roots, 16,721 legal root actions, 165 mating actions. All three
variants' complete legal/check/mate/stalemate sets agree with frozen
python-chess labels. Every witness is legal. Nested classification covered all
16,721 actions, with 29 positive immediate-opponent-mate witnesses, and agreed
with a scalar exhaustive reference.

After kernel freeze, the parent independently ran the reserved CERT corpus:
659 roots, 14,834 legal root actions. All three variants agreed with independent
python-chess legal/check/mate/stalemate sets. All 14,834 nested mate-existence
answers agreed, and all 79 positive witnesses were independently legal and mate.
No CERT positions were used to tune or change the frozen kernel.

Frozen hashes:

```
lib.rs 7b404dd4604f918b2711e96b5f5a25a43c84029c8236fa8f209ad564211fce26
generic binary 556cc95910ec4d641e50895c5713f6d8b732f4be9513728f05792d7dec10b539
```

Raw evidence is under the primary checkout's ignored
`results/radical-round1/relation/`:

* `frozen-dev-validation.jsonl`, `frozen-dev-oracle-check.json`.
* `cert-root-audit.jsonl`, `cert-independent.json` (parent's independent gate).
* `matrix/matrix-manifest.json`, `matrix/summary.json`, `matrix/families.json`,
  and 24 raw matrix runs.
* `baseline.jsonl` and `lazy-early-exit.jsonl` are exploratory earlier observations,
  with slightly different development corpus versions; do not pool them into the
  frozen matrix.

Classification JSONL records have `kind=classification`, `id`, `variant`
(`Stream`, `Full`, `Grouped`), and complete `legal`, `checking`, `mates`,
`stalemates`, `unknown` arrays. `ROOT` rows report current board-local terminal
status. `THREAT` rows contain root action, mating-reply witness or `-`, and
completeness. Budget, timing, and counter records are separate.

## Controlled CPU matrix

AMD Ryzen 5 3600, CPU 3 affinity, CPU only. Generic and znver2-native builds use
the same frozen source. Four paired order runs per ISA/front-end combination,
100 repeats over the 727 DEV roots: 24 process runs total. Variant order and ISA
order reverse on alternating runs. The host is shared, with no frequency/thermal
isolation. There is visible occasional scheduling noise; the table reports
medians, and the raw manifest retains binary hashes and full commands. One early
validation process briefly overlapped CPU 3 during the first matrix run, so small
percentage differences should not be promoted as significant.

The scalar control uses Chen legal root moves and `has_legal_move` after each
root action: it stops at the first legal reply. This is a serious scalar
existence baseline, though not a newly optimized specialized check-evasion
engine. `Full` does extra work relative to this existence-only control.

End-to-end all-action classification, microseconds/root:

| Root front end | Generic scalar | Generic stream | Native scalar | Native stream |
| --- | ---: | ---: | ---: | ---: |
| Independent relational singleton generator | 6.448 | 6.147 | 5.789 | 5.908 |
| Existing complete codec relation | 6.343 | 4.914 | 5.779 | 5.039 |
| Orthodox legal generator | 6.394 | 3.686 | 5.826 | 3.603 |

Scalar values vary because the same control is rerun in each process; scalar
code does not change with the chosen experimental front end. The orthodox-front
result deliberately retains conventional root generation. It demonstrates a
potential integration seam, not a fully independent engine architecture.

Prepared actions supplied free to both consumers, microseconds/root (own-front
process medians):

| Query demand | ISA | Scalar existence | Stream | Full | Grouped |
| --- | --- | ---: | ---: | ---: | ---: |
| All legal root actions | generic | 5.190 | 1.668 | 3.715 | 2.226 |
| All legal root actions | native | 4.653 | 1.595 | 3.502 | 1.930 |
| Checking actions only | generic | 0.813 | 1.099 | 1.688 | 1.394 |
| Checking actions only | native | 0.736 | 1.033 | 1.567 | 1.248 |

The broad shared kernel win is substantial; its front-end cost and sparse-query
loss are equally material. For checking-only end-to-end work, independent stream
is 5.627 us native versus scalar 2.711 us. Computing fewer answers leaves fewer
active lanes and fails to amortize representation/template setup.

Across one full DEV pass, `Stream` tests 27,030 active root/reply pairs in 5,936
operator words; `Full` tests 414,354 pairs in 20,567 words. `Grouped` tests 24,829
pairs in 3,735 words. Average active lanes per tested word are approximately
4.55, 20.15, and 6.65 respectively. Most root lanes retire early, leaving small
hard classes. Sorting improves logical work but loses wall time. Full retained
row capacity totals 451,648 bytes across the corpus; streaming has no retained
row array. These are allocation totals, not peak RSS, and exclude common scratch,
packets and outputs. Piece planes alone are 6,144 bytes per word; occupancy,
rights and en-passant planes add 1,056 bytes before small metadata. No overall
memory-compression advantage is claimed.

DEV operator-budget experiment (all known answers checked against the oracle):

| Operator-word budget | Decided actions / 16,721 | Proven mates / 165 |
| --- | ---: | ---: |
| 0 | 0 | 0 |
| 1 | 9,842 | 0 |
| 2 | 13,250 | 0 |
| 4 | 15,567 | 73 |
| 8 | 16,195 | 104 |
| 16 | 16,489 | 152 |
| 32 | 16,705 | 165 |

All other actions remain explicitly unknown. This illustrates a proof frontier;
it does not provide a strength/time tradeoff or guarantee useful move selection.
Nested certificate timings in validation output include instrumented scalar
validation and printing on the reference side; they are **not** a fair
performance comparison. Only its exactness result is promoted here.

## Cross-branch interpretation and next experiments

The parent reports that the causal branch's fourteen variants mostly lose to
maintenance overhead (one deferred-SIMD trajectory around 125 ns versus 67 ns
scalar); explicit AVX2 band gathers also lose on broad attack production. These
are compatible with this branch: modern hardware does not reward maintaining or
gathering facts that the consumer can cheaply avoid requesting.

The parent also reports promising target-conditioned band obligations (roughly
45 versus 71 ns, pending stable timing), local 2x2 leaper consequences unchanged
on 47.7% of legal moves, and 78,125 king-query inputs collapsing to 14–48 outputs
per file. These findings motivate **query-specific residual functions**, but
identical current output is not a continuation equivalence. Two states with the
same attack bit can react differently when a blocker moves. Any exact quotient
must preserve the chosen operator/query composition, not only a scalar answer.

The original JPEG proposal remains only partially explored: topology and exact
shared predicates now exist, but overlapping learned/finite-state tactical
blocks, lossy strategic channels, and reusable continuation equivalences do not.
The earlier policy's reported CERT result (22/68 mate-positive choices, versus
68/68 for depth-one no-quiescence control) reinforces that representation
correctness is not a decision-quality result.

Three next architectures worth distinct branches:

1. **Quantified residual decision diagrams.** Compile each checking/evasion
   obligation into a shared DAG of occupancy/source predicates and existential
   eliminations. Hash-cons residual functions under a declared finite reply
   vocabulary; persist reusable suffixes across root actions. The question is
   whether composition-equivalent obligations merge, not whether leaves have
   equal answers. Gate: exact closed-corpus mate and immediate-refutation
   certificates; report nodes/obligations before and after reduction; beat the
   scalar and streamed controls including construction on sparse checking-only
   and broad workloads. Reject if DAG/cache overhead dominates or reuse occurs
   only on repeated identical roots. Worst-case exponential growth remains.
2. **Learned channel decision tables with exact certificate filters.** Train
   small CPU decision tables or sparse factored linear/tree models on topology
   descriptors and their action deltas, using a fixed teacher protocol. Select
   directly in channel space, with exact terminal/refutation predicates taking
   precedence. Include cross-channel features deliberately rather than assuming
   independent bands encode tactics. Gate: zero avoidable mate-one omissions,
   controlled held-out tactical and equal-time match improvement over the same
   static policy and a size/latency-matched flat-feature model. Teacher agreement
   alone is insufficient. This investigates the lossy decision side of the
   original proposal instead of another exact move-generator variant.
3. **Demand-driven proof subscriptions.** Subscribe only current proof obligations
   (king escape, checker removal, relevant interposition) to typed causal events.
   Cache residual ray/band functions, invalidate on dependency changes, and stop
   when the externally observed residual is unchanged. Combine the band branch's
   target-conditioned predicates with this relation's witness retirement. Gate:
   measure update plus consumer cost over real batches and trajectories, validate
   every reused certificate, and show an amortization crossover on held-out
   workloads. Reject global attack-map maintenance or outputs-equal caching
   without a sound invalidation argument. This differs from merely accelerating
   the existing root front end.

No variant here is ready to replace Chen's search backend. The immediate reusable
result is the exact shared-reply operator and its measured broad-query advantage.

## Reproduction

```
cargo test --release --manifest-path experiments/radical_round1/relation/Cargo.toml
cargo clippy --manifest-path experiments/radical_round1/relation/Cargo.toml --all-targets -- -D warnings
cargo build --release --manifest-path experiments/radical_round1/relation/Cargo.toml

taskset -c 3 experiments/radical_round1/relation/target/release/chen-relation-r1 \
  results/radical-round1/corpus/dev.tsv 100 --labels --budgets --threats

CARGO_TARGET_DIR=/tmp/chen-r1-relation-native RUSTFLAGS='-C target-cpu=znver2' \
  cargo build --release --manifest-path experiments/radical_round1/relation/Cargo.toml

python3 experiments/radical_round1/relation/run_matrix.py \
  --corpus results/radical-round1/corpus/dev.tsv \
  --out results/radical-round1/relation/matrix \
  --generic experiments/radical_round1/relation/target/release/chen-relation-r1 \
  --native /tmp/chen-r1-relation-native/release/chen-relation-r1 --runs 4 --repeats 100
```

`--frontend=own|codec|orthodox` selects the explicit root-generation ablation;
`--reverse` reverses method order. The corpus/oracle files are generated by the
parent's round-one harness and are intentionally not checked in with this branch.
