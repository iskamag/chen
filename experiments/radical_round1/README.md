# Radical chess: architecture round 1

This round tests three substantially different interpretations of the proposed
modern-hardware chess engine. The earlier one-step codec and its untrained
policy are controls, not the definition of the research objective.

| Branch | Main hypothesis | Distinct mechanisms |
| --- | --- | --- |
| Causal | Computation follows changed observable consequences | Persistent source/line summaries, dependency subscriptions, event stopping, demand-driven proof support |
| Relation | Adversarial obligations can be discharged over shared alternatives | Reply operators, existential witness masks, exhaustive certificates, packed or streamed root/reply work |
| Bands | Chess operators admit small sufficient representations in different bases | 2x2 semantic blocks, 2x8 source bands, overlapping directional lines, transfer automata, hierarchical composition, explicit CPU vector kernels |

Implementations live in independent standalone Rust crates under this directory.
They depend on Chen for common state/oracles; experiment runtime/control roles
must be declared. No GPU execution is part of this round. AVX2 is available on
the current Ryzen 5 3600, but vectorization is an experimental factor, not an
assumed advantage. Generic/native and explicit-vector comparisons must match
flags on both candidate and control and distinguish latency from throughput.

## Shared gates

1. Compare complete semantic answers, not just checksums. Query contracts may
   include full attacks, king/check/pin constraints, legal root actions,
   candidate mate/stalemate and bounded reply/refutation certificates.
2. Use a strong control with the same query obligations. Separate prepared-state
   kernels from fresh ingress, sibling updates/undo and actual trajectories.
   Do not compare a rich attack/pin/evasion packet against attack masks alone.
3. Preserve negative results and rejected designs. Record memory/table footprint,
   initialized bytes, allocations, propagated channels, active lanes and shared
   or retired obligations when applicable, as well as elapsed time.
4. A primitive without a decision procedure has no strength result. A complete
   tactical procedure must recognize immediate mates correctly. Budget exhaustion
   is unknown, never a proof of absence. History-dependent game adjudication
   remains distinct from board-local legal/mate queries.
5. Freeze named variants before root-run certification. The root agent replays
   final binaries serially and independently audits output. Agent exploratory
   timings use separate physical CPU cores but remain shared-host diagnostics.
6. End the round with a joint comparison of original-proposal coverage, exactness,
   decision ability, cost, failures and proposed new architectural branches.

## Corpus

`harness/make_corpus.py` generates deterministic DEV and certification TSVs,
transition trajectories and exact python-chess legal/check/mate-in-one/stalemate
labels. It requires `chess==1.11.2`. Example:

```sh
python experiments/radical_round1/harness/make_corpus.py --out results/radical-round1/corpus
```

Root TSV fields are `id|split|family|fen`. Transition fields are
`id|parent_id|family|fen|uci|after_fen`; consecutive rows within a game form a
real legal trajectory. Parent IDs on the first event are descriptive sentinels.
The final generator emits 727 DEV roots / 3210 transitions and 659 certification
roots / 3036 transitions. DEV includes the existing special-rule regressions
and their color mirrors. Certification uses different trajectory and sparse
construction seeds, excluding DEV board identities. A 99/100 halfmove-clock
regression deliberately retains both clock contexts.

The reachable roots use uniform, capture-biased, quiet-biased and checking walks.
Constructed sparse positions are locally valid, not claimed reachable. Tactical
positive/negative quotas deliberately create a challenge distribution: this is
not a representative game-position or rating sample. The two sets contain 76/68
mate-positive roots, 165/95 mating actions and 16721/14834 legal root actions.
There is no supplied repetition history. Board-local mate/stalemate labels and
draw-clock/insufficient-material fields have separate meanings.

The compact corpus declaration in `docs/research/round1/corpus-manifest.json`
binds generator, seeds, inventories and complete file hashes. Full corpora,
labels, binaries and traces remain under ignored `results/radical-round1/` in
the primary checkout. `.ready` is a generation-complete marker, not evidence
that a process or experiment remains live.

## Work ownership and retained scope

The completed round's joint findings and proposed next branches are in
[`COMPARISON.md`](../../docs/research/round1/COMPARISON.md). In particular,
staggered 2x8/8x2 band stores remain untested, and no playing-strength result is
claimed for the research kernels.

Each agent owns its architecture crate and its corresponding report in
`docs/research/round1/`. The root owns corpus generation, independent output
audits, cross-branch comparison and later integration decisions. The production
engine and browser defaults are outside this experimental round unless evidence
later justifies an explicit reviewed promotion.
