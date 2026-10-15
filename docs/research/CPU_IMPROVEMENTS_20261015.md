# CPU integration and counterfactual experiment, 2026-10-15

Three initial agent worktrees independently implemented search allocation reuse,
exact evaluator tables and a counterfactual channel relation. A second isolated
codec refinement and independent comparator review then improved the mechanism
and corrected its performance comparison. The integration keeps ordinary
alpha-beta as the default and adds a named, experimental searchless consumer.
No GPU path or neural training was executed.

## Default engine result

The final CPU binary was compared with frozen `c4c9096`, using identical ordinary
release settings on an AMD Ryzen 5 3600 with rustc 1.99.0. Both executables run
`chen bench` with fresh searchers/TTs, depth 64 and 200,000 nodes over the six
existing positions. PVS off and on are measured separately. Eight alternating
rounds include two warm-up rounds and six measured rounds, pinned to CPU 2.
No other agent/build was running during this final measurement; the machine
remained a shared host without frequency/thermal isolation.

Every observed search matches completed depth, score, best move, nodes, qnodes,
stop status and full PV. All 192 rows agree with their corresponding position/
PVS signature. The source changes preserve the fixed-work search trajectory;
they change how much allocation and evaluation work implements it.

| Policy | Median baseline / candidate suite-time ratio | Less elapsed time |
| --- | ---: | ---: |
| Default PVS off | 1.1232 | 10.97% |
| PVS on | 1.1227 | 10.93% |

The default off lane's per-position throughput ratios were 1.167 start, 1.116
Kiwipete, 1.101 rook ending, 1.160 middlegame and 1.066 promotion. The mate fixture
finishes too quickly for a useful timing conclusion (observed ratio 0.976).
These are workload observations, not an updated rating or an equal-time match.

[Compact integrated evidence](evidence/cpu-integrated-20261015.json) retains
all per-position timing rows, complete search signatures, binary/source/suite/
runner hashes, compiler output and run configuration. The separate
[search allocation diagnostic](CPU_SEARCH_20261015.md) reports 29.8–42.4% fewer
allocation/reallocation calls. [Evaluator controls](CPU_EVAL_20261015.md) compare
the old arithmetic implementation directly against the exact tables.

The additional `BookSeed=0` repair changes the previously incorrect selection
behavior to the documented most-common edge, with stable authored-order ties.
Nonzero seeds retain weighted variety. It has a focused weight/order regression;
book selection was disabled in the implementation-performance comparison.

## Radical path and limits

The [codec](CHESS_CODEC.md) transposes sparse actions into bit lanes. Shared
line-prefix transfers and piece-specific channels compute attacks and legality
for many alternatives together without candidate `Position` construction.
Whole-set Boolean queries can intersect exact consequences directly; the
[policy/query guide](CODEC_POLICY.md) includes runnable examples.

The first codec looked faster against repeated per-square attacker queries, but
an independent source-piece-union control showed that comparison was too weak.
The [refinement](CODEC_TRANSFER.md) improves the codec itself about fourfold.
The final integrated binary, including normalized whole-alternative queries,
measured 3.905 us/root against 3.545 us/root for the strong scalar control in one
512-position/20-repeat diagnostic: **10.1% slower**, with identical complete attack
answers. The [final observation](evidence/codec-final-20261015.json) pins that
exact source/executable. Fixed method order, one sample and shared-host effects
limit precise comparisons. This does not establish a scalar speed advantage.

The independent python-chess oracle agrees on all 15,641 sampled positions,
414,731 legal actions and 22,399 checking effects, including 263 castlings,
75 en passant captures and 1,432 promotions. [Oracle evidence](evidence/codec-oracle-20261015.json)
binds the final binary, source, corpus hash and exact coverage. This is rules
coverage, not exhaustive proof or strength measurement.

`codec-policy` is a static experimental consumer, with untrained exposure
coefficients and no opponent-refutation model. It even misses a mate-in-one
fixture, documented explicitly in its guide. It is a way to exercise this
representation through UCI, not evidence of competitive searchless chess.
A playing-strength breakthrough has **not** been demonstrated. The concrete
result is an exact, executable alternative computation model with explicit
costs and an interface for composing consequence classes.

## Final checks and retained artifacts

Passed on the integrated source:

- 99 default-feature release engine tests, plus the fair-comparator example test.
- Strict all-target/all-feature Clippy (compile-only for optional GPU code),
  formatting and whitespace checks.
- Seven release web tests, nine API/worker smoke checks and client cancellation/
  capture-reaction checks.
- The independent oracle and the fixed-node binary comparison above.
- A real CLI query selecting checking alternatives through whole-word
  intersections, and parsed `codec-policy` benchmark telemetry over all six roots.

The source was not deployed to the live browser service. Full raw observations,
executed baseline/candidate/probe/oracle binaries, test output and a SHA256
manifest are preserved under ignored `results/improve-20261015/` in the primary
checkout. Compact summaries stay in Git; worktrees and their branches are kept.
