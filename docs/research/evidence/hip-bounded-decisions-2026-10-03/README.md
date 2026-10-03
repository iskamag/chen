# One-second classical decision diagnostic

PASS:18 decisions (six fixed benchmark FENs × three policies), all returned moves
and reported PVs legal. Balanced cyclic backend order, persistent UCI processes,
fresh root/newgame, Tapered evaluation,1s wall allowance. Engines ran onCPU5;
no other GPU jobs overlapped. Root's CPU0 match was concurrent, so CPU/system
power/cache were uncontrolled. This is a small reused-position diagnostic,
not a match, rating, closed holdout, converged quality verdict or causal isolation.

| Root | CPU alpha-beta | Selective GPU | Scalar bounded GPU |
| --- | --- | --- | --- |
| Start |g1f3,rank3,gap22|a2a4,rank15,gap58|b1c3,rank8,gap34;completed depth3|
| Kiwipete |d5e6,rank2,gap42|d5e6,rank2,gap42|a2a3,rank30,gap623;fallback depth0|
| Rook ending |b4f4,rank1,gap0|b4f4,rank1,gap0|b4f4,rank1,gap0;depth3|
| Middlegame |d4c5,rank1,gap0|d4c5,rank1,gap0|a2a4,rank9,gap36;depth1|
| Promotion |a7a8q,rank1,gap0|a7a8q,rank1,gap0|a7a8q,rank1,gap0;depth5|
| Mate |f7g7,mate1|f7g7,mate1|f7g7,mate1|

Rank/gap refer to the selected ordinary full-support Stockfish19 packet.
Gap is reported UCI integer CP difference, not Chen's evaluation scale, game
regret or exact minimax. Root-material/reference58 conversion preserves score
intervals in summary.json; Kiwipete's scalar fallback gap is about579.42..581.28
reference CP. No interval is made from bound/mate evidence. The mate query was
rejected by ordinary policy; the mate1 decision is separately checked by rules.
Five eligible packets had common depths9..13; support covered every legal action.
The selected historical rank1 packet may differ from final Stockfish bestmove.

Policies are materially different: CPU has TT/ordering/qsearch (no optionalPVS
in this frozen binary); selective GPU uses greedy mixed-depth static leaf work
with32leaves/1024candidates/cap1M; bounded GPU uses scalar canonical alpha-beta,
noTT/PVS, qply32/main16/totalnodecap1M and only completed Exact root publication.
Depths and node counts are therefore not comparable throughput units. Reported
nodes in the table artifacts are the last UCI report; bounded attempted/aborted
counts are separately available. Bounded unfinished-work fractions for start,
Kiwipete, ending, middlegame and promotion were72.86%,100%,81.94%,99.22%,66.47%.
Losing whole unfinished depths and executing one scalar worker remain the
architecture targets. No extra scalar launch/occupancy tuning is proposed.

Engine phase exited0 after16.59s, host9150 and all three engine children retired.
Its approximate UTC bounds10:31:04.092..10:31:20.680 come from pre-launch empty
stderr and post-retirement terminal file timestamps, explicitly not instrumented
device spans. CPU4-only Stockfish phase exited0 after4.00s, host10274/child9954
retired, instrumented10:33:09.914897..10:33:13.916769UTC. GPU visibility was empty
for the teacher phase. All six predeclared roots were retained; no rejected root
was replaced or relabeled. Authored FENs use empty history and are not synthetic
reachable game families or training data.

Evidence retains predeclared code/binary/library/helper/teacher/source/net pins,
12 fabricated packet admission controls, exact launch argv, raw output, complete
engine/teacher traces, typed scored emissions, packet provenance, score intervals,
retirement and summary. manifest.json hashes every payload, excluding itself.
Source helpers are pinned CPU packet/transport/score-frame code; no NN model
training or GPU calls occur in teacher/replay. Compiled engine/library remain
those from [the verified integration](../hip-bounded-integration-2026-10-03/README.md).

The next controlled candidate is a GPU root-child work pool: bootstrap all legal
children onGPU, prioritize the previous completed PV, keep finished child bounds
and resumable DFS work across bounded epochs, share a sound completed incumbent
for competitor cutoffs, and publish an Exact root only after every competitor is
dominated or completed. Independent oracle and equal-time decisions must precede
adoption. Parallel root workers must replace this one-thread reference floor.
