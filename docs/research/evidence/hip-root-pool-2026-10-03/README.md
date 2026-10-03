# Concurrent root-child pool v1 diagnostic

Named `hip-root-pool` ABI3 candidate; no production backend adoption or playing-strength claim.

Native v0 passed23 exhaustive fixtures and5 controls. Review found its published qnodes were cumulative while published nodes were per-iteration; v0 source/output preserved. V1 fixes only that telemetry and rejects Rust epoch exhaustion, then replays23 exhaustive roots,5 native controls,7 actual Rust/UCI controls and18 decisions. All PASS.

Each depth primes the previous completed PV for one bounded epoch, then runs one independent scalar DFS per root-child GPU workgroup. Frames and completed bounds persist across epochs. Native canonical tie identity is deterministic legal-generation ordinal. Resource-aborted children supply no bound; v1 globally stops the iteration on their error.

## Same1s decisions

| Root | CPU PVS/aspiration25 | Scalar bounded | Concurrent pool | Pool stop |
|---|---|---|---|---|
| start | g1f3 d7 | b1c3 d3 | b1c3 d4 | deadline |
| kiwipete | d5e6 d5 | a2a3 d0 | a2a3 d0 | qply |
| rook-ending | b4f4 d9 | b4f4 d3 | b4f4 d4 | deadline |
| middlegame | d4c5 d6 | a2a4 d1 | a2a4 d2 | deadline |
| promotion | a7a8q d10 | a7a8q d5 | a7a8q d5 | deadline |
| mate | f7g7 d1 | f7g7 d1 | f7g7 d1 | mate |

Pool completes one extra depth on start/rook-ending/middlegame, but all its moves match the scalar control. Kiwipete stops after49.77ms with unresolved qply32 and no completed root iteration; all47 other workers remain paused. No child completed Exact before the resource exit. It preserves canonical fallback a2a3. The same unchanged ordinary complete-root teacher packet ranks that action30th with623 reportedCP gap (converted reference interval579.42..581.28CP); this is a bounded teacher estimate, not game regret or an Elo result.

Pool attempted node counts: start149064, ending127152, middlegame61724, promotion52438. These are different explored trees/windows from the scalar and CPU controls; they are not comparable useful-throughput proof. Latest unfinished iteration nodes remain92159/91417/49610/37455 respectively. Completed depth, work distribution and decision failures are the meaningful evidence.

The next variant must leave resource-aborted children without bounds, allow other workers to form a completed Exact incumbent, and retry a failed child only after a strictly tighter admissible original window is available. No cap increase or fabricated checked-leaf score is justified by this packet.

## Closure and limits

Final native400963b8da019b1560c751660db2e491d82302ce8134a795b603a0aa2d1a49e5; Rust64e27fc935904ca2e636eb040e725d34861e80c3b91952ed61449952cbd2b453.61-file source closure binds current c4dd06a GPU rules, including tactical exports. Actual loaded HIP provider7.2.70204 is recorded in probe-results.json; no Torch runtime is used. Wall-clock rate100000kHz, persistent scratch24446976 bytes.

Combined v1 window15.44s, all three phase launch/terminal records and engine retirement packets saved; container retirement shows only init+inspection ps. No other GPU kernel window overlapped. CPU3 exchange producer could share package power/cache; no causal timing attribution from that possibility.

Six authored FEN roots with empty history are diagnostic, not reachable-family TRAIN data or a fresh closed holdout. CPU differs by TT/order/PVS/aspiration/q policy; GPU policies remain canonical finite reference controls. Selected prior teacher packets are reused unchanged, bound by summary.json SHA and source commit8eb2681; no duplicate teacher payload or search rerun. Cold audit checks saved move/PV legality, packet/counters and summary only, not general strength.
