# Milestone handoff — 2026-10-03

The user requested wrapping up at the next milestone and cleanup. GPU search
resource recovery, complete tactical edge buckets and the first neural learning
chunk are saved. No new experiment starts after this boundary. The full target
remains a modular gfx1030 engine with learned GPU-resident search, adjustable
play and eventual 2500 Lichess performance; it is not achieved.

## Retained ownership

- `chen` / `master`: Rust engine, exact rules, HIP search/producer interfaces,
  harness, canonical direction and integrated evidence.
- `../chen-neural` / `research/neural`: model/data/training experiments. Latest
  saved learning milestone is `861b3d3`, following implementation `58cd90c` and
  bucket wrapper `21ea33f`. No neural model is admitted to engine deployment.
- Temporary implementation branches remain available after their clean
  worktrees are removed. Historical source/runtime archives are immutable.

## Saved results and their limits

The [concurrent classical GPU pool](HIP_ROOT_POOL.md), main `7fef232`, improves
completed depth on three diagnostic positions at one second, with unchanged
chosen moves. Main `a3d7a68` adds the separately named
[hip-root-retry](HIP_ROOT_RETRY.md), making q/stack resource
failures local to a child. A failed child supplies no bound; a retry needs a
strictly tighter original beta from a completed Exact incumbent, and all work
counts against the same total budget. Its hardware fixture confirms other
children continue and retries happen. Kiwipete still cannot form any Exact
incumbent in one second: 15 failed/33 paused children, no retries, depth0 fallback.
This is a failure-handling result, not a strength improvement.

The [complete tactical buckets](TACTICAL_BUCKETS.md), main `a9415d2`, reduce the
measured batch20 producer/copy/captured-inference path from27.801 to6.082ms at
width128 and58.761 to11.848ms at width256. All measured rows fit E512 without
retry; batch-one benefits are smaller. This is one untrained opening-child
workload, not engine throughput or Elo. ABI1 remains the conservative control.

The [restricted-exchange study](../../chen-neural/docs/EXCHANGE_STUDY.md) has
two actual one-step typed HPO fits saved. Best DEV macro-family errors are
12.705/13.506 versus immediate-material129.292. These targets describe a
restricted recapture game, not unrestricted strategic chess value. TEST stays
unlabeled and closed. No HPO arm or architecture winner has been selected.

The requested [fresh CPU evaluation](harness/cpu-rating-current-2026-10-03.md)
is saved: 64 games/32 complete pairs, no errors/caps, all owned processes retired.
Default PVS-off scores 5W/2D/9L against configured1800, proxy1711. PVS-on's
configured1600/1800/2000 rows score 11W/3D/2L, 6W/4D/6L, 4W/4D/8L, with separate
pooled proxy1844. The older production GPU selective proxy1254 remains historical.
None measures the new GPU retry worker, neural quality, or Lichess rating;
no confidence interval or default change follows. Do not add the small direct
PVS match's relative estimate to these proxies. GPU/neural work remains paused.

## Next bounded work

1. Implement the isolated deterministic no-pruning move-ordering candidate in
   [the ordering handoff](research/ROOT_MOVE_ORDER_HANDOFF.md). Preserve immutable
   native root generation ordinals for certificate ties. Compare it against the
   unchanged retry control on exhaustive fixtures and one-second Kiwipete,
   reporting first-incumbent time, completed children/root depth and failures.
   More attempted nodes alone cannot select the candidate. TT/PVS and further
   parallel allocation experiments follow useful completed searches.
2. Resume the unchanged neural queue: one-step typed LR0.0003 with weight decay0
   then0.0001, seed6101. Only after all four HPO controls plateau may this arm
   select a pair and run6102/6103. Recurrence1/2/4/8 and typed/untyped arms are
   declared; only2 of48 planned fits have retired. Exact-child and scalable
   transformer comparisons still need their own admitted learning runs.
3. Once a frozen model/control comparison justifies it, measure its full native
   producer/inference/search path and paired equal-time games. Precision,
   inference latency, restricted-task quality and engine strength are separate
   comparisons. The eventual Lichess rating gate remains uncalibrated.

## Neural resume and cleanup boundaries

The tested GPU retry executable/library remain in `cpp-rocm` at
`/tmp/chen-root-retry-20261003/chen` and
`/tmp/chen-root-retry-20261003/libchen_root_retry.so`; the unchanged pool control
remains in `/tmp/chen-root-pool-v1-20261003`. These are frozen executed artifacts,
not a claim that later main builds have been replayed on hardware.

The frozen runtime is retained at
`/tmp/chen-exchange-learning-20261003.8wxbyhiu` on the host and in `cpp-rocm`.
Its declaration SHA256 is
`2e3e79a693db9daa7cfe2a8e2dc36a9f29209626f4b41ca7cd45cf5f17abcdfa`.
The saved state has `active_job=null`, exactly2 completed fits and1 group;
group000's process is retired. Do not restart either fit. Before resuming, bind
the unchanged source/data/runtime hashes and inspect the saved terminal state.
The existing supervisor creates `control/worker.log` exclusively: archive/rotate
closed group000 control logs before a new group, without changing the committed
archive or overwriting the previous journal. Continue with a fresh owned
watchdog and exclusive GPU lease, not an automatic fit restart.

Cleanup removes only retired temporary worktrees and disposable bytecode caches.
Five clean temporary worktrees were removed; all five branch heads are retained.
Keep the two original worktrees, experiment branches, evidence, release/build
caches, declared data/checkpoints and resumable runtime. Pre-existing untracked
semantic drafts in the neural worktree and symbolic-prior extraction evidence
on main are preserved; they are not part of the new verified milestones.
No container is stopped and no other user's process is killed.
