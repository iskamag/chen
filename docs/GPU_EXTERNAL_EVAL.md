# Synchronous external leaf evaluation

The independently versioned ABI1 in `gpu/chen_external.h` lets a trusted native
callback score resident GPU leaf packets without transferring boards or score
payloads through the host. The rules context borrows its existing position
buffer and int32 score scratch only for each invocation; these pointers never
become a persistent public arena view. The callback must finish its work on all
streams before returning, including failure. Its owner and code stay live through
the serialized synchronous call.

Before use, all five HIP runtime tokens must match the rules context's actual
bound allocation/query/synchronization functions. The first external invocation
locks evaluator2, callback, owner, and maximum batch until reset; classical and
external selectors cannot mix. A descriptor may not silently switch a model
owner or batch contract during a tree. The callback owner is trusted to preserve
its model identity and obey the borrowed-pointer protocol.

Evaluation preflights the entire submitted range for invalid expanded phases,
then performs rules-owned legal/terminal classification. Mate/stalemate comes
before draw rules; exact terminal scores remain untouched by external outputs.
Classification may commit terminal scores/counts even if later inference fails.
No static external scores commit on such a failure.

The GPU initializes scratch to an invalid sentinel and counts pending unscored
positions. If none remain, repeated evaluation returns existing telemetry and
never calls the callback. Otherwise the callback receives bounded contiguous
chunks, possibly including terminal or already-scored positions whose output is
ignored. Every chunk must succeed; a GPU preflight then validates every needed
score in [-29000,29000]. Only after that whole-range check does a GPU commit
scores and evaluated counts. Partial writes, out-of-range scores, or a callback
failure leave all static node scores uncommitted. The caller discards failed
iterations; validation/callback errors can retry the same locked descriptor,
while a genuine HIP fault requires context recreation.

Rules synchronize their private stream before callback entry and restore the
context device afterward. An event on that stream is completed before each
callback; a second event is recorded after its synchronous completion. The
cumulative `kernel_ms` therefore includes the callback event interval: inference,
host dispatch/scalar validation, and barrier/enqueue gaps. It is not pure active
GPU kernel time. Existing classical/replay event semantics remain unchanged.
Status/count/telemetry reads are bounded scalar transfers; callback model
implementation and its own stream completion are outside the rules library's
ability to enforce.

## Executed contract fixture

`gpu/external_probe.hip` supplies a device-only synthetic callback for lifecycle
and failure testing. It writes known integer scores with a HIP kernel, can leave
sentinels unwritten, emit an invalid score, or fail after a configured chunk. It
is explicitly not a learned inference model.

`gpu/external_probe.py` executed on actual gfx1030 and checked a failure on the
second chunk after GPU writes with zero static score commits, subsequent retry,
missing/out-of-range rejection, exact known leaf-score17 to root-score-17
negamax, idempotent and terminal/empty no-callback paths, runtime/owner/batch
locks, expanded phase rejection, and mate-before-halfmove draw ordering.
The separate `gpu/evidence/external-gfx1030-2026-09-30.json` fingerprints the
final hook, callback fixture, and observed log. Actual learned checkpoint
inference, feature parity, dynamic-batch behavior, and engine composition belong
to the native plugin/Rust integration evidence; this contract fixture does not
establish those or any playing strength.
