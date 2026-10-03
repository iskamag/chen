# Experimental hip-bounded backend

The `research/bounded-gpu-integration` branch/worktree adds a runnable scalar
GPU alpha-beta reference, named `hip-bounded`. GPU code owns legal generation,
apply, full-history adjudication, Tapered/material scoring, main search and
qsearch. Rust schedules iterative full-window depths and publishes a completed
Exact root. The existing `hip-selective` implementation is unchanged. This
reference establishes a different search-policy baseline; it is not the final
parallel architecture or a playing-strength claim.

```sh
bash gpu/experiments/bounded_search_timed_build.sh /tmp/libchen_bounded_timed.so
cargo build --release --features hip
export CHEN_BOUNDED_LIBRARY=/tmp/libchen_bounded_timed.so
target/release/chen bench --backend hip-bounded --movetime 1000 --depth 16
```

UCI exposes `Backend hip-bounded`; `go movetime`, `nodes`, `depth`, `stop` and
ordinary root-history moves work through the existing transport. CPU application
of user moves is transport/state preparation; search does not generate CPU legal
moves or score CPU leaves. Native support is admitted for actual gfx1030 with
no architecture override. Unsupported/missing libraries fail explicitly.

## Semantics and limits

The new [timed ABI2](../gpu/experiments/bounded_search_timed.h) embeds the exact
legacy ABI1 job/result and retains that API unchanged. It allocates persistent
native job/result/scratch/history capacity once; the Rust backend uses one job
slot. All DFS work executes on one device thread. Canonical legal order, no TT,
no PVS, no previous-PV reordering and no selective pruning are deliberate
controls. Main depth is at most16, qply32, stack64 and total search-node allowance
is at most1M across iterations. Requested depths/nodes above those limits are
clipped by this experimental backend and disclosed as the bounded telemetry.
Every checked qsearch evasion is searched; an unresolved q cap aborts. Mate has
precedence over draw counters, and mate scores retain the root score frame.
History contains previous positions only; current root/local path live onGPU.

Root bootstrap is a separate one-node GPU admission with a50ms device allowance.
It supplies a canonical legal fallback and terminal classification even with
zero requested nodes/time. Bootstrap is charged to wall/event telemetry but
excluded from `SearchResult.nodes`; `bootstrap_nodes` discloses it. That fallback
has depth0 and no optimality claim. A complete Exact iteration replaces it.
Every deadline, node, qply, stack or cancellation abort keeps the previous
completed root score, move, PV and depth. Attempted/aborted nodes still accumulate
in resource telemetry. Full-window Lower/Upper or malformed packets are rejected,
not published as root decisions. Runtime/configuration failure returns an error;
uncertain HIP retirement retains native resources and requires process retirement
rather than unloading live kernel code. The Rust owner aborts if cleanup cannot
prove retirement.

## Deadline and cancellation

`elapsed_ns` in ABI2 is a device elapsed-work allowance, not a host absolute
clock. Its range is1ns..5s per launch. Native admission queries
`hipDeviceAttributeWallClockRate`; `wall_clock64()` is the constant-frequency
steady counter in the installed HIP7.2.4 `amd_device_functions.h:630`, distinct
from frequency-varying core `clock64()`. Conversion is ceil(ns*kHz/1e6), with
admitted operands bounded against overflow. Rust subtracts bootstrap/history/
prior iteration time from the requested host wall allowance before each launch.
Device polling occurs every32 DFS transitions and before publication. A legal
move generation/evaluation is indivisible, so a hard nanosecond completion bound
is not promised. Copies, stream fences and the joined monitor add host overhead.

The cancellation monitor only writes a native host atomic into explicitly
`hipHostMallocMapped|hipHostMallocCoherent` storage; the device uses a system-
scope acquire atomic. It never runs CPU chess work or host HIP launches. The
owner clears cancellation before the monitor starts; run does not clear it.
The monitor is joined before context teardown and is released even on unwind.
The [HIP programming manual](https://rocm.docs.amd.com/projects/HIP/en/docs-6.2.0/how-to/programming_manual.html)
describes fine-grained coherent host allocation; installed headers and actual
cancel tests establish this candidate's concrete supported path.

## Evidence and next direction

Six focused Rust tests verify exact publication vs abort, zero budgets/history,
FFI layout, malformed/bound rejection, terminal bootstrap and concurrent stop.
The strict HIP build plus [actual gfx1030 packet](research/evidence/hip-bounded-integration-2026-10-03/README.md)
passed33 independent semantic cases/19 exhaustive controls and8 UCI cases.
Wall-clock admission100000kHz agreed with realized1/20/50ms deadlines; measured
overshoot was1.536/0.235/0.286ms. Mid-kernel cancellation response was1.35ms and
UCI stop response1.11ms. A1s start-position move retained Exact depth3 in1001.65ms.
This executes using native HIP7.2.4, separately bound from prior Torch runtime.

That move spent8034 of10877 attempted nodes (73.86%) in aborted depth4. Approximate
10.8knps on one scalar worker is the reference floor. The next architecture must
change work allocation: parallel root-child tasks with typed completed bounds,
shared root incumbents/cutoffs, retained completed child work and complete
competitor domination before an Exact root publication. It must preserve history,
mate frames, qsearch completeness and cancellation/last-completed semantics.
Graph launch microtuning does not address this serial/search-work limitation.
No equal-time match, calibrated rating or production promotion is claimed here.
