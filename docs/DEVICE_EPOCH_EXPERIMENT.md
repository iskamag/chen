# Bounded device-owned classical search epochs

Status: research prototype has executed on gfx1030; no production adoption.
The [matched causal dispatch experiment](CAUSAL_PROFILING.md) verifies identical
graph/queued trajectories and warmed timing. Earlier broader resident controls
remain partial work pending final evidence review; they are not adopted.
Production remains unchanged. The frozen control is the 11-file packed-cache
closure in `gpu/evidence/selective-cache-gfx1030-2026-10-02/sources/candidate`,
semantically equivalent to the GPU sources at commit `4e5782a`; that commit's
selective header has a documentation-only difference, recorded by staging.
The archived cache library is
`a47b8c4e5704703280834b4de864d1daf059b63ce79eb1f1fb22462994d3edaf`.

The current classical advance already keeps choose, child expansion/classify/
evaluation, sparse backup and admission on the GPU. Its repeated host boundary
is one completion copy/fence and the next dispatch per advance. The experiment
groups at most two or four advances under device-owned used/counter/admission
state and returns once. It does not transfer boards or score arrays to the host.

## Protocol

Reset uses the existing exact root/history/config admission and owns a separate
epoch state. A request supplies only the absolute node cap, evaluator0/1 and
advance bound2/4. Evaluator identity remains locked until reset. Every epoch
starts with an active gate and attempt count zero, retaining cumulative device
used/classified/evaluated/steps/expanded counters and the last complete result.

Each attempt performs parallel reservation/cache clear, single-thread canonical
choose, parallel complete sibling generation/classification/evaluation, then
single-thread sparse backup and publication. Root bootstrap uses an exact copy
of the frozen classify helper. Device code reads its current used/counters for
every attempt; host launch dimensions are fixed upper bounds. An absolute cap
still admits only complete sibling sets and never overshoots node admission.
Terminal/history/halfmove/insufficient-material logic and child ties are unchanged.

Publication builds a pending stats/root/PV packet, validates its complete chain,
then commits it to the last-result slot. Every completed attempt is also recorded
in a bounded four-entry trajectory, including a clean stop attempt. The host
compares these packets to one-step baseline calls, excluding timing fields.
A clean budget/capacity/candidate-scan/closure stop deactivates later attempts;
inactive phases leave counters, sticky error and last complete packet unchanged.
Any device/runtime failure invalidates the context, refuses finish publication
until reset and leaves the caller's previous host result untouched.

Host cancellation/deadline checks stay between bounded epoch calls. An in-flight
admitted sibling set always completes evaluation/backup. An epoch can therefore
finish up to two/four advances after the last host check. The harness must record
both the completed-advance difference and wall-clock deadline overshoot against
one-step control; no stronger cancellation claim precedes measurement.

## Four controls

1. **Gated graph:** two normal host-launched HIP graphs contain begin plus
   clear→choose→children→complete phases repeated two/four times, built at reset.
   The request chooses its bound; device gates handle early stops. Edges provide phase ordering;
   the host never reads intermediate admission or updates a kernel node.
2. **One resident workgroup:** one block loops over the same stages, using
   uniform `__syncthreads` on every path. Children and clear work use block-stride
   iteration. It is a useful control for dispatch savings versus lost parallelism.
3. **Cooperative resident grid:** the same loop uses HIP cooperative groups'
   grid synchronization. Admission must check the actual combined kernel's
   occupancy/resource use and cooperative capability, and bound the grid to
   resident blocks. Every thread reaches every grid barrier even after error or
   clean stop. There is no hand-written global spin barrier.
4. **Ordinary queued graph phases:** mode3 executes exactly the graph's kernels,
   dimensions, ordering and host completion protocol. It isolates graph dispatch
   from changes in device bookkeeping/publication and is a causal timing control.

Installed HIP7.2.4 headers provide ordinary graph kernel nodes and cooperative
launch. `hipGraphInstantiateWithFlags` explicitly warns that its flags are not
supported; this experiment does not infer working device graph launch from the
presence of an enum. Device phase control here means device gates/counters and
graph dependencies, with one host graph launch per bounded epoch.

## Numerical and timing acceptance

The prototype staging script binds all 11 committed baseline source hashes and
records differences from the archived cache closure before creating a new temporary candidate.
Only that copy gains extracted device-callable classify/choose/children/complete
helpers and the new experiment include. Original host kernels stay present for
in-process one-step control. Production files are not edited.

Before timing, compare every completed trajectory's integer score, full PV,
counter/depth/open/publication/stop fields and final arena/rules behavior. Include
both movers, root terminal/checkmate/stalemate, repetition history, halfmove100,
insufficient material, maximum ply, zero/exact/tiny node caps, arena capacity,
candidate scan, invalid admission, evaluator lock, sticky failure and reset.
Fixed-work timing compares equal completed work; one-second/deadline experiments
record exact resulting work and overshoot separately. Graph build/setup is
excluded from warmed repeated epochs and reported separately. Actual kernel
occupancy/resource admission is saved for each resident control.

## Future GPU neural phases

The exact symbolic producer can become another native graph phase while using
the same device-resident position/history buffers. INT8 static-weight packing
is setup work; native packed affine and fused dynamic/sourceK affine launches
can join a graph once tensor scratch/lifetimes and evaluator shape are declared.
Their existing C ABI launchers are host functions; resident kernels would need
device-callable arithmetic helpers and cooperative phase mappings rather than
calling a launcher from device code.

The current external Torch evaluator callback is host-owned and cannot execute
inside a resident HIP workgroup/grid. It needs an explicit epoch boundary before
the callback, a separately validated captured GPU graph with fixed ownership/
shapes, or a GPU-native semantic evaluator. Python/Torch admission, finite
checks, weight/model selection and runtime identity remain host setup boundaries.
The full GPU search plus symbolic residual neural target remains intact.
