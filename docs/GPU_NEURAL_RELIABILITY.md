# Eager runtime stall investigation

The final production eager run passed 18 independent minimax comparisons,
11,982 iterative leaf evaluations, both UCI option orders and resource/stop
checks, then exceeded the unchanged 30-second watchdog on its first neural
game move. The same production graph run completed 40 legal game plies.
These are bounded observations; long-play reliability remains an open gate.

## Reproduction and observed waits

The [diagnostic report](../gpu/evidence/neural-eager-stall-diagnostic-gfx1030-2026-10-01.json)
retains six successful isolated paired first-move trials and a failing replay
of the full prelude. An exec launcher grants debugger access to its own engine
child. Two attached thread snapshots, taken before retirement, place the worker
in `rocblas_set_workspace → hipFree → SyncAllStreams → awaitCompletion`.
The main thread waits for stdin. This identifies the observed wait location;
it does not identify why completion never arrives.

[Torch 2.10's handle implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CublasHandlePool.cpp)
installs a workspace when acquiring the current BLAS handle and uses a rocBLAS
workspace call on ROCm. That source motivated a narrowly scoped candidate:
initialize the worker's handle between checked idle fences before eager decode.
The model, numerical operations, ownership rules and watchdog were preserved.

## Rejected candidate and runtime control

The [isolated priming candidate](../gpu/evidence/inference-blas-priming-candidate-gfx1030-2026-10-01.json)
completed all 18 numerical cases in both fresh runs. One full run still timed
out on the first game move; the other completed 40 plies. The failed run's two
stacks instead show HSA code-cache invalidation/executable freezing during
Tensile's lazy code loading. Priming is rejected as a reliability fix and has
not entered production. Its source/build snapshots and failed/successful raw
records remain archived. Its 27-case native matrix was not expanded after the
candidate failed its reliability criterion.

An [original-plugin runtime control](../gpu/evidence/neural-eager-direct-dispatch0-control-gfx1030-2026-10-01.json)
sets `AMD_DIRECT_DISPATCH=0` before Python/Torch starts, with the same environment
inherited by the engine children. It passed 18 numerical comparisons, option/
budget checks and 40 legal plies. [HIP 7.0 documents the setting](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/reference/env_variables.html).
This single pass does not validate a workaround or establish a causal link.
Production configuration remains unchanged. A next predeclared experiment needs
repeated original-plugin controls for both dispatch settings, with identical
artifacts, watchdogs and workload, plus latency/copy checks before any adoption.
The [repeated control attempt](GPU_DISPATCH_STUDY.md) subsequently ended partially
after an external container stop. Timeout stacks survive for both requested
settings, but complete replacement logs were lost; no completed six-run or
replayable numerical-parity result is claimed. The production policy remains
unchanged. The subsequent [parent-context control](GPU_PARENT_CONTEXT.md)
completes all six declared CPU-only/live-idle-HIP consumers with durable logs:
108 numerical comparisons and 240 legal game plies pass. Neither arm reproduces
the timeout, leaving its cause and long-play reliability unresolved. Device
history and oracle work remain confounded; no production workaround is adopted.

## Diagnostic scope

[`gpu/eager_game_probe.py`](../gpu/eager_game_probe.py) records process/thread
state, maps and two native stacks on timeout, then retires only its own children.
It reuses the frozen numerical/protocol oracle. Debugger packages were isolated
under `/tmp`; no system installation or GPU reset was performed. The reports
retain debugger positive-control evidence, actual stderr, artifact hashes and
environment scope. Current local model/package/plugin snapshots are prerequisites;
a clean clone must regenerate the deployment artifacts described in
[GPU_NEURAL](GPU_NEURAL.md).

Example using those existing container snapshots and an available debugger:

```sh
docker cp gpu/eager_game_probe.py cpp-rocm:/tmp/chen-native-plugin/eager_game_probe.py
docker exec -w /tmp/chen-native-plugin cpp-rocm \
  python3 eager_game_probe.py --case full --gdb /path/to/gdb \
  --output-dir /tmp/chen-eager-diagnosis-new
```
