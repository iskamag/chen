# Learned inference on gfx1030

The unchanged native callback is limited by dispatch overhead at small batches.
An executable fixed-shape graph prototype preserves all integer scores and the
range guard, and reduces synchronous latency by 1.75–2.15× at batch 1. At batch
256 the improvement falls to 1.05–1.06×. This is a measured prototype result;
the engine's default remains eager. An explicit native graph mode now implements
owned packet staging, exact-count caching and the same guard/error contract.

Evidence: [full report](gpu/evidence/inference-perf-gfx1030-2026-10-01.json),
[raw observations](gpu/evidence/inference-perf-gfx1030-2026-10-01.jsonl),
[actual stderr](gpu/evidence/inference-perf-gfx1030-2026-10-01.stderr).
The report fingerprints the executed plugin, its compiled source snapshot,
Torch/HIP runtime, benchmark, model archives, manifests and exact teacher
checkpoints. Source scope is commit `d245386`; the archived native plugin was
compiled before later comment-only source revisions.

## Controlled comparison

All three trained architectures use width 32 and depth 1, with their original
frozen TorchScript modules, float32 operations, three-logit utility readout,
nearest-even rounding and int32 centipawn output. Six position packets cover
both mover colors, castling, en passant, promotion and a populated middlegame.
Each batch rotates those fixtures. All 15 model/batch combinations passed exact
eager, scripted, native and graph integer parity. Changing the packet contents
at the same device addresses also changed the scores and matched eager output.

The benchmark uses five warmup calls and 12 measured calls per path. It then
collects 12 native/graph pairs, alternating which path runs first. The table
shows paired wall-clock medians in milliseconds; each call completes before
the timer stops.

| Model | Batch | Native callback | Graph with guard | Speedup |
|---|---:|---:|---:|---:|
| Square | 1 | 0.2940 | 0.1682 | 1.75× |
| Square | 8 | 0.3558 | 0.1682 | 2.12× |
| Square | 32 | 0.3545 | 0.1984 | 1.79× |
| Square | 128 | 0.4052 | 0.3539 | 1.15× |
| Square | 256 | 0.5167 | 0.4880 | 1.06× |
| Relation | 1 | 0.4556 | 0.2331 | 1.95× |
| Relation | 8 | 0.4795 | 0.2594 | 1.85× |
| Relation | 32 | 0.5785 | 0.3646 | 1.59× |
| Relation | 128 | 0.9344 | 0.8233 | 1.13× |
| Relation | 256 | 1.4322 | 1.3603 | 1.05× |
| Plan | 1 | 0.6633 | 0.3081 | 2.15× |
| Plan | 8 | 0.6768 | 0.3287 | 2.06× |
| Plan | 32 | 0.6924 | 0.4356 | 1.59× |
| Plan | 128 | 1.0818 | 0.9707 | 1.11× |
| Plan | 256 | 1.6551 | 1.5599 | 1.06× |

The native callback includes feature decoding, JIT execution, output shape and
range validation, a bounded scalar read, device-to-device score copy and device
completion. Eager and scripted model-only timings omit decode, guard and copy;
the report labels these distinct scopes. The graph captures decode, the same
frozen module, range predicate and score copy. Its bounded scalar validation and
completion remain outside capture. No guard or rounding contract was removed.

HIP events and host wall times are both recorded per observation. Event intervals
include gaps while the host submits work; they cannot establish pure GPU kernel
time. The smaller graph interval supports a dispatch bottleneck, but does not
separate interpreter time, enqueue time and device scheduling individually.
The shrinking benefit at larger batches predicts diminishing returns as device
computation dominates. A native staged graph that fails to reproduce the small
batch gain would falsify the proposed engine optimization.

CPU affinity, environment, process snapshot and start/end load averages are in
the report. End load averages were 1.25/1.38/1.65; PyTorch and BLAS used one host
thread. GPU clocks were not pinned. Fixed path order is supplemented by the
alternating paired measurements. Per-path first invocation and native context
creation times are retained separately. Native first batch-1 calls were
3.46/6.77/12.49 ms for square/relation/plan in an already initialized shared
process; these are not cold engine startup timings.

## Reproduce

Inside the existing ROCm container, use the isolated package/model/plugin
snapshots and ensure this benchmark is the only GPU workload:

```sh
docker cp gpu/inference_bench.py cpp-rocm:/tmp/chen-native-plugin/inference_bench.py
docker exec -w /tmp/chen-native-plugin \
  -e OMP_NUM_THREADS=1 -e OPENBLAS_NUM_THREADS=1 cpp-rocm \
  python3 inference_bench.py
```

The script rejects missing ROCm or a different GPU architecture, verifies all
five runtime identities and archive/checkpoint fingerprints, and fails score
parity rather than substituting CPU inference. Capture failures are explicit
records and are not reported as successful graph measurements.

## Native graph mode

The native implementation and [functional evidence](gpu/evidence/inference-perf-native-graph-functional-gfx1030-2026-10-01.json)
now use a separately versioned execution API. Configuration is 24 bytes and
telemetry is 88 bytes; [the header](gpu/chen_neural.h) defines the exact contract.
Configuration becomes immutable on the first nonzero evaluation. Eager is the
default, and graph mode has no padding or automatic fallback.

The context owns a maximum-batch packet staging buffer and its existing feature
storage. Every callback copies the incoming packet bytes D2D into this storage.
Each graph entry owns its exact-count output and range predicate, an independent
private allocator pool and its graph executable. Replay validates the bounded
scalar before copying scores into the caller's output. Cache-hit staging and
launch run on the same stream; capture misses complete staging before side-stream
warmup/capture. All success/failure paths retire work before owner reclamation.

The LRU cache bounds shapes and physical retained private-pool segment bytes,
including allocator rounding. It reports hits, misses, captures, evictions,
staging logical bytes and capture wall time. Pool admission occurs after capture,
so the retained limit is not a transient peak bound. Model, feature, staging and
runtime allocations are outside this budget. Observed pool bytes include the
old entries plus the candidate before eviction, rather than every intermediate
allocator high-water mark. Pools are never shared between graphs.

Raw HIP capture uses Torch 2.10's native allocator begin/end/release protocol and
unique Torch-generated pool IDs. Capture-ID/status matching also identifies
joined capture streams. Expandable segments and optional capture record-stream
reuse are rejected. Only the trusted generated deterministic static evaluators
are supported; the small IR screen is not a validator for arbitrary TorchScript.
The pinned [allocator implementation](https://github.com/pytorch/pytorch/blob/v2.10.0/c10/cuda/CUDACachingAllocator.cpp)
defines pool reference ownership and balanced begin/end/release; the installed
HIP headers expose that version's corresponding API. The standard
[Torch graph wrapper](https://github.com/pytorch/pytorch/blob/v2.10.0/aten/src/ATen/cuda/CUDAGraph.cpp)
documents unresolved allocator/generator states after capture failures, which
is why this implementation balances the deterministic raw-capture protocol and
uses conservative process poison instead of promising arbitrary retry.
Plugin GPU operations serialize process-wide. Capture requires that no other
same-process Torch or resident-rules GPU operation run concurrently outside this
plugin.

All 27 native model/count combinations, including 127/255 partial tails, matched
eager and scripted int32 scores on two distinct borrowed input allocations.
Each model's two-slot LRU recorded nine misses, nine hits and seven evictions.
Budget rejection preserved the output sentinel. Four deliberately injected host
failure stages recovered in the same context after balanced cleanup. An actual
GPU scalar-read operation unsupported during capture poisoned future eager,
graph, configuration and creation calls in a child process. An injected
unprovable retirement exited that child with code 70 before storage reclamation.
Actual capture faults require a process restart; these tests do not prove
recovery from arbitrary driver faults. Library lifetime must retain that poison
state across backend recreation.

Raw C callers must retain one native plugin DSO for the process lifetime and
reuse that same loaded instance for every eager or graph context. The poison
flag and plugin mutex belong to that DSO; loading another file copy creates
independent state and can bypass both protections. Retaining the original DSO
with `RTLD_NODELETE` alone does not prevent this. Rust integration must enforce
one plugin file identity before loading any replacement. After an actual capture
fault, raw callers must stop inference and unrelated Torch GPU work and restart
the process; backend recreation or another plugin copy is not recovery. The
negative child probe uses bounded diagnostic reads and retirement to inspect its
specific, successfully retired unsupported-capture failure, rather than
establishing a general post-fault computation contract.

The [native paired benchmark](gpu/evidence/inference-perf-native-graph-gfx1030-2026-10-01.json)
compares eager and graph callbacks from the same production library, including
input staging, output copy and guards. Twelve alternating pairs show:

| Model | Batch | Native eager ms | Native graph ms | Speedup |
|---|---:|---:|---:|---:|
| Square | 1 | 0.3002 | 0.1716 | 1.75× |
| Square | 8 | 0.3364 | 0.1724 | 1.95× |
| Square | 32 | 0.3193 | 0.2002 | 1.60× |
| Square | 128 | 0.3881 | 0.3515 | 1.10× |
| Square | 256 | 0.5179 | 0.4899 | 1.06× |
| Relation | 1 | 0.4800 | 0.2348 | 2.04× |
| Relation | 8 | 0.5200 | 0.2621 | 1.98× |
| Relation | 32 | 0.5131 | 0.3649 | 1.41× |
| Relation | 128 | 0.9157 | 0.8341 | 1.10× |
| Relation | 256 | 1.4028 | 1.3421 | 1.05× |
| Plan | 1 | 0.6142 | 0.3078 | 2.00× |
| Plan | 8 | 0.6840 | 0.3316 | 2.06× |
| Plan | 32 | 0.7436 | 0.4409 | 1.69× |
| Plan | 128 | 1.1002 | 0.9818 | 1.12× |
| Plan | 256 | 1.6423 | 1.5565 | 1.06× |

The first native graph invocation at counts 1/8/32 cost approximately 450–1,242
ms, whereas first counts 128/256 cost 4–12 ms in this process. Five shapes cost
1.41/3.55/3.64 seconds total capture time for square/relation/plan. These cold
costs can dominate short searches or constantly changing tails. At batch 1,
amortizing the observed capture time requires roughly 3,700–4,700 repeated
callbacks using the paired warm savings. Cache misses and evictions reset that
opportunity. Four cached relation/plan shapes retained about 247 MB of private
segments; packet staging separately occupied 32 KiB of logical storage.

Thus small-batch warmed inference improves, while arbitrary exact-shape search
can regress. Engine cold/warm searches and their actual shape distributions need
separate evidence before choosing graph mode for a playing configuration.

To benchmark the implemented path after building the plugin:

```sh
python3 inference_bench.py \
  --library /tmp/chen-graph-plugin/libchen_neural.so \
  --source-dir /tmp/chen-graph-plugin/gpu --native-graphs
python3 graph_probe.py /tmp/chen-graph-plugin/libchen_neural.so
```

Failure injection requires an isolated `CHEN_NEURAL_TESTING` build; ordinary
libraries reject the diagnostic injection API. The raw records contain actual
child stdout/stderr and retain separate artifact fingerprints.

PyTorch's [version-pinned graph guidance](https://docs.pytorch.org/docs/2.10/notes/cuda.html#cuda-graphs)
requires stable addresses and shapes, side-stream warmup and capture-safe work.
The [HIP 7.0 event API](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/doxygen/html/group___event.html)
defines stream event timing; host submission gaps remain part of this measured
interval. Neither these documents nor this experiment establish playing strength.
