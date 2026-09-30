# HIP resident tree to Torch replay

The independently versioned export API in `gpu/chen_replay.h` copies exact
resident parent-child transitions directly into Torch-owned GPU buffers.
It uploads no board packets and returns scalar status, epoch, count, and event
timing. It preserves board, node, cached legal move, and legal count storage;
private scratch and timing telemetry can change. This is a replay producer,
not neural inference within the UCI search engine or playing-strength evidence.

## Storage and selection

`ChenReplaySink` is 136 bytes: version and descriptor size, device, legal slot
count, row capacity, then fourteen uint64 device addresses. The header specifies
the exact order, shapes, and element types. Boards and metadata retain absolute
white/black piece ordering and raw EP/clocks; unsigned values preserve their
bits in signed Torch tensors. Canonical moves rank-flip black squares with XOR56,
encode `(from*64+to)*5+promotion`, and sort exact legal support into an int16
prefix with -1 padding. An undersized legal prefix errors instead of truncating.

Selections are contiguous existing child IDs. Their source parents must be
expanded and nonterminal in a search-mode arena. Successors can be expanded,
classified, or unclassified. Terminal labels use actual successor legal moves
and retained ancestry: mate/stalemate first, then halfmove/material/repetition
adjudication. This does not include prospective next-move draw claims. Export
never invokes arena classification or changes its scored state. The material
proxy is `tanh(current-mover Material centipawns/800)`; it is not a game result.

Each row carries parent/child hashes and node IDs plus a context epoch. Epochs
advance on successful reset and accepted generic legal/make arena invalidation;
perft resets the arena. Export requires the expected epoch before validating
or writing output. Epochs never silently wrap. A caller context UUID separates
IDs across contexts, and root/history manifests preserve adjudication provenance.

## Runtime and ownership boundary

The producer and Torch must bind to the same HIP allocation/query/synchronization
runtime. Compiling the producer with ROCm 7.2 does not prove this: the installed
Torch package bundles another HIP library. A diagnostic preload that eagerly
loaded system HIP caused two active bindings; allocation-range queries through
the other instance accepted an interval beyond a raw Torch-runtime allocation.
That run was rejected as an invalid interoperability setup.

The adapter promotes Torch's already-loaded `libamdhip64.so` with `RTLD_GLOBAL`
before loading the producer. It compares all five producer identity tokens
from `chen_replay_runtime_identity(selector)` against Torch's explicit bound
functions: pointer attributes, allocation range, allocation, free, and stream
synchronization. Mismatches fail before context creation. The probe records
`dladdr` runtime paths and library SHA256 values. Loaded library names alone do
not establish actual symbol binding. The standalone Rust path continues to use
system HIP; these are distinct runtime environments.

Before export, Torch completes its producer/consumer work on the device and
retains every tensor allocation owner. C validates device memory type, device,
alignment, actual allocation base/extent, full advertised output byte ranges,
and nonoverlap with other outputs and private arena buffers. HIP allocation
bounds cannot prove a Torch logical view's shape, dtype, writable capacity,
contiguity, or exclusive ownership; the Python adapter validates those.

The private HIP stream drains before return, including preflight/runtime errors.
Selection and bounded-support preflight precede output writes. Runtime faults
can leave partial output; discard it and recreate the context after a genuine
HIP runtime fault. Validation/capacity failures are recoverable. After success,
the snapshot is independent of arena reset/destruction. Neural targets, replay
publication fences, and training are the Python adapter's responsibility.

## Executed oracle and copy diagnostic

`gpu/replay_probe.py` independently compares output with python-chess, then
performs pointer/schema/epoch/ownership boundary tests. Diagnostic output
inspection downloads tensors after export; it is outside the producer path.
`gpu/evidence/replay-gfx1030-2026-09-30.json` records the final executed source,
library, runtime, oracle, and trace fingerprints.

The copy diagnostic interposes `hipMemcpy` and `hipMemcpyAsync` around only the
synchronous C export call. The marked successful calls each issue two 16-byte
D2H summary/status copies, no H2D copies, and no D2D copies through those APIs.
A separate negative control intentionally copies 128 bytes through each API;
the diagnostic catches both, totaling 256 bytes. This validates that payload
leakage through these APIs would fail the oracle. It is not a complete trace of
HIP runtime internals, PCIe transactions, or every possible transfer API.

The helper uses system C++ with HIP headers and `dlsym`, without linking HIP.
Preload the helper first and the actual Torch HIP library second to preserve
runtime binding:

```sh
g++ -O2 -std=c++17 -D__HIP_PLATFORM_AMD__ -I/opt/rocm/include \
  -fPIC -shared -Wall -Wextra -Werror gpu/replay_copy_trace.cpp \
  -ldl -o libreplay_copy_trace.so
LD_PRELOAD="$PWD/libreplay_copy_trace.so:/usr/local/lib/python3.12/dist-packages/torch/lib/libamdhip64.so" \
  python3 gpu/replay_probe.py gpu/build/libchen_gpu.so --trace-copies
```

The captured run uses actual gfx1030 hardware. One physical GPU limits foreign
GPU allocation tests; a wrong-device descriptor is rejected, but a real allocation
on a second GPU was not tested. The finite fixture/transition suite is scoped
correctness evidence, not exhaustive state-space verification or an optimized
producer-throughput comparison.

The [combined integration report](../gpu/evidence/bridge-integration-2026-09-30.json)
records the later wrapper build: 62 actual-GPU neural tests without skips and
six Rust hardware tests after the extension. The Python adapter and its
[stream/ownership contract](../../chen-neural/docs/VRAM_BRIDGE_CONTRACT.md)
live in the separate neural worktree. Run the combined suite with the
`neural-test` wrapper mode described in [GPU integration](GPU_INTEGRATION.md).

HIP's official [memory API reference](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/doxygen/html/group___memory.html)
provides pointer attributes and allocation base/size queries. Installed header
signatures, actual runtime identity, and executed boundary tests determine the
concrete implementation used here.
