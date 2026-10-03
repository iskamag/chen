# Native topology correctness packet

Status: **PASS**, four actual gfx1030 tests, 755 independently compared rows.
The process exited zero and retired; `retirement.txt` contains only container
init and the inspection process. No timing or training was performed.

The separate 80-byte typed sink exports all 64 source pseudoattack masks,
mover/opponent pins, current-mover checkers, both-color pinners and arena identity.
Original symbolic ABI1 declarations and implementation are byte-identical after
removing the additive topology sections, as checked in `summary.json`. ABI1's
seven exact topology channels were also checked on 24 mirrored fixtures.

- `source-binding.json`: pre-build base commit, 13-file local compiler closure,
  frozen probe and staging paths. Git retains the source; this packet does not
  duplicate old source archives or store a shared-library binary.
- `declaration.json`: frozen library hash, 27 reference Python module hashes,
  exact launch argv, bounded watchdog and acceptance scope before launch.
- `compiler.txt` / `build.json`: strict compiler identity, build command and result.
- `stdout.txt` / `stderr.txt`: unchanged admission and unittest output.
- `terminal.json` / `retirement.txt`: elapsed time, exit and process retirement.
- `oracle-binding.json`: CPU-only post-run Python/chess package identity.
- `summary.json`: independent counts, runtime binding, source-preservation audit
  and limitations.
- `manifest.json`: SHA256 of every payload byte in this directory, excluding
  the manifest itself. Source-file digests are separately in source-binding.

Compiled shared library:
`/tmp/chen-native-topology-20261003-_ooq6w3i/libchen_gpu_topology.so` in `cpp-rocm`,
SHA256 `6b9cda01670239318cc282bbe5702e11f2a26ac7de1bc396b52d87f3eda639ea`.
It was built with HIP7.2.4/Clang22; the executed Torch process loaded Torch's
ROCm7.0 HIP provider. Path, provider hash and runtime version are in raw stdout
and summary. This is a tested cross-version ABI combination, not a general
compatibility guarantee. The `/opt/amdgpu/share/libdrm/amdgpu.ids` diagnostic is
preserved in stderr; actual architecture admission and all tests passed.

Rebuild with `bash gpu/build.sh /tmp/libchen_gpu_topology.so`. The focused probe
requires Python-chess, Torch/ROCm and the compatible neural worktree bridge on
`PYTHONPATH`; set `CHEN_GPU_LIBRARY` and its actual `CHEN_GPU_LIBRARY_SHA256`.
Consult [the integration contract](../../../docs/GPU_TOPOLOGY.md) for canonical
indices, empty-square/own-blocker semantics and failure handling. Correctness on
these fixtures does not imply exhaustive coverage, faster inference, a deployed
sparse consumer or stronger play.
