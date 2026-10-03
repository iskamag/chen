# Actual gfx1030 tactical graph packet

PASS: four initial tests, 45 complete CPU-reference graph rows and all 14
relations; one later four-state ABI1 composition test. The resident processor
forward/backward check uses width8/steps2, no optimizer. These are mechanics and
integration checks, not training, latency, architecture-selection or Elo evidence.

- `declaration.json` is the unchanged initial pre-run declaration. Its original
  source-binding hash refers to `original-focused-source-binding.json`, the
  broader temporary module inventory. The compact local execution closure is
  independently retained and pinned here.
- `initial-sources.tar.gz` / `initial-source-binding.json` preserve the exact
  initial local compiler/Python dependency closure (35 files). This includes the
  initial probe/producer versions corresponding to initial stdout/stderr.
- Apply `composition-overlay.tar.gz` over that tree for the current producer and
  five-method probe. `composition-source-binding.json` binds these two changed
  Python files; numerical native source and compiled library did not change.
- `stdout.txt` / `stderr.txt` are the initial unchanged outputs; `composition-*`
  are the later ABI composition outputs, including runtime/device admission.
- `runtime.json`, `compiler.txt`, `external-dependencies.json` bind compiler,
  native binary and actually loaded Torch HIP provider. External Torch and
  python-chess packages are not redistributed in the source archive.
- `terminal.json` binds the two owned terminal exec handles, exit codes,
  instrumented unittest body durations and exact tested scope. Start/end wall
  timestamps were not instrumented. `retirement.txt` is the subsequent container
  process inventory: only init and inspection, no candidate worker.
- `summary.json` lists checked boundaries and limitations. `manifest.json`
  hashes all packet payload files, excluding itself.

Native library in `cpp-rocm`:
`/tmp/chen-native-tactical-20261003-dj0vjgka/libchen_gpu_tactical.so`, SHA256
`177b600836cf4bef2944ffdb31e9ea2202751b3f51eac524b3b8422a6637f80d`.
HIP7.2.4 compiler; actual provider Torch ROCm7.0, actual architecture gfx1030
with no override. The preserved amdgpu.ids diagnostic did not prevent admission
or checks. No shared library is stored in Git.

Rebuild a new library with `bash gpu/build.sh /tmp/libchen_gpu_tactical.so`.
Run the probe with the main GPU and neural source trees both on PYTHONPATH,
Python-chess1.11.2, Torch/ROCm, `CHEN_GPU_LIBRARY` and its actual
`CHEN_GPU_LIBRARY_SHA256`:

```sh
PYTHONDONTWRITEBYTECODE=1 timeout --kill-after=5s 30s \
  python -m unittest gpu.test_native_tactical -v
```

The archived current probe can be reproduced by extracting the initial archive
and composition overlay into one directory and using it as PYTHONPATH. A rebuild
is a new binary; pin its own hash rather than reusing the measured hash.
