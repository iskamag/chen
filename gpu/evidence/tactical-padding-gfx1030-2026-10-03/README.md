# Actual GPU padding diagnostic

Revision1 PASS: six same-checkpoint shapes, eager and captured full/tight arms,
all prediction/capture parity checks. Initial attempt failed before timing on a
non-tensor metadata comparison; failed source/declaration/stderr are preserved.
Both owned exec handles are terminal; no observation timeout caused a restart.

`declaration.json` fixes shapes, source pins, library, sampling and execution scope.
`source.py` is the executed revision1 probe. `source-closure-binding.json` binds
the compact source archive and overlay already stored in the native graph packet;
they are not duplicated. `runtime.json` records actual provider/device and native
manifest plus exact row slot counts. `results.json`, stdout/stderr are unchanged
raw outputs. `terminal.json` records terminal handles/body duration; `retirement.txt`
is the subsequent container process inventory. `manifest.json` hashes all payloads.

The native binary is the original177b6008...7f80d graph producer. Actual gfx1030,
Torch2.10.0+ROCm7.0 provider; no override, no concurrent GPU work. Shapes use
untrained frozen FP32 weights. Three small samples and shared-clock/power effects
limit timing scope. No labels, optimizer, architecture selection or match occurs.

See [the interpretation and table](../../../docs/TACTICAL_PADDING.md). Tight
capacity is a cold intervention; export/copy/setup/capture costs are excluded.
The native emitter's current admission and default allocation remain unchanged.

Replay against a compatible native library and source closure:

```sh
PYTHONDONTWRITEBYTECODE=1 timeout --kill-after=5s 90s \
  python -m gpu.profile_tactical_padding \
  --library /path/to/libchen_gpu.so --library-sha256 ACTUAL_SHA \
  --output /path/to/fresh/results
```
