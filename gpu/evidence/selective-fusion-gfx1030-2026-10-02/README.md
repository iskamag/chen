# Classical selective fusion hardware evidence

The declared direct-process study passed on the actual gfx1030 device. See `report.json` for the bounded conclusion and `study/declaration.json` for the frozen benchmark protocol. `study/` retains every native command, exit/retirement record, stdout and stderr. `commands.json` records builds, runtime checks and the Docker invocation.

The benchmark uses the original verified baseline binary, whose required SHA-256 remains unchanged, and the freshly built fused binary. Both are preserved, together with the identical Rust engine/test executables and their full Rust source closure. Fresh baseline recompilations are separately retained; their differing hashes are explicitly not reproducible-build evidence. All compiler logs are empty and all three compilation commands exited successfully.

`runtime-library-maps.json` records a separate dynamic-load-only process, without a GPU API call. The device probe and native classical processes use this same isolated system-HIP library in the same container; no Torch, HSA architecture override or preloaded runtime was admitted. The device's API-reported compute-unit count is preserved as observed.

The manifest binds every payload. These measurements concern exact-work classical search latency on six positions, with three alternating repetitions per arm. They do not establish neural inference reliability or playing strength.
