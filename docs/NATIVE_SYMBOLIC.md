# Native symbolic observations

`neural.native_symbolic.NativeSymbolicProducer` exports exact current-state
features and symbolic channels from the HIP rules arena into owned Torch VRAM.
It requires native symbolic ABI1, implemented in main-worktree commit `d2a54be`.
The [shared contract and raw GPU test records](../../chen/docs/GPU_SYMBOLIC_RESIDUAL.md)
define history, orientation, ownership and numerical scope.

`CompactSymbolicState` carries normalized `[N,64,12]` channels, schema-1 board
features, raw history and identity sidecars. Both `SymbolicResidualAnchor` and
the separate multihead transformer consume it through their existing residual
projections. The dense `SymbolicState` remains an oracle/control representation.
Native export constructs no dense attack graph and never reconstructs boards
on the host. Library fingerprints, runtime identity, source epoch and output
allocation ownership are admitted before use.

Six actual gfx1030 tests pass, checking 940 independent oracle rows and consumer
gradients. `neural.tests.test_native_symbolic` requires an actual ROCm GPU and
`CHEN_GPU_LIBRARY` plus `CHEN_GPU_LIBRARY_SHA256`; a requested hardware run fails
if the GPU is absent. Independent oracle downloads occur outside the audited
producer/model hot path. The native build instructions are `gpu/build.sh` in
the main worktree. Existing HIP–Torch runtime binding applies.

This is a synchronous snapshot bridge with host admission and completion
barriers. It establishes mechanics, not learning, strength or an inference
speedup. Buffer reuse and an asynchronous native graph phase remain the next
execution experiment; do not capture the synchronous bridge as a hot GPU phase.
