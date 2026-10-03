# Runnable bounded GPU reference integration

PASS: strict HIP build;33 independently traversed cases with19 exhaustive small
references; realized device deadlines; coherent cancellation; four unchanged-
output rejections; eight actual Rust/UCI cases. Exit0 after1.88s, host1856 and
containerPGID9761 retired; no candidate process remained in the final scan.

The new timed ABI2 leaves legacy bounded ABI1 unchanged. Its persistent context
retains scratch/history/device packets and explicitly coherent mapped cancel
storage. Device wall-clock admission returned100000kHz; the executed native HIP
provider was /opt/rocm-7.2.4/lib/libamdhip64.so.7.2.70204, runtime70253211.
Detailed path/hash and realized timing are in native-results.json, not inferred
from the compiler version.

| Device allowance | Observed elapsed | Overshoot |
| --- | --- | --- |
|1ns|0.01016ms|0.010159ms|
|1ms|2.53568ms|1.53568ms|
|20ms|20.235ms|0.235ms|
|50ms|50.28592ms|0.28592ms|

The host-issued cancellation reached an already searching kernel (217 nodes)
and returned in1.35ms after request. UCI cancellation retained completed depth2
and returned1.11ms after stop. At1s from start position, completed Exact depth3
chose b1c3; depth4 aborted, retaining the previous score/PV. Wall time1001.65ms.
8034/10877 attempted nodes (73.86%) belonged to the unfinished depth. This is a
scalar correctness/reference floor, not an efficient GPU architecture or match.
Root's CPU0 match was concurrent; shared power/cache were uncontrolled.

Payloads retain the failed first compile source/error, source/binary bindings,
pre-launch declaration, independent cases, compiler identities, raw stdout/
stderr, structured native/UCI results, terminal/retirement and summary.
manifest.json hashes every payload byte, excluding itself. Git retains the
reviewable implementation; old source archives/binaries are not duplicated.

Native library ec56be9e3588eb224b3723f727ef3842d0fc1da4e44b7b57d7ebe87b879b9123
and Rust ccebec06e558432d1694fb4536fb41ee719df762e83c6399a7d51c8ad9efc3f2
remain staged in cpp-rocm:/tmp/chen-bounded-timed-v2-20261003-crnnqh9f/.
See [integration contract](../../../HIP_BOUNDED.md) for commands, boundaries,
telemetry and the next parallel work allocation direction.
