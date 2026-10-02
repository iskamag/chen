# Closed corrected semantic GPU study

This is one fresh, separately reviewed bounded attempt after the earlier
checkpoint-bundle size failure. The earlier failed attempt remains closed.
This run completed all 8 anchor and 32 transition phases (9,216 updates) on
actual gfx1030 ROCm, froze all 40 checkpoint files under a 33KB manifest,
then evaluated the declared validation split. The unchanged joint gate selected
NONE. TEST remained closed; there is no model promotion or Elo claim.

The owned watchdog recorded exit0 after 144.173732 seconds; runner elapsed was
142.158113 seconds and summed training phase time was 107.045102 seconds.
Watchdog PID5099, training PID5105 and PGID5105 were independently observed
absent after completion. Stderr contains only the pinned missing amdgpu.ids
metadata warning. No retry or resume occurred.

`runtime-payloads.tar.gz` preserves exact isolated sources, all three immutable
packs, old native baseline sources/artifacts, path-rebased witness, declarations,
seals, logs and all individual states. `runtime-members.json` pins each regular
member. `manifest.json` pins the review payloads. Baseline core/source is the
historical f7e967 witness closure, before later main SWAR adoption.

`validation-results.json` contains every baseline and all 16 arm x 2 seed metrics
and diagnostics. `independent-terminal-replay.json` independently checks all40
file hashes, finite typed states, architecture keys and state fingerprints,
complete updates, frozen metric bindings and strict all-seed/all-baseline gate.
It reads the already published metrics; it does not recalculate quality from
corpus or run any GPU work. The first cold audit mistakenly required empty
stderr; its failed log is retained, and the final audit explicitly verifies the
exact harmless metadata warning.

Some anchors improve scalar error and some transitions improve root regret;
the per-seed optimal agreement condition still prevents promotion. Conditioned
and transport arms have equal root ranking metrics in this run, while their
latent diagnostics differ. This is an observation, not evidence that typed
routing is generally useless. Future research needs separately declared exact
child inference and transformer controls; current validation is development
evidence and must not become an untouched final evaluation claim.
