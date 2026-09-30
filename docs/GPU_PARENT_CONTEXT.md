# Parent-context reliability control

This experiment is prepared in source and has not run on the GPU. It asks whether the unchanged eager engine behaves differently when its controller has no Torch/HIP context versus a live context that becomes idle during the UCI games. The earlier interrupted dispatch comparison is documented separately in [GPU_DISPATCH_STUDY.md](GPU_DISPATCH_STUDY.md).

A single GPU producer computes frozen oracle answers for the three unchanged trained archives and six benchmark positions at depths one and two. It records CP scores, optimal tie sets, visited/evaluated counts, exact archive and feature-source hashes, and its PID/start identity. The producer must exit before any consumer starts. Its cache is used only to compare engine results: the engine still runs its original GPU model and resident rules.

Six fresh consumer processes are declared in three pairs: `CPU,HIP`, `HIP,CPU`, `CPU,HIP`. All use `AMD_DIRECT_DISPATCH=1`, eager execution, the same pinned production artifacts, a 30-second UCI watchdog, and the same engine prelude. The source copies five helpers and the complete option/limits/game protocol from reference SHA `d8c580b796e694ce5f80eb91c7d93d0b4706bc15797a73eaabd61c216d7ada9c`; their separate fingerprints are embedded in the new probe.

The CPU consumer imports standard-library modules and python-chess. It verifies that neither Torch/neural modules nor HIP/Torch shared libraries are present in its own process, before and after the prelude. Reading and hashing the pinned HIP library does not load it. It compares benchmark output with the cache. The HIP consumer conditionally imports Torch, repeats the original per-model GPU oracle work at the corresponding prelude points, then keeps its context live but idle throughout the UCI option/limit/game checks. Both verify exact engine scores, optimal moves, node counts, legal PVs, backend/evaluator/execution metadata, and the model SHA. Child environment observation occurs after `readyok`; unavailable observations are explicitly recorded rather than treated as launch failures.

The difference includes GPU warming/history and the time spent computing oracle answers, as well as context presence. Device history is not reset or randomized. These controls cannot distinguish those effects by themselves, and six finite trials cannot establish a failure rate or production workaround. No playing-strength conclusion follows from the two 20-ply protocol games.

Each command writes a local JSONL journal with `flush` and `fsync` at every phase and also flushes the same records to stdout. The host `mirror` command uses nonblocking pipe reads and copies stdout and stderr into new binary files under repository `results/`, flushing and syncing every received chunk. Its declared wall deadline is 600 seconds, with at most one second of pipe draining after the transport exits. A wall timeout, inherited open pipe, or output-write failure triggers bounded TERM/KILL of the local process group (three seconds per wait). It records remote retirement as unknown, requires ending the study, and never replays a command. Filesystem operations must remain responsive for these process/pipe bounds to hold. An ordinary nonzero consumer exit is recorded separately; its durable ownership journal must be inspected before deciding whether the declared schedule can continue.

Timeout snapshots include raw debugger output and selected proc data directly in the journal, rather than referring only to host `/tmp` files. The consumer retires its own UCI children on failure; the host transport cannot guarantee retirement inside a stopped or unreachable container. An external interruption ends the study partially, with no automatic restart. Remaining pipe bytes may be discarded when retiring a failed transport, and that limitation is recorded.

Preparation is CPU-only:

```bash
results/qa-venv/bin/python gpu/parent_context_probe.py declare \
  --declaration results/parent-context-new/declaration.json \
  --journal results/parent-context-new/declaration-journal.jsonl
```

After the GPU slot is explicitly released, stage the frozen probe in `/tmp/chen-parent-context/` with the existing neural package available on `PYTHONPATH`. Use a separate producer process with `AMD_DIRECT_DISPATCH=1`; retain its completed cache and emitted SHA in repository results before beginning the six consumers. Each `consume` invocation must supply the declared trial/parent, the same cache SHA, and a fresh local journal. Wrap every remote command with host `mirror`, using repository result paths for both output streams. The declaration binds the probe SHA, so source changes require a new declaration before any trial.

The remote producer command is:

```bash
docker exec -e AMD_DIRECT_DISPATCH=1 -e CHEN_NEURAL_EXECUTION=eager \
  -e PYTHONPATH=/tmp/chen-native-plugin cpp-rocm \
  python /tmp/chen-parent-context/parent_context_probe.py produce \
  --oracle /tmp/chen-parent-context/oracle.json \
  --journal /tmp/chen-parent-context/producer.jsonl
```

For example, the first consumer command uses `consume --parent cpu --trial 1 --declaration /tmp/chen-parent-context/declaration.json --oracle /tmp/chen-parent-context/oracle.json --oracle-sha CACHE_SHA --journal /tmp/chen-parent-context/trial-1.jsonl`. Pass that entire Docker command after `--exec` to the host probe's `mirror` command, specifying new `--journal`, `--mirror-stdout`, and `--mirror-stderr` paths under repository `results/`. Use the declared parent order for trials two through six. Do not reuse output paths or resume a trial after an infrastructure interruption.

The producer/consumer paths remain unverified on hardware. CPU checks have verified the import guard, exact helper fingerprints, durable stdout/stderr mirror including a partial final line, bounded transport timeout, inherited-pipe retirement, nonzero-exit reporting, two timeout snapshots, owned-child retirement, and an empty owner registry after a deliberate protocol timeout. The timeout fixture uses a short test-only deadline; the declared engine watchdog remains 30 seconds. The [prepared declaration](../gpu/evidence/parent-context-config-gfx1030-2026-10-01.json) and [CPU verification archive](../gpu/evidence/parent-context-cpu-preparation-gfx1030-2026-10-01/verification.json) are separate from any future hardware results.
