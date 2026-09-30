# Eager dispatch reliability study on gfx1030

This study compares the original eager inference plugin with `AMD_DIRECT_DISPATCH=1` and `0`. The production Rust binary, plugin, rules library, trained archives, and 30-second UCI watchdog are fixed. The comparison is exploratory; the observed trials do not establish a failure rate, playing strength, or a permanent workaround.

HIP 7.0 documents `0` as disabling direct kernel dispatch and `1` as enabling it on Linux. ROCm's command implementation has distinct direct-submission and host-queue branches, including event notification. These sources define the control; they do not identify the cause of the earlier stalls. [HIP environment variables](https://rocm.docs.amd.com/projects/HIP/en/docs-7.0.0/reference/env_variables.html), [ROCm 7.0 command implementation](https://raw.githubusercontent.com/ROCm/clr/rocm-7.0.0/rocclr/platform/command.cpp).

The replacement schedule was declared before its first launch: three matched pairs in orders `[1,0]`, `[0,1]`, and `[1,0]`. Each trial starts a fresh parent Python process with the flag already set, then inherits it into benchmark and UCI children. A full prelude checks 18 model/position minimax comparisons, option ordering and configuration recovery, node/level/stop behavior, and two 20-ply protocol games. Timeout diagnostics attach twice before retiring the failed child. The run was stopped after an external infrastructure interruption; it was not resumed.

The runner records requested configuration and selected child `/proc/environ` fields after `readyok`. Missing observations are explicitly unavailable after at most three attempts and 10 milliseconds of retry delays. Environment observation does not read HIP's internal runtime flag; that flag is not exported from the installed runtime. Benchmark environments are recorded at construction, and their output metadata verifies `hip-resident`, neural evaluation, eager execution, and the exact model archive hash.

An initial six-launch batch is excluded. Its instrumentation asserted on an empty immediate `/proc/environ` observation inside the `Popen` constructor, aborting the intended checks and bypassing ordinary child ownership cleanup. Those assertions are harness failures. The invalid logs and original runner are retained separately; no reliability comparison uses them. A CPU fixture verified corrected observations after a child's `readyok` for both flag values before the replacement schedule began.

The study ended partially when the container stopped externally at **2026-09-30 21:17:28 UTC**, exit code 137 with Docker `OOMKilled=false`. Host `/tmp` disappeared and the orchestration session became unavailable. This event is not classified as an engine failure. The container's diagnostic files were copied before root restarted the same container; no study continuation followed.

| Trial | Requested dispatch | Surviving replacement evidence |
| --- | --- | --- |
| 1 | 1 | Two timeout-triggered debugger captures: HSA code-object freeze/blit wait during the Torch score-range scalar comparison. |
| 2 | 0 | Two captures: BLAS workspace installation/free, HIP `SyncAllStreams`, and event completion wait. |
| 3 | 0 | Two captures at the same BLAS workspace/free wait boundary. |
| 4 | 1 | Two captures: HSA code-object freeze/blit wait during a Torch mean reduction. |
| 5 | 1 | Replacement startup debugger control exists; externally interrupted without a preserved terminal outcome. |
| 6 | 0 | No replacement startup/result evidence. Its directory contains only a control from the excluded earlier instrumentation batch. |

The complete replacement stdout, stderr, and host end-event journal were under host `/tmp` and were lost. Earlier live tool inspections observed 18 comparison records and first-game waits, but the full numerical results and ready-time environment records cannot now be independently rechecked. They are **not counted as archived, replayable oracle verification**. The actual surviving debugger files demonstrate GPU/HIP/Torch runtime waits; they do not supply the missing score metadata or establish CPU-substitution checks for this interrupted comparison.

The planned six-run comparison was not completed. A successful earlier dispatch-zero control did not establish a permanent workaround, and this partial archive cannot support a failure-rate or speed estimate. Device history and caches were not reset or randomized. Earlier passing/failing eager runs, the rejected BLAS-priming candidate, and the single dispatch-zero control remain separate experiments with their own fingerprints.

[Partial results and limitations](../gpu/evidence/dispatch-study-partial-results-gfx1030-2026-10-01.json), [recovery manifest and file hashes](../gpu/evidence/dispatch-study-recovery-gfx1030-2026-10-01.json), [replacement declaration](../gpu/evidence/dispatch-study-replacement-config-gfx1030-2026-10-01.json), [excluded instrumentation attempts](../gpu/evidence/dispatch-study-invalid-launches-gfx1030-2026-10-01.json).
