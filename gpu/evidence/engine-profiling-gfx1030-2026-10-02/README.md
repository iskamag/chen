# Actual CPU/GPU engine profiles

`summary.json` reconstructs 21 bounded one-second UCI workloads, independent legal replay of every selected move and 935 reported PVs, 1792 CPU worker samples and three GPU worker traces. The flat CPU samples exclude main-thread initialization. GPU summaries exclude main-thread initialization and preserve overlapping host waits/device work as separate measurements.

`raw-records.tar.gz` retains all per-process UCI output, declarations, argv, retirement results, raw `.perf.data`, exported flat samples/reports, independent hardware counter runs, raw rocprof CSV and tooling/runtime bindings. `raw-member-index.json` binds every member. Full engine and GPU source closure plus scripts/docs are copied under `sources/`; exact executed binaries are already preserved in the referenced fusion archive.

Actual profile inputs are frozen: engine `a31be5d1...`, fused library `04447c45...`, Tapered, CPU5, UCI depth64/movetime1000; GPU batch32/capacity250000. Nine CPU cycle-sampling searches and three GPU trace searches are distinct from six plain timing/search observations and three counter searches. No equal-work CPU/GPU speedup, strength, rating, inclusive CPU callchain or additive API/device-duration claim is made.

No host package was installed. Perf/libpfm were extracted from official Artix packages under `/tmp`; the original stale cached perf URL returned HTTP404, then the current mirror package was fetched. Existing ROCm profiler libdw dependencies were reused. Clock/governor settings were not changed. The concurrent CPU0 strength match and shared cache/power limitations are recorded.
