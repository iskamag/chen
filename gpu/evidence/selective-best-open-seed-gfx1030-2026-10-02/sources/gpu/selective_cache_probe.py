#!/usr/bin/env python3
"""Pinned sparse-baseline/packed-cache exact-work comparison; direct local children."""
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import statistics

from selective_fusion_probe import TEST_NAMES, normalized, run, sha, write_json


ORDER = ("baseline", "candidate", "candidate", "baseline", "baseline", "candidate")
WORKLOADS = (
    {"name": "original-six", "depth": 4, "nodes": 2000, "batch": 8, "capacity": 250000},
    {"name": "large-arena-six", "depth": 64, "nodes": 50000, "batch": 32, "capacity": 250000},
)


def parse_rows(stdout, count, config):
    rows = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    if len(rows) != count:
        raise ValueError("row count changed")
    for row in rows:
        s = row["selective"]
        def finite(x):
            if isinstance(x, dict):
                return all(finite(v) for v in x.values())
            if isinstance(x, list):
                return all(finite(v) for v in x)
            return not isinstance(x, float) or math.isfinite(x)
        if not finite(row) or row["backend"] != "hip-selective" or row["eval"] != "tapered":
            raise ValueError("invalid or nonclassical benchmark metadata")
        if row["neural_execution"] is not None or row["evaluator_artifact_sha256"] is not None:
            raise ValueError("unexpected neural admission")
        if (row["requested_depth"], row["requested_nodes"], row["max_positions"], s["batch_leaves"]) != (
                config["depth"], config["nodes"], config["capacity"], config["batch"]):
            raise ValueError("requested workload changed")
        if (s["abi_version"], s["struct_size"], s["candidate_limit"]) != (1, 96, 1024):
            raise ValueError("ABI/scheduler changed")
        if row["nodes"] != s["classified_positions"] or row["nodes"] > config["nodes"]:
            raise ValueError("node accounting mismatch/overshoot")
        if not isinstance(row["elapsed_ns"], int) or row["elapsed_ns"] <= 0:
            raise ValueError("missing positive search time")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "tests", "baseline", "candidate", "suite", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("engine", "tests", "baseline", "candidate"):
        parser.add_argument("--" + name + "-sha256", required=True)
    parser.add_argument("--cpu", type=int, default=5)
    args = parser.parse_args()
    paths = {name: getattr(args, name).resolve(strict=True)
             for name in ("engine", "tests", "baseline", "candidate", "suite")}
    pins = {name: getattr(args, name + "_sha256") for name in ("engine", "tests", "baseline", "candidate")}
    for name, wanted in pins.items():
        if sha(paths[name]) != wanted:
            parser.error(name + " fingerprint mismatch")
    if pins["baseline"] == pins["candidate"]:
        parser.error("artifacts must differ")
    for key in ("HSA_OVERRIDE_GFX_VERSION", "CHEN_NEURAL_MODEL", "CHEN_NEURAL_LIBRARY", "LD_PRELOAD"):
        if os.environ.get(key):
            parser.error("unexpected inherited " + key)
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU outside permitted mask")
    args.output.mkdir(parents=True, exist_ok=False)
    count = sum(bool(line.strip()) and not line.lstrip().startswith("#")
                for line in paths["suite"].read_text().splitlines())
    declaration = {"kind": "classical_packed_cache_baseline_candidate_exact_work_comparison",
                   "variant": "packed_best_open_child_cache_v1",
                   "source_sha256": sha(__file__), "helper_sha256": sha(Path(__file__).with_name("selective_fusion_probe.py")),
                   "artifacts": {name: {"path": str(p), "sha256": sha(p)} for name, p in paths.items()},
                   "tests": list(TEST_NAMES), "workloads": WORKLOADS, "order": ORDER,
                   "rows_per_run": count, "cpu_affinity": [args.cpu], "timeout_per_child_seconds": 30,
                   "scope": "Per-step portable oracle tests and alternating cold-process identical-work latency; no neural or strength claim"}
    write_json(args.output / "declaration.json", declaration)
    results = []
    try:
        def verify():
            for name, p in paths.items():
                if sha(p) != declaration["artifacts"][name]["sha256"]:
                    raise ValueError("artifact changed after declaration: " + name)
        for arm in ("baseline", "candidate"):
            verify()
            env = dict(os.environ, CHEN_GPU_LIBRARY=str(paths[arm]))
            text, _ = run(["taskset", "-c", str(args.cpu), str(paths["tests"]), "--ignored", "--test-threads=1", "--nocapture"],
                          env, args.output / ("tests-" + arm), 30)
            if not all("test " + name + " ... ok" in text for name in TEST_NAMES):
                raise ValueError("portable oracle tests did not both pass")
        for config in WORKLOADS:
            reference = None
            timing = []
            for index, arm in enumerate(ORDER):
                verify()
                env = dict(os.environ, CHEN_GPU_LIBRARY=str(paths[arm]))
                argv = ["taskset", "-c", str(args.cpu), str(paths["engine"]), "bench", "--backend", "hip-selective",
                        "--eval", "tapered", "--depth", str(config["depth"]), "--nodes", str(config["nodes"]),
                        "--selective-batch", str(config["batch"]), "--max-positions", str(config["capacity"]),
                        "--suite", str(paths["suite"])]
                text, rec = run(argv, env, args.output / (config["name"] + "-" + str(index + 1) + "-" + arm), 30)
                rows = parse_rows(text, count, config)
                comparable = [normalized(x) for x in rows]
                if reference is None:
                    reference = comparable
                if reference != comparable:
                    raise ValueError("score/PV/counters/admission parity failed; no timing conclusion")
                timing.append({"arm": arm, "run": index + 1,
                               "search_sum_ms": sum(x["elapsed_ns"] for x in rows) / 1e6,
                               "wall_seconds": rec["wall_seconds"], "exact_work_parity": True})
            medians = {arm: statistics.median(x["search_sum_ms"] for x in timing if x["arm"] == arm)
                       for arm in ("baseline", "candidate")}
            results.append({"workload": config, "runs": timing, "median_search_sum_ms": medians,
                            "baseline_over_candidate_ratio": medians["baseline"] / medians["candidate"]})
        verify()
        write_json(args.output / "result.json", {"status": "passed", "workloads": results, "scope": declaration["scope"]})
    except Exception as error:
        write_json(args.output / "result.json", {"status": "failed_or_partial", "error": str(error),
                                               "completed_workloads": results, "timing_conclusion_admitted": False})
        raise


if __name__ == "__main__":
    main()
