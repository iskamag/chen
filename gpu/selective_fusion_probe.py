#!/usr/bin/env python3
"""Future direct-process classical HIP comparison; this file itself is CPU-only.

Run inside an authorized GPU environment with explicit native artifact paths.
It launches no Docker/SSH wrapper and stops on any failure, without replay.
"""
import argparse
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

TEST_NAMES = (
    "actual_selective_stepwise_arena_parity_and_complete_horizons",
    "actual_selective_atomic_limits_history_invalidation_and_thread_transfer",
)
ORDER = ("baseline", "fused", "fused", "baseline", "baseline", "fused")
BASELINE_SHA256 = "0cf4171f1e056e281f2cca318b6b67a7616b9a7dca688001542eed8b4da2963a"


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    with Path(path).open("w") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())


def run(argv, env, stem, timeout):
    """Bounded local leader/pipe retirement; escaped descendants are not owned."""
    started = time.monotonic()
    p = subprocess.Popen(argv, env=env, stdout=subprocess.PIPE,
                         stderr=subprocess.PIPE, start_new_session=True)
    record = {"argv": argv, "pid": p.pid, "requested_library": env["CHEN_GPU_LIBRARY"],
              "timeout_seconds": timeout, "phase": "started"}
    write_json(str(stem) + ".json", record)
    timed_out = False
    cleanup_error = None
    try:
        stdout, stderr = p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as initial:
        timed_out = True
        try:
            os.killpg(p.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            stdout, stderr = p.communicate(timeout=5)
        except subprocess.TimeoutExpired as final:
            stdout = final.output or initial.output or b""
            stderr = final.stderr or initial.stderr or b""
            cleanup_error = "pipe drain deadline exceeded; descendant retirement unknown"
            p.stdout.close()
            p.stderr.close()
            try:
                p.wait(timeout=1)
            except subprocess.TimeoutExpired:
                cleanup_error += "; leader retirement unknown"
    try:
        os.killpg(p.pid, 0)
        group_absent = False
    except ProcessLookupError:
        group_absent = True
    Path(str(stem) + ".stdout").write_bytes(stdout)
    Path(str(stem) + ".stderr").write_bytes(stderr)
    record.update(phase="finished", exit_code=p.poll(), timed_out=timed_out,
                  wall_seconds=time.monotonic() - started, cleanup_error=cleanup_error,
                  owned_leader_retired=p.poll() is not None, process_group_absent=group_absent)
    write_json(str(stem) + ".json", record)
    if timed_out or p.returncode != 0 or cleanup_error or not group_absent:
        raise RuntimeError(f"native run failed/retirement not established: {stem}; study ends")
    return stdout.decode("utf-8"), record


def normalized(row):
    """Compare every field except explicit timing/rate telemetry."""
    result = copy.deepcopy(row)
    for name in ("elapsed_ns", "nps", "kernel_ms"):
        result.pop(name, None)
    result["selective"].pop("kernel_ms", None)
    return result


def rows(stdout, expected_rows):
    data = [json.loads(line) for line in stdout.splitlines() if line.startswith("{")]
    if len(data) != expected_rows:
        raise ValueError("benchmark output row count changed")
    for row in data:
        def finite(value):
            if isinstance(value, dict):
                return all(finite(v) for v in value.values())
            if isinstance(value, list):
                return all(finite(v) for v in value)
            return not isinstance(value, float) or math.isfinite(value)
        s = row.get("selective")
        if not finite(row) or row.get("backend") != "hip-selective" or row.get("eval") != "tapered":
            raise ValueError("invalid metadata or unexpected backend/evaluator")
        if not s or s.get("abi_version") != 1 or s.get("struct_size") != 96:
            raise ValueError("selective ABI metadata missing")
        if (row.get("requested_depth"), row.get("requested_nodes"), row.get("max_positions"),
                s.get("batch_leaves"), s.get("candidate_limit")) != (4, 2000, 250000, 8, 1024):
            raise ValueError("requested search parameters changed")
        if row.get("neural_execution") is not None or row.get("evaluator_artifact_sha256") is not None:
            raise ValueError("classical benchmark unexpectedly admitted neural inference")
        if not isinstance(row.get("elapsed_ns"), int) or row["elapsed_ns"] <= 0:
            raise ValueError("elapsed search timing missing")
        if row["nodes"] != s["classified_positions"] or row["nodes"] > 2000:
            raise ValueError("node accounting mismatch/overshoot")
    return data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("engine", "tests", "baseline", "fused", "suite", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--prepare-only", action="store_true", help="declare/fingerprint without launching")
    parser.add_argument("--timeout", type=float, default=30)
    args = parser.parse_args()
    if not 0 < args.timeout <= 120:
        parser.error("timeout must be in (0,120]")
    paths = {n: getattr(args, n).resolve(strict=True) for n in
             ("engine", "tests", "baseline", "fused", "suite")}
    if paths["baseline"] == paths["fused"] or sha(paths["baseline"]) == sha(paths["fused"]):
        parser.error("baseline and fused artifacts must be distinct")
    if sha(paths["baseline"]) != BASELINE_SHA256:
        parser.error("baseline library does not match the verified frozen artifact")
    for name in ("HSA_OVERRIDE_GFX_VERSION", "CHEN_NEURAL_MODEL", "LD_PRELOAD"):
        if os.environ.get(name):
            parser.error(f"unexpected inherited {name}; refuse ambiguous experiment")
    args.output.mkdir(parents=True, exist_ok=False)
    expected_rows = sum(1 for line in paths["suite"].read_text().splitlines()
                        if line.strip() and not line.lstrip().startswith("#"))
    declaration = {"kind": "classical_selective_baseline_fused_direct_process_comparison",
                   "source_sha256": sha(__file__),
                   "artifacts": {n: {"path": str(p), "sha256": sha(p)} for n, p in paths.items()},
                   "search": {"depth": 4, "nodes": 2000, "max_positions": 250000,
                              "batch_leaves": 8, "candidate_limit": 1024, "eval": "tapered"},
                   "tests": list(TEST_NAMES), "benchmark_order": list(ORDER),
                   "rows_per_run": expected_rows, "timeout_seconds": args.timeout,
                   "scope": "Per-step Rust oracle tests and identical-work cold-process search timing; "
                            "not warm throughput, strength, or external neural validation",
                   "transport": "Direct local child process groups; no remote wrapper; "
                                "timeout or uncertain retirement ends study without replay"}
    write_json(args.output / "declaration.json", declaration)
    if args.prepare_only:
        print("Prepared only; no child or GPU execution")
        return
    results = []
    def verify_artifacts():
        for name, path in paths.items():
            if sha(path) != declaration["artifacts"][name]["sha256"]:
                raise ValueError(f"artifact changed after declaration: {name}")
    try:
        for mode in ("baseline", "fused"):
            verify_artifacts()
            env = dict(os.environ, CHEN_GPU_LIBRARY=str(paths[mode]))
            stdout, record = run([str(paths["tests"]), "--ignored", "--test-threads=1", "--nocapture"],
                                 env, args.output / ("tests-" + mode), args.timeout)
            if not all("test " + name + " ... ok" in stdout for name in TEST_NAMES):
                raise ValueError("same Rust per-step oracle tests did not both pass")
        reference = None
        for i, mode in enumerate(ORDER):
            verify_artifacts()
            env = dict(os.environ, CHEN_GPU_LIBRARY=str(paths[mode]))
            argv = [str(paths["engine"]), "bench", "--backend", "hip-selective", "--eval", "tapered",
                    "--depth", "4", "--nodes", "2000", "--selective-batch", "8",
                    "--max-positions", "250000", "--suite", str(paths["suite"])]
            stdout, record = run(argv, env, args.output / f"bench-{i + 1}-{mode}", args.timeout)
            data = rows(stdout, expected_rows)
            comparable = [normalized(r) for r in data]
            if reference is None:
                reference = comparable
            if comparable != reference:
                raise ValueError("score/PV/admission/counter parity failed; timings not admitted")
            results.append({"mode": mode, "run": i + 1, "same_work_parity": True,
                            "search_sum_ms": sum(r["elapsed_ns"] for r in data) / 1e6,
                            "kernel_span_sum_ms": sum(r["kernel_ms"] for r in data),
                            "wall_seconds": record["wall_seconds"]})
        verify_artifacts()
        medians = {m: statistics.median(r["search_sum_ms"] for r in results if r["mode"] == m)
                   for m in ("baseline", "fused")}
        write_json(args.output / "result.json", {"status": "passed", "runs": results,
                   "median_search_sum_ms": medians,
                   "baseline_over_fused_ratio": medians["baseline"] / medians["fused"],
                   "scope": declaration["scope"]})
    except Exception as error:
        write_json(args.output / "result.json", {"status": "failed_or_partial", "error": str(error),
                   "completed_benchmark_runs": results, "timing_conclusion_admitted": False})
        raise


if __name__ == "__main__":
    main()
