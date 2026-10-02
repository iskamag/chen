#!/usr/bin/env python3
"""Read-only reconstruction of best-open-seed exact-work, native and trace evidence.

Requires python-chess for independent PV legality. Extract raw-records.tar.gz
into an empty directory, then pass that directory. This makes no GPU calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from analyze_engine_profile import gpu
from selective_fusion_probe import normalized
import chess


def load(path):
    return json.loads(path.read_text())


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_pv(fen, moves):
    board = chess.Board(fen)
    for text in moves:
        move = chess.Move.from_uci(text)
        assert move in board.legal_moves, (fen, text)
        board.push(move)


def analyze(root):
    declaration = load(root / "exact-work/declaration.json")
    exact = []
    rows_checked = 0
    for config in declaration["workloads"]:
        reference, timings = None, {"baseline": [], "candidate": []}
        for index, arm in enumerate(declaration["order"], 1):
            path = root / "exact-work" / (config["name"] + "-" + str(index) + "-" + arm + ".stdout")
            rows = [json.loads(line) for line in path.read_text().splitlines() if line.startswith("{")]
            assert len(rows) == declaration["rows_per_run"]
            comparable = [normalized(x) for x in rows]
            if reference is None:
                reference = comparable
            assert reference == comparable
            timings[arm].append(sum(x["elapsed_ns"] for x in rows) / 1e6)
            rows_checked += len(rows)
        medians = {arm: statistics.median(values) for arm, values in timings.items()}
        exact.append({"workload": config, "median_search_sum_ms": medians,
                      "baseline_over_candidate_ratio": medians["baseline"] / medians["candidate"],
                      "reduction_percent": 100 * (1 - medians["candidate"] / medians["baseline"])})
    native = load(root / "deep/result.json")
    assert native["status"] == "passed" and len(native["cases"]) == 33
    native_steps = native_pvs = native_moves = 0
    for case in native["cases"]:
        assert case["exact_per_step_parity"]
        for snapshot in case["snapshots"]:
            assert snapshot["classified"] <= case["absolute_limit"] and snapshot["arena"] <= 6000
            assert snapshot["maximum"] <= case["max_ply"]
            if "pv" in snapshot:
                audit_pv(case["fen"], snapshot["pv"])
                native_pvs += 1
                native_moves += len(snapshot["pv"])
            native_steps += 1
        if case["repeated_root_history"]:
            final = case["snapshots"][-1]
            assert final["root_flags"] & 2 and final["score"] == 0 and final["stop"] == 1
    bounds = [json.loads(line) for line in (root / "seed-bounds.stdout").read_text().splitlines()]
    assert len(bounds) == 14 and all(x["status"] == "passed" and x["guard_words"] == 8 for x in bounds)
    assert any(x["depth"] == 64 and x["batch"] == 256 for x in bounds)
    observations, uci_pvs = {}, 0
    for mode in ("uci", "trace"):
        record = load(root / mode / "result.json")
        assert record["status"] == "passed"
        arms = {"baseline": {}, "candidate": {}}
        for run in record["runs"]:
            assert run["exit_code"] == 0 and run["process_group_absent"] and run["movetime_ms"] == 1000
            audit_pv(run["fen"], [run["bestmove"]])
            info = run["last_info"]
            fields = {key: int(re.search(r"(?:^| )" + key + r" (\d+)(?: |$)", info).group(1))
                      for key in ("nodes", "time", "depth", "seldepth")}
            arms[run["arm"]].setdefault(run["name"], []).append(fields)
        observations[mode] = arms
        # Executed run records retain their original absolute argv paths. The
        # archive is relocatable; inspect all matching local worker stdout files.
        for stdout in (root / mode).rglob("1-*.stdout"):
            name = stdout.stem.removeprefix("1-")
            fen = next(x["fen"] for x in record["runs"] if x["name"] == name)
            for line in stdout.read_text().splitlines():
                if line.startswith("info depth ") and " pv " in line:
                    audit_pv(fen, line.split(" pv ", 1)[1].split())
                    uci_pvs += 1
    assert uci_pvs == sum(load(root / mode / "result.json")["legal_reported_pvs"] for mode in ("uci", "trace"))
    traces = {arm: gpu(root / "trace" / arm) for arm in ("baseline", "candidate")}
    for arm, workloads in traces.items():
        for work in workloads:
            assert work["stream_fence_calls"] == 3 * work["published_steps"] + 3
            assert work["worker_allocation_calls"] == 0
    fixture = [json.loads(line) for line in (root / "seed-trace.stdout").read_text().splitlines()]
    assert len(fixture) == 48 and all(x["status"] == "passed" for x in fixture)
    isa = load(root / "isa/extraction-report.json")
    assert isa["status"] == "passed"
    resources = {}
    for artifact in isa["records"]:
        assert digest(root / "isa" / artifact["elf"]) == artifact["sha256"]
        if artifact["kernels"]:
            resources[artifact["arm"]] = artifact["kernels"]
    assert set(resources) == {"baseline", "candidate"}
    per_position = []
    for config in declaration["workloads"]:
        by_arm = {}
        for arm in ("baseline", "candidate"):
            by_arm[arm] = [[json.loads(line) for line in (root / "exact-work" /
                (config["name"] + "-" + str(index) + "-" + arm + ".stdout")).read_text().splitlines()
                if line.startswith("{")] for index, scheduled in enumerate(declaration["order"], 1) if scheduled == arm]
        for position in range(declaration["rows_per_run"]):
            times = {arm: statistics.median(run[position]["elapsed_ns"] / 1e6 for run in runs)
                for arm, runs in by_arm.items()}
            per_position.append({"workload": config["name"], "position": position,
                "fen": by_arm["baseline"][0][position]["fen"], "median_ms": times,
                "reduction_percent": 100 * (1 - times["candidate"] / times["baseline"])})
    trajectories = []
    for baseline in traces["baseline"]:
        name = baseline["name"]
        candidate = next(x for x in traces["candidate"] if x["name"] == name)
        def infos(arm):
            stdout = root / "trace" / arm / "gpu-trace" / ("1-" + name + ".stdout")
            return [re.sub(r" (?:time|nps) \d+", "", line) for line in stdout.read_text().splitlines()
                if line.startswith("info depth ")]
        same = infos("baseline") == infos("candidate")
        if same:
            assert baseline["classified_nodes"] == candidate["classified_nodes"]
            assert baseline["published_steps"] == candidate["published_steps"]
        trajectories.append({"name":name, "complete_non_timing_trajectory_equal":same,
            "baseline_steps":baseline["published_steps"],"candidate_steps":candidate["published_steps"],
            "baseline_nodes":baseline["classified_nodes"],"candidate_nodes":candidate["classified_nodes"]})
    return {"status": "passed", "kind": "classical_best_open_seed_gfx1030_reconstructed_evidence",
            "variant": declaration["variant"], "artifacts": declaration["artifacts"],
            "independent_rules_version": chess.__version__,
            "exact_work_rows": rows_checked, "exact_work": exact,
            "native_cases": len(native["cases"]), "native_per_step_snapshots": native_steps,
            "native_legal_pvs": native_pvs, "native_legal_pv_moves": native_moves,
            "maximum_mechanism_bounds": bounds, "uci_workloads": 18, "uci_legal_reported_pvs": uci_pvs,
            "one_second_observations": observations, "traces": traces, "kernel_isa": resources, "selector_mechanism_cases": fixture,
            "per_position_exact_work": per_position, "profile_trajectories": trajectories,
            "limits": "Small authored suites and machine-local cold-process medians; no significance, strength, rating, neural-runtime or long-run fault-recovery claim. Profile trajectory equality is reported separately; instrumentation changes timing. Host waits overlap device time. Code-object metadata/static opcodes are not occupancy or executed instruction counts."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print("Reconstructed", result["exact_work_rows"], "exact-work rows;", result["native_per_step_snapshots"],
          "native steps;", result["uci_legal_reported_pvs"], "legal UCI PVs")


if __name__ == "__main__":
    main()
