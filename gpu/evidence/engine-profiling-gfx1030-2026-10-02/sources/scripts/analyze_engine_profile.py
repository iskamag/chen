#!/usr/bin/env python3
"""Reconstruct flat CPU sampling and GPU worker-thread trace summaries.

No inclusive CPU callchain or additive host-wait/device-time claim is made.
The independent chess package checks every reported PV and selected move.
"""
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import statistics


SAMPLE = re.compile(r"^\s*(\d+)/(\d+)\s+([0-9.]+):\s+(\d+) cycles:u:\s+([0-9a-f]+) (.*?) \((.*?)\)\s*$")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def correlation(xs, ys):
    x, y = statistics.mean(xs), statistics.mean(ys)
    numerator = sum((a - x) * (b - y) for a, b in zip(xs, ys))
    denom = math.sqrt(sum((a - x) ** 2 for a in xs) * sum((b - y) ** 2 for b in ys))
    return numerator / denom if denom else None


def union_ns(intervals):
    total, end = 0, None
    for start, stop in sorted(intervals):
        if stop < start:
            raise ValueError("negative duration")
        if end is None or start > end:
            total += stop - start
        elif stop > end:
            total += stop - end
        end = max(stop, end if end is not None else stop)
    return total


def group(rows, field):
    groups = defaultdict(list)
    for row in rows:
        groups[row[field]].append(int(row["End_Timestamp"]) - int(row["Start_Timestamp"]))
    total = sum(sum(v) for v in groups.values())
    return [{"name": name, "calls": len(v), "total_ms": sum(v) / 1e6,
             "mean_us": statistics.mean(v) / 1e3, "maximum_ms": max(v) / 1e6,
             "domain_duration_percent": 100 * sum(v) / total if total else 0}
            for name, v in sorted(groups.items(), key=lambda x: -sum(x[1]))]


def audit_workloads(root):
    import chess
    count = pv_count = 0
    snapshots = {}
    for dirname in ("cpu-unprofiled", "cpu-perf", "cpu-counters", "gpu-unprofiled", "gpu-trace"):
        record = json.loads((root / dirname / "result.json").read_text())
        if record["status"] != "passed":
            raise ValueError("incomplete workload set: " + dirname)
        snapshots[dirname] = []
        for run in record["runs"]:
            if run["exit_code"] != 0 or not run["process_group_absent"] or run["movetime_ms"] != 1000:
                raise ValueError("workload retirement/budget mismatch")
            board = chess.Board(run["fen"])
            if chess.Move.from_uci(run["bestmove"]) not in board.legal_moves:
                raise ValueError("illegal selected move")
            stem = root / dirname / (str(run["repeat"]) + "-" + run["name"])
            for line in stem.with_suffix(".stdout").read_text().splitlines():
                if not line.startswith("info depth ") or " pv " not in line:
                    continue
                b = board.copy(stack=False)
                for text in line.split(" pv ", 1)[1].split():
                    move = chess.Move.from_uci(text)
                    if move not in b.legal_moves:
                        raise ValueError("illegal reported PV: " + line)
                    b.push(move)
                pv_count += 1
            count += 1
            snapshots[dirname].append({key: run[key] for key in
                                       ("name", "repeat", "bestmove", "last_info", "search_response_wall_seconds")})
    return {"completed_uci_workloads": count, "legal_reported_pvs": pv_count,
            "independent_rules_version": chess.__version__, "workload_observations": snapshots}


def cpu(root):
    symbols, categories = defaultdict(Counter), defaultdict(Counter)
    samples, periods = Counter(), Counter()
    main_samples = 0
    reports = []
    for p in sorted((root / "cpu-perf").glob("*.perf.leaf.txt")):
        # Every export has exactly one UCI search worker. Startup/main samples
        # are separate; do not treat their allocations as search hotspots.
        rows = []
        for line in p.read_text().splitlines():
            match = SAMPLE.fullmatch(line)
            if match is None:
                raise ValueError("unexpected perf leaf grammar")
            pid, tid, timestamp, period, ip, symbol, dso = match.groups()
            if pid == tid:
                main_samples += 1
                continue
            rows.append((tid, int(period), symbol))
        workers = {row[0] for row in rows}
        if len(workers) != 1 or not rows:
            raise ValueError("CPU workload has ambiguous/missing search worker")
        name = p.name.split("-", 1)[1].removesuffix(".perf.leaf.txt")
        for _, period, symbol in rows:
            symbols[name][symbol] += period
            periods[name] += period
            samples[name] += 1
            if "::make_move" in symbol:
                category = "make_move"
            elif "::legal_moves" in symbol:
                category = "legal_moves"
            elif "sliding_attacks" in symbol:
                category = "sliding_attacks"
            elif "::attackers_to" in symbol:
                category = "attackers_to"
            elif "::effective_ep_file" in symbol:
                category = "effective_ep_file"
            elif "::evaluate" in symbol:
                category = "evaluation"
            elif "::tt::" in symbol:
                category = "transposition_table"
            elif "::capture" in symbol:
                category = "capture_lookup"
            elif any(x in symbol for x in ("::ordered", "::sort::")):
                category = "ordering"
            else:
                category = "other"
            categories[name][category] += period
        report = p.with_name(p.name.removesuffix(".leaf.txt") + ".report.txt")
        lost = re.search(r"Total Lost Samples:\s*(\d+)", report.read_text())
        if not lost or int(lost.group(1)) != 0:
            raise ValueError("CPU samples lost or loss record missing")
        reports.append({"path": str(p.relative_to(root)), "sha256": sha(p),
                        "worker_samples": len(rows), "worker_tid": next(iter(workers))})
    output = []
    for name in sorted(symbols):
        total = periods[name]
        output.append({"name": name, "search_worker_samples": samples[name],
                       "sampled_cycle_period_sum": total,
                       "flat_categories_percent": {key: 100 * value / total for key, value in categories[name].most_common()},
                       "flat_symbols": [{"name": key, "sampled_cycle_period": value, "percent": 100 * value / total}
                                        for key, value in symbols[name].most_common()]})
    counter_data = []
    for p in sorted((root / "cpu-counters").glob("*.stat.jsonl")):
        raw = [json.loads(line) for line in p.read_text().splitlines() if line.startswith("{")]
        values = {row["event"]: float(row["counter-value"]) for row in raw}
        if len(values) != 5 or any(row["pcnt-running"] != 100 for row in raw):
            raise ValueError("missing or multiplexed hardware counter")
        counter_data.append({"name": p.name.removeprefix("1-").removesuffix(".stat.jsonl"),
                             "raw_counts": values,
                             "instructions_per_cycle": values["instructions:u"] / values["cycles:u"],
                             "branch_miss_percent": 100 * values["branch-misses:u"] / values["branches:u"],
                             "cache_misses_per_thousand_instructions": 1000 * values["cache-misses:u"] / values["instructions:u"]})
    return {"method": "cycles:u 199Hz flat instruction-pointer attribution, three 1000ms searches per position",
            "callchain_limit": "Recorded DWARF callchains contain unresolved/implausible frames; inclusive attribution is not admitted. Flat symbol/IP attribution is used.",
            "main_thread_samples_excluded": main_samples, "leaf_exports": reports,
            "positions": output, "counters": counter_data,
            "counter_scope": "Separate whole-process counts, including startup; all five events ran 100%, no scaling. Not sampled worker counts."}


def gpu(root):
    results = []
    for directory in sorted((root / "gpu-trace").glob("*-trace")):
        name = directory.name.removeprefix("1-").removesuffix("-trace")
        with (directory / "trace_kernel_trace.csv").open() as f:
            all_kernels = list(csv.DictReader(f))
        with (directory / "trace_hip_api_trace.csv").open() as f:
            all_apis = list(csv.DictReader(f))
        workers = {x["Thread_Id"] for x in all_kernels if "selective_complete_classical" in x["Kernel_Name"]}
        if len(workers) != 1:
            raise ValueError("ambiguous GPU search worker")
        tid = next(iter(workers))
        kernels = [row for row in all_kernels if row["Thread_Id"] == tid]
        apis = [row for row in all_apis if row["Thread_Id"] == tid]
        phase = {key: sorted([row for row in kernels if key in row["Kernel_Name"]],
                             key=lambda row: int(row["Start_Timestamp"]))
                 for key in ("selective_complete_classical", "selective_choose_classical", "selective_children_classical")}
        if len({len(x) for x in phase.values()}) != 1:
            raise ValueError("phase count disagreement")
        count = len(phase["selective_complete_classical"])
        infos = [line for line in (root / "gpu-trace" / ("1-" + name + ".stdout")).read_text().splitlines()
                 if line.startswith("info depth ")]
        if len(infos) != count:
            raise ValueError("published-root/advance count mismatch")
        used = [int(re.search(r" nodes (\d+) ", line).group(1)) for line in infos]
        if any(b < a for a, b in zip(used, used[1:])):
            raise ValueError("arena count not monotonic")
        complete = [int(x["End_Timestamp"]) - int(x["Start_Timestamp"]) for x in phase["selective_complete_classical"]]
        first = min(int(x["Start_Timestamp"]) for x in apis)
        last = max(int(x["End_Timestamp"]) for x in apis)
        api_summary = group(apis, "Function")
        kernel_summary = group(kernels, "Kernel_Name")
        intervals = [(int(x["Start_Timestamp"]), int(x["End_Timestamp"])) for x in kernels]
        fences = [row for row in apis if row["Function"] == "hipStreamSynchronize"]
        allocations = [row for row in apis if row["Function"] in ("hipMalloc", "hipHostMalloc", "hipFree", "hipHostFree")]
        quartile = max(1, count // 4)
        results.append({"name": name, "worker_tid": tid, "classified_nodes": used[-1],
                        "published_steps": count, "worker_api_span_ms": (last - first) / 1e6,
                        "device_dispatch_union_ms": union_ns(intervals) / 1e6,
                        "kernel_dispatch_count": len(kernels), "api_call_count": len(apis),
                        "worker_allocation_calls": len(allocations), "stream_fence_calls": len(fences),
                        "stream_fence_duration_ms": sum(int(x["End_Timestamp"]) - int(x["Start_Timestamp"]) for x in fences) / 1e6,
                        "api": api_summary, "kernels": kernel_summary,
                        "backup_duration_retained_nodes_pearson_r": correlation(used, complete),
                        "backup_first_quartile_mean_ms": statistics.mean(complete[:quartile]) / 1e6,
                        "backup_last_quartile_mean_ms": statistics.mean(complete[-quartile:]) / 1e6,
                        "phase_dispatches": [{"step": i + 1, "classified_nodes": used[i],
                                              **{key: (int(rows[i]["End_Timestamp"]) - int(rows[i]["Start_Timestamp"])) / 1e6
                                                 for key, rows in phase.items()}} for i in range(count)],
                        "scope": "Single instrumented search; dispatch duration and host API duration overlap and must not be added. Main-thread setup/allocation excluded."})
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = {"kind": "measured_cpu_gpu_engine_profile", "analyzer_sha256": sha(__file__),
              "workloads": audit_workloads(args.root), "cpu": cpu(args.root), "gpu": gpu(args.root),
              "limits": "Small authored suite, machine-local instrumentation, concurrent isolated-core strength match, shared cache/power uncontrolled. No CPU/GPU speedup, strength, rating or counter-based occupancy claim."}
    with args.output.open("x") as f:
        json.dump(result, f, indent=2, allow_nan=False)
        f.write("\n")
    print("Audited", result["workloads"]["completed_uci_workloads"], "workloads;",
          result["workloads"]["legal_reported_pvs"], "legal PVs;",
          sum(x["search_worker_samples"] for x in result["cpu"]["positions"]), "CPU worker samples")


if __name__ == "__main__":
    main()
