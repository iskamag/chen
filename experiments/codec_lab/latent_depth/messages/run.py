#!/usr/bin/env python3
"""Run fixed-input message depth curves; labels are consumed only by auditing."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import statistics
import time

from solver import Graph, explain_round, stages, status
from audit import deduction_errors, feasible_edge_domains, maximum_matching

DEPTHS = [1, 2, 4, 8, 16, 19, 32]
MODES = ["coupled_synchronous", "input_only_onepass", "repeat_original_no_feedback",
         "target_only_no_capacity_messages", "rowwise_shuffled_resource_coupling"]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def shuffled(graph, identity):
    columns = []
    for i, mask in enumerate(graph.columns):
        rng = random.Random(int(hashlib.sha256(f"1729:{identity}:{i}".encode()).hexdigest(), 16))
        permutation = list(range(graph.resources))
        rng.shuffle(permutation)
        columns.append(sum(1 << permutation[r] for r in range(graph.resources) if mask & (1 << r)))
    return Graph(graph.resources, tuple(columns))


def deductions(graph, state):
    return {"chosen_edges": sum((one & ~zero).bit_count() for zero, one in zip(state.zero, state.one)),
            "forbidden_edges": sum((edges & ~one).bit_count() for edges, one in zip(graph.columns, state.one))}


def trace(row, graph, sequence):
    rounds = []
    for depth, state in enumerate(sequence):
        rounds.append({"round": depth, "status": status(graph, state),
                       "zero_allowed": state.zero, "one_allowed": state.one,
                       "new_chosen": [one & ~zero & ~(sequence[depth - 1].one[i] & ~sequence[depth - 1].zero[i])
                                      for i, (zero, one) in enumerate(zip(state.zero, state.one))] if depth else [],
                       "proof": state.contradiction,
                       "incoming_messages": explain_round(graph, sequence[depth - 1]) if depth else None,
                       **deductions(graph, state)})
    return {"id": row["id"], "family": row["family"], "targets": row["targets"],
            "resources": graph.resources, "columns": graph.columns, "offline_feasible": row["feasible"],
            "rounds": rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    os.sched_setaffinity(0, {4})
    corpus = json.loads(args.corpus.read_text())
    rows = [r for r in corpus["rows"] if r["split"] in ["eval", "structure"]]
    assert rows and len({r["id"] for r in rows}) == len(rows)
    args.out.mkdir(parents=True, exist_ok=False)
    counts = defaultdict(Counter)
    convergence = defaultdict(Counter)
    audit_counts = defaultdict(Counter)
    examples = {}
    graphs = []
    for row in rows:
        graph = Graph(row["resources"], tuple(row["columns"]))
        assert len(graph.columns) == row["targets"]
        assert all(0 <= mask < 1 << graph.resources for mask in graph.columns)
        exact = maximum_matching(graph.columns, graph.resources)
        assert exact == row["maximum_matching"] and (exact == row["targets"]) == row["feasible"]
        edge_gold = feasible_edge_domains(graph.columns, graph.resources) if row["feasible"] else None
        graphs.append((row, graph))
        for mode in MODES:
            changed = mode == "rowwise_shuffled_resource_coupling"
            actual_graph = shuffled(graph, row["id"]) if changed else graph
            sequence = stages(actual_graph, 32, mode)
            first_certificate = next((i for i, state in enumerate(sequence) if status(actual_graph, state) != "UNKNOWN"), None)
            first_fixed = next((i for i in range(1, 33) if sequence[i] == sequence[i - 1]), None)
            key = (mode, row["split"], row["family"], row["targets"], row["resources"])
            convergence[key]["roots"] += 1
            convergence[key]["first_certificate_" + str(first_certificate)] += 1
            convergence[key]["first_fixed_point_" + str(first_fixed)] += 1
            for depth in DEPTHS:
                state = sequence[depth]
                result = status(actual_graph, state)
                target = counts[key + (depth,)]
                target["roots"] += 1
                target[result] += 1
                target["offline_feasible_roots"] += int(row["feasible"])
                target["false_feasible"] += int(result == "FEASIBLE" and not row["feasible"])
                target["false_infeasible"] += int(result == "INFEASIBLE" and row["feasible"])
                target.update(deductions(actual_graph, state))
                if edge_gold and not changed:
                    bad_remove, bad_force = deduction_errors(graph, state, *edge_gold)
                    target["removed_supported_edges"] += bad_remove
                    target["forced_nonrequired_edges"] += bad_force
            if edge_gold and not changed:
                bad_remove, bad_force = deduction_errors(graph, sequence[-1], *edge_gold)
                audit_counts[mode].update({"feasible_roots": 1, "removed_supported_edges": bad_remove,
                                           "forced_nonrequired_edges": bad_force})
            if mode == "coupled_synchronous" and row["split"] == "structure":
                bucket = (row["family"], bool(row["feasible"]))
                if bucket not in examples or row["targets"] > examples[bucket]["targets"]:
                    examples[bucket] = trace(row, graph, sequence)
    results = []
    for key, metrics in sorted(counts.items()):
        mode, split, family, n, m, depth = key
        entry = {"mode": mode, "split": split, "family": family, "targets": n,
                        "resources": m, "depth": depth,
                        "trusted_original_graph_certificates": mode != "rowwise_shuffled_resource_coupling",
                        **{k: metrics[k] for k in ["roots", "FEASIBLE", "INFEASIBLE", "UNKNOWN", "offline_feasible_roots",
                            "false_feasible", "false_infeasible", "chosen_edges", "forbidden_edges",
                            "removed_supported_edges", "forced_nonrequired_edges"]}}
        if mode == "rowwise_shuffled_resource_coupling":
            entry["removed_supported_edges"] = entry["forced_nonrequired_edges"] = None
        results.append(entry)
    convergence_rows = []
    for key, metrics in sorted(convergence.items()):
        mode, split, family, n, m = key
        convergence_rows.append({"mode": mode, "split": split, "family": family, "targets": n,
                                 "resources": m, "counts": dict(sorted(metrics.items()))})
    deterministic = {"corpus_sha256": sha(args.corpus), "rows": len(rows), "depths": DEPTHS,
                     "curves": results, "convergence": convergence_rows,
                     "feasible_deduction_audit": dict(audit_counts)}
    (args.out / "curves.json").write_text(json.dumps(deterministic, indent=2) + "\n")
    (args.out / "examples.json").write_text(json.dumps(list(examples.values()), indent=2) + "\n")
    if args.verify:
        assert json.loads(args.verify.read_text()) == deterministic
    # Setup, JSON, labels, independent matching audits and rewiring are OUTSIDE
    # these timings. Literal requested rounds remain inside, including repeated
    # original-input passes. No hidden convergence loop or assignment decisions.
    selected = [(r, g) for r, g in graphs if r["split"] == "eval"][:512]
    selected += [(r, g) for r, g in graphs if r["split"] == "structure"]
    timings = []
    for mode in MODES:
        inputs = [shuffled(g, r["id"]) if mode == "rowwise_shuffled_resource_coupling" else g for r, g in selected]
        for depth in DEPTHS:
            samples = []
            for _ in range(5):
                started = time.perf_counter_ns()
                # Python eagerly constructs and retains all states; returned
                # state equality is tested separately, never substituted here.
                for graph in inputs:
                    final = stages(graph, depth, mode)[-1]
                    assert len(final.one) == len(graph.columns)
                samples.append((time.perf_counter_ns() - started) / len(inputs))
            timings.append({"mode": mode, "depth": depth, "roots": len(inputs),
                            "median_ns_per_root": statistics.median(samples),
                            "min_ns": min(samples), "max_ns": max(samples), "repetitions": 5})
    (args.out / "timings.json").write_text(json.dumps(timings, indent=2) + "\n")
    here = Path(__file__).resolve().parent
    metadata = {"python": platform.python_version(), "affinity": [4], "corpus_sha256": sha(args.corpus),
                "source_sha256": {name: sha(here / name) for name in ["solver.py", "audit.py", "run.py", "protocol.json"]},
                "curves_sha256": sha(args.out / "curves.json"), "examples_sha256": sha(args.out / "examples.json")}
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
