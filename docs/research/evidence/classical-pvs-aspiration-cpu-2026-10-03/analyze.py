#!/usr/bin/env python3
"""Replay fixed-depth observations; ratios do not establish playing strength."""
import argparse
import csv
import json
from pathlib import Path
import statistics


def analyze(directory):
    depths = {}
    for path in sorted(directory.glob("depth*.csv")):
        with path.open() as source:
            rows = list(csv.DictReader(source))
        for row in rows:
            for name in row.keys() - {"position", "policy", "bestmove"}:
                row[name] = int(row[name])
        assert len(rows) == 96
        policies = ["baseline", "pvs", "aspiration25", "pvs_aspiration25"]
        positions = sorted({r["position"] for r in rows})
        assert len(positions) == 6
        index = {(r["round"], r["position"], r["policy"]): r for r in rows}
        assert len(index) == len(rows)
        for round in range(4):
            for position in positions:
                group = [index[round, position, policy] for policy in policies]
                assert len({(r["depth"], r["score"]) for r in group}) == 1
        summary = {}
        for policy in policies:
            totals = {key: sum(r[key] for r in rows if r["policy"] == policy)
                      for key in ("nodes", "qnodes", "elapsed_ns", "pvs_probes",
                                  "pvs_researches", "aspiration_retries")}
            per_position = {}
            for position in positions:
                paired = [index[round, position, policy]["elapsed_ns"] /
                          index[round, position, "baseline"]["elapsed_ns"] for round in range(4)]
                repetitions = [index[round, position, policy] for round in range(4)]
                assert len({(r["nodes"], r["qnodes"], r["score"], r["depth"]) for r in repetitions}) == 1
                per_position[position] = dict(
                    nodes_ratio=repetitions[0]["nodes"] / index[0, position, "baseline"]["nodes"],
                    wall_ratio=sum(r["elapsed_ns"] for r in repetitions) /
                               sum(index[round, position, "baseline"]["elapsed_ns"] for round in range(4)),
                    paired_wall_ratios=paired, paired_wall_median=statistics.median(paired))
            summary[policy] = dict(totals=totals, per_position=per_position)
        baseline = summary["baseline"]["totals"]
        for policy in policies:
            totals = summary[policy]["totals"]
            summary[policy]["aggregate_nodes_ratio"] = totals["nodes"] / baseline["nodes"]
            summary[policy]["aggregate_wall_ratio"] = totals["elapsed_ns"] / baseline["elapsed_ns"]
            summary[policy]["round_total_wall_ratios"] = [
                sum(index[round, p, policy]["elapsed_ns"] for p in positions) /
                sum(index[round, p, "baseline"]["elapsed_ns"] for p in positions) for round in range(4)]
        depths[path.stem] = summary
    assert len(depths) == 2
    return dict(status="PASS", rows=192, depths=depths,
                scope="Fixed-depth fresh-TT CPU5 observations, four balanced-order rounds; no strength or universal latency claim")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(json.dumps(analyze(args.directory), indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
