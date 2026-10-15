#!/usr/bin/env python3
"""Alternate portable/native frozen binaries; retain every raw JSONL run."""
import argparse
import json
import os
from pathlib import Path
import statistics
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("--portable", required=True)
parser.add_argument("--native", required=True)
parser.add_argument("--corpus", required=True)
parser.add_argument("--out", required=True, type=Path)
parser.add_argument("--runs", type=int, default=3)
parser.add_argument("--cpu", type=int, default=4)
args = parser.parse_args()
os.sched_setaffinity(0, {args.cpu})
args.out.mkdir(parents=True, exist_ok=True)
records = []
for run in range(args.runs):
    labels = ["portable", "native"] if run % 2 == 0 else ["native", "portable"]
    for label in labels:
        output = args.out / f"{label}-{run}.jsonl"
        command = [getattr(args, label), "--corpus", args.corpus, "--out", str(output), "--iters", "100", "--all-only"]
        with output.with_suffix(".stdout").open("w") as stdout:
            subprocess.run(command, check=True, stdout=stdout, env={**os.environ, "CHEN_BUILD_TARGET": label})
        records.extend({"build": label, "run": run, **json.loads(line)} for line in output.read_text().splitlines())
summary = {}
for label in ["portable", "native"]:
    group = [r for r in records if r["build"] == label]
    variants = sorted({r["variant"] for r in group if r["kind"] == "benchmark"})
    summary[label] = {}
    for variant in variants:
        values = [r for r in group if r["kind"] == "benchmark" and r["variant"] == variant]
        summary[label][variant] = {
            key: statistics.median(r[key] for r in values)
            for key in ["attacks_query_ns", "attacks_build_query_ns", "complete_packet_ns"]
        }
    for kind in ["constraints", "transitions"]:
        values = [r for r in group if r["kind"] == kind]
        summary[label][kind] = {
            key: statistics.median(r[key] for r in values)
            for key in values[0] if key.endswith("_ns")
        }
(args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
