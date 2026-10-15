#!/usr/bin/env python3
"""Build a portable CPU experiment; separate deterministic answers from timings."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify", type=Path, help="Compare deterministic correctness with a frozen run")
    args = parser.parse_args()
    here = Path(__file__).resolve().parent
    args.out.mkdir(parents=True, exist_ok=False)
    env = dict(os.environ)
    env.pop("CARGO_ENCODED_RUSTFLAGS", None)
    env["RUSTFLAGS"] = "-C target-cpu=x86-64"
    subprocess.run(["cargo", "build", "--release", "--offline", "--locked", "--manifest-path", str(here / "Cargo.toml")], env=env, check=True)
    binary = here / "target/release/chen-codec-overlap-lab"
    raw = subprocess.check_output(["taskset", "-c", "4", str(binary)])
    result = json.loads(raw)
    timings = {key: result.pop(key) for key in ["timing", "class_timing_ns_per48queries"]}
    for name, data in [("correctness.json", result), ("timings.json", timings)]:
        (args.out / name).write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")
    if args.verify:
        assert json.loads(args.verify.read_text()) == result, "Deterministic answer audit differs"
    source = [here / p for p in ["protocol.json", "Cargo.toml", "Cargo.lock", "src/main.rs", "run.py"]]
    metadata = {
        "source_sha256": {str(p.relative_to(here)): digest(p) for p in source},
        "binary_sha256": digest(binary),
        "compiler": subprocess.check_output(["rustc", "-Vv"], text=True).strip(),
        "cpu": subprocess.check_output(["lscpu"], text=True),
        "affinity": [4], "rustflags": env["RUSTFLAGS"],
        "correctness_sha256": digest(args.out / "correctness.json"),
        "timings_sha256": digest(args.out / "timings.json"),
        "verified_against": str(args.verify) if args.verify else None,
    }
    (args.out / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "correctness_sha256": metadata["correctness_sha256"]}))


if __name__ == "__main__":
    main()
