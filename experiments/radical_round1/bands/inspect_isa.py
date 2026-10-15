#!/usr/bin/env python3
"""Record actual hot-function instructions from the measured ELF executable."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument("binary", type=Path)
parser.add_argument("--out", required=True, type=Path)
args = parser.parse_args()
assembly = subprocess.check_output(
    ["objdump", "-d", "-C", "--no-show-raw-insn", str(args.binary)], text=True
)
labels = list(re.finditer(r"^([0-9a-f]+) <([^>]+)>:$", assembly, re.MULTILINE))
functions = {}
for i, label in enumerate(labels):
    name = label[2]
    if name not in [
        "chen_radical_bands::slider_avx2",
        "chen_radical_bands::slider_scalar",
        "chen_radical_bands::union_control",
        "chen_radical_bands::obligations_control",
        "chen_radical_bands::obligations_impl",
    ]:
        continue
    stop = labels[i + 1].start() if i + 1 < len(labels) else len(assembly)
    body = assembly[label.start() : stop]
    counts = Counter(re.findall(r"^\s*[0-9a-f]+:\s+(\w+)", body, re.MULTILINE))
    functions[name] = {
        "instructions": sum(counts.values()),
        "vector_mnemonics": {k: v for k, v in counts.items() if k.startswith("v")},
    }
    args.out.with_suffix("." + name.rsplit("::", 1)[-1] + ".txt").write_text(body)
assert "vpgatherdd" in functions["chen_radical_bands::slider_avx2"]["vector_mnemonics"]
result = {
    "binary": str(args.binary.resolve()),
    "sha256": hashlib.sha256(args.binary.read_bytes()).hexdigest(),
    "functions": functions,
    "scope": "static hot-function disassembly, not sampled dynamic instruction counts",
}
args.out.write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
