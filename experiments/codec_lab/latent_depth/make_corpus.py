#!/usr/bin/env python3
"""Generate a small shared interaction benchmark with independent exact labels."""

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random

HERE = Path(__file__).resolve().parent


def hall_maximum(columns):
    unions = [0] * (1 << len(columns))
    deficiency = 0
    for subset in range(1, len(unions)):
        bit = subset & -subset
        unions[subset] = unions[subset ^ bit] | columns[bit.bit_length() - 1]
        deficiency = max(deficiency, subset.bit_count() - unions[subset].bit_count())
    return len(columns) - deficiency


def subset_maximum(columns):
    reachable = {0}
    for column in columns:
        following = set(reachable)  # This target can remain unmatched.
        for used in reachable:
            remaining = column & ~used
            while remaining:
                bit = remaining & -remaining
                following.add(used | bit)
                remaining ^= bit
        reachable = following
    return max(used.bit_count() for used in reachable)


def generate(protocol):
    rng = random.Random(protocol["corpus_seed"])
    rows, seen = [], set()

    def key(columns, resources):
        return resources, tuple(columns)

    def add(columns, resources, split, family, **metadata):
        identity = key(columns, resources)
        if identity in seen:
            return False
        maximum = hall_maximum(columns)
        assert maximum == subset_maximum(columns)
        seen.add(identity)
        rows.append({
            "id": f"depth-{len(rows):05d}", "split": split, "family": family,
            "targets": len(columns), "resources": resources, "columns": columns,
            "maximum_matching": maximum, "feasible": maximum == len(columns),
            **metadata,
        })
        return True

    def permuted_pair(templates, size, family):
        accepted = 0
        for _ in range(128):
            target_order = rng.sample(range(size), size)
            resource_order = rng.sample(range(size), size)
            cases = [
                [sum(1 << resource_order[d] for d in range(size) if columns[t] & (1 << d))
                 for t in target_order]
                for columns in templates
            ]
            if any(key(columns, size) in seen for columns in cases):
                continue
            pair_id = f"{family}-{size}-{accepted}"
            for variant, columns in enumerate(cases):
                add(columns, size, "structure", family,
                    template_variant=variant, pair_id=pair_id)
            accepted += 1
            if accepted == 8:
                break

    for size in protocol["structure_sizes"]:
        chain = [1] + [(1 << (t - 1)) | (1 << t) for t in range(1, size)]
        missing = chain.copy()
        missing[-1] &= ~(1 << (size - 1))
        permuted_pair([chain], size, "forced_chain")
        permuted_pair([missing], size, "forced_chain_missing_last_resource")
        cycle = [(1 << t) | (1 << ((t + 1) % size)) for t in range(size)]
        permuted_pair([cycle], size, "cycle")
        deficit = [(1 << (t % (size - 1))) | (1 << ((t + 1) % (size - 1)))
                   for t in range(size)]
        permuted_pair([deficit], size, "cycle_resource_deficit")
        if size >= 4:
            core = [3, 3, 3] + [(1 << (t - 1)) | (1 << t) for t in range(3, size)]
            rewired = core.copy()
            rewired[0] = 6
            rewired[3] = (1 << 0) | (1 << 3)
            assert hall_maximum(core) == size - 1
            assert hall_maximum(rewired) == size
            assert [c.bit_count() for c in core] == [c.bit_count() for c in rewired]
            assert [sum(bool(c & (1 << d)) for c in core) for d in range(size)] == [
                sum(bool(c & (1 << d)) for c in rewired) for d in range(size)]
            permuted_pair([core, rewired], size, "Hall_core_degree_pair")
    permuted_pair([[3, 5, 9, 3], [3, 5, 6, 3]], 4, "Gram_equal_pair")

    densities = (0.15, 0.25, 0.4, 0.55, 0.7, 0.85)
    for size_text, splits in protocol["random_per_size"].items():
        size = int(size_text)
        for split, amount in splits.items():
            counts = [0, 0]
            attempts = 0
            while sum(counts) < amount:
                attempts += 1
                if attempts > 200000:
                    raise RuntimeError(f"Sampling exhausted size={size} split={split}")
                density = rng.choice(densities)
                columns = [sum(1 << d for d in range(size) if rng.random() < density)
                           for _ in range(size)]
                if key(columns, size) in seen:
                    continue
                feasible = hall_maximum(columns) == size
                if counts[int(feasible)] == amount // 2:
                    continue
                if add(columns, size, split, "random_small" if size <= 6 else "random_larger"):
                    counts[int(feasible)] += 1
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    protocol_path = HERE / "protocol.json"
    protocol = json.loads(protocol_path.read_text())
    rows = generate(protocol)
    counts = Counter((row["split"], row["family"], row["targets"]) for row in rows)
    result = {
        "version": 1,
        "generator_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "protocol_sha256": hashlib.sha256(protocol_path.read_bytes()).hexdigest(),
        "split_contract": "Distinct raw graphs; no isomorphism holdout claim. Structure templates are held out from fitting but random data may contain related motifs.",
        "oracle_crosscheck": "Hall deficiency equals independently implemented resource-subset DP on every row",
        "counts": [{"split": s, "family": f, "targets": n, "rows": count}
                   for (s, f, n), count in sorted(counts.items())],
        "rows": rows,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, separators=(",", ":")) + "\n")
    print(json.dumps({"path": str(args.out), "rows": len(rows),
                      "splits": dict(Counter(row["split"] for row in rows)),
                      "sha256": hashlib.sha256(args.out.read_bytes()).hexdigest()}))


if __name__ == "__main__":
    main()
