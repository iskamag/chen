#!/usr/bin/env python3
"""Independent parent checks for the small codec-composition experiments."""

import argparse
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import random

import numpy as np


HERE = Path(__file__).resolve().parent


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def audit_phase():
    phase = load_module("phase_experiment", HERE / "phase/run.py")
    rng = random.Random(2026101505)
    worst = 0.0
    cases = 24
    for _ in range(cases):
        source = np.array([[rng.randrange(-2, 3) for _ in range(8)] for _ in range(8)])
        target = np.array([[rng.randrange(-2, 3) for _ in range(8)] for _ in range(8)])
        # Pair enumeration accumulates a displacement histogram, independently
        # of the agent's overlap slicing and FFT implementations.
        expected = np.zeros((15, 15), dtype=np.int64)
        for sy in range(8):
            for sx in range(8):
                for ty in range(8):
                    for tx in range(8):
                        expected[ty - sy + 7, tx - sx + 7] += (
                            source[sy, sx] * target[ty, tx]
                        )
        assert np.array_equal(phase.direct(source, target), expected)
        actual = phase.query(source, target, "complex_phase_padded16")
        error = float(np.max(np.abs(actual - expected)))
        assert error < 1e-9
        worst = max(worst, error)
    source = np.zeros((8, 8))
    source[0, 0] = 1
    target = np.zeros((8, 8))
    target[1, 6] = 1
    assert abs(phase.knight(phase.query(source, target, "complex_phase_padded16"))) < 1e-9
    circular = phase.knight(phase.query(source, target, "complex_phase_circular8"))
    assert abs(circular - 1) < 1e-9
    return {
        "independent_pair_enumeration_cases": cases,
        "full_outputs_per_case": 225,
        "maximum_absolute_error": worst,
        "negative_control_circular_false_knight_count": circular,
    }


def audit_mdct():
    output = {}
    length = 64
    for count in (8, 16):
        # Assemble a complete global analysis operator using scalar math,
        # independently of run.py's block synthesis and overlap-add code.
        rows = []
        for start in range(0, length + 1, count):
            for frequency in range(count):
                row = []
                for location in range(length):
                    local = location + count - start
                    if 0 <= local < 2 * count:
                        value = math.sin(math.pi * (local + 0.5) / (2 * count)) * math.cos(
                            math.pi / count * (local + 0.5 + count / 2) * (frequency + 0.5)
                        )
                    else:
                        value = 0.0
                    row.append(value)
                rows.append(row)
        analysis = np.array(rows)
        gram = 2.0 / count * analysis.T @ analysis
        error = float(np.max(np.abs(gram - np.eye(length))))
        rank = int(np.linalg.matrix_rank(analysis))
        assert rank == length
        assert error < 1e-12
        output[str(count)] = {
            "signal_samples": length,
            "analysis_shape": list(analysis.shape),
            "analysis_rank": rank,
            "synthesis_analysis_identity_max_error": error,
        }
    return output


def audit_overlap():
    result = json.loads((HERE / "overlap/results/correctness.json").read_text())
    checked = 0
    for variants in result["correctness"].values():
        for row in variants.values():
            witness = row["first_counterexample"]
            if witness is None:
                continue
            line = witness["line"]
            expected = []
            # Closed path predicate, independently of source ray walking,
            # destination sweeps, or local boundary transfers.
            for source, symbol in enumerate(line):
                mask = 0
                if symbol == 2:
                    for target in range(len(line)):
                        lo, hi = sorted((source, target))
                        if target != source and all(cell == 0 for cell in line[lo + 1 : hi]):
                            mask |= 1 << target
                expected.append(mask)
            assert expected == witness["expected"]
            assert expected != witness["actual"]
            checked += 1
    assert checked == 12
    for variant in ("boundary4_lut", "lapped_lifting_inverse_boundary4"):
        assert sum(rows[variant]["whole_relation_errors"] for rows in result["correctness"].values()) == 0
    return {
        "independently_checked_counterexamples": checked,
        "oracle": "closed intervening-cell emptiness predicate",
        "exact_line_cases": sum(rows["boundary4_lut"]["roots"] for rows in result["correctness"].values()),
        "substitution_queries_per_variant": result["action_classes"]["boundary4_shared_descriptor_edits"]["queries"],
    }


def resource_stages(columns, defenders):
    """Direct coefficient-domain nilpotent product, not the signed-basis operator."""
    coefficients = {0: 1}
    stages = [1]
    for choices in columns:
        following = {}
        for used, count in coefficients.items():
            for defender in range(defenders):
                bit = 1 << defender
                if choices & bit and not used & bit:
                    following[used | bit] = following.get(used | bit, 0) + count
        coefficients = following
        stages.append(sum(coefficients.values()))
    return stages


def audit_resource_example():
    # Targets A,D share {a,b}; the other targets change their higher-order
    # coupling while keeping every per-target count and every pairwise overlap.
    star = [0b0011, 0b0101, 0b1001, 0b0011]
    triangle = [0b0011, 0b0101, 0b0110, 0b0011]

    def gram(columns):
        return [[(a & b).bit_count() for b in columns] for a in columns]

    assert gram(star) == gram(triangle)
    assert resource_stages(star, 4) == [1, 2, 3, 4, 2]
    assert resource_stages(triangle, 4) == [1, 2, 3, 2, 0]
    return {
        "oracle": "direct subset-coefficient composition",
        "star_columns": star,
        "triangle_columns": triangle,
        "equal_pairwise_gram": gram(star),
        "star_joint_assignment_count_stages": resource_stages(star, 4),
        "triangle_joint_assignment_count_stages": resource_stages(triangle, 4),
    }


def audit_identity():
    identity = load_module("identity_experiment", HERE / "identity/run.py")
    rng = random.Random(2026101506)
    checked = 0
    for defenders in (3, 4):
        graph_count = 1 << (defenders * 4)
        graph_ids = sorted({0, graph_count - 1, *rng.sample(range(graph_count), 64)})
        # Independently unpack the declared incidence bits.
        incidence = np.array([
            [[(graph >> (4 * d + t)) & 1 for t in range(4)] for d in range(defenders)]
            for graph in graph_ids
        ], dtype=np.int64)
        encoded = identity.signed(incidence)
        tensor = identity.creation_tensor(defenders)
        for disabled in (None, *range(defenders)):
            actual = identity.compose_signed(identity.remove_in_signed_basis(encoded, disabled), tensor)
            for index, matrix in enumerate(incidence):
                columns = [
                    sum(1 << d for d in range(defenders) if d != disabled and matrix[d, t])
                    for t in range(4)
                ]
                for demand in range(16):
                    selected = [column for t, column in enumerate(columns) if demand & (1 << t)]
                    expected = resource_stages(selected, defenders)[-1]
                    assert actual[index, demand] == expected * 8 ** demand.bit_count()
                    checked += 1
    return {
        "independent_subset_composition_count_queries": checked,
        "matching_count_mismatches": 0,
        "scope": "Deterministic independent graph sample, all demands and labelled removal conditions; full exhaustive replay is separate",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = {
        "audit_source_sha256": digest(Path(__file__)),
        "phase": audit_phase(),
        "mdct": audit_mdct(),
        "overlap": audit_overlap(),
        "resource_example": audit_resource_example(),
        "identity": audit_identity(),
        "files": {
            str(path.relative_to(HERE)): digest(path)
            for lane in ("phase", "mdct", "overlap", "identity")
            for path in sorted((HERE / lane).rglob("*"))
            if path.is_file() and (path.suffix in (".py", ".rs", ".json", ".toml") or path.name == "Cargo.lock")
            and "target" not in path.parts
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"audit": str(args.out), "status": "pass"}))


if __name__ == "__main__":
    main()
