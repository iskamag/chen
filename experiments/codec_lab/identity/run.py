#!/usr/bin/env python3
"""Exact finite interaction lab. No chess engine, learner, or floating oracle.

The experimental operator multiplies resource polynomials in a changed basis.
Its readout is a collective matching count, not reconstruction quality.
"""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import argparse
import hashlib
import itertools
import json
import math
from pathlib import Path
import platform
import time
import numpy as np

HERE = Path(__file__).resolve().parent
TARGETS = 4
H = np.array([[(-1) ** ((f & d).bit_count()) for d in range(4)]
              for f in range(4)], dtype=np.int64)
VARIANTS = {
    "counts_after": 4, "energy_after": 4, "signed_dc_after": 4,
    "signed_two_after": 8, "signed_three_after": 12, "signed_full_after": 16,
    "magnitude_full_after": 16, "pooled_signed_after": 4,
    "counts_plus_pooled_after": 8, "gram_after": 10, "gram_root": 10,
    "independently_relabelled_full_after": 16, "incidence_after": 2,
}

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def incidence(n):
    ids = np.arange(1 << (n * TARGETS), dtype=np.uint32)
    shifts = np.arange(n * TARGETS).reshape(n, TARGETS)
    return ids, ((ids[:, None, None] >> shifts) & 1).astype(np.int64)


def oracle_counts(n, ids, disabled):
    """Independent explicit assignments, never the experimental state algebra."""
    available = [d for d in range(n) if d != disabled]
    out = np.zeros((len(ids), 16), dtype=np.int64)
    for demand in range(16):
        targets = [t for t in range(4) if demand & (1 << t)]
        for resources in itertools.permutations(available, len(targets)):
            required = sum(1 << (d * 4 + t) for t, d in zip(targets, resources))
            out[:, demand] += (ids & required) == required
    return out


def signed(a):
    padded = np.zeros((len(a), 4, 4), dtype=np.int64)
    padded[:, :a.shape[1], :] = a
    return np.einsum("fd,ndt->ntf", H, padded, optimize=False)


def remove_in_signed_basis(f, defender):
    """Rank-one event projection; there is no incidence reconstruction here."""
    if defender is None:
        return f.copy()
    column = H[:, defender]
    numerator = 4 * f - np.einsum("ntf,f->nt", f, column)[:, :, None] * column
    assert not np.any(numerator % 4)
    return numerator // 4


def creation_tensor(n):
    """Transform nilpotent resource creation, not XOR multiplication.

    x_d^2=0 enforces that one labelled defender cannot serve twice.
    The coefficient-state Walsh basis has size 2^n. Each target input has
    four signed components. The fixed normalization per target is 1/8.
    """
    size = 1 << n
    c = np.zeros((4, size, size), dtype=np.int64)
    for f in range(4):
        for k in range(size):
            for d in range(n):
                weight = H[f, d] * (1 if not (k & (1 << d)) else -1)
                c[f, k, k] += weight
                c[f, k, k ^ (1 << d)] += weight
    return c


def compose_signed(f, tensor):
    """Return scaled DC for every demand set, directly in transformed state.

    No inverse transform, source-membership decode, or matching oracle call.
    Retain exact integer scaling 8^number_of_targets instead of dividing.
    """
    count, _, _ = f.shape
    size = tensor.shape[1]
    states = [None] * 16
    states[0] = np.ones((count, size), dtype=np.int64)
    dc = np.ones((count, 16), dtype=np.int64)
    for demand in range(1, 16):
        bit = demand & -demand
        target = bit.bit_length() - 1
        previous = states[demand ^ bit]
        result = np.zeros_like(previous)
        for frequency in range(4):
            values = f[:, target, frequency]
            if np.any(values):
                result += (previous @ tensor[frequency].T) * values[:, None]
        dc[:, demand] = result[:, 0]
        # A state containing target 0 is never a parent under this ordering.
        if not (demand & 1):
            states[demand] = result
    return dc


def gram(a):
    g = np.einsum("ndt,nds->nts", a, a, optimize=False)
    return g[:, np.triu_indices(4)[0], np.triu_indices(4)[1]]


def group_info(key):
    key = np.ascontiguousarray(key.reshape(len(key), -1))
    unique, inverse, counts = np.unique(key, axis=0, return_inverse=True,
                                        return_counts=True)
    return key, unique, inverse, counts


def ambiguity(info, labels):
    key, unique, inverse, counts = info
    positives = np.bincount(inverse, weights=labels, minlength=len(counts)).astype(np.int64)
    negatives = counts - positives
    mixed = (positives > 0) & (negatives > 0)
    result = {
        "groups": len(counts), "ambiguous_groups": int(mixed.sum()),
        "ambiguous_cases": int(counts[mixed].sum()),
        "minimum_decoder_errors": int(np.minimum(positives, negatives).sum()),
    }
    witness = None
    if np.any(mixed):
        group = int(np.flatnonzero(mixed)[0])
        members = np.flatnonzero(inverse == group)
        yes = int(members[np.flatnonzero(labels[members])[0]])
        no = int(members[np.flatnonzero(~labels[members])[0]])
        witness = {"yes_graph": yes, "no_graph": no,
                   "same_key": key[yes].tolist()}
    return result, witness


def matrix_rows(a):
    return ["".join(str(int(v)) for v in row) for row in a]


def aggregate(records, names):
    out = []
    for name in names:
        rows = [r for r in records if r["representation"] == name]
        cases = sum(r["cases"] for r in rows)
        ambiguous = sum(r["ambiguous_cases"] for r in rows)
        minimum = sum(r["minimum_decoder_errors"] for r in rows)
        out.append({"representation": name, "fixed_storage_bytes": VARIANTS[name],
                    "query_cases": cases, "ambiguous_cases": ambiguous,
                    "ambiguous_fraction": ambiguous / cases,
                    "minimum_decoder_errors": minimum,
                    "minimum_decoder_error_fraction": minimum / cases,
                    "ambiguous_groups": sum(r["ambiguous_groups"] for r in rows),
                    "min_log2_groups": min(math.log2(r["groups"]) for r in rows),
                    "max_log2_groups": max(math.log2(r["groups"]) for r in rows)})
    return out


def operator_variants(f, n, disabled):
    yield "full_signed", f
    for kept in (1, 2, 3):
        truncated = np.zeros_like(f)
        truncated[:, :, :kept] = f[:, :, :kept]
        yield f"zero_fill_{kept}", truncated
    yield "magnitude_as_signed", np.abs(f)
    completed = np.zeros_like(f)
    completed[:, :, :3] = f[:, :, :3]
    zero = 3 if n == 3 else disabled
    if zero is not None:
        completed[:, :, 3] = -H[3, zero] * np.einsum(
            "ntf,f->nt", completed[:, :, :3], H[:3, zero])
    yield "three_with_known_zero_completion", completed


def alignment(n, a, oracles):
    ids = np.arange(len(a), dtype=np.uint32)
    global_swap = (ids & ~np.uint32(255)) | ((ids & 15) << 4) | ((ids >> 4) & 15)
    independent = ids ^ ((((ids >> 0) ^ (ids >> 4)) & 1) * np.uint32(17))
    strengths = a.sum(axis=1)
    assert np.array_equal(strengths, strengths[global_swap])
    assert np.array_equal(strengths, strengths[independent])
    result = []
    witness = None
    conditions = [None] + list(range(n))
    for i, condition in enumerate(conditions):
        mapped = {0: 1, 1: 0}.get(condition, condition)
        remapped_index = conditions.index(mapped)
        before = oracles[i]
        coherent = oracles[remapped_index][global_swap]
        assert np.array_equal(before, coherent)
        changed = oracles[i][independent]
        effective = a.copy()
        if condition is not None:
            effective[:, condition] = 0
        counts = effective.sum(axis=1)
        count_changed = np.any(counts != counts[independent], axis=1)
        row = {"disabled": condition, "query_cases": int(before.size),
               "coherent_count_changes": int(np.count_nonzero(before != coherent)),
               "independent_count_changes": int(np.count_nonzero(before != changed)),
               "independent_feasibility_changes": int(np.count_nonzero((before > 0) != (changed > 0))),
               "graphs_with_available_strength_changes": int(count_changed.sum()),
               "root_strength_changes": 0}
        if condition is None:
            assert not np.any(count_changed)
            changed_labels = (before > 0) != (changed > 0)
            if np.any(changed_labels):
                graph, demand = np.argwhere(changed_labels)[0]
                other = int(independent[graph])
                f = signed(a[[graph, other]])
                witness = {"graph": int(graph), "changed_graph": other,
                           "demand": int(demand), "before_rows": matrix_rows(a[graph]),
                           "after_rows": matrix_rows(a[other]),
                           "root_strengths": strengths[graph].tolist(),
                           "signed_target_vectors": f.tolist(),
                           "matching_counts": [int(before[graph, demand]), int(changed[graph, demand])]}
        result.append(row)
    return {"conditions": result, "unconditional_witness": witness}


def worked_cases():
    examples = []
    cases = [
        ("same_counts_and_source_margins", ["1100", "0011", "0010", "0001"],
         ["1010", "0101", "0010", "0001"], 15, None),
        ("same_full_gram_different_higher_interaction", ["1111", "1001", "0100", "0010"],
         ["1101", "1011", "0110", "0000"], 15, None),
        ("same_root_gram_fixed_identity_removal", ["1000", "0100", "0000"],
         ["0100", "1000", "0000"], 1, 0),
    ]
    for name, left, right, demand, condition in cases:
        a = np.array([[list(map(int, row)) for row in rows] for rows in (left, right)])
        n = a.shape[1]
        ids = np.array([sum(int(a[i, d, t]) << (d * 4 + t)
                            for d in range(n) for t in range(4)) for i in range(2)], dtype=np.uint32)
        effective = a.copy()
        if condition is not None:
            effective[:, condition] = 0
        f = remove_in_signed_basis(signed(a), condition)
        tensor = creation_tensor(n)
        raw = compose_signed(f, tensor)
        oracle = oracle_counts(n, ids, condition)
        stages = [0]
        for target in range(4):
            if demand & (1 << target):
                stages.append(stages[-1] | (1 << target))
        traces = [[int(raw[i, mask] // (8 ** mask.bit_count())) for mask in stages]
                  for i in range(2)]
        for mask in range(16):
            assert np.array_equal(raw[:, mask], oracle[:, mask] * (8 ** mask.bit_count()))
        shared = np.einsum("ndt,nds->nts", effective, effective)
        triple = np.prod(effective[:, :, :3], axis=2).sum(axis=1)
        examples.append({"name": name, "rows": [left, right], "disabled": condition,
                         "demand": demand, "root_target_strengths": a.sum(axis=1).tolist(),
                         "available_target_strengths": effective.sum(axis=1).tolist(),
                         "source_degrees": effective.sum(axis=2).tolist(),
                         "root_gram": np.einsum("ndt,nds->nts", a, a).tolist(),
                         "available_gram": shared.tolist(),
                         "triple_intersection_targets_012": triple.tolist(),
                         "signed_vectors": f.tolist(), "demand_stage_masks": stages,
                         "transformed_DC_matching_count_stages": traces,
                         "final_matching_counts": oracle[:, demand].tolist()})
    return examples


def run_domain(n):
    started = time.perf_counter()
    ids, a = incidence(n)
    root_f = signed(a)
    root_g = gram(a)
    tensor = creation_tensor(n)
    records, operator_results, witnesses, oracles = [], [], {}, []
    oracle_seconds = 0.0
    projection_errors = 0
    conditions = [None] + list(range(n))
    for condition in conditions:
        oracle_started = time.perf_counter()
        exact = oracle_counts(n, ids, condition)
        oracle_seconds += time.perf_counter() - oracle_started
        oracles.append(exact)
        labels = exact > 0
        effective = a.copy()
        if condition is not None:
            effective[:, condition] = 0
        f = remove_in_signed_basis(root_f, condition)
        projection_errors += int(np.count_nonzero(f != signed(effective)))
        counts = effective.sum(axis=1)
        energy = np.sum(f * f, axis=2) // 4
        assert np.array_equal(counts, energy)
        canonical = (np.arange(n)[None, :, None] < counts[:, None, :]).astype(np.int64)
        packed = (effective * (1 << np.arange(n * 4).reshape(n, 4))).sum(axis=(1, 2))[:, None]
        keys = {"counts_after": counts, "energy_after": energy,
                "signed_dc_after": f[:, :, :1], "signed_two_after": f[:, :, :2],
                "signed_three_after": f[:, :, :3], "signed_full_after": f,
                "magnitude_full_after": np.abs(f), "gram_after": gram(effective),
                "gram_root": root_g, "independently_relabelled_full_after": signed(canonical),
                "incidence_after": packed}
        static_groups = {name: group_info(key) for name, key in keys.items()}
        for demand in range(16):
            targets = [t for t in range(4) if demand & (1 << t)]
            pooled = f[:, targets, :].sum(axis=1) if targets else np.zeros((len(a), 4), dtype=np.int64)
            groups = dict(static_groups)
            groups["pooled_signed_after"] = group_info(pooled)
            groups["counts_plus_pooled_after"] = group_info(np.concatenate([counts, pooled], axis=1))
            for name in VARIANTS:
                stats, witness = ambiguity(groups[name], labels[:, demand])
                record = {"representation": name, "disabled": condition, "demand": demand,
                          "demand_size": demand.bit_count(), "cases": len(a),
                          "positive_cases": int(labels[:, demand].sum()), **stats}
                records.append(record)
                if witness is not None and name not in witnesses:
                    yes, no = witness["yes_graph"], witness["no_graph"]
                    witnesses[name] = {"disabled": condition, "demand": demand, **witness,
                                       "yes_rows": matrix_rows(a[yes]), "no_rows": matrix_rows(a[no]),
                                       "matching_counts": [int(exact[yes, demand]), int(exact[no, demand])]}
        for name, view in operator_variants(f, n, condition):
            tick = time.perf_counter()
            result = compose_signed(view, tensor)
            elapsed = time.perf_counter() - tick
            predicted = result > 0
            row = {"operator": name, "disabled": condition, "cases": int(labels.size),
                   "false_positives": int(np.count_nonzero(predicted & ~labels)),
                   "false_negatives": int(np.count_nonzero(~predicted & labels)),
                   "negative_DC_outputs": int(np.count_nonzero(result < 0)),
                   "seconds": elapsed}
            if name == "full_signed":
                mismatches = sum(np.count_nonzero(result[:, d] != exact[:, d] * (8 ** d.bit_count()))
                                 for d in range(16))
                row["matching_count_mismatches"] = int(mismatches)
                assert mismatches == 0
            operator_results.append(row)
        print(f"defenders={n} disabled={condition}: exact/ambiguity/operators complete", flush=True)
    assert projection_errors == 0
    return {"defenders": n, "targets": 4, "graphs": len(a),
            "query_cases": len(a) * len(conditions) * 16,
            "summary": aggregate(records, VARIANTS), "contexts": records,
            "witnesses": witnesses, "operators": operator_results,
            "conditional_projection_mismatches": projection_errors,
            "creation_tensor_entries": int(tensor.size),
            "creation_tensor_nonzero": int(np.count_nonzero(tensor)),
            "creation_tensor_compact_int8_bytes": int(tensor.size),
            "signed_state_coordinates": 1 << n,
            "alignment": alignment(n, a, oracles),
            "oracle_seconds": oracle_seconds, "total_seconds": time.perf_counter() - started}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(HERE / "results.json"))
    parser.add_argument("--cpu", type=int, default=1)
    args = parser.parse_args()
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, {args.cpu})
    started = time.perf_counter()
    result = {"source_sha256": sha(__file__), "protocol_sha256": sha(HERE / "protocol.json"),
              "alignment_addendum_sha256": sha(HERE / "alignment_addendum.json"),
              "python": platform.python_version(), "numpy": np.__version__,
              "cpu_affinity": sorted(os.sched_getaffinity(0)),
              "scope": "Exact joint resource compatibility, not chess legality/strength.",
              "worked_cases": worked_cases(), "domains": [run_domain(3), run_domain(4)]}
    result["total_seconds"] = time.perf_counter() - started
    Path(args.out).write_text(json.dumps(result, indent=2) + "\n")
    print(f"wrote {args.out}; elapsed={result['total_seconds']:.3f}s", flush=True)


if __name__ == "__main__":
    main()
