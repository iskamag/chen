#!/usr/bin/env python3
"""Parent checks for fixed-input latent-depth experiments; no training."""

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import torch

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def maximum_matching(columns, resources):
    """Offline augmenting-path oracle, independent of Hall/subset generation."""
    owner = [-1] * resources
    assigned = [-1] * len(columns)
    for root in range(len(columns)):
        queue, seen_targets = [root], {root}
        parent = [-1] * resources
        free = -1
        for target in queue:
            for resource in range(resources):
                if not columns[target] & (1 << resource) or parent[resource] != -1:
                    continue
                parent[resource] = target
                if owner[resource] == -1:
                    free = resource
                    break
                following = owner[resource]
                if following not in seen_targets:
                    seen_targets.add(following)
                    queue.append(following)
            if free != -1:
                break
        while free != -1:
            target = parent[free]
            previous = assigned[target]
            assigned[target], owner[free] = free, target
            free = previous
    return sum(resource != -1 for resource in assigned)


def audit_algebra():
    path = HERE / "algebra/results.json"
    result = json.loads(path.read_text())
    assert result["corpus_sha256"] == sha(HERE / "corpus.json")
    assert result["source_sha256"] == sha(HERE / "algebra/run.py")
    assert result["protocol_sha256"] == sha(HERE / "algebra/protocol.json")
    for row in result["records"]:
        maximum = maximum_matching(row["columns"], row["resources"])
        assert maximum == row["maximum_matching"]
        stages = row["variants"]["nilpotent_closure"]
        assert stages[3]["max_covered_targets"] == maximum
        for stage in stages:
            assert stage["max_covered_targets"] <= maximum
            if stage["answer"] != "unknown":
                assert (stage["answer"] == "feasible") == (maximum == row["targets"])
    return {"independent_oracle_cases": len(result["records"]),
            "all_maximum_cardinality_facts_complete_by_round": 3,
            "incorrect_exact_certificates": 0}


def audit_messages():
    result = json.loads((HERE / "messages/results/curves.json").read_text())
    assert result["corpus_sha256"] == sha(HERE / "corpus.json")
    recorded_sources = json.loads((HERE / "messages/results/run.json").read_text())["source_sha256"]
    for name, digest in recorded_sources.items():
        assert sha(HERE / "messages" / name) == digest
    for row in result["curves"]:
        if row["trusted_original_graph_certificates"]:
            assert row["false_feasible"] == row["false_infeasible"] == 0
    chains = [row for row in result["convergence"]
              if row["mode"] == "coupled_synchronous" and row["family"] == "forced_chain"]
    for row in chains:
        count = row["counts"]
        assert count[f'first_certificate_{2 * row["targets"] - 1}'] == count["roots"]
    return {"cases": result["rows"], "incorrect_trusted_certificates": 0,
            "forced_chain_formula_checked": "2*n-1",
            "feasible_edge_audit": result["feasible_deduction_audit"]["coupled_synchronous"]}


def audit_learned(rows, replay_root):
    folder = HERE / "learned/results"
    result = json.loads((folder / "results.json").read_text())
    assert result["corpus_sha256"] == sha(HERE / "corpus.json")
    assert result["source_sha256"] == sha(HERE / "learned/run.py")
    assert result["protocol_sha256"] == sha(HERE / "learned/protocol.json")
    selected = [row for row in rows if row["split"] != "train"]
    labels = np.array([row["feasible"] for row in selected], dtype=bool)
    replay_fits = {}
    if replay_root:
        replay_result = json.loads((replay_root / "results.json").read_text())
        replay_fits = {(fit["regime"], fit["seed"]): fit for fit in replay_result["fits"]}
    fits = []
    for fit in result["fits"]:
        checkpoint = folder / fit["checkpoint"]
        assert sha(checkpoint) == fit["checkpoint_sha256"]
        state = torch.load(checkpoint, weights_only=True)
        parameters = sum(value.numel() for value in state.values())
        assert parameters == fit["parameters"] == 16897
        if replay_root:
            replay_fit = replay_fits[fit["regime"], fit["seed"]]
            assert fit["metrics"] == replay_fit["metrics"]
            assert fit["transitions"] == replay_fit["transitions"]
        path = folder / fit["predictions"]
        assert sha(path) == fit["predictions_sha256"]
        with np.load(path) as saved:
            assert list(saved["ids"]) == [row["id"] for row in selected]
            assert np.array_equal(saved["labels"].astype(bool), labels)
            logits = saved["logits"]
            depths, variants = list(saved["depths"]), list(saved["variants"])
            for metric in fit["metrics"]:
                split, family = metric["group"].split("/", 1)
                positions = [i for i, row in enumerate(selected)
                             if row["split"] == split and (family == "ALL" or row["family"] == family)]
                values = logits[variants.index(metric["variant"]), depths.index(metric["depth"]), positions]
                truth = labels[positions]
                predicted = values >= 0
                assert len(positions) == metric["count"]
                assert float(np.mean(predicted == truth)) == metric["accuracy"]
                assert np.count_nonzero(predicted & ~truth) == metric["false_positives"]
                assert np.count_nonzero(~predicted & truth) == metric["false_negatives"]
            reset = logits[variants.index("reset_every_round")]
            assert all(np.array_equal(reset[0], row) for row in reset)
            assert np.array_equal(reset[0], logits[variants.index("feedback"), depths.index(1)])
            if replay_root:
                assert sha(replay_root / fit["checkpoint"]) == sha(checkpoint)
                with np.load(replay_root / fit["predictions"]) as replay:
                    assert saved.files == replay.files
                    assert all(np.array_equal(saved[key], replay[key]) for key in saved.files)
        fits.append({"regime": fit["regime"], "seed": fit["seed"], "parameters": parameters,
                     "checkpoint_sha256": sha(checkpoint), "recorded_accuracy_recomputed": True,
                     "parent_prediction_replay_identical": bool(replay_root)})
    spec = importlib.util.spec_from_file_location("latent_learned_probe", HERE / "learned/run.py")
    probe = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = probe
    spec.loader.exec_module(probe)
    original = rows[-8:]
    poisoned = [{**row, "feasible": not row["feasible"], "maximum_matching": -999,
                 "family": "poison", "split": "poison"} for row in original]
    assert all(torch.equal(a, b) for a, b in zip(probe.pack(original), probe.pack(poisoned), strict=True))
    return {"fits": fits, "oracle_metadata_absent_from_encoder": True,
            "selection_rows": len(selected), "retraining_replayed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--algebra-replay", type=Path)
    parser.add_argument("--messages-replay", type=Path)
    parser.add_argument("--learned-replay-root", type=Path)
    args = parser.parse_args()
    rows = json.loads((HERE / "corpus.json").read_text())["rows"]
    assert len(rows) == 3800
    assert len({(row["resources"], tuple(row["columns"])) for row in rows}) == len(rows)
    for row in rows:
        maximum = maximum_matching(row["columns"], row["resources"])
        assert maximum == row["maximum_matching"]
        assert (maximum == row["targets"]) == row["feasible"]
    if args.algebra_replay:
        assert args.algebra_replay.read_bytes() == (HERE / "algebra/results.json").read_bytes()
    if args.messages_replay:
        assert (args.messages_replay / "curves.json").read_bytes() == (HERE / "messages/results/curves.json").read_bytes()
        assert (args.messages_replay / "examples.json").read_bytes() == (HERE / "messages/results/examples.json").read_bytes()
    result = {
        "audit_source_sha256": sha(Path(__file__)), "corpus_sha256": sha(HERE / "corpus.json"),
        "independent_augmenting_path_oracle_rows": len(rows),
        "algebra": audit_algebra(), "messages": audit_messages(),
        "learned": audit_learned(rows, args.learned_replay_root),
        "parent_replays": {"algebra": bool(args.algebra_replay), "messages": bool(args.messages_replay),
                           "learned_frozen_inference": bool(args.learned_replay_root)},
        "separate_source_review": "Astra checked synchronous old-state updates, tied learned parameters, no labels/oracles in inference, train/dev/eval boundaries and sound message certificates. Corruption-control confounds documented.",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"audit": str(args.out), "status": "pass", "corpus_rows": len(rows)}))


if __name__ == "__main__":
    main()
