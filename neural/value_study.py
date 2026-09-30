"""Predeclared resident value-head study; classical search labels, no outcomes."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .compact import decode_features
from .models import MODELS
from .teacher import (MATE_BAND, SCALE_CP, comparable_model, material_cp,
                      produce_dataset, split_root_families, _replay_fingerprint)
from .value_heads import ValueAdapter

HEADS = ("three_logit", "direct_cp", "material_residual")
VALUE_FIELDS = ("search_score_cp", "teacher_root_id", "parent_node")


def gather_values(replay, indices):
    """Gather exact current observations only; no legal mask/next-state corpus copy."""
    storage = replay.compact.storage
    pieces = storage["pieces"].index_select(0, indices)
    state = storage["state"].index_select(0, indices)
    return {"features": decode_features(pieces, state),
            "game_ids": storage["game_ids"].index_select(0, indices),
            "sequence_ids": replay.compact.sequence_ids.index_select(0, indices),
            **{key: replay.labels[key].index_select(0, indices) for key in VALUE_FIELDS}}


def value_loss(adapter, batch):
    """Same normalized-CP SmoothL1 for all heads, with exact mate exclusion."""
    output = adapter(batch["features"])
    target = batch["search_score_cp"].float() / SCALE_CP
    ordinary = batch["search_score_cp"].to(torch.int64).abs() < MATE_BAND
    rows = F.smooth_l1_loss(output["value_cp"] / SCALE_CP, target, reduction="none")
    loss = (rows * ordinary).sum() / ordinary.sum().clamp_min(1)
    return loss, output


def value_step(adapter, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    loss, _ = value_loss(adapter, batch)
    loss.backward()
    norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), 1.0)
    optimizer.step()
    return loss.detach(), norm.detach()


def paired_family_interval(rows, *, seed=20261001, draws=2000):
    """Cold aggregate-only paired bootstrap: each source-game family is a block."""
    if not rows:
        return {"families": 0, "mean_network_minus_material_cp_mae": None, "ci95": None}
    differences = [row["network_mae"] - row["material_mae"] for row in rows]
    rng = random.Random(seed)
    means = sorted(sum(rng.choice(differences) for _ in differences) / len(differences)
                   for _ in range(draws))
    return {"families": len(rows),
            "mean_network_minus_material_cp_mae": sum(differences) / len(differences),
            "ci95": [means[int(draws * .025)], means[min(draws - 1, int(draws * .975))]],
            "scope": "paired equally weighted families; initialization-seed variation reported separately"}


def _ordering(prediction, teacher, roots):
    """Ordinary nonterminal child subset; lower child-mover CP is better at root."""
    _, inverse = torch.unique(roots, sorted=True, return_inverse=True)
    groups = torch.unique(roots).numel()
    indexes = torch.arange(teacher.numel(), device=teacher.device)
    minima = torch.full((groups,), torch.inf, device=teacher.device).scatter_reduce_(
        0, inverse, teacher, reduce="amin", include_self=True)
    predicted_min = torch.full_like(minima, torch.inf).scatter_reduce_(
        0, inverse, prediction, reduce="amin", include_self=True)
    candidates = torch.where(prediction == predicted_min.index_select(0, inverse), indexes, teacher.numel())
    chosen = torch.full((groups,), teacher.numel(), dtype=torch.int64, device=teacher.device).scatter_reduce_(
        0, inverse, candidates, reduce="amin", include_self=True)
    counts = torch.zeros(groups, device=teacher.device).scatter_add_(0, inverse, torch.ones_like(teacher))
    eligible = counts >= 2
    regret = teacher.index_select(0, chosen) - minima
    correct = (regret == 0).float()
    # All unordered within-root pairs, teacher ties excluded; predicted ties get half credit.
    concordant, pairs = torch.zeros((), device=teacher.device), torch.zeros((), device=teacher.device)
    for start in range(0, teacher.numel(), 256):
        stop = min(start + 256, teacher.numel())
        truth = teacher[start:stop, None] - teacher[None]
        guessed = prediction[start:stop, None] - prediction[None]
        valid = ((roots[start:stop, None] == roots[None]) &
                 (indexes[start:stop, None] < indexes[None]) & (truth != 0))
        concordant += (((guessed.sign() == truth.sign()).float() + .5 * (guessed == 0)) * valid).sum()
        pairs += valid.sum()
    return torch.stack((eligible.sum(), (regret * eligible).sum(), (correct * eligible).sum(), concordant, pairs))


def metrics(prediction, utility, teacher, features_material, game_ids, root_ids):
    """Resident reductions followed by explicit aggregate-only reporting boundary."""
    ordinary = teacher.to(torch.int64).abs() < MATE_BAND
    cp = prediction[ordinary]
    truth = teacher[ordinary].float()
    util = utility[ordinary]
    anchor = features_material[ordinary]
    families = game_ids[ordinary]
    roots = root_ids[ordinary]
    error = cp - truth
    count = truth.numel()
    if not count:
        raise ValueError("evaluation has no ordinary teacher scores")
    total = torch.stack((error.abs().mean(), error.square().mean().sqrt(), error.mean(),
                         (util - torch.tanh(truth / SCALE_CP)).square().mean().sqrt(),
                         error.abs().quantile(.95), cp.abs().max(),
                         (util.abs() >= .99).float().mean(),
                         (torch.tanh(truth / SCALE_CP).abs() >= .99).float().mean(),
                         (cp.abs() >= MATE_BAND).float().mean(),
                         (anchor - truth).abs().mean()))
    keys = ("ordinary_centipawn_mae", "ordinary_centipawn_rmse", "signed_centipawn_bias",
            "utility_rmse", "p95_absolute_centipawn_error", "max_absolute_raw_prediction_cp",
            "predicted_utility_abs_ge_099_fraction", "teacher_utility_abs_ge_099_fraction",
            "predicted_static_band_fraction", "material_centipawn_mae")
    report = dict(zip(keys, total.cpu().tolist()))
    report.update(records=teacher.numel(), ordinary_records=count, mate_band_records=teacher.numel() - count)
    order = _ordering(cp, truth, roots).cpu().tolist()
    base_order = _ordering(anchor, truth, roots).cpu().tolist()
    def order_report(values):
        groups, regret, optimal, concordant, pairs = values
        return {"eligible_roots": int(groups), "mean_selected_child_regret_cp": regret / max(groups, 1),
                "optimal_child_fraction": optimal / max(groups, 1), "non_tied_pairs": int(pairs),
                "pairwise_order_concordance": concordant / max(pairs, 1),
                "scope": "observed ordinary nonterminal children only; excludes terminal and mate-band root actions"}
    report["ordering"] = order_report(order)
    report["material_ordering"] = order_report(base_order)
    unique, inverse = torch.unique(families, sorted=True, return_inverse=True)
    family_count = torch.zeros_like(unique, dtype=torch.float32).scatter_add_(0, inverse, torch.ones_like(truth))
    family_error = torch.zeros_like(family_count).scatter_add_(0, inverse, error.abs())
    family_base = torch.zeros_like(family_count).scatter_add_(0, inverse, (anchor - truth).abs())
    # Only O(number of families) summaries cross this cold diagnostics boundary.
    family_rows = torch.stack((unique.float(), family_count, family_error / family_count,
                               family_base / family_count), 1).cpu().tolist()
    report["family_rows"] = [{"family_id": int(row[0]), "records": int(row[1]),
                              "network_mae": row[2], "material_mae": row[3]} for row in family_rows]
    report["paired_family_interval"] = paired_family_interval(report["family_rows"])
    bins = []
    for label, low, high in (("abs_cp_lt200", 0, 200), ("200_to800", 200, 800),
                             ("800_to2000", 800, 2000), ("2000_to_mate_band", 2000, MATE_BAND)):
        mask = (truth.abs() >= low) & (truth.abs() < high)
        bins.append(torch.stack((mask.sum(), (error.abs() * mask).sum(), ((anchor - truth).abs() * mask).sum())))
    report["teacher_magnitude_bins"] = [
        {"bin": label, "records": int(n), "network_mae": err / max(n, 1), "material_mae": base / max(n, 1)}
        for label, (n, err, base) in zip(("abs_cp_lt200", "200_to800", "800_to2000", "2000_to_mate_band"),
                                       torch.stack(bins).cpu().tolist())]
    imbalance = []
    for mask in (anchor.abs() <= 100, anchor.abs() > 100):
        imbalance.append(torch.stack((mask.sum(), (error.abs() * mask).sum(), ((anchor - truth).abs() * mask).sum())))
    report["material_imbalance_bins"] = [
        {"bin": label, "records": int(n), "network_mae": err / max(n, 1), "material_mae": base / max(n, 1)}
        for label, (n, err, base) in zip(("abs_material_cp_le100", "abs_material_cp_gt100"), torch.stack(imbalance).cpu().tolist())]
    return report


@torch.no_grad()
def evaluate_values(adapter, replay, batch_size=256):
    adapter.eval()
    outputs, fields = [], []
    for start in range(0, replay.compact.size, batch_size):
        batch = gather_values(replay, torch.arange(start, min(start + batch_size, replay.compact.size), device=replay.device))
        outputs.append(adapter(batch["features"]))
        fields.append({key: batch[key] for key in ("search_score_cp", "game_ids", "teacher_root_id")})
    joined = {key: torch.cat([out[key] for out in outputs]) for key in ("value_cp", "utility", "material_cp")}
    labels = {key: torch.cat([row[key] for row in fields]) for key in fields[0]}
    return metrics(joined["value_cp"], joined["utility"], labels["search_score_cp"], joined["material_cp"],
                   labels["game_ids"], labels["teacher_root_id"])


def summarize(runs):
    result = []
    for kind in MODELS:
        for head in HEADS:
            selected = [run for run in runs if run["architecture"] == kind and run["head"] == head]
            if not selected:
                continue
            maes = [run["after_heldout"]["ordinary_centipawn_mae"] for run in selected]
            regrets = [run["after_heldout"]["ordering"]["mean_selected_child_regret_cp"] for run in selected]
            mean = sum(maes) / len(maes)
            result.append({"architecture": kind, "head": head, "seeds": len(selected),
                           "heldout_cp_mae_mean": mean,
                           "heldout_cp_mae_seed_sd": (sum((value - mean) ** 2 for value in maes) / max(len(maes) - 1, 1)) ** .5,
                           "heldout_root_regret_cp_mean": sum(regrets) / len(regrets),
                           "paired_family_cp_mae_mean_differences": [run["after_heldout"]["paired_family_interval"] for run in selected]})
    return result


def run_study(*, library, output, producer_manifest, train_families=128, heldout_families=32,
              corpus_seed=20261001, seeds=(20261001, 20261002, 20261003), steps=2048,
              batch_size=128, width=32, depth=1, tree_depth=3, evaluator=1,
              arena_capacity=100000, checkpoint_dir="results/neural/value-study",
              architectures=MODELS, heads=HEADS, device="cuda", pilot=False):
    if any(type(v) is not int or v <= 0 for v in (steps, batch_size, width, depth)):
        raise ValueError("positive integer study configuration required")
    if not seeds or len(set(seeds)) != len(seeds) or any(type(seed) is not int or seed < 0 for seed in seeds):
        raise ValueError("distinct nonnegative integer run seeds required")
    if not architectures or any(name not in MODELS for name in architectures) or not heads or any(head not in HEADS for head in heads):
        raise ValueError("invalid study models/heads")
    producer = json.loads(Path(producer_manifest).read_text())
    library_sha = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    if producer.get("teacher_library_sha256") != library_sha:
        raise ValueError("producer library/evidence fingerprint mismatch")
    started = time.perf_counter()
    train_fams, heldout_fams, split = split_root_families(train_families, heldout_families, corpus_seed)
    cpu_split_seconds = time.perf_counter() - started
    train, train_manifest = produce_dataset(train_fams, library=library, device=device, arena_capacity=arena_capacity,
                                             tree_depth=tree_depth, evaluator=evaluator)
    heldout, heldout_manifest = produce_dataset(heldout_fams, library=library, device=device, arena_capacity=arena_capacity,
                                                 tree_depth=tree_depth, evaluator=evaluator)
    schedules = {seed: torch.randint(train.compact.size, (steps, batch_size), device=train.device,
                                     generator=torch.Generator(device=train.device).manual_seed(seed + 2)) for seed in seeds}
    report = {"kind": "resident_search_value_head_study", "strength_evidence": False, "pilot": bool(pilot),
              "config": {"train_families": train_families, "heldout_families": heldout_families,
                         "corpus_seed": corpus_seed, "seeds": list(seeds), "steps": steps, "batch_size": batch_size,
                         "width": width, "depth": depth, "tree_depth": tree_depth, "remaining_parent_horizon": tree_depth - 1,
                         "evaluator": evaluator, "arena_capacity": arena_capacity,
                         "architectures": list(architectures), "heads": list(heads),
                         "loss": "ordinary-only SmoothL1(value_cp/800, teacher_cp/800)",
                         "optimizer": "AdamW", "learning_rate": 3e-4, "grad_clip": 1., "scale_cp": SCALE_CP,
                         "mate_band_excluded": MATE_BAND,
                         "sample_schedule": "same resident with-replacement indices for each head/architecture within seed",
                         "model_initialization": "same comparable encoder per architecture/seed; zero new scalar heads; existing3head seeded"},
              "environment": {"torch": str(torch.__version__), "torch_hip": torch.version.hip,
                              "architecture": torch.cuda.get_device_properties(train.device).gcnArchName},
              "cpu_split_seconds": cpu_split_seconds, "split": split, "train_corpus": train_manifest,
              "heldout_corpus": heldout_manifest, "producer_evidence": producer,
              "library": {"path": library, "sha256": library_sha}, "runs": []}
    destination = Path(output); destination.parent.mkdir(parents=True, exist_ok=True)
    def publish():
        destination.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    publish()
    for seed in seeds:
        for name in architectures:
            for head in heads:
                adapter = ValueAdapter(comparable_model(name, width=width, depth=depth, seed=seed), head).to(train.device)
                before = evaluate_values(adapter, heldout)
                optimizer = torch.optim.AdamW(adapter.active_parameters(), lr=3e-4)
                torch.cuda.synchronize(train.device); started = time.perf_counter()
                losses, norms = [], []
                adapter.train()
                for indexes in schedules[seed]:
                    loss, norm = value_step(adapter, optimizer, gather_values(train, indexes))
                    losses.append(loss); norms.append(norm)
                torch.cuda.synchronize(train.device)
                seconds = time.perf_counter() - started
                after = evaluate_values(adapter, heldout)
                training = evaluate_values(adapter, train)
                diagnostics = torch.stack((torch.stack(losses)[0], torch.stack(losses)[-1],
                                           torch.stack(norms).mean(), torch.stack(norms).max(),
                                           (torch.stack(norms) > 1).float().mean())).cpu().tolist()
                # Cold explicit checkpoint boundary, separate kind from legacy teacher artifacts.
                from .value_heads import save_value_checkpoint
                path = Path(checkpoint_dir) / f"{name}-{head}-{seed}.pt"
                checkpoint = save_value_checkpoint(adapter, path, provenance={
                    "teacher": train_manifest, "study_config": report["config"], "run_seed": seed,
                    "split_sha256": split["source_sha256"], "strength_evidence": False})
                run = {"architecture": name, "head": head, "seed": seed,
                       "parameters": sum(parameter.numel() for parameter in adapter.parameters()),
                       "active_value_parameters": sum(parameter.numel() for parameter in adapter.active_parameters()),
                       "before_heldout": before, "after_heldout": after, "after_training": training,
                       "training_seconds": seconds, "loss_first": diagnostics[0], "loss_last": diagnostics[1],
                       "preclip_grad_norm_mean": diagnostics[2], "preclip_grad_norm_max": diagnostics[3],
                       "gradient_clipped_fraction": diagnostics[4], "checkpoint": checkpoint}
                report["runs"].append(run)
                publish()
                print(json.dumps({"architecture": name, "head": head, "seed": seed, "training_seconds": seconds,
                                  "heldout_cp_mae": after["ordinary_centipawn_mae"],
                                  "material_cp_mae": after["material_centipawn_mae"],
                                  "root_regret_cp": after["ordering"]["mean_selected_child_regret_cp"]}), flush=True)
                del adapter, optimizer, losses, norms
    report["sample_stream_sha256"] = {str(seed): hashlib.sha256(bytes(schedule.cpu().contiguous().view(torch.uint8).flatten().tolist())).hexdigest()
                                        for seed, schedule in schedules.items()}
    report["data_fingerprints"] = {"train": _replay_fingerprint(train), "heldout": _replay_fingerprint(heldout),
                                   "scope": "explicit post-training cold payload downloads for checksums; never training ingress"}
    report["summary"] = summarize(report["runs"])
    report["source_sha256"] = {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in
                               map(Path, ("neural/value_study.py", "neural/value_heads.py", "neural/teacher.py",
                                          "neural/compact.py", "neural/models.py", "neural/features.py"))}
    report["limitations"] = ["single fixed synthetic legal-game corpus, finite-depth classical teacher, no outcomes or match strength",
                              "initialization/sample seeds vary; games held fixed, heldout metrics not independent across seeds",
                              "ordering excludes terminal/mate-band root actions; no full engine policy claim",
                              "heads differ in initialization and parameter count; base shared within architecture/seed",
                              "scalar producer/control synchronization allowed; payload checksums/checkpoints are cold explicit downloads"]
    publish()
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", required=True)
    parser.add_argument("--producer-manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--train-families", type=int, default=128)
    parser.add_argument("--heldout-families", type=int, default=32)
    parser.add_argument("--corpus-seed", type=int, default=20261001)
    parser.add_argument("--seeds", type=int, nargs="+", default=[20261001, 20261002, 20261003])
    parser.add_argument("--steps", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--evaluator", type=int, choices=(0, 1), default=1)
    parser.add_argument("--arena-capacity", type=int, default=100000)
    parser.add_argument("--architectures", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--heads", nargs="+", choices=HEADS, default=list(HEADS))
    parser.add_argument("--checkpoint-dir", default="results/neural/value-study")
    parser.add_argument("--pilot", action="store_true")
    run_study(**vars(parser.parse_args()))


if __name__ == "__main__":
    main()
