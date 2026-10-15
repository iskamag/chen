#!/usr/bin/env python3
"""CPU-only listwise training; no runtime teacher, child boards or value backup."""
import argparse
from collections import defaultdict
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
import torch
from torch.nn import functional as F

from model import ChannelPolicy, exported_forward


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def teacher_value(item):
    if item.get("mate") is not None:
        mate = item["mate"]
        return float((10000 - min(100, mate) * 10) if mate > 0 else (-10000 + min(100, -mate) * 10))
    return float(item["cp"])


class Dataset:
    def __init__(self, labels, features):
        self.feature_manifest = json.loads((features / "manifest.json").read_text())
        self.dim = self.feature_manifest["dim"]
        self.channels = self.feature_manifest["channels"]
        index = {row["id"]: row for row in read_rows(features / "index.jsonl")}
        raw = np.fromfile(features / "features.f32", dtype="<f4")
        assert raw.size % self.dim == 0
        self.features = torch.from_numpy(raw.reshape(-1, self.dim))
        assert torch.isfinite(self.features).all()
        self.groups = {}
        for split in ["train", "dev"]:
            groups = []
            for record in read_rows(labels / f"{split}.jsonl"):
                row = index[record["id"]]
                assert row["count"] == len(row["moves"]) > 0
                mapping = {move: i for i, move in enumerate(row["moves"])}
                assert record["hard_best_uci"] in mapping
                record["offset"] = row["offset"]
                record["count"] = row["count"]
                record["action_inventory"] = row["moves"]
                record["hard_index"] = mapping[record["hard_best_uci"]]
                record["labeled_indices"] = [mapping[x["uci"]] for x in record["moves"]]
                record["labeled_values"] = [teacher_value(x) for x in record["moves"]]
                record["pairs"] = [(mapping[p["good"]], mapping[p["bad"]], p["margin"])
                                   for p in record.get("regret_pairs", [])]
                assert all(math.isfinite(x) for x in record["labeled_values"])
                groups.append(record)
            self.groups[split] = groups
        assert self.groups["train"] and self.groups["dev"]
        assert not ({r["split_component"] for r in self.groups["train"]} & {r["split_component"] for r in self.groups["dev"]})
        assert not ({r["canonical_fen_key"] for r in self.groups["train"]} & {r["canonical_fen_key"] for r in self.groups["dev"]})
        # The closed TEST labels/FENs are deliberately not loaded by this class.

    def normalization(self):
        rows = torch.cat([torch.arange(r["offset"], r["offset"] + r["count"]) for r in self.groups["train"]])
        x = self.features[rows]
        mean = x.mean(0)
        std = x.std(0, correction=0)
        inverse = torch.where(std >= 1e-4, std.reciprocal(), torch.ones_like(std))
        return mean, inverse

    def batch(self, groups, temperature):
        n = max(r["count"] for r in groups)
        rows = torch.zeros((len(groups), n), dtype=torch.long)
        valid = torch.zeros((len(groups), n), dtype=torch.bool)
        known = torch.zeros_like(valid)
        values = torch.zeros((len(groups), n))
        soft = torch.tensor([r["soft_eligible"] for r in groups], dtype=torch.bool)
        hard = torch.tensor([r["hard_index"] for r in groups])
        pairs = []
        for i, record in enumerate(groups):
            count = record["count"]
            rows[i, :count] = torch.arange(record["offset"], record["offset"] + count)
            valid[i, :count] = True
            known[i, record["labeled_indices"]] = True
            values[i, record["labeled_indices"]] = torch.tensor(record["labeled_values"])
            pairs.extend((i, good, bad, margin) for good, bad, margin in record["pairs"])
        targets = torch.softmax((values / temperature).masked_fill(~known, -1e9), dim=-1)
        return self.features[rows], valid, hard, known, soft, values, targets, pairs


def objective(logits, valid, hard, known, soft, values, targets, pairs, temperature):
    logits = logits.masked_fill(~valid, -1e9)
    loss = 0.3 * F.cross_entropy(logits, hard)
    if soft.any():
        # For truncated top-K packets this normalizes ONLY within observed
        # labels. Unobserved actions receive neither invented zeros nor ranks.
        logp = F.log_softmax(logits[soft].masked_fill(~known[soft], -1e9), dim=-1)
        loss = loss + 0.2 * (-(targets[soft] * logp).sum(-1)).mean()
        packet_best = values[soft].masked_fill(~known[soft], -1e9).argmax(-1, keepdim=True)
        center = logits[soft].gather(1, packet_best)
        best_score = values[soft].masked_fill(~known[soft], -1e9).max(-1, keepdim=True).values
        gap = ((best_score - values[soft]) / temperature).clamp(0, 8)
        observed = known[soft]
        if observed.any():
            loss = loss + 0.05 * F.smooth_l1_loss((center - logits[soft])[observed], gap[observed])
    if pairs:
        group, good, bad, margin = zip(*pairs)
        penalty = F.softplus(logits[list(group), list(bad)] - logits[list(group), list(good)] + torch.tensor(margin))
        loss = loss + penalty.mean()
    return loss


@torch.no_grad()
def evaluate(model, data, batch_size, temperature):
    model.eval()
    correct = top3 = covered = base_changes = 0
    ce = regret = 0.0
    full_regret = []
    by_game = defaultdict(list)
    pair_loss = pair_fail = pair_bad_selected = pair_count = 0
    selection_sum = 0.0
    for start in range(0, len(data.groups["dev"]), batch_size):
        groups = data.groups["dev"][start:start + batch_size]
        x, valid, hard, known, soft, values, targets, pairs = data.batch(groups, temperature)
        logits, base = model(x, parts=True)
        logits = logits.masked_fill(~valid, -1e9)
        base = base.masked_fill(~valid, -1e9)
        selected = logits.argmax(-1)
        correct += (selected == hard).sum().item()
        top3 += (logits.topk(min(3, logits.shape[1]), dim=-1).indices == hard[:, None]).any(-1).sum().item()
        base_changes += (selected != base.argmax(-1)).sum().item()
        ce += F.cross_entropy(logits, hard, reduction="sum").item()
        selection_sum += objective(logits, valid, hard, known, soft, values, targets, [], temperature).item() * len(groups)
        for group, good, bad, margin in pairs:
            pair_count += 1
            pair_loss += F.softplus(logits[group, bad] - logits[group, good] + margin).item()
            pair_fail += int(logits[group, bad] >= logits[group, good])
            pair_bad_selected += int(selected[group] == bad)
        for i, r in enumerate(groups):
            by_game[r["split_component"]].append(float(selected[i] == hard[i]))
            choice = selected[i].item()
            label_index = {index: j for j, index in enumerate(r["labeled_indices"])}
            # Numerical regret uses ordinary comparable labels only, and only
            # where the selected action is actually labeled. Missing top-K
            # entries are unknown, not silently assigned the worst score.
            ordinary = all(x.get("mate") is None for x in r["moves"])
            if r["soft_eligible"] and ordinary and choice in label_index:
                gap = max(r["labeled_values"]) - r["labeled_values"][label_index[choice]]
                regret += gap
                covered += 1
                if r["all_legal"]:
                    full_regret.append(gap)
    n = len(data.groups["dev"])
    return {"roots": n, "hard_ce": ce / n, "top1": correct / n, "top3": top3 / n,
            "cost_sensitive_selection": selection_sum / n + (pair_loss / pair_count if pair_count else 0),
            "regret_pairs": pair_count, "pair_softplus": pair_loss / pair_count if pair_count else None,
            "pair_order_fail_fraction": pair_fail / pair_count if pair_count else None,
            "known_bad_pair_selected_fraction": pair_bad_selected / pair_count if pair_count else None,
            "macro_component_top1": float(np.mean([np.mean(x) for x in by_game.values()])),
            "base_full_choice_changes": base_changes,
            "ordinary_covered_regret_cp": regret / covered if covered else None,
            "ordinary_regret_covered_roots": covered,
            "complete_support_regret_cp": float(np.mean(full_regret)) if full_regret else None,
            "complete_support_regret_roots": len(full_regret), "closed_test_opened": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--features", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--architecture", choices=["interaction", "additive", "flat"], default="interaction")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=0.002)
    parser.add_argument("--temperature", type=float, default=100)
    parser.add_argument("--seed", type=int, default=83137)
    parser.add_argument("--cpu", type=int, default=0)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    assert torch.version.cuda is None and torch.version.hip is None
    data = Dataset(args.labels, args.features)
    assert data.feature_manifest.get("schema_sha256"), "schema fingerprint required"
    mean, inverse = data.normalization()
    model = ChannelPolicy(data.channels, data.dim, args.architecture, mean, inverse)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.001)
    args.out.mkdir(parents=True, exist_ok=False)
    run = {"arguments": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
           "parameters": sum(p.numel() for p in model.parameters()),
           "torch": torch.__version__, "device": "cpu", "cuda": None, "hip": None,
           "feature_manifest_sha256": sha(args.features / "manifest.json"),
           "feature_schema_sha256": data.feature_manifest.get("schema_sha256"),
           "label_manifest_sha256": sha(args.labels / "manifest.json"),
           "source_sha256": {p.name: sha(p) for p in [Path(__file__), Path(__file__).with_name("model.py")]},
           "train_roots": len(data.groups["train"]), "dev_roots": len(data.groups["dev"]),
           "selection": "minimum DEV weighted hard CE(0.3)+observed soft CE(0.2)+gap Huber(0.05)+large-regret pair softplus(1); closed TEST untouched",
           "pair_qualification": "CP gap>=200 and both depth>=8; mate outcome classes separately; forced64k vs packet128k approximate, not proven regret",
           "strength_claim": False, "runtime": "direct action logits only; teacher offline"}
    (args.out / "run.json").write_text(json.dumps(run, indent=2) + "\n")
    best = math.inf
    best_epoch = 0
    started = time.monotonic()
    log = (args.out / "epochs.jsonl").open("w")
    for epoch in range(1, args.epochs + 1):
        model.train()
        groups = list(data.groups["train"])
        random.shuffle(groups)
        total = 0.0
        for start in range(0, len(groups), args.batch):
            batch = groups[start:start + args.batch]
            x, valid, hard, known, soft, values, targets, pairs = data.batch(batch, args.temperature)
            optimizer.zero_grad(set_to_none=True)
            loss = objective(model(x), valid, hard, known, soft, values, targets, pairs, args.temperature)
            assert torch.isfinite(loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.item() * len(batch)
        dev = evaluate(model, data, args.batch, args.temperature)
        record = {"epoch": epoch, "train_loss": total / len(groups), "dev": dev, "seconds": time.monotonic() - started}
        log.write(json.dumps(record) + "\n"); log.flush()
        print(json.dumps(record), flush=True)
        if dev["cost_sensitive_selection"] < best - 1e-5:
            best, best_epoch = dev["cost_sensitive_selection"], epoch
            metadata = {**run, "epoch": epoch, "dev": dev}
            exported = model.export(metadata)
            probe = data.features[:min(128, len(data.features))]
            actual = model(probe).detach().numpy()
            expected, _ = exported_forward(exported, probe.numpy())
            error = float(np.max(np.abs(actual - expected)))
            assert error < 2e-3, error
            metadata["export_max_abs_error"] = error
            temporary = args.out / "best.tmp.json"
            temporary.write_text(json.dumps(exported, separators=(",", ":")) + "\n")
            temporary.replace(args.out / "best.json")
            torch.save(model.state_dict(), args.out / "best-state.pt")
        if epoch - best_epoch >= args.patience:
            break
    log.close()
    result = {**run, "best_epoch": best_epoch, "best_dev_cost_sensitive_selection": best,
              "elapsed_seconds": time.monotonic() - started, "model_sha256": sha(args.out / "best.json")}
    (args.out / "summary.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
