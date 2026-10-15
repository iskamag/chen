#!/usr/bin/env python3
"""Frozen small learned latent-depth control; labels only train/evaluate offline."""
import os
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import argparse
import hashlib
import json
from pathlib import Path
import random
import time
import numpy as np
import torch
from torch import nn

HERE = Path(__file__).resolve().parent
DEPTHS = (1, 2, 4, 8, 16, 32)
VARIANTS = ("feedback", "reset_every_round", "zero_coupling",
            "independent_target_alignment")
WIDTH = 24
SLOTS = 10


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def pack(rows):
    """Input encoder reads only columns, dimensions and id for corruption RNG."""
    adjacency = np.zeros((len(rows), SLOTS, SLOTS), dtype=np.float32)
    resource_mask = np.zeros((len(rows), SLOTS), dtype=np.float32)
    target_mask = np.zeros_like(resource_mask)
    corrupt = np.zeros_like(adjacency)
    for i, row in enumerate(rows):
        nr, nt = row["resources"], row["targets"]
        assert 0 < nr <= SLOTS and 0 < nt <= SLOTS
        assert len(row["columns"]) == nt
        resource_mask[i, :nr] = 1
        target_mask[i, :nt] = 1
        for t, column in enumerate(row["columns"]):
            assert 0 <= column < 1 << nr
            adjacency[i, :nr, t] = [(column >> d) & 1 for d in range(nr)]
            key = hashlib.sha256(f'{row["id"]}:{t}'.encode()).digest()
            order = np.random.default_rng(int.from_bytes(key[:8], "little")).permutation(nr)
            corrupt[i, :nr, t] = adjacency[i, order, t]
    resource_features = np.stack([resource_mask, adjacency.sum(2) / 10,
                                  resource_mask], axis=2)
    target_features = np.stack([target_mask, adjacency.sum(1) / 10,
                                target_mask], axis=2)
    glob = np.stack([target_mask.sum(1) / 10, resource_mask.sum(1) / 10,
                     adjacency.sum((1, 2)) / 100], axis=1)
    return tuple(torch.from_numpy(x) for x in
                 (adjacency, resource_features, target_features,
                  resource_mask, target_mask, glob, corrupt))


def subset(data, indices):
    return tuple(x[indices] for x in data)


class Probe(nn.Module):
    def __init__(self):
        super().__init__()
        self.resource_initial = nn.Linear(3, WIDTH)
        self.target_initial = nn.Linear(3, WIDTH)
        self.resource_message = nn.Linear(WIDTH, WIDTH)
        self.target_message = nn.Linear(WIDTH, WIDTH)
        self.resource_cell = nn.GRUCell(2 * WIDTH, WIDTH)
        self.target_cell = nn.GRUCell(2 * WIDTH, WIDTH)
        self.readout = nn.Sequential(nn.Linear(4 * WIDTH + 3, 48), nn.ReLU(),
                                     nn.Linear(48, 1))

    def snapshots(self, data, depths, variant="feedback"):
        adj, rf, tf, rm, tm, glob, corrupt = data
        if variant == "independent_target_alignment":
            adj = corrupt
        r0 = torch.relu(self.resource_initial(rf)) * rm[..., None]
        t0 = torch.relu(self.target_initial(tf)) * tm[..., None]
        r, t = r0, t0
        output, times = {}, {}
        tick = time.perf_counter()
        for depth in range(1, max(depths) + 1):
            if variant == "reset_every_round":
                r, t = r0, t0
            if variant == "zero_coupling":
                mr, mt = torch.zeros_like(r), torch.zeros_like(t)
            else:
                mr = torch.bmm(adj, torch.relu(self.target_message(t))) / 10
                mt = torch.bmm(adj.transpose(1, 2), torch.relu(self.resource_message(r))) / 10
            # Synchronous: neither update sees the other's newly updated state.
            rnew = self.resource_cell(torch.cat([r0, mr], 2).flatten(0, 1),
                                      r.flatten(0, 1)).reshape_as(r)
            tnew = self.target_cell(torch.cat([t0, mt], 2).flatten(0, 1),
                                    t.flatten(0, 1)).reshape_as(t)
            r, t = rnew * rm[..., None], tnew * tm[..., None]
            if depth in depths:
                pooled = torch.cat([r.sum(1) / 10, t.sum(1) / 10,
                                    r.masked_fill(~rm.bool()[..., None], -1e9).amax(1),
                                    t.masked_fill(~tm.bool()[..., None], -1e9).amax(1), glob], 1)
                output[depth] = self.readout(pooled).squeeze(1)
                times[depth] = time.perf_counter() - tick
        return output, times

    def forward(self, data, depth):
        return self.snapshots(data, (depth,))[0][depth]


def metrics(logits, labels):
    logits = np.asarray(logits, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    probabilities = 1 / (1 + np.exp(-np.clip(logits, -80, 80)))
    predicted = logits >= 0
    positives, negatives = int(labels.sum()), int((~labels).sum())
    fp = int(np.count_nonzero(predicted & ~labels))
    fn = int(np.count_nonzero(~predicted & labels))
    return {"count": len(labels), "positives": positives, "negatives": negatives,
            "accuracy": float(np.mean(predicted == labels)),
            "balanced_accuracy": (1 - (fp / negatives + fn / positives) / 2
                                  if positives and negatives else None),
            "false_positives": fp, "false_negatives": fn,
            "brier": float(np.mean((probabilities - labels) ** 2)),
            "logloss": float(np.mean(np.logaddexp(0, logits) - labels * logits)),
            "mean_probability": float(probabilities.mean())}


@torch.no_grad()
def predict(model, data, depths=DEPTHS, variant="feedback", batch=256):
    model.eval()
    logits = {d: [] for d in depths}
    times = {d: 0.0 for d in depths}
    for start in range(0, len(data[0]), batch):
        output, elapsed = model.snapshots(subset(data, slice(start, start + batch)), depths, variant)
        for depth in depths:
            logits[depth].append(output[depth].numpy())
            times[depth] += elapsed[depth]
    return {d: np.concatenate(logits[d]) for d in depths}, times


def train_one(data, labels, rows, regime, seed, out):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model = Probe()
    train = np.array([i for i, r in enumerate(rows) if r["split"] == "train"])
    dev = np.array([i for i, r in enumerate(rows) if r["split"] == "dev"])
    train_y = labels[train]
    positive_weight = float((1 - train_y).sum() / train_y.sum())
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight))
    optimizer = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.0001)
    dev_data = subset(data, dev)
    rng = np.random.default_rng(seed)
    best_key = None
    checkpoint = out / f"{regime}-seed{seed}.pt"
    log, selected_step = [], None
    tick = time.perf_counter()
    for step in range(1, 601):
        model.train()
        indices = train[rng.integers(0, len(train), size=128)]
        depth = 4 if regime == "fixed4" else int(rng.integers(1, 9))
        optimizer.zero_grad(set_to_none=True)
        loss = loss_fn(model(subset(data, indices), depth), torch.from_numpy(labels[indices]))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 100 == 0:
            dev_depth = 4 if regime == "fixed4" else 8
            logits, _ = predict(model, dev_data, (dev_depth,))
            stat = metrics(logits[dev_depth], labels[dev])
            key = (stat["balanced_accuracy"], -stat["logloss"])
            if best_key is None or key > best_key:
                best_key, selected_step = key, step
                torch.save(model.state_dict(), checkpoint)
            record = {"step": step, "depth": depth, "train_batch_loss": float(loss.detach()),
                      "dev": stat, "selected_step_so_far": selected_step,
                      "elapsed_seconds": time.perf_counter() - tick}
            log.append(record)
            print(json.dumps({"regime": regime, "seed": seed, **record}), flush=True)
    # Reload frozen selected weights. Save/load itself is checked numerically.
    loaded = Probe()
    loaded.load_state_dict(torch.load(checkpoint, weights_only=True))
    with torch.no_grad():
        small = subset(dev_data, slice(0, 8))
        before = loaded(small, 4).clone()
        torch.save(loaded.state_dict(), out / "numeric-roundtrip.pt")
        replay = Probe()
        replay.load_state_dict(torch.load(out / "numeric-roundtrip.pt", weights_only=True))
        after = replay(small, 4)
        error = float((before - after).abs().max())
        assert error == 0
    (out / "numeric-roundtrip.pt").unlink()
    return {"regime": regime, "seed": seed, "selected_step": selected_step,
            "checkpoint": checkpoint.name, "checkpoint_sha256": sha(checkpoint),
            "parameters": sum(p.numel() for p in loaded.parameters()),
            "positive_weight": positive_weight, "roundtrip_max_logit_error": error,
            "train_seconds": time.perf_counter() - tick, "training_log": log}


@torch.no_grad()
def permutation_check(model, rows):
    changed = []
    for row in rows:
        nr, nt = row["resources"], row["targets"]
        # Reverse both node orders; feature encoder must transform with them.
        columns = [sum(((column >> d) & 1) << (nr - 1 - d) for d in range(nr))
                   for column in reversed(row["columns"])]
        changed.append({**row, "columns": columns})
    original = predict(model, pack(rows), (8,))[0][8]
    permuted = predict(model, pack(changed), (8,))[0][8]
    error = float(np.max(np.abs(original - permuted)))
    assert error < 1e-5, error
    return error


def evaluate(fit, data, labels, rows, out):
    model = Probe()
    model.load_state_dict(torch.load(out / fit["checkpoint"], weights_only=True))
    selected = np.array([i for i, r in enumerate(rows) if r["split"] != "train"])
    selected_rows = [rows[i] for i in selected]
    selected_labels, selected_data = labels[selected], subset(data, selected)
    groups = {}
    for split in sorted({r["split"] for r in selected_rows}):
        groups[f"{split}/ALL"] = [i for i, r in enumerate(selected_rows) if r["split"] == split]
        for family in sorted({r["family"] for r in selected_rows if r["split"] == split}):
            groups[f"{split}/{family}"] = [i for i, r in enumerate(selected_rows)
                                           if r["split"] == split and r["family"] == family]
    records, transition_records, arrays = [], [], []
    for variant in VARIANTS:
        logits, elapsed = predict(model, selected_data, variant=variant)
        arrays.append(np.stack([logits[d] for d in DEPTHS]))
        for group, indices in groups.items():
            idx = np.array(indices)
            for depth in DEPTHS:
                records.append({"variant": variant, "depth": depth, "group": group,
                                **metrics(logits[depth][idx], selected_labels[idx])})
            for previous, depth in zip(DEPTHS, DEPTHS[1:]):
                a, b = logits[previous][idx], logits[depth][idx]
                transition_records.append({"variant": variant, "from_depth": previous,
                    "to_depth": depth, "group": group,
                    "classification_flips": int(np.count_nonzero((a >= 0) != (b >= 0))),
                    "mean_abs_probability_change": float(np.mean(np.abs(
                        1 / (1 + np.exp(-a)) - 1 / (1 + np.exp(-b)))))})
        if variant == "reset_every_round":
            assert all(np.array_equal(logits[1], logits[d]) for d in DEPTHS)
        fit.setdefault("prefix_cumulative_inference_seconds", {})[variant] = elapsed
    stem = f'{fit["regime"]}-seed{fit["seed"]}'
    predictions = out / f"{stem}-logits.npz"
    np.savez_compressed(predictions, logits=np.stack(arrays),
                        ids=np.array([r["id"] for r in selected_rows]),
                        labels=selected_labels, depths=np.array(DEPTHS),
                        variants=np.array(VARIANTS))
    fit.update({"metrics": records, "transitions": transition_records,
                "predictions": predictions.name, "predictions_sha256": sha(predictions),
                "coherent_permutation_max_logit_error": permutation_check(
                    model, [r for r in rows if r["split"] == "dev"][:16])})
    print(f"evaluated frozen {stem}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--cpu", type=int, default=1)
    parser.add_argument("--replay", action="store_true", help="Use existing frozen checkpoints; no training")
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    torch.use_deterministic_algorithms(True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = json.loads(Path(args.corpus).read_text())["rows"]
    data = pack(rows)
    labels = np.array([r["feasible"] for r in rows], dtype=np.float32)
    tick = time.perf_counter()
    if args.replay:
        summary = json.loads((out / "results.json").read_text())
        assert summary["corpus_sha256"] == sha(args.corpus)
        fits = summary["fits"]
        for fit in fits:
            assert sha(out / fit["checkpoint"]) == fit["checkpoint_sha256"]
    else:
        fits = []
        for regime in ("fixed4", "variable1to8"):
            for seed in (17, 29):
                fits.append(train_one(data, labels, rows, regime, seed, out))
                (out / "frozen-checkpoints.json").write_text(json.dumps(fits, indent=2) + "\n")
    # No EVAL/STRUCTURE predictions are computed before all4 checkpoints freeze.
    for fit in fits:
        evaluate(fit, data, labels, rows, out)
    summary = {"protocol_sha256": sha(HERE / "protocol.json"), "source_sha256": sha(__file__),
               "corpus_sha256": sha(args.corpus), "torch": torch.__version__,
               "numpy": np.__version__, "cpu_affinity": sorted(os.sched_getaffinity(0)),
               "threads": torch.get_num_threads(), "replay_only": args.replay,
               "rounds": DEPTHS, "variants": VARIANTS, "fits": fits,
               "total_seconds": time.perf_counter() - tick}
    (out / "results.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(f"complete {out}; seconds={summary['total_seconds']:.3f}", flush=True)


if __name__ == "__main__":
    main()
