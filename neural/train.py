"""Reproducible architecture smoke comparisons on identical legal toy data.

Run ``python -m neural.train --device cuda --models square relation plan``.
This is a mechanics check, not a strength run or a learned chess engine.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import time

import torch
from torch.nn import functional as F

from .data import DeviceReplay, generate_synthetic
from .features import SCHEMA_VERSION
from .models import MODELS, build_model, checkpoint_buffers


def seed_all(seed: int):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def loss_terms(model, batch):
    """Rule-supervised mechanics loss with no claimed WDL outcomes or teacher search."""
    outputs = model(batch["features"], batch["legal_mask"])
    policy = F.cross_entropy(outputs["policy_logits"], batch["target_actions"])
    probabilities = outputs["wdl_logits"].softmax(-1)
    value = F.mse_loss(probabilities[:, 2] - probabilities[:, 0], batch["material_proxy"])
    board = F.binary_cross_entropy_with_logits(outputs["board_logits"], batch["features"][:, :12])
    successor = model.transition(outputs["latent"], batch["actions"])
    # Decode only the board for dynamics loss: avoid constructing an unused policy.
    successor_board = model.decode_board(successor)
    next_board = F.binary_cross_entropy_with_logits(successor_board, batch["next_features"][:, :12])
    def rules_loss(prediction, features):
        castling = F.binary_cross_entropy_with_logits(prediction["castling_logits"], features[:, 12:16, 0, 0])
        ep_plane = features[:, 16].flatten(1)
        ep_target = torch.where(ep_plane.sum(-1) > 0, ep_plane.argmax(-1), 64)
        ep = F.cross_entropy(prediction["ep_logits"], ep_target)
        clock = F.mse_loss(prediction["halfmove"], features[:, 17, 0, 0])
        return castling + ep + clock
    rules = rules_loss(model.decode_rules(outputs["latent"]), batch["features"])
    next_rules = rules_loss(model.decode_rules(successor), batch["next_features"])
    with torch.no_grad():
        target_latent = model.encode(batch["next_features"])
    consistency = F.mse_loss(F.normalize(successor, dim=-1), F.normalize(target_latent, dim=-1))
    total = policy + value + 0.25 * (board + next_board) + 0.1 * (rules + next_rules + consistency)
    return {"total": total, "policy": policy, "material_proxy": value,
            "board": board, "successor_board": next_board, "rules": rules,
            "successor_rules": next_rules, "latent_consistency": consistency}


def train_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    losses = loss_terms(model, batch)
    losses["total"].backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {name: value.detach() for name, value in losses.items()}


def synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def save_checkpoint(model, path, *, corpus_manifest, training_config):
    """Persist an inference-ready smoke snapshot at an explicit artifact boundary."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"name": model.kind, "width": model.width, "depth": model.depth,
              "plan_tokens": model.plans.shape[0] if model.kind == "plan" else 8}
    payload = {"checkpoint_version": 1, "schema_version": SCHEMA_VERSION,
               "kind": "synthetic_legal_smoke_checkpoint", "strength_evidence": False,
               "model_config": config, "corpus_manifest": corpus_manifest,
               "training_config": training_config,
               "state_dict": {name: tensor.detach().cpu().clone()
                              for name, tensor in model.state_dict().items()}}
    torch.save(payload, path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kind": payload["kind"], "strength_evidence": False}


def load_checkpoint(path, *, device="cpu"):
    """Load schema-v1 tensors/primitives only, without an unrestricted pickle fallback.

    Deserialization starts on CPU; only a successfully validated model is moved
    to the requested device. Return (model, metadata), excluding tensor weights.
    """
    path = Path(path)
    if path.stat().st_size > 64 * 1024 * 1024:
        raise ValueError("smoke checkpoint exceeds 64 MiB size limit")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("checkpoint failed restricted tensor-only loading") from error
    if not isinstance(payload, dict) or type(payload.get("checkpoint_version")) is not int or payload["checkpoint_version"] != 1:
        raise ValueError("unsupported checkpoint format")
    if type(payload.get("schema_version")) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("checkpoint feature/action schema mismatch")
    if payload.get("kind") != "synthetic_legal_smoke_checkpoint" or payload.get("strength_evidence") is not False:
        raise ValueError("unsupported checkpoint evidence status")
    config = payload.get("model_config")
    if not isinstance(config, dict) or set(config) != {"name", "width", "depth", "plan_tokens"}:
        raise ValueError("checkpoint model configuration missing or malformed")
    if config["name"] not in MODELS:
        raise ValueError("unknown checkpoint architecture")
    for key, low, high in (("width", 8, 512), ("depth", 1, 32), ("plan_tokens", 1, 128)):
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError(f"checkpoint {key} outside supported smoke limits")
    weights = payload.get("state_dict")
    if not isinstance(weights, dict) or not all(isinstance(k, str) and isinstance(v, torch.Tensor)
                                               for k, v in weights.items()):
        raise ValueError("checkpoint requires a tensor state_dict")
    if not all(bool(torch.isfinite(v).all()) for v in weights.values() if v.is_floating_point()):
        raise ValueError("checkpoint has nonfinite weights")
    for key in ("corpus_manifest", "training_config"):
        if not isinstance(payload.get(key), dict):
            raise ValueError(f"checkpoint missing {key}")
    model = build_model(**config)
    # Geometry and canonical orientation are architecture definitions, not learned
    # checkpoint state. A valid-shaped but corrupted buffer can otherwise pass
    # strict loading and change rules or fail only on a later transition.
    for name, expected in checkpoint_buffers(model):
        actual = weights.get(name)
        if actual is None or actual.dtype != expected.dtype or actual.shape != expected.shape or not torch.equal(actual, expected):
            raise ValueError(f"checkpoint deterministic buffer mismatch: {name}")
    try:
        model.load_state_dict(weights, strict=True)
    except RuntimeError as error:
        raise ValueError("checkpoint weights do not match model configuration") from error
    model.to(device).eval()
    return model, {key: value for key, value in payload.items() if key != "state_dict"}


def run_comparison(*, device="cpu", models=MODELS, samples=192, steps=8,
                   batch_size=16, seed=20260930, width=32, depth=1,
                   max_plies=48, output=None,
                   checkpoint_dir="results/neural/checkpoints", no_checkpoints=False):
    if steps <= 0 or batch_size <= 0:
        raise ValueError("steps and batch size must be positive")
    # Data is generated once, then copied once into a replay shared by all models.
    corpus = generate_synthetic(samples, seed, max_plies=max_plies, device=device)
    replay = DeviceReplay(samples, device)
    replay.append(corpus.tensors)
    device = replay.device
    # All variants consume exactly the same sample-index stream. Independent model
    # initialization is reseeded separately so architecture shape cannot alter data.
    environment = {"torch": torch.__version__, "hip": torch.version.hip,
                   "device": str(device), "seed": seed,
                   "deterministic_algorithms": torch.are_deterministic_algorithms_enabled()}
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment.update(gpu=properties.name,
                           architecture=getattr(properties, "gcnArchName", "unknown"))
    results = []
    for name in models:
        seed_all(seed)
        model = build_model(name, width=width, depth=depth).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        generator = torch.Generator(device=device).manual_seed(seed + 1)
        model.train()
        # A separate warm-up forward does not consume a sampling or optimizer step.
        warmup = {key: value[:min(batch_size, samples)] for key, value in corpus.tensors.items()}
        with torch.no_grad():
            loss_terms(model, warmup)
        synchronize(device)
        start = time.perf_counter()
        first = final = None
        sample_digest = hashlib.sha256()
        for step in range(steps):
            batch = replay.sample(batch_size, generator)
            losses = train_step(model, optimizer, batch)
            if step == 0:
                first = losses
            final = losses
            # Audit rows are retained on device during the timed loop. Their small
            # fingerprint is copied only after timing, never used to feed a batch.
            if step == 0:
                identities = [batch["sequence_ids"]]
            else:
                identities.append(batch["sequence_ids"])
        synchronize(device)
        elapsed = time.perf_counter() - start
        first_values = {key: float(value) for key, value in first.items()}
        final_values = {key: float(value) for key, value in final.items()}
        if not all(torch.isfinite(torch.tensor(list(final_values.values())))):
            raise RuntimeError(f"nonfinite losses for {name}")
        sample_digest.update(json.dumps(torch.stack(identities).cpu().tolist()).encode())
        model.eval()
        with torch.no_grad():
            probe = model(warmup["features"], warmup["legal_mask"])
            chosen = probe["policy_logits"].argmax(-1)
            legal = warmup["legal_mask"].gather(1, chosen[:, None]).all()
            if not bool(legal):
                raise RuntimeError(f"illegal greedy policy action for {name}")
            accuracy = float((chosen == warmup["target_actions"]).float().mean())
        checkpoint = None
        if checkpoint_dir and not no_checkpoints:
            checkpoint = save_checkpoint(model, Path(checkpoint_dir) / f"{name}.pt",
                                         corpus_manifest=corpus.manifest,
                                         training_config={"seed": seed, "steps": steps,
                                                          "batch_size": batch_size,
                                                          "optimizer": "AdamW", "learning_rate": 3e-4,
                                                          "objective": "rule_proxy_v1",
                                                          "loss_weights": {"policy": 1, "material_proxy": 1,
                                                                           "board": 0.25, "successor_board": 0.25,
                                                                           "rules": 0.1, "successor_rules": 0.1,
                                                                           "latent_consistency": 0.1},
                                                          "torch": str(torch.__version__),
                                                          "hip": torch.version.hip,
                                                          "sample_sequence_sha256": sample_digest.hexdigest()})
        results.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                        "width": width, "depth": depth, "steps": steps,
                        "batch_size": batch_size, "seconds": elapsed,
                        "training_positions_per_second": steps * batch_size / elapsed,
                        "first_loss": first_values, "final_loss": final_values,
                        "training_heuristic_policy_accuracy": accuracy,
                        "sample_sequence_sha256": sample_digest.hexdigest(),
                        "checkpoint": checkpoint,
                        "greedy_moves_legal": True})
        del optimizer, model
    if len({row["sample_sequence_sha256"] for row in results}) > 1:
        raise RuntimeError("architecture comparison consumed different replay sequences")
    report = {"kind": "synthetic_legal_smoke", "strength_evidence": False,
              "environment": environment, "data": corpus.manifest,
              "replay": replay.manifest(), "results": results,
              "limitations": ["random legal games, no teacher search or outcome targets",
                              "training-set metrics, no held-out strength evidence",
                              "CPU data ingestion, device-resident replay sampling",
                              "timing includes first optimizer-state allocation",
                              "seeded reproducibility; backend determinism not guaranteed"]}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--samples", type=int, default=192)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--max-plies", type=int, default=48)
    parser.add_argument("--output")
    parser.add_argument("--checkpoint-dir", default="results/neural/checkpoints")
    parser.add_argument("--no-checkpoints", action="store_true")
    args = vars(parser.parse_args())
    print(json.dumps(run_comparison(**args), indent=2))


if __name__ == "__main__":
    main()
