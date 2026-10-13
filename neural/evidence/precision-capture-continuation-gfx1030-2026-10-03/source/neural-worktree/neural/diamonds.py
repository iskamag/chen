"""Oracle-certified quiet chess diamonds and controlled latent-composition smoke.

Run: python -m neural.diamonds --device cpu --samples 16 --steps 4
This experiment tests mechanics and synthetic held-out drift, never strength.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .features import SCHEMA_VERSION, encode_fen, move_to_index
from .models import MODELS, build_model


@dataclass(frozen=True)
class CertifiedDiamond:
    root_fen: str
    origin_fen: str
    root_history: tuple[str, ...]
    paths: tuple[tuple[str, ...], tuple[str, ...]]
    path_fens: tuple[tuple[str, ...], tuple[str, ...]]
    endpoint_fen: str
    draw_context: tuple[bool, ...]


@dataclass(frozen=True)
class DiamondCorpus:
    features: torch.Tensor  # [B,2,5,18,8,8], includes exact root and every ply
    actions: torch.Tensor   # [B,2,4], each ply in that ply's canonical frame
    records: tuple[CertifiedDiamond, ...]
    manifest: dict[str, object]

    def to(self, device: str | torch.device) -> DiamondCorpus:
        return DiamondCorpus(self.features.to(device), self.actions.to(device), self.records, self.manifest)


def feature_identity(fen: str) -> str:
    """Identity of the actual schema-1 model input, independent of FEN spelling."""
    return hashlib.sha256(json.dumps(encode_fen(fen).flatten().tolist()).encode()).hexdigest()


def _draw_context(board) -> tuple[bool, ...]:
    return (board.is_repetition(2), board.is_repetition(3), board.can_claim_threefold_repetition(),
            board.can_claim_fifty_moves(), board.is_fivefold_repetition(), board.is_seventyfive_moves())


def _safe_node(board):
    if board.is_check() or board.is_game_over(claim_draw=True) or any(_draw_context(board)):
        raise ValueError("check, terminal, repeated, or draw-claim state rejected")


def _quiet_moves(board):
    import chess
    if board.is_check():
        return []
    result = []
    for move in board.legal_moves:
        piece = board.piece_at(move.from_square)
        if piece.piece_type in (chess.PAWN, chess.KING) or board.is_capture(move) or board.gives_check(move) or move.promotion:
            continue
        rights = board.castling_rights
        board.push(move)
        unchanged = board.castling_rights == rights
        board.pop()
        if unchanged:
            result.append(move)
    return sorted(result, key=lambda move: move.uci())


def certify_diamond(root, path_a: tuple[str, ...], path_b: tuple[str, ...]) -> CertifiedDiamond:
    """Reject presumed commutation unless both histories pass the exact oracle.

    Retains root move-stack. Reject any repeated root-history state and roots
    within ten plies of the fifty-move threshold. FEN alone cannot certify equal
    future repetition histories; this function certifies an unclaimed current
    context, preserves histories, and refuses near-boundary examples.
    """
    import chess
    if len(path_a) != 4 or len(path_b) != 4 or tuple(path_b) != (path_a[2], path_a[3], path_a[0], path_a[1]) or tuple(path_a) == tuple(path_b):
        raise ValueError("expected two distinct per-color swapped four-ply paths")
    if not root.is_valid() or root.halfmove_clock >= 90:
        raise ValueError("invalid root or fifty-move boundary")
    history = root.copy(stack=True)
    while True:
        if history.is_repetition(2):
            raise ValueError("root history includes a repeated state")
        if not history.move_stack:
            break
        history.pop()
    _safe_node(root)
    endings, fens = [], []
    for path in (path_a, path_b):
        board = root.copy(stack=True)
        path_fens = [board.fen(en_passant="fen")]
        for uci in path:
            move = chess.Move.from_uci(uci)
            if move not in _quiet_moves(board):
                raise ValueError(f"illegal or unsafe commutation move {uci}")
            board.push(move)
            _safe_node(board)
            path_fens.append(board.fen(en_passant="fen"))
        endings.append(board)
        fens.append(tuple(path_fens))
    if endings[0].fen(en_passant="legal") != endings[1].fen(en_passant="legal") or fens[0][-1] != fens[1][-1] or _draw_context(endings[0]) != _draw_context(endings[1]):
        raise ValueError("endpoints differ in full effective-EP FEN or current draw context")
    return CertifiedDiamond(root.fen(en_passant="fen"), root.root().fen(en_passant="fen"), tuple(move.uci() for move in root.move_stack),
                            (tuple(path_a), tuple(path_b)), tuple(fens), endings[0].fen(en_passant="legal"), _draw_context(endings[0]))


def generate_diamonds(count: int, seed: int = 20260930, max_root_plies: int = 32,
                      device: str | torch.device = "cpu") -> DiamondCorpus:
    """Generate distinct diamonds from independently sampled reachable roots."""
    import chess
    if count < 1 or max_root_plies < 1:
        raise ValueError("positive count and root-ply limit required")
    rng, records, seen = random.Random(seed), [], set()
    attempts = 0
    while len(records) < count and attempts < count * 400:
        attempts += 1
        root = chess.Board()
        for _ in range(rng.randrange(max_root_plies + 1)):
            if root.is_game_over(claim_draw=True):
                break
            root.push(rng.choice(sorted(root.legal_moves, key=lambda move: move.uci())))
        board, path = root.copy(stack=True), []
        for _ in range(4):
            moves = _quiet_moves(board)
            if not moves:
                break
            move = rng.choice(moves)
            path.append(move.uci())
            board.push(move)
        if len(path) != 4:
            continue
        alternate = (path[2], path[3], path[0], path[1])
        try:
            record = certify_diamond(root, tuple(path), alternate)
        except ValueError:
            continue
        identity = (record.root_fen, tuple(sorted(record.paths)))
        if identity in seen:
            continue
        seen.add(identity)
        records.append(record)
    if len(records) != count:
        raise RuntimeError(f"found only {len(records)}/{count} legal diamonds after {attempts} roots")
    features = torch.stack([torch.stack([torch.stack([encode_fen(fen) for fen in path]) for path in record.path_fens]) for record in records])
    actions = torch.tensor([[[move_to_index(move, record.path_fens[path][ply].split()[1]) for ply, move in enumerate(moves)]
                             for path, moves in enumerate(record.paths)] for record in records], dtype=torch.long)
    digest = hashlib.sha256(json.dumps([asdict(record) for record in records], sort_keys=True).encode()).hexdigest()
    manifest = {"kind": "oracle_quiet_legal_diamonds", "schema_version": SCHEMA_VERSION, "seed": seed,
                "count": count, "max_root_plies": max_root_plies, "attempted_roots": attempts,
                "oracle": f"python-chess {chess.__version__}", "sha256": digest,
                "feature_ep": "raw FEN target, matching schema 1 replay", "endpoint_equivalence_ep": "effective legal target",
                "strength_evidence": False, "history_scope": "retained root history, no repeated states or current/next-move claims; future histories not interchangeable"}
    return DiamondCorpus(features, actions, tuple(records), manifest).to(device)


def _reconstruction(model, latent, features):
    board = F.binary_cross_entropy_with_logits(model.decode_board(latent), features[:, :12])
    rules = model.decode_rules(latent)
    rights = F.binary_cross_entropy_with_logits(rules["castling_logits"], features[:, 12:16, 0, 0])
    ep_plane = features[:, 16].flatten(1)
    ep = F.cross_entropy(rules["ep_logits"], torch.where(ep_plane.sum(-1) > 0, ep_plane.argmax(-1), 64))
    clock = F.mse_loss(rules["halfmove"], features[:, 17, 0, 0])
    return board, rights + ep + clock


def diamond_losses(model, features: torch.Tensor, actions: torch.Tensor) -> dict[str, torch.Tensor]:
    """Resident differentiable ordinary supervision + four-ply refresh/composition.

    No host extraction occurs here. Refresh targets are detached exact encodings;
    board/rule reconstruction anchors representation against trivial collapse.
    """
    if features.ndim != 6 or tuple(features.shape[1:]) != (2, 5, 18, 8, 8) or actions.shape != (features.shape[0], 2, 4):
        raise ValueError("expected features [B,2,5,18,8,8], actions [B,2,4]")
    batch = features.shape[0]
    exact = model.encode(features.reshape(-1, 18, 8, 8)).reshape(batch, 2, 5, 64, model.width)
    one_step = model.transition(exact[:, :, :4].reshape(-1, 64, model.width), actions.reshape(-1))
    one_board, one_rules = _reconstruction(model, one_step, features[:, :, 1:].reshape(-1, 18, 8, 8))
    current_board, current_rules = _reconstruction(model, exact.reshape(-1, 64, model.width), features.reshape(-1, 18, 8, 8))
    ordinary = one_board + current_board + 0.1 * (one_rules + current_rules)
    ordinary = ordinary + F.mse_loss(F.normalize(one_step, dim=-1), F.normalize(exact[:, :, 1:].reshape(-1, 64, model.width).detach(), dim=-1))
    roots = exact[:, :, 0].reshape(-1, 64, model.width)
    endpoints = model.rollout(roots, actions.reshape(-1, 4))[:, -1].reshape(batch, 2, 64, model.width)
    target = exact[:, :, -1].detach()
    refresh = F.mse_loss(F.normalize(endpoints, dim=-1), F.normalize(target, dim=-1))
    composition = F.mse_loss(F.normalize(endpoints[:, 0], dim=-1), F.normalize(endpoints[:, 1], dim=-1))
    endpoint_flat = endpoints.reshape(-1, 64, model.width)
    endpoint_features = features[:, :, -1].reshape(-1, 18, 8, 8)
    endpoint_board, endpoint_rules = _reconstruction(model, endpoint_flat, endpoint_features)
    with torch.no_grad():
        piece_logits = model.decode_board(endpoint_flat)
        confidence, predicted_piece = piece_logits.max(1)
        predicted_piece = torch.where(confidence > 0, predicted_piece, 12)
        occupied = endpoint_features[:, :12].sum(1) > 0
        expected_piece = torch.where(occupied, endpoint_features[:, :12].argmax(1), 12)
        correct_piece = (predicted_piece == expected_piece).float()
        rules = model.decode_rules(endpoint_flat)
        ep_plane = endpoint_features[:, 16].flatten(1)
        expected_ep = torch.where(ep_plane.sum(-1) > 0, ep_plane.argmax(-1), 64)
        metrics = {"endpoint_piece_accuracy": correct_piece.mean(),
                   "endpoint_occupied_piece_accuracy": (correct_piece * occupied).sum() / occupied.sum().clamp_min(1),
                   "endpoint_castling_accuracy": ((rules["castling_logits"] > 0) == (endpoint_features[:, 12:16, 0, 0] > 0)).float().mean(),
                   "endpoint_ep_accuracy": (rules["ep_logits"].argmax(-1) == expected_ep).float().mean(),
                   "endpoint_clock_mae": (rules["halfmove"] - endpoint_features[:, 17, 0, 0]).abs().mean()}
        first_steps = F.normalize(one_step.reshape(batch, 2, 4, 64, model.width)[:, :, 0], dim=-1)
        exact_first = F.normalize(exact[:, :, 1], dim=-1)
        separation = F.mse_loss(first_steps[:, 0], first_steps[:, 1])
        exact_separation = F.mse_loss(exact_first[:, 0], exact_first[:, 1])
        metrics.update(legal_action_latent_separation=separation,
                       exact_legal_action_latent_separation=exact_separation,
                       action_separation_ratio=separation / exact_separation.clamp_min(1e-12),
                       state_latent_variance=exact[:, 0, 0].var(dim=0, unbiased=False).mean())
    return {"ordinary": ordinary, "refresh": refresh, "composition": composition,
            "endpoint_board": endpoint_board, "endpoint_rules": endpoint_rules,
            "spatial_latent_variance": exact.var(dim=3).mean(), **metrics}


def run_smoke(*, device="cpu", samples=16, steps=4, batch_size=4, width=16, depth=1,
              models=MODELS, seed=20260930, max_root_plies=32, output=None):
    if samples < 2 or steps < 1 or batch_size < 1:
        raise ValueError("need samples>=2, steps>=1, batch_size>=1")
    train = generate_diamonds(samples, seed, max_root_plies, device)
    heldout = generate_diamonds(max(2, samples // 4), seed + 1, max_root_plies, device)
    # Reject root-state overlap, independent of the ordering or sampling seed.
    # Fullmove number and absolute color are invisible to schema 1. Compare
    # canonical features, not FEN spelling or unencoded repetition history.
    train_states = {feature_identity(fen) for record in train.records for path in record.path_fens for fen in path}
    keep = [index for index, record in enumerate(heldout.records)
            if not train_states.intersection(feature_identity(fen) for path in record.path_fens for fen in path)]
    if len(keep) != len(heldout.records):
        # Initial-root fixtures recur naturally; held-out policy excludes those roots.
        if not keep:
            raise RuntimeError("held-out roots overlap; choose a larger corpus or another seed")
        indexes = torch.tensor(keep, device=device)
        retained_records = tuple(heldout.records[i] for i in keep)
        digest = hashlib.sha256(json.dumps([asdict(record) for record in retained_records], sort_keys=True).encode()).hexdigest()
        heldout = DiamondCorpus(heldout.features[indexes], heldout.actions[indexes], retained_records,
                                {**heldout.manifest, "source_sha256": heldout.manifest["sha256"], "sha256": digest,
                                 "filtered_state_overlap": len(heldout.records) - len(keep), "count": len(keep)})
    generator = torch.Generator(device=device).manual_seed(seed + 2)
    schedule = torch.randint(samples, (steps, batch_size), device=device, generator=generator)
    results = []
    for name in models:
        for control in ("ordinary", "refresh", "diamond"):
            torch.manual_seed(seed)
            model = build_model(name, width=width, depth=depth).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
            with torch.no_grad():
                before = {key: float(value) for key, value in diamond_losses(model, heldout.features, heldout.actions).items()}
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            start = time.perf_counter()
            for indexes in schedule:
                losses = diamond_losses(model, train.features[indexes], train.actions[indexes])
                loss = losses["ordinary"]
                if control != "ordinary":
                    loss = loss + losses["refresh"] + 0.25 * losses["endpoint_board"] + 0.025 * losses["endpoint_rules"]
                if control == "diamond":
                    loss = loss + losses["composition"]
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1)
                optimizer.step()
            if torch.device(device).type == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
            with torch.no_grad():
                after = {key: float(value) for key, value in diamond_losses(model, heldout.features, heldout.actions).items()}
            if not all(torch.isfinite(torch.tensor(list(after.values())))):
                raise RuntimeError("nonfinite diamond result")
            results.append({"model": name, "control": control, "parameters": sum(p.numel() for p in model.parameters()),
                            "seconds": elapsed, "heldout_before": before, "heldout_after": after,
                            "peak_allocated_bytes": torch.cuda.max_memory_allocated() if torch.device(device).type == "cuda" else None})
    report = {"kind": "quiet_diamond_mechanics", "strength_evidence": False,
              "environment": {"torch": torch.__version__, "hip": torch.version.hip, "device": str(device)},
              "source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("diamonds.py", "models.py", "features.py")},
              "train": train.manifest, "heldout": heldout.manifest,
              "corpus_records": {"train": [asdict(record) for record in train.records],
                                 "heldout": [asdict(record) for record in heldout.records]},
              "config": {"seed": seed, "steps": steps, "batch_size": batch_size, "width": width, "depth": depth},
              "sample_schedule_sha256": hashlib.sha256(json.dumps(schedule.cpu().tolist()).encode()).hexdigest(), "results": results,
              "limitations": ["tiny synthetic quiet-only corpus; no strength inference", "seeded, not guaranteed deterministic kernels",
                              "equal sample schedule, different backward graphs/time", "endpoint history maps differ despite safe current draw context"]}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--samples", type=int, default=16)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--width", type=int, default=16)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-root-plies", type=int, default=32)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--output")
    print(json.dumps(run_smoke(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
