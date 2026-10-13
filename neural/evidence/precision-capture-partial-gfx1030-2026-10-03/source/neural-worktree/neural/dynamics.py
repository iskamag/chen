"""Exact local transport with separately learned strategic move dynamics.

The simulator owns observed state and legal-move effects. Neural prediction never
reads exact successor features; these enter only supervision and observed state.
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
from torch import nn
from torch.nn import functional as F

from .features import encode_fen, move_to_index
from .models import build_model
from .diamonds import DiamondCorpus, feature_identity, generate_diamonds


EFFECT_FIELDS = ("source", "target", "promotion", "captured", "rook_source", "rook_target")


def move_effects(board, move) -> dict[str, int]:
    """Trusted ingress oracle; supports standard chess, excludes Chess960."""
    import chess
    if board.chess960 or move not in board.legal_moves:
        raise ValueError("effects require legal standard-chess move")
    flip = 0 if board.turn else 56
    source, target = move.from_square ^ flip, move.to_square ^ flip
    captured = target
    if board.is_en_passant(move):
        captured = (move.to_square - (8 if board.turn else -8)) ^ flip
    rook_source = rook_target = -1
    if board.is_castling(move):
        rook_source, rook_target = (7, 5) if board.is_kingside_castling(move) else (0, 3)
    promotion = 0 if move.promotion is None else {chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]
    return dict(source=source, target=target, promotion=promotion, captured=captured,
                rook_source=rook_source, rook_target=rook_target)


def batch_effects(rows: list[dict[str, int]], device="cpu") -> dict[str, torch.Tensor]:
    return {key: torch.tensor([row[key] for row in rows], device=device, dtype=torch.long) for key in EFFECT_FIELDS}


def transport_slots(slots: torch.Tensor, effects: dict[str, torch.Tensor], empty: torch.Tensor | None = None) -> torch.Tensor:
    """Differentiable exact square-slot transport, then next-mover rank flip.

    Captured slots are removed, including EP; castling transports the rook too.
    Promotion identity and color perspective require representation-specific code.
    Inputs are already validated at the simulator ingress boundary.
    """
    if slots.ndim != 3 or slots.shape[1] != 64 or set(effects) != set(EFFECT_FIELDS):
        raise ValueError("expected [B,64,C] slots and exact move-effect fields")
    if any(value.shape != (slots.shape[0],) or value.dtype != torch.long or value.device != slots.device for value in effects.values()):
        raise ValueError("effect tensors must be resident int64 [B]")
    batch = torch.arange(slots.shape[0], device=slots.device)
    square = torch.arange(64, device=slots.device)[None, :, None]
    source, target = effects["source"], effects["target"]
    source_slot = slots[batch, source]
    rook_slot = slots[batch, effects["rook_source"].clamp_min(0)]
    rook_active = (effects["rook_source"] >= 0)[:, None, None]
    cleared = (square == source[:, None, None]) | (square == effects["captured"][:, None, None])
    cleared = cleared | ((square == effects["rook_source"][:, None, None]) & rook_active)
    blank = torch.zeros_like(slots[:, :1]) if empty is None else empty.view(1, 1, -1)
    result = torch.where(cleared, blank, slots)
    result = torch.where(square == target[:, None, None], source_slot[:, None], result)
    result = torch.where((square == effects["rook_target"][:, None, None]) & rook_active, rook_slot[:, None], result)
    return result[:, torch.arange(64, device=slots.device) ^ 56]


def transport_pieces(features: torch.Tensor, effects: dict[str, torch.Tensor]) -> torch.Tensor:
    """Exact 12-plane piece algebra. Rights, EP, clocks/history remain simulator-owned."""
    slots = features[:, :12].flatten(2).transpose(1, 2)
    transported = transport_slots(slots, effects)
    square = torch.arange(64, device=features.device)[None]
    promotion_mask = (effects["promotion"] > 0)[:, None] & (square == (effects["target"] ^ 56)[:, None])
    promoted = F.one_hot(effects["promotion"], 12).to(features.dtype)
    transported = torch.where(promotion_mask[:, :, None], promoted[:, None], transported)
    # Old opponent is next mover, and old mover is next opponent.
    transported = torch.cat((transported[:, :, 6:], transported[:, :, :6]), -1)
    return transported.transpose(1, 2).reshape(-1, 12, 8, 8)


@dataclass(frozen=True)
class AnchoredState:
    observed: torch.Tensor
    strategic: torch.Tensor


class StrategicTransport(nn.Module):
    """Learned strategic residual over known move transport, no successor shortcut."""
    def __init__(self, width: int, hidden_width: int | None = None, use_transport: bool = True):
        super().__init__()
        self.width = width
        self.use_transport = use_transport
        self.empty = nn.Parameter(torch.zeros(width))
        self.empty.requires_grad_(use_transport)
        self.perspective = nn.Linear(width, width)
        with torch.no_grad():
            self.perspective.weight.copy_(torch.eye(width))
            self.perspective.bias.zero_()
        self.promotion = nn.Embedding(5, width)
        self.piece_condition = nn.Linear(12, width)
        hidden = width * 2 if hidden_width is None else hidden_width
        self.residual = nn.Sequential(nn.Linear(width * 2 + 3, hidden), nn.GELU(), nn.Linear(hidden, width))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.norm = nn.LayerNorm(width)

    def predict(self, strategic: torch.Tensor, observed: torch.Tensor, effects: dict[str, torch.Tensor]) -> torch.Tensor:
        moved = transport_slots(strategic, effects, self.empty) if self.use_transport else strategic[:, torch.arange(64, device=strategic.device) ^ 56]
        batch = torch.arange(strategic.shape[0], device=strategic.device)
        piece = observed[:, :12].flatten(2).transpose(1, 2)[batch, effects["source"]]
        condition = self.piece_condition(piece) + self.promotion(effects["promotion"])
        square = torch.arange(64, device=strategic.device)[None]
        flags = torch.stack((square == (effects["source"] ^ 56)[:, None], square == (effects["target"] ^ 56)[:, None],
                             square == (effects["captured"] ^ 56)[:, None]), -1).to(strategic.dtype)
        oriented = self.perspective(moved)
        inputs = torch.cat((oriented, condition[:, None].expand(-1, 64, -1), flags), -1)
        return self.norm(oriented + self.residual(inputs))

    def transition(self, state: AnchoredState, effects: dict[str, torch.Tensor], observed_successor: torch.Tensor) -> AnchoredState:
        # Exact observations do not enter predict(). History stays with the
        # external simulator; schema 1 does not encode complete repetition state.
        return AnchoredState(observed_successor, self.predict(state.strategic, state.observed, effects))


@dataclass(frozen=True)
class CounterfactualCorpus:
    roots: torch.Tensor
    successors: torch.Tensor
    actions: torch.Tensor
    effects: dict[str, torch.Tensor]
    records: tuple[dict[str, object], ...]
    manifest: dict[str, object]


def _pack_corpus(records, manifest, device):
    roots = torch.stack([encode_fen(row["root_fen"]) for row in records]).to(device)
    successors = torch.stack([torch.stack([encode_fen(fen) for fen in row["successors"]]) for row in records]).to(device)
    actions = torch.tensor([[move_to_index(move, row["root_fen"].split()[1]) for move in row["moves"]] for row in records], device=device)
    effects = {key: torch.tensor([[item[key] for item in row["effects"]] for row in records], device=device) for key in EFFECT_FIELDS}
    manifest = {**manifest, "sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()}
    return CounterfactualCorpus(roots, successors, actions, effects, tuple(records), manifest)


def generate_counterfactuals(count=64, seed=20260930, max_plies=48, device="cpu", excluded_states=frozenset()):
    import chess
    if count < 1 or max_plies < 1:
        raise ValueError("positive corpus size and ply bound required")
    rng, records, seen = random.Random(seed), [], set()
    for _ in range(count * 500):
        root = chess.Board()
        for _ in range(rng.randrange(1, max_plies + 1)):
            if root.is_game_over(claim_draw=True):
                break
            root.push(rng.choice(sorted(root.legal_moves, key=lambda move: move.uci())))
        if root.is_game_over(claim_draw=True):
            continue
        root_fen = root.fen(en_passant="fen")
        identity = feature_identity(root_fen)
        if identity in seen or identity in excluded_states:
            continue
        moves = list(root.legal_moves)
        if len(moves) < 2:
            continue
        moves = rng.sample(sorted(moves, key=lambda move: move.uci()), 2)
        successors, effects = [], []
        for move in moves:
            effects.append(move_effects(root, move))
            child = root.copy(stack=True)
            child.push(move)
            successors.append(child.fen(en_passant="fen"))
        if any(feature_identity(fen) in excluded_states for fen in successors):
            continue
        records.append({"origin_fen": root.root().fen(en_passant="fen"), "history": [move.uci() for move in root.move_stack],
                        "root_fen": root_fen, "moves": [move.uci() for move in moves], "successors": successors, "effects": effects})
        seen.add(identity)
        if len(records) == count:
            break
    if len(records) != count:
        raise RuntimeError("insufficient independent counterfactual roots")
    manifest = {"kind": "legal_sibling_actions", "strength_evidence": False, "seed": seed, "count": count,
                "oracle": f"python-chess {chess.__version__}"}
    return _pack_corpus(records, manifest, device)


def tactical_counterexamples(device="cpu"):
    """Declared untrained rare-move stress roots, each with two legal alternatives."""
    import chess
    fixtures = (
        ("castle", "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ("e1g1", "e1c1")),
        ("en_passant", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ("e5d6", "e5e6")),
        ("promotion", "4k3/P7/8/8/8/8/8/4K3 w - - 0 1", ("a7a8q", "a7a8n")),
        ("capture_promotion", "1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", ("a7b8q", "a7b8n")),
        ("check_evasion_capture", "4k3/8/8/8/8/2q5/2R5/4K3 w - - 0 1", ("c2c3", "e1f1")),
    )
    records = []
    for name, fen, moves in fixtures:
        root = chess.Board(fen)
        successors, effects = [], []
        for uci in moves:
            move = chess.Move.from_uci(uci)
            effects.append(move_effects(root, move))
            child = root.copy()
            child.push(move)
            successors.append(child.fen(en_passant="fen"))
        records.append(dict(name=name, origin_fen=fen, history=[], root_fen=fen, moves=list(moves), successors=successors, effects=effects))
    return _pack_corpus(records, {"kind": "declared_special_move_counterexamples", "count": len(records), "strength_evidence": False}, device)


def sibling_contrast(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Explicit observed legal-action delta and correct-vs-sibling target margin."""
    predicted, target = F.normalize(predicted, dim=-1), F.normalize(target.detach(), dim=-1)
    true_distance = (predicted - target).square().mean((-1, -2))
    wrong_distance = (predicted - target.flip(1)).square().mean((-1, -2))
    target_distance = (target[:, 0] - target[:, 1]).square().mean((-1, -2))
    margin = F.relu(true_distance - wrong_distance + 0.25 * target_distance[:, None]).mean()
    return margin + F.mse_loss(predicted[:, 0] - predicted[:, 1], target[:, 0] - target[:, 1])


def _predict(base, transport, corpus, indexes=None):
    roots = corpus.roots if indexes is None else corpus.roots[indexes]
    actions = corpus.actions if indexes is None else corpus.actions[indexes]
    effects = {key: value if indexes is None else value[indexes] for key, value in corpus.effects.items()}
    z = base.encode(roots)[:, None].expand(-1, 2, -1, -1).reshape(-1, 64, base.width)
    if transport is None:
        predicted = base.transition(z, actions.reshape(-1))
    else:
        predicted = transport.predict(z, roots[:, None].expand(-1, 2, -1, -1, -1).reshape(-1, 18, 8, 8), {key: value.reshape(-1) for key, value in effects.items()})
    return predicted.reshape(-1, 2, 64, base.width)


def _metrics(base, predicted, observed):
    target = base.encode(observed.reshape(-1, 18, 8, 8)).reshape_as(predicted)
    normalized_prediction, normalized_target = F.normalize(predicted, dim=-1), F.normalize(target, dim=-1)
    separation = (normalized_prediction[:, 0] - normalized_prediction[:, 1]).square().mean()
    exact_separation = (normalized_target[:, 0] - normalized_target[:, 1]).square().mean()
    board_logits = base.decode_board(predicted.reshape(-1, 64, base.width))
    truth = observed[:, :, :12].reshape(-1, 12, 8, 8)
    confidence, piece = board_logits.max(1)
    occupied = truth.sum(1) > 0
    correct = torch.where(confidence > 0, piece, 12) == torch.where(occupied, truth.argmax(1), 12)
    return {"refresh": F.mse_loss(normalized_prediction, normalized_target),
            "board_bce": F.binary_cross_entropy_with_logits(board_logits, truth),
            "piece_classification": _piece_classification(board_logits, truth),
            "occupied_accuracy": (correct * occupied).sum() / occupied.sum().clamp_min(1),
            "action_separation_ratio": separation / exact_separation.clamp_min(1e-12),
            "action_delta_error": F.mse_loss(normalized_prediction[:, 0] - normalized_prediction[:, 1], normalized_target[:, 0] - normalized_target[:, 1]),
            "contrast": sibling_contrast(predicted, target)}


def _piece_classification(logits, truth):
    # BCE over 12 planes sees mostly zeros. A 13th fixed empty class makes the
    # occupied-piece decision explicit, with a lower weight for empty squares.
    occupied = truth.sum(1) > 0
    labels = torch.where(occupied, truth.argmax(1), 12)
    classes = torch.cat((logits, torch.zeros_like(logits[:, :1])), 1)
    weights = logits.new_tensor([1.] * 12 + [0.25])
    return F.cross_entropy(classes, labels, weight=weights)


def _diamond_metrics(base, transport, corpus):
    import chess
    batch = corpus.features.shape[0]
    latent = base.encode(corpus.features[:, :, 0].reshape(-1, 18, 8, 8))
    for ply in range(4):
        if transport is None:
            latent = base.transition(latent, corpus.actions[:, :, ply].reshape(-1))
        else:
            rows = []
            for record in corpus.records:
                for path_index in range(2):
                    board = chess.Board(record.path_fens[path_index][ply])
                    rows.append(move_effects(board, chess.Move.from_uci(record.paths[path_index][ply])))
            latent = transport.predict(latent, corpus.features[:, :, ply].reshape(-1, 18, 8, 8), batch_effects(rows, latent.device))
    paired = latent.reshape(batch, 2, 64, base.width)
    metrics = _metrics(base, paired, corpus.features[:, :, -1])
    for key in ("action_separation_ratio", "action_delta_error", "contrast"):
        metrics.pop(key)
    normalized = F.normalize(paired, dim=-1)
    metrics["crosspath_disagreement"] = F.mse_loss(normalized[:, 0], normalized[:, 1])
    return metrics


def run_comparison(*, device="cpu", samples=64, steps=16, batch_size=8, width=16, seed=20260930, output=None):
    if samples < 2 or steps < 1 or batch_size < 1:
        raise ValueError("positive sizes and at least two samples required")
    train = generate_counterfactuals(samples, seed, device=device)
    excluded = {feature_identity(row["root_fen"]) for row in train.records} | {feature_identity(fen) for row in train.records for fen in row["successors"]}
    heldout = generate_counterfactuals(max(4, samples // 4), seed + 1, device=device, excluded_states=excluded)
    tactical = tactical_counterexamples(device)
    candidates = generate_diamonds(16, seed + 3, device=device)
    keep = [index for index, record in enumerate(candidates.records)
            if not excluded.intersection(feature_identity(fen) for path in record.path_fens for fen in path)][:4]
    if len(keep) != 4:
        raise RuntimeError("insufficient canonical-disjoint diamond validation roots")
    retained = tuple(candidates.records[index] for index in keep)
    indexes = torch.tensor(keep, device=device)
    diamonds = DiamondCorpus(candidates.features[indexes], candidates.actions[indexes], retained,
                              {**candidates.manifest, "count": 4,
                               "sha256": hashlib.sha256(json.dumps([asdict(record) for record in retained], sort_keys=True).encode()).hexdigest()})
    generator = torch.Generator(device=device).manual_seed(seed + 2)
    schedule = torch.randint(samples, (steps, batch_size), generator=generator, device=device)
    results = []
    for control in ("baseline", "observed_residual", "transport", "transport_contrast"):
        torch.manual_seed(seed)
        base = build_model("square", width=width, depth=1).to(device)
        for parameter in base.parameters():
            parameter.requires_grad_(False)
        # Keep encoder and exact latent target fixed. Train only predictors and
        # piece decoder; this isolates action dynamics from representation collapse.
        for parameter in base.board.parameters():
            parameter.requires_grad_(True)
        transport = None
        if control == "baseline":
            for module in (base.promotion, base.action_condition, base.dynamics):
                for parameter in module.parameters():
                    parameter.requires_grad_(True)
        else:
            # Match the baseline's active parameter budget to the nearest hidden
            # unit, keeping observed input and the frozen encoder identical.
            prototype = StrategicTransport(width)
            baseline_budget = sum(parameter.numel() for module in (base.board, base.promotion, base.action_condition, base.dynamics) for parameter in module.parameters())
            current_budget = sum(parameter.numel() for parameter in prototype.parameters()) + sum(parameter.numel() for parameter in base.board.parameters())
            hidden = max(1, width * 2 + round((baseline_budget - current_budget) / (3 * width + 4)))
            transport = StrategicTransport(width, hidden_width=hidden, use_transport=control != "observed_residual").to(device)
        parameters = [parameter for parameter in base.parameters() if parameter.requires_grad]
        if transport is not None:
            parameters.extend(parameter for parameter in transport.parameters() if parameter.requires_grad)
        optimizer = torch.optim.AdamW(parameters, lr=1e-3)
        with torch.no_grad():
            before = {key: float(value) for key, value in _metrics(base, _predict(base, transport, heldout), heldout.successors).items()}
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for indexes in schedule:
            predicted = _predict(base, transport, train, indexes)
            observed = train.successors[indexes]
            target = base.encode(observed.reshape(-1, 18, 8, 8)).reshape_as(predicted).detach()
            metrics = _metrics(base, predicted, observed)
            anchor = _piece_classification(base.decode_board(target.reshape(-1, 64, width)), observed[:, :, :12].reshape(-1, 12, 8, 8))
            loss = 10 * metrics["refresh"] + metrics["piece_classification"] + anchor
            if control == "transport_contrast":
                loss = loss + 10 * metrics["contrast"]
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(parameters, 1)
            optimizer.step()
        if torch.device(device).type == "cuda":
            torch.cuda.synchronize()
        seconds = time.perf_counter() - start
        with torch.no_grad():
            after = {key: float(value) for key, value in _metrics(base, _predict(base, transport, heldout), heldout.successors).items()}
            full_refresh = base.encode(heldout.successors.reshape(-1, 18, 8, 8)).reshape(-1, 2, 64, width)
            reference = {key: float(value) for key, value in _metrics(base, full_refresh, heldout.successors).items()}
            diamond = {key: float(value) for key, value in _diamond_metrics(base, transport, diamonds).items()}
            tactical_predictions = _predict(base, transport, tactical)
            tactical_metrics = {record["name"]: {key: float(value) for key, value in _metrics(base, tactical_predictions[index:index + 1], tactical.successors[index:index + 1]).items()}
                                for index, record in enumerate(tactical.records)}
        if not all(torch.isfinite(torch.tensor(list(after.values())))):
            raise RuntimeError("nonfinite strategic dynamics result")
        results.append({"control": control, "trainable_parameters": sum(parameter.numel() for parameter in parameters),
                        "seconds": seconds, "before": before, "after": after,
                        "always_refresh_reference": reference, "four_ply_diamond": diamond, "tactical_counterexamples": tactical_metrics})
    report = {"kind": "observed_piece_transport_strategic_dynamics", "strength_evidence": False,
              "environment": {"torch": torch.__version__, "hip": torch.version.hip, "device": device},
              "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "config": dict(samples=samples, steps=steps, batch_size=batch_size, width=width, seed=seed),
              "train": train.manifest, "heldout": heldout.manifest, "diamonds": diamonds.manifest, "tactical": tactical.manifest,
              "corpora": {"train": list(train.records), "heldout": list(heldout.records), "diamonds": [asdict(record) for record in diamonds.records], "tactical": list(tactical.records)},
              "schedule_sha256": hashlib.sha256(json.dumps(schedule.cpu().tolist()).encode()).hexdigest(), "results": results,
              "limitations": ["frozen random encoder, uncalibrated latent semantics", "same seeds/data/schedule, unequal parameters and time",
                              "exact prior observations and special-move effects supplied by rules", "no successor observations fed to strategic predictor",
                              "small synthetic corpora and no strength evidence"]}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name, default in (("samples", 64), ("steps", 16), ("batch-size", 8), ("width", 16), ("seed", 20260930)):
        parser.add_argument(f"--{name}", type=int, default=default)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output")
    print(json.dumps(run_comparison(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
