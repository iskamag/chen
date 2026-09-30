"""Legal synthetic trajectories and a device-resident, bounded replay buffer.

The CPU chess oracle is used only at ingestion. Sampling and batch gathering
operate on Torch tensors on the configured device, including ROCm ``cuda``.
These labels exercise learning mechanics; they are not chess-strength evidence.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import random
from typing import Mapping

import torch

from .features import encode_fen, move_to_index

ACTION_COUNT = 64 * 64 * 5
FEATURE_SHAPE = (18, 8, 8)
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class SyntheticCorpus:
    tensors: dict[str, torch.Tensor]
    records: tuple[dict[str, object], ...]
    manifest: dict[str, object]


def replay_schema(feature_dtype: torch.dtype = torch.float32):
    if feature_dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("features require float32, float16, or bfloat16")
    return {
        "features": (FEATURE_SHAPE, feature_dtype),
        "next_features": (FEATURE_SHAPE, feature_dtype),
        "legal_mask": ((ACTION_COUNT,), torch.bool),
        "actions": ((), torch.int64),
        "target_actions": ((), torch.int64),
        "material_proxy": ((), torch.float32),
        "terminal": ((), torch.bool),
        "game_ids": ((), torch.int64),
    }


class DeviceReplay:
    """Single-owner ring; append validates ingress, sample never copies to host.

    Append accepts only already-resident, exact-schema tensors and overwrites the
    oldest records. Gathered batches own storage and survive later overwrites.
    ``sequence_ids`` identify the stream insert generation, including oversized
    appends; physical row indexes alone must not be used as persistent handles.
    Concurrent writers/readers on different streams require external ordering.
    """

    def __init__(self, capacity: int, device: str | torch.device,
                 feature_dtype: torch.dtype = torch.float32,
                 max_bytes: int | None = None):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.schema = replay_schema(feature_dtype)
        self.bytes_per_record = 8 + sum(
            math.prod(shape) * torch.empty((), dtype=dtype).element_size()
            for shape, dtype in self.schema.values()
        )
        if max_bytes is not None and capacity * self.bytes_per_record > max_bytes:
            raise ValueError("replay capacity exceeds configured memory budget")
        self.device = torch.empty(0, device=device).device
        self.storage = {
            name: torch.empty((capacity, *shape), dtype=dtype, device=self.device)
            for name, (shape, dtype) in self.schema.items()
        }
        self.sequence_ids = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.size = self.cursor = self.total_inserted = 0

    def clear(self):
        """Invalidate rows without reallocating VRAM or reusing stream identities."""
        self.size = self.cursor = 0

    @torch.no_grad()
    def append(self, batch: Mapping[str, torch.Tensor]):
        if set(batch) != set(self.schema):
            raise ValueError("batch keys must exactly match replay schema")
        if not isinstance(batch["features"], torch.Tensor) or batch["features"].ndim != 4:
            raise ValueError("features must be a batched tensor [B,18,8,8]")
        count = batch["features"].shape[0]
        for name, (shape, dtype) in self.schema.items():
            value = batch[name]
            if not isinstance(value, torch.Tensor):
                raise ValueError(f"{name}: expected a tensor")
            if value.shape != (count, *shape):
                raise ValueError(f"{name}: wrong shape {tuple(value.shape)}")
            if value.dtype != dtype or value.device != self.device:
                raise ValueError(f"{name}: dtype/device mismatch")
        if count == 0:
            return
        # These scalar synchronizations occur at the explicit ingestion boundary,
        # never in sample(). They make malformed data fail before ring mutation.
        masks = batch["legal_mask"]
        if not bool(masks.any(dim=1).all()):
            raise ValueError("current positions must have a legal action")
        for name in ("actions", "target_actions"):
            actions = batch[name]
            if not bool(((actions >= 0) & (actions < ACTION_COUNT)).all()):
                raise ValueError(f"{name}: out of range")
            if not bool(masks.gather(1, actions[:, None]).all()):
                raise ValueError(f"{name}: illegal action")
        for name in ("features", "next_features", "material_proxy"):
            if not bool(torch.isfinite(batch[name]).all()):
                raise ValueError(f"{name}: nonfinite data")
        if not bool((batch["material_proxy"].abs() <= 1).all()):
            raise ValueError("material proxy must be in [-1, 1]")

        kept = min(count, self.capacity)
        skip = count - kept
        indexes = (torch.arange(kept, device=self.device) + self.cursor + skip) % self.capacity
        for name, values in batch.items():
            self.storage[name].index_copy_(0, indexes, values[skip:])
        identities = torch.arange(self.total_inserted + skip,
                                  self.total_inserted + count, device=self.device)
        self.sequence_ids.index_copy_(0, indexes, identities)
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.total_inserted += count

    def sample(self, count: int, generator: torch.Generator | None = None):
        """Uniform sampling with replacement; random indexes and gather stay resident."""
        if self.size == 0 or count <= 0:
            raise ValueError("sampling requires a nonempty replay and positive count")
        indexes = torch.randint(self.size, (count,), device=self.device, generator=generator)
        batch = {name: values.index_select(0, indexes) for name, values in self.storage.items()}
        batch["sequence_ids"] = self.sequence_ids.index_select(0, indexes)
        return batch

    def manifest(self):
        return {"schema_version": SCHEMA_VERSION, "device": str(self.device),
                "capacity": self.capacity, "size": self.size, "cursor": self.cursor,
                "total_inserted": self.total_inserted,
                "bytes_per_record": self.bytes_per_record,
                "allocated_bytes": self.capacity * self.bytes_per_record}


def sample_legal(logits: torch.Tensor, legal_mask: torch.Tensor,
                 generator: torch.Generator | None = None):
    """Strict legal categorical sampling; reject terminal rows and malformed scores.

    Validation can synchronize the device. Search should validate an exact mask
    once at ingress, then use its own asynchronous fast path for repeated draws.
    """
    if logits.ndim != 2 or legal_mask.shape != logits.shape:
        raise ValueError("logits and mask must have equal [batch, actions] shapes")
    if legal_mask.dtype != torch.bool or legal_mask.device != logits.device:
        raise ValueError("mask must be bool and resident with logits")
    if not logits.is_floating_point():
        raise ValueError("logits must be floating point")
    if not bool(legal_mask.any(dim=1).all()):
        raise ValueError("terminal row cannot be sampled")
    if not bool(torch.isfinite(logits[legal_mask]).all()):
        raise ValueError("legal logits must be finite")
    weights = torch.softmax(logits.masked_fill(~legal_mask, -torch.inf), dim=-1)
    return torch.multinomial(weights, 1, generator=generator).squeeze(-1)


def generate_synthetic(count: int, seed: int = 20260930, max_plies: int = 96,
                       device: str | torch.device = "cpu",
                       feature_dtype: torch.dtype = torch.float32) -> SyntheticCorpus:
    """Generate legal random-play transitions with explicitly weak rule labels.

    chess==1.11.2 is an optional development oracle, not a runtime engine dependency.
    Current state is nonterminal; successors may be terminal. Game ids are tracked
    so downstream splitting cannot interleave adjacent trajectory records.
    """
    if count <= 0 or max_plies <= 0:
        raise ValueError("count and max_plies must be positive")
    import chess

    rng = random.Random(seed)
    values = (0, 1, 3, 3, 5, 9, 0)
    schema = replay_schema(feature_dtype)
    tensors = {name: torch.empty((count, *shape), dtype=dtype)
               for name, (shape, dtype) in schema.items()}
    records = []
    board = chess.Board()
    game_id = ply = 0
    for row in range(count):
        if ply >= max_plies or board.is_game_over(claim_draw=True):
            board = chess.Board()
            game_id += 1
            ply = 0
        side = board.turn
        side_name = "w" if side else "b"
        fen = board.fen(en_passant="fen")
        legal = sorted(board.legal_moves, key=lambda m: m.uci())
        mask = tensors["legal_mask"][row]
        mask.zero_()
        for move in legal:
            mask[move_to_index(move.uci(), side_name)] = True
        # Label a simple rule exercise, not a best chess move. It ignores recapture,
        # positional play and search. Checkmate is exact, capture/check is heuristic.
        def immediate_score(move):
            victim = board.piece_at(move.to_square)
            capture = values[victim.piece_type] if victim else int(board.is_en_passant(move))
            promotion = values[move.promotion] - 1 if move.promotion else 0
            checking = board.gives_check(move)
            board.push(move)
            mating = board.is_checkmate()
            board.pop()
            return 1000 * mating + capture + promotion + 0.25 * checking
        target = max(legal, key=immediate_score)
        move = rng.choice(legal)
        material = sum(values[p.piece_type] * (1 if p.color == side else -1)
                       for p in board.piece_map().values())
        tensors["features"][row] = encode_fen(fen).to(dtype=feature_dtype)
        tensors["actions"][row] = move_to_index(move.uci(), side_name)
        tensors["target_actions"][row] = move_to_index(target.uci(), side_name)
        tensors["material_proxy"][row] = math.tanh(material / 8)
        tensors["game_ids"][row] = game_id
        board.push(move)
        next_fen = board.fen(en_passant="fen")
        tensors["next_features"][row] = encode_fen(next_fen).to(dtype=feature_dtype)
        tensors["terminal"][row] = board.is_game_over(claim_draw=True)
        records.append({"game_id": game_id, "ply": ply, "fen": fen,
                        "action": move.uci(), "next_fen": next_fen,
                        "target_action": target.uci()})
        ply += 1
    fingerprint = hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest()
    manifest = {"schema_version": SCHEMA_VERSION, "kind": "synthetic_legal_smoke",
                "seed": seed, "count": count, "max_plies": max_plies,
                "games": game_id + 1, "oracle": f"chess=={chess.__version__}",
                "fingerprint_sha256": fingerprint,
                "labels": "immediate capture/check/mate heuristic + material proxy",
                "strength_evidence": False, "ingestion_device": str(device)}
    # The one explicit ingress transfer. Replay refuses implicit transfers later.
    tensors = {name: tensor.to(device=device) for name, tensor in tensors.items()}
    return SyntheticCorpus(tensors, tuple(records), manifest)
