"""Resident finite-depth minimax teachers and controlled score/policy learning.

Search scores are current-mover centipawns, not outcomes or calibrated WDL.
The bounded utility head reuses the three existing logits only as a regressor.
"""
from __future__ import annotations

import argparse
import ctypes as c
from dataclasses import dataclass, asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch import nn
from torch.nn import functional as F

from .bridge import (BridgeError, ExportedTransitions, ResidentSpan, RuleArena,
                     position_from_fen, validate_output_tensors)
from .compact import CompactReplay, decode_batch, decode_features
from .features import SCHEMA_VERSION
from .models import MODELS, build_model
from .train import seed_all

SCALE_CP = 800.0
MATE_BAND = 29000
START_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
TEACHER_SCHEMA = {"scores": ((), torch.int32), "best_actions": ((), torch.int64),
                  "parent_ply": ((), torch.int32)}
SEARCH_LABEL_DTYPES = {"search_score_cp": torch.int32, "teacher_parent_ply": torch.int32,
                       "material_greedy_action": torch.int64,
                       **{name: torch.int64 for name in ("parent_hash", "child_hash", "parent_node", "child_node", "root_epoch", "teacher_root_id")}}


class TeacherSink(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32),
                ("device", c.c_int32), ("reserved", c.c_uint32), ("capacity", c.c_uint64),
                ("scores", c.c_uint64), ("best_actions", c.c_uint64), ("parent_ply", c.c_uint64)]


class TeacherResult(c.Structure):
    _fields_ = [("rows", c.c_uint64), ("root_epoch", c.c_uint64), ("kernel_ms", c.c_double),
                ("evaluator", c.c_int32), ("label_kind", c.c_uint32)]


assert c.sizeof(TeacherSink) == 48
assert c.sizeof(TeacherResult) == 32


@dataclass(frozen=True)
class TeacherLabels:
    tensors: dict[str, torch.Tensor]
    manifest: dict[str, object]


def export_teacher(arena: RuleArena, span: ResidentSpan):
    """Owned resident exact labels for the same selected replay edge rows."""
    with arena._lock:
        arena._require_open()
        source = arena._validate_span(span)
        if source.start <= 0:
            raise ValueError("teacher selection must contain child edges")
        version = arena._lib.chen_teacher_abi_version
        version.argtypes, version.restype = [], c.c_uint32
        if version() != 1:
            raise ValueError("unsupported teacher ABI")
        fn = arena._lib.chen_rules_export_teacher
        fn.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64,
                       c.POINTER(TeacherSink), c.POINTER(TeacherResult), c.c_char_p, c.c_uint64]
        fn.restype = c.c_int32
        tensors = {name: torch.empty((source.count,), dtype=dtype, device=arena.device)
                   for name, (_, dtype) in TEACHER_SCHEMA.items()}
        validate_output_tensors(tensors, TEACHER_SCHEMA, source.count, arena.device)
        sink = TeacherSink(1, c.sizeof(TeacherSink), arena.device.index, 0, source.count,
                           *(tensors[name].data_ptr() for name in TEACHER_SCHEMA))
        result = TeacherResult()
        torch.cuda.synchronize(arena.device)
        arena._call("export_teacher", source.start, source.count, source.epoch,
                    c.byref(sink), c.byref(result))
        if result.rows != source.count or result.root_epoch != source.epoch or result.label_kind != 1 or result.evaluator not in (-1, 0, 1):
            raise RuntimeError("teacher success returned inconsistent metadata")
        return TeacherLabels(tensors, {"context_uuid": arena.context_uuid, "root_epoch": source.epoch,
                                      "child_start": source.start, "rows": source.count,
                                      "evaluator": result.evaluator, "label_kind": result.label_kind,
                                      "kernel_ms": result.kernel_ms,
                                      "score_kind": "scored_resident_finite_tree_classical_minimax_cp",
                                      "perspective": "current parent mover; mate uses absolute root ply"})


def backed_up_tree(arena, root, history, *, tree_depth=3, evaluator=1):
    """Uniform full BFS, static deepest frontier, complete bottom-up reductions."""
    if type(tree_depth) is not int or tree_depth < 2 or tree_depth > 5:
        raise ValueError("tree_depth must be an integer in [2,5]")
    root_manifest = arena.reset(root, history)
    root_span = ResidentSpan(arena.context_uuid, root_manifest["root_epoch"], 0, 1)
    levels, schedule = [root_span], []
    for level in range(tree_depth):
        if not levels[-1].count:
            break
        stats = arena.expand(levels[-1])
        schedule.append({"operation": "expand", "source_level": level,
                         "source_span": asdict(levels[-1]), "next_span": asdict(stats["span"]),
                         "arena_positions": stats["arena_positions"]})
        levels.append(stats["span"])
    if levels[-1].count:
        arena.evaluate(levels[-1], evaluator)
        schedule.append({"operation": "evaluate", "source_level": len(levels) - 1,
                         "span": asdict(levels[-1]), "evaluator": evaluator})
    for level in range(len(levels) - 2, -1, -1):
        arena.reduce(levels[level])
        schedule.append({"operation": "reduce", "source_level": level, "span": asdict(levels[level])})
    if len(levels) < 3 or not levels[2].count:
        raise ValueError("root produces no nonterminal first-ply parent teacher records")
    records = arena.export(levels[2])
    labels = export_teacher(arena, levels[2])
    if labels.manifest["evaluator"] not in (-1, evaluator):
        raise RuntimeError("teacher evaluator differs from declared construction")
    construction = {"tree_depth": tree_depth, "source_parent_level": 1,
                    "remaining_parent_horizon": tree_depth - 1,
                    "evaluator": evaluator, "schedule": schedule,
                    "early_terminal_branches": "exact scored terminals, not static horizon truncation"}
    return records, labels, construction


def material_cp(features):
    weights = torch.tensor([100, 320, 330, 500, 900, 0], device=features.device, dtype=features.dtype)
    counts = features[:, :12].sum((-1, -2))
    return ((counts[:, :6] - counts[:, 6:12]) * weights).sum(-1).clamp(-29000, 29000)


def deduplicate_parents(records, labels):
    """One device record/vote per parent; all sibling-edge labels must agree."""
    if any(records.manifest[key] != labels.manifest[key] for key in ("context_uuid", "root_epoch", "child_start", "rows")):
        raise ValueError("teacher labels do not describe the same replay edge selection")
    parent = records.sidecar["parent_node"]
    unique, inverse, counts = torch.unique_consecutive(parent, return_inverse=True, return_counts=True)
    first = torch.cat((torch.zeros(1, dtype=torch.int64, device=parent.device), counts.cumsum(0)[:-1]))
    if not bool((counts == records.tensors["legal_counts"].index_select(0, first)).all()):
        raise ValueError("deduplication requires all sibling edges for each parent")
    for name, value in labels.tensors.items():
        if not bool((value == value.index_select(0, first).index_select(0, inverse)).all()):
            raise ValueError(f"inconsistent sibling teacher labels: {name}")
    if not bool((labels.tensors["parent_ply"] == 1).all()):
        raise ValueError("first-ply teacher construction expected parent_ply1")
    # A one-ply greedy material policy is an independently defined control.
    next_cp = -material_cp(decode_features(records.tensors["next_pieces"], records.tensors["next_state"]))
    maxima = torch.full((unique.shape[0],), -30001.0, device=parent.device).scatter_reduce_(0, inverse, next_cp, reduce="amax", include_self=True)
    indexes = torch.arange(parent.shape[0], device=parent.device)
    candidates = torch.where(next_cp == maxima.index_select(0, inverse), indexes, parent.shape[0])
    chosen = torch.full((unique.shape[0],), parent.shape[0], device=parent.device, dtype=torch.int64).scatter_reduce_(0, inverse, candidates, reduce="amin", include_self=True)
    tensors = {key: value.index_select(0, first) for key, value in records.tensors.items()}
    tensors["target_actions"] = labels.tensors["best_actions"].index_select(0, first)
    teachers = {"search_score_cp": labels.tensors["scores"].index_select(0, first),
                "teacher_parent_ply": labels.tensors["parent_ply"].index_select(0, first),
                "material_greedy_action": records.tensors["actions"].index_select(0, chosen)}
    sidecar = {key: value.index_select(0, first) for key, value in records.sidecar.items()}
    return tensors, teachers, sidecar


class SearchReplay:
    """Compact owned ring plus explicit resident search-label storage."""
    def __init__(self, capacity, device):
        self.compact = CompactReplay(capacity, device)
        self.device = self.compact.device
        self.labels = {name: torch.empty(capacity, dtype=dtype, device=self.device)
                       for name, dtype in SEARCH_LABEL_DTYPES.items()}

    def append(self, tensors, labels):
        count = tensors["pieces"].shape[0]
        if set(labels) != set(self.labels):
            raise ValueError("search label keys mismatch")
        for name, storage in self.labels.items():
            value = labels[name]
            if not isinstance(value, torch.Tensor) or value.shape != (count,) or value.dtype != storage.dtype or value.device != self.device:
                raise ValueError(f"{name}: search label schema/device mismatch")
        if not bool((labels["search_score_cp"].to(torch.int64).abs() <= 30000).all()) or not bool(((labels["teacher_parent_ply"] >= 0) & (labels["teacher_parent_ply"] <= 64)).all()):
            raise ValueError("search label score/ply range invalid")
        support = tensors["legal_actions"]
        active = torch.arange(self.compact.max_slots, device=self.device)[None] < tensors["legal_counts"][:, None]
        if not bool(((support == labels["material_greedy_action"][:, None]) & active).any(-1).all()):
            raise ValueError("material baseline action absent from exact support")
        kept, skip = min(count, self.compact.capacity), max(0, count - self.compact.capacity)
        indexes = (torch.arange(kept, device=self.device) + self.compact.cursor + skip) % self.compact.capacity
        try:
            self.compact.append(tensors)
            for name, value in labels.items():
                self.labels[name].index_copy_(0, indexes, value[skip:])
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)

    def sample(self, count, generator=None):
        if type(count) is not int or count <= 0 or not self.compact.size:
            raise ValueError("sample requires positive count and nonempty search replay")
        indexes = torch.randint(self.compact.size, (count,), generator=generator, device=self.device)
        return self.gather(indexes)

    def gather(self, indexes):
        packed = {key: value.index_select(0, indexes) for key, value in self.compact.storage.items()}
        packed["sequence_ids"] = self.compact.sequence_ids.index_select(0, indexes)
        batch = decode_batch(packed)
        batch.update({key: value.index_select(0, indexes) for key, value in self.labels.items()})
        return batch


def search_outputs(model, batch):
    output = model(batch["features"], batch["legal_mask"])
    head = output["wdl_logits"].softmax(-1)
    return {"search_utility": head[:, 2] - head[:, 0], "policy_logits": output["policy_logits"]}


def comparable_model(name, *, width=32, depth=1, seed=20260930):
    """Same shared input/position/norm/value/policy initialization across kinds."""
    seed_all(seed)
    reference = build_model("square", width=width, depth=depth)
    seed_all(seed)
    model = build_model(name, width=width, depth=depth)
    shared_prefixes = ("input.", "position", "latent_norm.", "wdl.", "policy_source.",
                       "policy_target.", "promo_source.", "promo_target.")
    state = model.state_dict()
    for key, value in reference.state_dict().items():
        if key.startswith(shared_prefixes):
            state[key] = value
    model.load_state_dict(state, strict=True)
    return model


def search_loss(model, batch):
    output = search_outputs(model, batch)
    target = torch.tanh(batch["search_score_cp"].float() / SCALE_CP)
    ordinary = batch["search_score_cp"].to(torch.int64).abs() < MATE_BAND
    value_rows = F.smooth_l1_loss(output["search_utility"], target, reduction="none")
    value = (value_rows * ordinary).sum() / ordinary.sum().clamp_min(1)
    policy = F.cross_entropy(output["policy_logits"], batch["target_actions"])
    return {"total": policy + value, "search_utility": value, "chosen_action": policy}


def search_step(model, optimizer, batch):
    optimizer.zero_grad(set_to_none=True)
    losses = search_loss(model, batch)
    losses["total"].backward()
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    optimizer.step()
    return {key: value.detach() for key, value in losses.items()}


@dataclass(frozen=True)
class RootSample:
    family_id: int
    ply: int
    fen: str
    previous_hashes: tuple[int, ...]


@dataclass(frozen=True)
class RootFamily:
    family_id: int
    origin_fen: str
    moves: tuple[str, ...]
    roots: tuple[RootSample, ...]


def canonical_position_key(board_or_fen):
    """Exact schema1 feature equivalence, including raw EP and clipped clocks.

    Uses lossless integer rule state during CPU root admission; verified against
    the actual encode_fen oracle. Fullmove and repetition are absent in schema1.
    """
    import chess
    board = chess.Board(board_or_fen) if isinstance(board_or_fen, str) else board_or_fen
    flip = board.turn == chess.BLACK
    planes = []
    for color in (board.turn, not board.turn):
        for piece in range(1, 7):
            bits = board.pieces_mask(piece, color)
            planes.append(int.from_bytes(bits.to_bytes(8, "little")[::-1], "little") if flip else bits)
    rights = [board.has_kingside_castling_rights(board.turn), board.has_queenside_castling_rights(board.turn),
              board.has_kingside_castling_rights(not board.turn), board.has_queenside_castling_rights(not board.turn)]
    ep = -1 if board.ep_square is None else board.ep_square ^ (56 if flip else 0)
    state = [*planes, *rights, ep, min(100, board.halfmove_clock)]
    return hashlib.sha256(json.dumps(state, separators=(",", ":")).encode()).hexdigest()


def _family_positions(family):
    """Conservative split set: roots, observed parents and stored successors."""
    import chess
    keys = set()
    for root in family.roots:
        board = chess.Board(root.fen)
        keys.add(canonical_position_key(board))
        for move in list(board.legal_moves):
            board.push(move)
            keys.add(canonical_position_key(board))
            for reply in list(board.legal_moves):
                board.push(reply)
                keys.add(canonical_position_key(board))
                board.pop()
            board.pop()
    return keys


def _candidate_family(family_id, seed, roots_at):
    import chess
    board, moves, hashes, roots = chess.Board(START_FEN), [], [], []
    rng = random.Random(seed + family_id * 1000003)
    for ply in range(max(roots_at) + 1):
        if board.is_game_over(claim_draw=True) or board.is_repetition(2) or board.halfmove_clock >= 90:
            return None
        if ply in roots_at:
            roots.append(RootSample(family_id, ply, board.fen(en_passant="fen"), tuple(hashes)))
        if ply == max(roots_at):
            break
        hashes.append(position_from_fen(board.fen(en_passant="fen")).hash)
        move = rng.choice(sorted(board.legal_moves, key=lambda action: action.uci()))
        moves.append(move.uci())
        board.push(move)
    return RootFamily(family_id, START_FEN, tuple(moves), tuple(roots))


def split_root_families(train_families=64, heldout_families=16, seed=20260930, roots_at=(12, 24)):
    """Game-lineage split with whole-family canonical position collision rejection."""
    if any(type(value) is not int or value <= 0 for value in (train_families, heldout_families)):
        raise ValueError("positive training and heldout family counts required")
    if not roots_at or any(type(ply) is not int or not 4 <= ply <= 80 for ply in roots_at) or len(set(roots_at)) != len(roots_at):
        raise ValueError("root plies must be unique integers in [4,80]")
    train, heldout, train_keys, heldout_keys = [], [], set(), set()
    rejected = {"terminal_or_repeated_history": 0, "canonical_collision": 0}
    limit = 20 * (train_families + heldout_families)
    for family_id in range(limit):
        family = _candidate_family(family_id, seed, roots_at)
        if family is None:
            rejected["terminal_or_repeated_history"] += 1
            continue
        keys = _family_positions(family)
        if keys & (train_keys | heldout_keys):
            rejected["canonical_collision"] += 1
            continue
        if len(train) < train_families:
            train.append(family); train_keys.update(keys)
        else:
            heldout.append(family); heldout_keys.update(keys)
        if len(heldout) == heldout_families:
            break
    if len(train) != train_families or len(heldout) != heldout_families:
        raise RuntimeError("unable to construct requested collision-free root families")
    records = {"train": [asdict(family) for family in train], "heldout": [asdict(family) for family in heldout]}
    manifest = {"seed": seed, "roots_at_plies": list(roots_at), "train_families": train_families,
                "heldout_families": heldout_families, "train_canonical_positions": len(train_keys),
                "heldout_canonical_positions": len(heldout_keys), "canonical_overlap": 0,
                "source_records": records, "rejected_candidates": rejected,
                "source_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                "split_scope": "disjoint source-game families and schema1-equivalent root/current/successor positions",
                "distribution": "seeded uniform legal move walks from START; shared untrained early ancestry; no real game outcomes"}
    return tuple(train), tuple(heldout), manifest


def produce_dataset(families, *, library, device="cuda", arena_capacity=100000,
                    tree_depth=3, evaluator=1):
    """Direct resident exact labels; root/history upload only, no CPU label path."""
    import chess
    capacity = sum(chess.Board(root.fen).legal_moves.count() for family in families for root in family.roots)
    if capacity <= 0:
        raise ValueError("no legal root families")
    replay = SearchReplay(capacity, device)
    provenance = []
    start = time.perf_counter()
    with RuleArena(library, arena_capacity, device) as arena:
        for root_id, root in enumerate(root for family in families for root in family.roots):
            records, teacher, construction = backed_up_tree(
                arena, position_from_fen(root.fen), root.previous_hashes,
                tree_depth=tree_depth, evaluator=evaluator)
            tensors, labels, sidecar = deduplicate_parents(records, teacher)
            tensors["game_ids"] = torch.full((tensors["pieces"].shape[0],), root.family_id,
                                               dtype=torch.int64, device=replay.device)
            labels.update(sidecar, teacher_root_id=torch.full((tensors["pieces"].shape[0],), root_id,
                                                              dtype=torch.int64, device=replay.device))
            replay.append(tensors, labels)
            provenance.append({"teacher_root_id": root_id, "root": asdict(root),
                               "source": records.manifest, "teacher": teacher.manifest,
                               "construction": construction, "deduplicated_parents": tensors["pieces"].shape[0]})
            del records, teacher, tensors, labels, sidecar
    torch.cuda.synchronize(replay.device)
    manifest = {"records": replay.compact.size, "capacity": capacity,
                "producer_seconds": time.perf_counter() - start,
                "teacher_kind": "finite_depth_classical_minimax_search_targets",
                "tree_depth": tree_depth, "parent_level": 1, "parent_horizon": tree_depth - 1,
                "evaluator": evaluator, "scale_cp": SCALE_CP, "mate_band": MATE_BAND,
                "deduplication": "one resident first-ply parent, not repeated sibling edges",
                "lineage": provenance, "input_source": "CPU root/history only; labels/observations exported and consumed resident",
                "replay_payload_bytes": replay.compact.bytes_per_record * capacity,
                "teacher_sidecar_payload_bytes": sum(value.numel() * value.element_size() for value in replay.labels.values()),
                "state_payload_download_in_producer": False}
    return replay, manifest


def utility_to_cp(utility):
    """Reporting conversion, not mate inference; exactly deployment scale/epsilon."""
    return (torch.atanh(utility.clamp(-1 + 1e-6, 1 - 1e-6)) * SCALE_CP).round().clamp(-29000, 29000)


def _metric_rows(predicted_utility, predicted_action, batch):
    target = torch.tanh(batch["search_score_cp"].float() / SCALE_CP)
    ordinary = batch["search_score_cp"].to(torch.int64).abs() < MATE_BAND
    cp_error = utility_to_cp(predicted_utility) - batch["search_score_cp"].float()
    return torch.stack((ordinary.sum(), ((predicted_utility - target).square() * ordinary).sum(),
                        (cp_error.abs() * ordinary).sum(),
                        (predicted_action == batch["target_actions"]).sum(),
                        (~ordinary).sum(), batch["target_actions"].numel() * torch.ones((), device=target.device)))


@torch.no_grad()
def evaluate_replay(model, replay, *, batch_size=128):
    model.eval()
    total = torch.zeros(6, device=replay.device)
    material = torch.zeros_like(total)
    for start in range(0, replay.compact.size, batch_size):
        batch = replay.gather(torch.arange(start, min(start + batch_size, replay.compact.size), device=replay.device))
        output = search_outputs(model, batch)
        total += _metric_rows(output["search_utility"], output["policy_logits"].argmax(-1), batch)
        base_utility = torch.tanh(material_cp(batch["features"]) / SCALE_CP)
        # Current-material value and one-ply material-greedy policy are explicit controls.
        material += _metric_rows(base_utility, batch["material_greedy_action"], batch)
    def metrics(values):
        count, squared, cp_mae, correct, mates, records = values.cpu().tolist()
        return {"records": int(records), "ordinary_value_records": int(count), "mate_band_records": int(mates),
                "utility_rmse": (squared / max(1, count)) ** 0.5,
                "ordinary_centipawn_mae": cp_mae / max(1, count),
                "exact_teacher_action_accuracy": correct / max(1, records)}
    return {"network": metrics(total), "material_value_oneply_policy_control": metrics(material)}


def save_teacher_checkpoint(model, path, *, teacher_manifest, training_config):
    """Separate search-teacher artifact; legacy synthetic checkpoint stays intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    config = {"name": model.kind, "width": model.width, "depth": model.depth,
              "plan_tokens": model.plans.shape[0] if model.kind == "plan" else 8}
    payload = {"checkpoint_version": 1, "schema_version": SCHEMA_VERSION,
               "kind": "search_teacher_cp_utility_checkpoint", "strength_evidence": False,
               "model_config": config,
               "utility_contract": {"kind": "bounded_search_cp_regression", "scale_cp": SCALE_CP,
                                    "epsilon": 1e-6, "mate_band": MATE_BAND, "mate_value_loss_excluded": True},
               "teacher_manifest": teacher_manifest, "training_config": training_config,
               "state_dict": {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}}
    torch.save(payload, path)
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kind": payload["kind"], "strength_evidence": False}


def load_teacher_checkpoint(path, *, device="cpu"):
    """Restricted CPU-first load, exact state dtypes and deterministic buffers."""
    path = Path(path)
    if path.stat().st_size > 128 * 1024 * 1024:
        raise ValueError("search checkpoint exceeds 128MiB limit")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("restricted search checkpoint loading failed") from error
    keys = {"checkpoint_version", "schema_version", "kind", "strength_evidence", "model_config",
            "utility_contract", "teacher_manifest", "training_config", "state_dict"}
    if not isinstance(payload, dict) or set(payload) != keys:
        raise ValueError("malformed search checkpoint envelope")
    if type(payload["checkpoint_version"]) is not int or payload["checkpoint_version"] != 1 or type(payload["schema_version"]) is not int or payload["schema_version"] != SCHEMA_VERSION:
        raise ValueError("search checkpoint schema/version mismatch")
    if payload["kind"] != "search_teacher_cp_utility_checkpoint" or payload["strength_evidence"] is not False:
        raise ValueError("unsupported search checkpoint status")
    expected_utility = {"kind": "bounded_search_cp_regression", "scale_cp": SCALE_CP,
                        "epsilon": 1e-6, "mate_band": MATE_BAND, "mate_value_loss_excluded": True}
    if payload["utility_contract"] != expected_utility or payload["utility_contract"].get("mate_value_loss_excluded") is not True:
        raise ValueError("search checkpoint utility contract mismatch")
    config = payload["model_config"]
    if not isinstance(config, dict) or set(config) != {"name", "width", "depth", "plan_tokens"} or config["name"] not in MODELS:
        raise ValueError("invalid search model configuration")
    for name, low, high in (("width", 8, 512), ("depth", 1, 32), ("plan_tokens", 1, 128)):
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f"search model {name} outside supported bounds")
    for name in ("teacher_manifest", "training_config"):
        if not isinstance(payload[name], dict):
            raise ValueError(f"search checkpoint missing {name}")
        try:
            json.dumps(payload[name], allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(f"search checkpoint nonprimitive/nonfinite {name}") from error
    teacher = payload["teacher_manifest"]
    if teacher.get("teacher_kind") != "finite_depth_classical_minimax_search_targets" or teacher.get("scale_cp") != SCALE_CP or type(teacher.get("parent_horizon")) is not int or teacher["parent_horizon"] < 1 or type(teacher.get("tree_depth")) is not int or teacher.get("tree_depth") != teacher["parent_horizon"] + 1 or type(teacher.get("evaluator")) is not int or teacher.get("evaluator") not in (0, 1):
        raise ValueError("invalid teacher construction metadata")
    model = build_model(**config)
    state = payload["state_dict"]
    expected_state = model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ValueError("search checkpoint state keys mismatch")
    for name, expected in expected_state.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype or value.layout != expected.layout:
            raise ValueError(f"search checkpoint tensor schema mismatch: {name}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"search checkpoint nonfinite tensor: {name}")
    for name, expected in model.named_buffers():
        if not torch.equal(state[name], expected):
            raise ValueError(f"search checkpoint deterministic buffer mismatch: {name}")
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {key: value for key, value in payload.items() if key != "state_dict"}


def _replay_fingerprint(replay):
    """Explicit post-learning audit boundary; never copied back into training."""
    digest = hashlib.sha256()
    count = replay.compact.size
    for name, value in sorted({**replay.compact.storage, **replay.labels}.items()):
        digest.update(name.encode())
        raw = value[:count].detach().cpu().contiguous().view(torch.uint8).flatten()
        digest.update(bytes(raw.tolist()))
    return digest.hexdigest()


def run_comparison(*, library=None, device="cuda", train_families=64, heldout_families=16,
                   steps=256, batch_size=64, seed=20260930, width=32, depth=1,
                   arena_capacity=100000, tree_depth=3, evaluator=1, output=None,
                   checkpoint_dir="results/neural/search-teacher", producer_manifest=None):
    if type(steps) is not int or steps <= 0 or type(batch_size) is not int or batch_size <= 0:
        raise ValueError("positive training steps and batch size required")
    import os
    library = library or os.environ.get("CHEN_GPU_LIBRARY")
    if not library:
        raise ValueError("pass the teacher HIP library or CHEN_GPU_LIBRARY")
    library_hash = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    producer = json.loads(Path(producer_manifest).read_text()) if producer_manifest else None
    if producer and producer.get("teacher_library_sha256") != library_hash:
        raise ValueError("teacher producer evidence/library hash mismatch")
    start = time.perf_counter()
    train_roots, heldout_roots, split = split_root_families(train_families, heldout_families, seed)
    cpu_admission_seconds = time.perf_counter() - start
    train, train_manifest = produce_dataset(train_roots, library=library, device=device,
                                            arena_capacity=arena_capacity, tree_depth=tree_depth, evaluator=evaluator)
    heldout, heldout_manifest = produce_dataset(heldout_roots, library=library, device=device,
                                                arena_capacity=arena_capacity, tree_depth=tree_depth, evaluator=evaluator)
    model_results = []
    for name in MODELS:
        model = comparable_model(name, width=width, depth=depth, seed=seed).to(train.device)
        before = evaluate_replay(model, heldout)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        generator = torch.Generator(device=train.device).manual_seed(seed + 2)
        torch.cuda.synchronize(train.device)
        start = time.perf_counter()
        losses, samples = [], []
        model.train()
        for _ in range(steps):
            batch = train.sample(batch_size, generator)
            losses.append(search_step(model, optimizer, batch))
            samples.append(batch["sequence_ids"])
        torch.cuda.synchronize(train.device)
        seconds = time.perf_counter() - start
        after = evaluate_replay(model, heldout)
        training_metrics = evaluate_replay(model, train)
        training_config = {"steps": steps, "batch_size": batch_size, "seed": seed,
                           "optimizer": "AdamW", "learning_rate": 3e-4,
                           "loss": "chosen-action CE + ordinary-score bounded-utility SmoothL1",
                           "split_source_sha256": split["source_sha256"]}
        checkpoint = save_teacher_checkpoint(model, Path(checkpoint_dir) / f"{name}.pt",
                                              teacher_manifest=train_manifest, training_config=training_config)
        model_results.append({"model": name, "parameters": sum(p.numel() for p in model.parameters()),
                              "before_heldout": before, "after_heldout": after,
                              "after_training": training_metrics, "training_seconds": seconds,
                              "first_loss": {key: float(value) for key, value in losses[0].items()},
                              "final_loss": {key: float(value) for key, value in losses[-1].items()},
                              "sample_stream_sha256": hashlib.sha256(json.dumps(torch.stack(samples).cpu().tolist()).encode()).hexdigest(),
                              "checkpoint": checkpoint})
        del model, optimizer, losses, samples, batch
    if len({result["sample_stream_sha256"] for result in model_results}) != 1:
        raise RuntimeError("architectures consumed different sample streams")
    report = {"kind": "resident_finite_depth_search_teacher_comparison", "strength_evidence": False,
              "environment": {"torch": str(torch.__version__), "torch_hip": torch.version.hip,
                              "architecture": getattr(torch.cuda.get_device_properties(train.device), "gcnArchName", "unknown")},
              "config": {"steps": steps, "batch_size": batch_size, "seed": seed, "width": width, "depth": depth,
                         "tree_depth": tree_depth, "parent_horizon": tree_depth - 1, "evaluator": evaluator,
                         "scale_cp": SCALE_CP, "mate_band_value_excluded": MATE_BAND,
                         "initialization": "shared input/position/norm/value/policy weights equal; architectures not parameter-matched"},
              "cpu_root_split_seconds": cpu_admission_seconds,
              "split": split, "train_corpus": train_manifest, "heldout_corpus": heldout_manifest,
              "data_fingerprints": {"train_sha256": _replay_fingerprint(train), "heldout_sha256": _replay_fingerprint(heldout),
                                    "scope": "post-learning payload audit download only, never returned to producer/training"},
              "models": model_results,
              "library": {"path": str(Path(library).resolve()), "sha256": library_hash},
              "producer_evidence": producer,
              "source_sha256": {f"neural/{name}": hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                for name in ("teacher.py", "bridge.py", "compact.py", "models.py", "features.py")},
              "limitations": ["uniform legal random-game root distribution, finite-depth classical teacher, no real outcomes",
                              "policy accuracy uses first-tie chosen action; equivalent tied actions may be scored wrong",
                              "scalar ingress/control synchronizes; payload diagnostic checksums download after learning",
                              "single seed, short bounded training; no engine matches or Lichess rating evidence"]}
    if output:
        path = Path(output); path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train-families", type=int, default=64)
    parser.add_argument("--heldout-families", type=int, default=16)
    parser.add_argument("--steps", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--width", type=int, default=32)
    parser.add_argument("--depth", type=int, default=1)
    parser.add_argument("--arena-capacity", type=int, default=100000)
    parser.add_argument("--tree-depth", type=int, default=3)
    parser.add_argument("--evaluator", type=int, choices=(0, 1), default=1)
    parser.add_argument("--checkpoint-dir", default="results/neural/search-teacher")
    parser.add_argument("--producer-manifest")
    parser.add_argument("--output")
    print(json.dumps(run_comparison(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
