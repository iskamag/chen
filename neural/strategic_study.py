"""Fresh complete-root strategic supervision, explicit action causality and splits."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .bridge import RuleArena, ResidentSpan, position_from_fen
from .compact import CompactReplay, decode_features
from .models import MODELS
from .teacher import (MATE_BAND, SCALE_CP, RootFamily, RootSample, _candidate_family,
                      _family_positions, backed_up_tree, comparable_model,
                      deduplicate_parents, export_teacher, canonical_position_key)
from .strategic_transport import StrategicPredictor, derive_effects, successor_pieces

CONTROLS = ("transport", "conditioned", "action_blind", "transport_fixed_encoder", "transport_zero_anchor")
LOSS_WEIGHTS = {"value": 1., "score_delta": .5, "refresh": .05, "latent_delta": .05}


def tapered_cp(features):
    """Exact integer source-equivalent static Tapered, canonical STM perspective."""
    pieces = features[:, :12].to(torch.int64).flatten(2)
    square = torch.arange(64, device=features.device)
    file, rank = square % 8, square // 8
    weights = torch.tensor([100, 320, 330, 500, 900, 0], device=features.device)
    phase_weights = torch.tensor([0, 1, 1, 2, 4, 0], device=features.device)
    mg, eg = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device), torch.zeros(features.shape[0], dtype=torch.int64, device=features.device)
    counts = pieces.sum(-1)
    phase = ((counts[:, :6] + counts[:, 6:]) * phase_weights).sum(-1).clamp_max(24)
    for color in range(2):
        relative_rank = rank if color == 0 else 7 - rank
        center = 14 - (2 * file - 7).abs() - (2 * relative_rank - 7).abs()
        m = torch.stack((relative_rank * 8 + center * 2, center * 6, center * 3 + relative_rank * 2,
                         relative_rank * 2 + 18 * (relative_rank == 6), center,
                         -center * 5 - relative_rank * 4 + 35 * ((relative_rank == 0) & ((file == 2) | (file == 6)))))
        e = torch.stack((relative_rank * 14 + center, center * 4, center * 3, center, center * 2, center * 6))
        sign = 1 if color == 0 else -1
        mg += sign * (pieces[:, color * 6:color * 6 + 6] * (weights[:, None] + m)).sum((1, 2))
        eg += sign * (pieces[:, color * 6:color * 6 + 6] * (weights[:, None] + e)).sum((1, 2))
        mg += sign * 25 * (counts[:, color * 6 + 2] >= 2)
        eg += sign * 40 * (counts[:, color * 6 + 2] >= 2)
    return torch.div(mg * phase + eg * (24 - phase), 24, rounding_mode="trunc").clamp(-29000, 29000).to(torch.int32)


def _family(record):
    roots = tuple(RootSample(row["family_id"], row["ply"], row["fen"], tuple(row["previous_hashes"])) for row in record["roots"])
    return RootFamily(record["family_id"], record["origin_fen"], tuple(record["moves"]), roots)


def fresh_split(train_count=96, validation_count=24, test_count=24, seed=20261003,
                spent_reports=(), roots_at=(12, 24)):
    """Three-way lineage/canonical split excluding every previously consumed input."""
    if any(type(count) is not int or count < 1 for count in (train_count, validation_count, test_count)):
        raise ValueError("positive train/validation/test family counts required")
    spent, spent_sources = set(), []
    for path in spent_reports:
        raw = Path(path).read_bytes(); report = json.loads(raw)
        if "canonical_position_fens" in report:
            spent.update(canonical_position_key(fen) for fen in report["canonical_position_fens"])
        else:
            records = report["split"]["source_records"]
            for families in records.values():
                for record in families:
                    spent.update(_family_positions(_family(record)))
        spent_sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
    groups = {"train": [], "validation": [], "test": []}
    counts = {"train": train_count, "validation": validation_count, "test": test_count}
    keys = {name: set() for name in groups}
    rejected = {"terminal_or_repeated_history": 0, "past_or_split_canonical_collision": 0}
    for family_id in range(30 * sum(counts.values())):
        candidate = _candidate_family(family_id, seed, roots_at)
        if candidate is None:
            rejected["terminal_or_repeated_history"] += 1; continue
        states = _family_positions(candidate)
        if states & (spent | keys["train"] | keys["validation"] | keys["test"]):
            rejected["past_or_split_canonical_collision"] += 1; continue
        name = next(name for name in groups if len(groups[name]) < counts[name])
        groups[name].append(candidate); keys[name].update(states)
        if all(len(groups[name]) == counts[name] for name in groups):
            break
    if any(len(groups[name]) != counts[name] for name in groups):
        raise RuntimeError("could not build requested fresh split")
    records = {name: [asdict(family) for family in families] for name, families in groups.items()}
    manifest = {"seed": seed, "roots_at": list(roots_at), "families": counts,
                "source_records": records, "canonical_counts": {name: len(values) for name, values in keys.items()},
                "canonical_overlap": 0, "previous_consumed_overlap": 0, "spent_canonical_positions": len(spent),
                "spent_report_fingerprints": spent_sources, "rejected_candidates": rejected,
                "source_sha256": hashlib.sha256(json.dumps(records, sort_keys=True).encode()).hexdigest(),
                "scope": "whole legal-game lineages and root/current/successor canonical feature identities; shared early unobserved ancestry"}
    return {name: tuple(families) for name, families in groups.items()}, manifest


def join_complete_root(first, child_packed, child_labels, child_sidecar, child_manifest, root_labels):
    """Exact root-edge -> child-parent node join; reject the whole incomplete root."""
    if any(first.manifest[key] != root_labels.manifest[key] for key in ("context_uuid", "root_epoch", "child_start", "rows")):
        raise ValueError("root labels do not share context/epoch")
    if any(first.manifest[key] != child_manifest[key] for key in ("context_uuid", "root_epoch")):
        raise ValueError("child labels do not share context/epoch")
    children = first.sidecar["child_node"]
    parents = child_sidecar["parent_node"]
    if not bool((first.sidecar["parent_node"] == 0).all()) or not bool((parents[1:] > parents[:-1]).all()):
        raise ValueError("root-edge/source-parent node order invalid")
    if not parents.numel():
        return None, "no_child_scores"
    lookup = torch.searchsorted(parents, children)
    safe = lookup.clamp_max(parents.numel() - 1)
    matched = (lookup < parents.numel()) & (parents.index_select(0, safe) == children)
    score = child_labels["search_score_cp"].index_select(0, safe)
    if not bool(matched.all()) or bool(first.tensors["terminal"].any()):
        return None, "terminal_or_missing_child_scores"
    if not bool((score.to(torch.int64).abs() < MATE_BAND).all()):
        return None, "mate_band_child_scores"
    if not bool((first.tensors["legal_counts"] == children.numel()).all()):
        raise ValueError("root legal support is incomplete")
    # Match the exact child state, not merely numeric node identifiers.
    for first_name, child_name in (("next_pieces", "pieces"), ("next_state", "state")):
        if not bool((first.tensors[first_name] == child_packed[child_name].index_select(0, safe)).all()):
            raise ValueError("joined parent state differs from exported root child")
    root_score = root_labels.tensors["scores"]
    if not bool((root_labels.tensors["parent_ply"] == 0).all()) or not bool((child_labels["teacher_parent_ply"].index_select(0, safe) == 1).all()):
        raise ValueError("root/child score ply mismatch")
    if not bool((root_score == -score.min()).all()):
        raise ValueError("root depth3 score is not maximum negated child horizon2 score")
    optimal = score == score.min()
    best = root_labels.tensors["best_actions"]
    if not bool(((first.tensors["actions"] == best) & optimal).any()):
        raise ValueError("native root best action does not attain child score minimum")
    return {"child_score_cp": score, "root_score_cp": root_score,
            "target_action": best}, None


class StrategicReplay:
    """Owned compact root->child records plus exact child scores and root groups."""
    def __init__(self, capacity, device):
        self.compact = CompactReplay(capacity, device)
        self.device = self.compact.device
        self.labels = {name: torch.empty(capacity, dtype=dtype, device=self.device) for name, dtype in
                       (("child_score_cp", torch.int32), ("root_score_cp", torch.int32),
                        ("root_ids", torch.int64), ("child_node", torch.int64))}
        self.root_sizes, self.family_ids = [], []

    def append_root(self, first, labels, family_id):
        count = first.tensors["pieces"].shape[0]
        if count < 2 or set(labels) != {"child_score_cp", "root_score_cp", "target_action"}:
            raise ValueError("complete root requires at least two actions and exact score fields")
        for name, dtype in (("child_score_cp", torch.int32), ("root_score_cp", torch.int32), ("target_action", torch.int64)):
            value = labels[name]
            if not isinstance(value, torch.Tensor) or value.shape != (count,) or value.dtype != dtype or value.device != self.device:
                raise ValueError("root label schema/device mismatch")
        if not bool((labels["child_score_cp"].to(torch.int64).abs() < MATE_BAND).all()):
            raise ValueError("root labels include mate band")
        if (not bool((labels["root_score_cp"] == -labels["child_score_cp"].min()).all()) or
                not bool((labels["target_action"] == labels["target_action"][0]).all()) or
                not bool(((first.tensors["actions"] == labels["target_action"]) &
                          (labels["child_score_cp"] == labels["child_score_cp"].min())).any())):
            raise ValueError("root score/action labels do not back up child scores")
        if (not bool((first.tensors["legal_counts"] == count).all()) or
                not bool((torch.sort(first.tensors["actions"]).values == first.tensors["legal_actions"][0, :count].long()).all()) or
                not bool((first.tensors["pieces"] == first.tensors["pieces"][0]).all()) or
                not bool((first.tensors["state"] == first.tensors["state"][0]).all())):
            raise ValueError("root group observations/actions do not cover one complete support")
        if self.compact.total_inserted + count > self.compact.capacity:
            raise ValueError("strategic corpus cannot overwrite an incomplete root group")
        root_id = len(self.root_sizes)
        tensors = dict(first.tensors)
        tensors["target_actions"] = labels["target_action"]
        tensors["game_ids"] = torch.full((count,), family_id, device=self.device, dtype=torch.int64)
        start = self.compact.total_inserted
        try:
            self.compact.append(tensors)
            self.labels["child_score_cp"][start:start + count].copy_(labels["child_score_cp"])
            self.labels["root_score_cp"][start:start + count].copy_(labels["root_score_cp"])
            self.labels["root_ids"][start:start + count].fill_(root_id)
            self.labels["child_node"][start:start + count].copy_(first.sidecar["child_node"])
        finally:
            if self.device.type == "cuda":
                torch.cuda.synchronize(self.device)
        self.root_sizes.append(count); self.family_ids.append(family_id)

    def seal(self):
        if not self.root_sizes:
            raise ValueError("no complete ordinary root groups retained")
        counts = torch.tensor(self.root_sizes, dtype=torch.int64, device=self.device)
        self.root_offsets = torch.cat((torch.zeros(1, device=self.device, dtype=torch.int64), counts.cumsum(0)[:-1]))
        self.root_counts = counts
        # Precompute informative same-material sibling pairs entirely on the device.
        pairs_a, pairs_b = [], []
        offset = 0
        for count in self.root_sizes:
            indexes = torch.arange(offset, offset + count, device=self.device)
            batch = self.gather(indexes)
            pieces = successor_pieces(batch["root_features"], derive_effects(batch["root_features"], batch["actions"]))
            material = piece_material_cp(pieces)
            truth = batch["child_score_cp"]
            pair = ((material[:, None] == material[None]) &
                    ((truth[:, None] - truth[None]).abs() >= 100) &
                    (indexes[:, None] < indexes[None]))
            a, b = pair.nonzero(as_tuple=True)
            pairs_a.append(indexes[a]); pairs_b.append(indexes[b]); offset += count
        self.same_material_a = torch.cat(pairs_a)
        self.same_material_b = torch.cat(pairs_b)

    def gather(self, indexes):
        storage = self.compact.storage
        return {"indices": indexes.clone(),
                "root_features": decode_features(storage["pieces"].index_select(0, indexes), storage["state"].index_select(0, indexes)),
                "successor_features": decode_features(storage["next_pieces"].index_select(0, indexes), storage["next_state"].index_select(0, indexes)),
                "actions": storage["actions"].index_select(0, indexes),
                "game_ids": storage["game_ids"].index_select(0, indexes),
                **{key: value.index_select(0, indexes) for key, value in self.labels.items()}}

    def pair_batch(self, pairs=16, generator=None):
        roots = torch.randint(len(self.root_sizes), (pairs,), device=self.device, generator=generator)
        counts = self.root_counts.index_select(0, roots)
        offsets = self.root_offsets.index_select(0, roots)
        a = offsets + (torch.rand(pairs, device=self.device, generator=generator) * counts).long()
        b = offsets + (torch.rand(pairs, device=self.device, generator=generator) * (counts - 1)).long()
        b += b >= a
        # Half broad root-uniform pairs, half informative pairs uniform over the available pair set.
        if self.same_material_a.numel():
            choices = torch.randint(self.same_material_a.numel(), (pairs,), device=self.device, generator=generator)
            a = torch.cat((a, self.same_material_a.index_select(0, choices)))
            b = torch.cat((b, self.same_material_b.index_select(0, choices)))
        indices = torch.stack((a, b), 1).flatten()
        return self.gather(indices)


def piece_material_cp(pieces):
    weights = pieces.new_tensor([100, 320, 330, 500, 900, 0])
    counts = pieces.sum((-1, -2))
    return ((counts[:, :6] - counts[:, 6:]) * weights).sum(-1).clamp(-29000, 29000)


def produce_strategic(families, *, library, device="cuda", arena_capacity=250000, tree_depth=3):
    import chess
    capacity = sum(chess.Board(root.fen).legal_moves.count() for family in families for root in family.roots)
    replay = StrategicReplay(capacity, device)
    coverage = {"input_roots": 0, "input_actions": 0, "retained_roots": 0, "retained_actions": 0, "omissions": {}}
    records = []
    started = time.perf_counter()
    with RuleArena(library, arena_capacity, device) as arena:
        for family in families:
            for root in family.roots:
                coverage["input_roots"] += 1
                legal_count = chess.Board(root.fen).legal_moves.count()
                coverage["input_actions"] += legal_count
                if legal_count < 2:
                    reason = "fewer_than_two_legal_actions"
                    coverage["omissions"][reason] = coverage["omissions"].get(reason, 0) + 1
                    records.append({"root": asdict(root), "omitted": reason}); continue
                try:
                    second, teacher, construction = backed_up_tree(arena, position_from_fen(root.fen), root.previous_hashes,
                                                                   tree_depth=tree_depth, evaluator=1)
                except ValueError as error:
                    if "no nonterminal first-ply parent" not in str(error):
                        raise
                    reason = "no_nonterminal_child_scores"
                    coverage["omissions"][reason] = coverage["omissions"].get(reason, 0) + 1
                    records.append({"root": asdict(root), "omitted": reason}); continue
                child_packed, child_labels, child_sidecar = deduplicate_parents(second, teacher)
                first_span = ResidentSpan(**construction["schedule"][0]["next_span"])
                first = arena.export(first_span)
                root_labels = export_teacher(arena, first_span)
                labels, reason = join_complete_root(first, child_packed, child_labels, child_sidecar, second.manifest, root_labels)
                if labels is None:
                    coverage["omissions"][reason] = coverage["omissions"].get(reason, 0) + 1
                    records.append({"root": asdict(root), "omitted": reason, "construction": construction}); continue
                root_features = decode_features(first.tensors["pieces"], first.tensors["state"])
                exact_pieces = successor_pieces(root_features, derive_effects(root_features, first.tensors["actions"]))
                observed = decode_features(first.tensors["next_pieces"], first.tensors["next_state"])
                if not bool((exact_pieces == observed[:, :12]).all()):
                    raise ValueError("resident legal-move algebra differs from simulator")
                replay.append_root(first, labels, family.family_id)
                coverage["retained_roots"] += 1; coverage["retained_actions"] += legal_count
                records.append({"root": asdict(root), "construction": construction,
                                "source": first.manifest, "root_teacher": root_labels.manifest,
                                "child_teacher": teacher.manifest})
    replay.seal()
    manifest = {"coverage": coverage, "lineage": records, "producer_seconds": time.perf_counter() - started,
                "root_horizon": tree_depth, "child_horizon": tree_depth - 1, "evaluator": 1,
                "same_material_gap_threshold_cp": 100, "same_material_pairs": replay.same_material_a.numel(),
                "root_sizes": replay.root_sizes, "family_ids": replay.family_ids,
                "input_source": "CPU roots/history only; root/child observations, actions and score labels exported resident",
                "score_perspective": "child side-to-move; root action value is negative child score",
                "coverage_policy": "whole root omitted if any terminal, missing or mate-band child target; all retained legal actions complete"}
    return replay, manifest


def strategic_loss(model, target_encoder, batch):
    """Successor observations enter frozen target branch only; prediction is causal."""
    output = model.predict(batch["root_features"], batch["actions"])
    truth = batch["child_score_cp"].float() / SCALE_CP
    value = F.smooth_l1_loss(output["value_cp"] / SCALE_CP, truth)
    pairs = output["value_cp"].reshape(-1, 2) / SCALE_CP
    target_pairs = truth.reshape(-1, 2)
    pair_pieces = output["material_cp"].reshape(-1, 2)
    same = (pair_pieces[:, 0] == pair_pieces[:, 1]) & ((target_pairs[:, 0] - target_pairs[:, 1]).abs() >= .125)
    delta_rows = F.smooth_l1_loss(pairs[:, 0] - pairs[:, 1], target_pairs[:, 0] - target_pairs[:, 1], reduction="none")
    delta = (delta_rows * same).sum() / same.sum().clamp_min(1)
    with torch.no_grad():
        target = target_encoder.encode(batch["successor_features"])
    predicted = F.normalize(output["latent"], dim=-1)
    target = F.normalize(target, dim=-1)
    refresh = F.mse_loss(predicted, target)
    p = predicted.reshape(-1, 2, 64, model.base.width)
    t = target.reshape_as(p)
    latent_delta = F.mse_loss(p[:, 0] - p[:, 1], t[:, 0] - t[:, 1])
    terms = {"value": value, "score_delta": delta, "refresh": refresh, "latent_delta": latent_delta}
    return {"total": sum(LOSS_WEIGHTS[key] * term for key, term in terms.items()), **terms}, output


@torch.no_grad()
def evaluate_strategic(model, target_encoder, replay, batch_size=128):
    """All retained legal actions; aggregate reporting is an explicit cold boundary."""
    from .value_study import metrics
    model.eval(); target_encoder.eval()
    cp, shuffled_cp, refreshed_cp, material, tapered = [], [], [], [], []
    latents, target_latents, scores, games, roots = [], [], [], [], []
    shuffled_actions = replay.compact.storage["actions"][:replay.compact.size].clone()
    offset = 0
    for count in replay.root_sizes:
        shuffled_actions[offset:offset + count] = shuffled_actions[offset:offset + count].roll(1)
        offset += count
    for start in range(0, replay.compact.size, batch_size):
        indexes = torch.arange(start, min(start + batch_size, replay.compact.size), device=replay.device)
        batch = replay.gather(indexes)
        prediction = model.predict(batch["root_features"], batch["actions"])
        changed = model.predict(batch["root_features"], shuffled_actions.index_select(0, indexes))
        refreshed = model.refresh(batch["successor_features"])
        cp.append(prediction["value_cp"])
        # Retain the true action's anchor, replacing only learned strategic residual.
        shuffled_cp.append(prediction["material_cp"] + changed["residual_cp"] if model.anchor_kind == "material" else changed["residual_cp"])
        refreshed_cp.append(refreshed["value_cp"])
        material.append(prediction["material_cp"])
        tapered.append(tapered_cp(prediction["pieces"]).float())
        latents.append(F.normalize(prediction["latent"], dim=-1))
        target_latents.append(F.normalize(target_encoder.encode(batch["successor_features"]), dim=-1))
        scores.append(batch["child_score_cp"]); games.append(batch["game_ids"]); roots.append(batch["root_ids"])
    cp, shuffled_cp, refreshed_cp, material, tapered = map(torch.cat, (cp, shuffled_cp, refreshed_cp, material, tapered))
    latent, target = torch.cat(latents), torch.cat(target_latents)
    scores, games, roots = map(torch.cat, (scores, games, roots))
    def score_metrics(values):
        result = metrics(values, (values / SCALE_CP).tanh(), scores, material, games, roots)
        result["ordering"]["scope"] = "all legal actions for retained complete ordinary nonterminal root groups; lower child-mover CP is better"
        result["material_ordering"]["scope"] = result["ordering"]["scope"]
        return result
    a, b = replay.same_material_a, replay.same_material_b
    if a.numel():
        predicted_gap = cp[a] - cp[b]
        true_gap = (scores[a] - scores[b]).float()
        shuffled_gap = shuffled_cp[a] - shuffled_cp[b]
        latent_pair_indices = torch.linspace(0, a.numel() - 1, min(a.numel(), 256), device=replay.device).long()
        la, lb = a[latent_pair_indices], b[latent_pair_indices]
        p_delta, t_delta = latent[la] - latent[lb], target[la] - target[lb]
        intervention = torch.stack(((predicted_gap - true_gap).square().mean().sqrt(),
                                    (shuffled_gap - true_gap).square().mean().sqrt(),
                                    ((predicted_gap.sign() == true_gap.sign()).float() + .5 * (predicted_gap == 0)).mean(),
                                    predicted_gap.abs().mean(), true_gap.abs().mean(),
                                    p_delta.square().mean().sqrt() / t_delta.square().mean().sqrt().clamp_min(1e-12),
                                    (p_delta - t_delta).square().mean().sqrt())).cpu().tolist()
        names = ("score_gap_rmse_cp", "shuffled_residual_score_gap_rmse_cp", "score_direction_concordance",
                 "predicted_mean_abs_score_gap_cp", "teacher_mean_abs_score_gap_cp",
                 "latent_action_separation_ratio", "latent_action_delta_rmse")
        subset = dict(zip(names, intervention))
    else:
        subset = {"score_gap_rmse_cp": None, "score_direction_concordance": None}
    subset.update(latent_pairs_sampled=min(a.numel(), 256),
                  latent_sampling="deterministic evenly spaced up to256 pairs; scalar score diagnostics use all pairs",
                  pairs=a.numel(), scope="same-material siblings with abs teacher gap>=100cp; all eligible pairs")
    diagnostics = torch.stack(((latent - target).square().mean().sqrt(),
                               latent.var(0, unbiased=False).mean(),
                               target.var(0, unbiased=False).mean())).cpu().tolist()
    return {"network": score_metrics(cp), "material": score_metrics(material), "static_tapered": score_metrics(tapered),
            "action_shuffle_true_anchor": score_metrics(shuffled_cp),
            "full_refresh_diagnostic": score_metrics(refreshed_cp),
            "same_material_teacher_gap": subset,
            "latent_refresh_rmse": diagnostics[0], "latent_state_variance": diagnostics[1],
            "fixed_target_state_variance": diagnostics[2]}


def validation_selection(runs):
    """Written gate over means of fixed seeds; cannot read or select on test metrics."""
    if not runs or any("after_validation" not in run or "after_test" in run for run in runs):
        raise ValueError("selection requires completed validation-only runs")
    candidates = []
    for architecture in MODELS:
        for control in CONTROLS:
            selected = [row for row in runs if row["architecture"] == architecture and row["control"] == control]
            if not selected:
                continue
            mae = sum(row["after_validation"]["network"]["ordinary_centipawn_mae"] for row in selected) / len(selected)
            regret = sum(row["after_validation"]["network"]["ordering"]["mean_selected_child_regret_cp"] for row in selected) / len(selected)
            material_mae = selected[0]["after_validation"]["material"]["ordinary_centipawn_mae"]
            tapered_mae = selected[0]["after_validation"]["static_tapered"]["ordinary_centipawn_mae"]
            material_regret = selected[0]["after_validation"]["material"]["ordering"]["mean_selected_child_regret_cp"]
            tapered_regret = selected[0]["after_validation"]["static_tapered"]["ordering"]["mean_selected_child_regret_cp"]
            candidates.append({"architecture": architecture, "control": control, "seeds": len(selected),
                               "validation_cp_mae_mean": mae, "validation_root_regret_mean": regret,
                               "mae_gate": min(material_mae, tapered_mae), "regret_gate": min(material_regret, tapered_regret),
                               "eligible": mae < min(material_mae, tapered_mae) and regret < min(material_regret, tapered_regret)})
    eligible = [row for row in candidates if row["eligible"]]
    winner = min(eligible, key=lambda row: (row["validation_root_regret_mean"], row["validation_cp_mae_mean"],
                                           row["architecture"], row["control"])) if eligible else None
    return {"kind": "sealed_validation_only_strategic_selection", "selected": winner, "candidates": candidates,
            "decision_rule": "seed-mean ordinary CP MAE AND selected-child regret must beat BOTH material and static Tapered; choose lowest regret then MAE then names",
            "test_labels_read": False, "strength_evidence": False}


def _atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def _tensor_fingerprint(values):
    digest = hashlib.sha256()
    for key, tensor in sorted(values.items()):
        digest.update(key.encode())
        digest.update(bytes(tensor.detach().cpu().contiguous().reshape(-1).view(torch.uint8).tolist()))
    return digest.hexdigest()


def _build(architecture, control, width, depth, seed):
    base = comparable_model(architecture, width=width, depth=depth, seed=seed)
    mode = "transport" if control.startswith("transport") else control
    return StrategicPredictor(base, mode=mode,
                              anchor_kind="zero" if control == "transport_zero_anchor" else "material",
                              freeze_encoder=control == "transport_fixed_encoder")


def study_configuration(*, train_families=96, validation_families=24, test_families=24,
                        corpus_seed=20261003, seeds=(20261003, 20261004), steps=1024, pairs=8,
                        width=16, depth=1, tree_depth=3, arena_capacity=250000):
    integer_options = (train_families, validation_families, test_families, steps, pairs, width, depth, arena_capacity)
    if any(type(value) is not int or value < 1 for value in integer_options) or type(tree_depth) is not int or tree_depth != 3:
        raise ValueError("positive integer study options and exact depth3 required")
    if type(corpus_seed) is not int or not seeds or any(type(seed) is not int for seed in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("integer corpus seed and distinct integer initialization seeds required")
    return {"families": {"train": train_families, "validation": validation_families, "test": test_families},
            "corpus_seed": corpus_seed, "seeds": list(seeds), "steps": steps, "pairs": pairs,
            "width": width, "depth": depth, "tree_depth": tree_depth, "root_horizon": tree_depth,
            "child_target_horizon": tree_depth - 1, "evaluator": "classical_tapered1",
            "arena_capacity": arena_capacity, "architectures": list(MODELS), "controls": list(CONTROLS),
            "loss_weights": LOSS_WEIGHTS, "loss_scale_cp": SCALE_CP, "teacher_gap_cp": 100,
            "optimizer": "AdamW", "learning_rate": .0003, "grad_clip": 1.,
            "latent_target": "frozen initial comparable encoder; never updated or optimized",
            "batch": f"{pairs} broad root-uniform sibling pairs plus{pairs} informative same-material pairs when available; {4 * pairs}rows with informative pool",
            "selection": "seed-mean CP MAE AND complete-root regret beat BOTH material and static Tapered; choose lowest regret then MAE then names",
            "test_protocol": "test labels produced only after validation selection JSON written and sealed; all final models evaluated once; no later optimization"}


def run_strategic(*, library, producer_manifest, predeclaration, output,
                  spent_reports, checkpoint_dir="results/neural/strategic-study", device="cuda", **configuration):
    """Bounded predeclared pipeline. Native production requires a coordinated GPU."""
    from .strategic_transport import save_strategic_checkpoint, load_strategic_checkpoint
    config = study_configuration(**configuration)
    declared = json.loads(Path(predeclaration).read_text())
    actual_spent = [{"path": str(path), "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()} for path in spent_reports]
    if declared.get("spent_reports") != actual_spent:
        raise ValueError("spent report selection/order/hash differs from predeclaration")
    if declared.get("configuration") != config:
        raise ValueError("study configuration differs from predeclaration")
    for path, digest in declared["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("predeclared source changed: " + path)
    if torch.device(device).type != "cuda" or not torch.cuda.is_available():
        raise ValueError("resident strategic study requires recovered/coordinated GPU; no CPU training fallback")
    producer_raw = Path(producer_manifest).read_bytes()
    producer_sha = hashlib.sha256(producer_raw).hexdigest()
    if producer_sha != declared["producer_manifest_sha256"]:
        raise ValueError("producer manifest differs from predeclaration")
    producer = json.loads(producer_raw)
    library_sha = hashlib.sha256(Path(library).read_bytes()).hexdigest()
    if library_sha != producer.get("teacher_library_sha256"):
        raise ValueError("producer evidence/library hash mismatch")
    started = time.perf_counter()
    families, split = fresh_split(train_count=config["families"]["train"],
                                  validation_count=config["families"]["validation"], test_count=config["families"]["test"],
                                  seed=config["corpus_seed"], spent_reports=spent_reports)
    split_seconds = time.perf_counter() - started
    # Configuration names differ deliberately only at the family-admission helper boundary.
    if any(len(families[name]) != config["families"][name] for name in families):
        raise ValueError("family counts disagree with predeclared configuration")
    train, train_manifest = produce_strategic(families["train"], library=library, device=device,
                                              arena_capacity=config["arena_capacity"], tree_depth=config["tree_depth"])
    validation, validation_manifest = produce_strategic(families["validation"], library=library, device=device,
                                                        arena_capacity=config["arena_capacity"], tree_depth=config["tree_depth"])
    report = {"kind": "fresh_resident_strategic_action_study", "status": "training", "strength_evidence": False,
              "configuration": config, "split": split, "cpu_split_seconds": split_seconds,
              "train_corpus": train_manifest, "validation_corpus": validation_manifest,
              "predeclaration_sha256": hashlib.sha256(Path(predeclaration).read_bytes()).hexdigest(),
              "producer_manifest_sha256": producer_sha, "producer_evidence": producer, "library": {"path": library, "sha256": library_sha},
              "environment": {"torch": str(torch.__version__), "torch_hip": torch.version.hip,
                              "architecture": torch.cuda.get_device_properties(train.device).gcnArchName}, "runs": []}
    output = Path(output); output.parent.mkdir(parents=True, exist_ok=True)
    def publish():
        _atomic_json(output, report)
        with output.with_suffix(".jsonl").open("a") as journal:
            journal.write(json.dumps({"status": report["status"], "completed_runs": len(report["runs"]),
                                      "test_evaluated_runs": sum("after_test" in row for row in report["runs"]),
                                      "report_sha256": hashlib.sha256(output.read_bytes()).hexdigest()}) + "\n")
            journal.flush()
    publish()
    for seed in config["seeds"]:
        for architecture in MODELS:
            target = comparable_model(architecture, width=config["width"], depth=config["depth"], seed=seed).to(train.device).eval()
            target.requires_grad_(False)
            target_sha = _tensor_fingerprint(target.state_dict())  # cold model-weight audit, not data ingress
            for control in CONTROLS:
                model = _build(architecture, control, config["width"], config["depth"], seed).to(train.device)
                before = evaluate_strategic(model, target, validation)
                optimizer = torch.optim.AdamW(model.active_parameters(), lr=config["learning_rate"])
                generator = torch.Generator(device=train.device).manual_seed(seed + 2)
                losses, norms, sample_stream = [], [], []
                torch.cuda.synchronize(train.device); started = time.perf_counter(); model.train()
                for _ in range(config["steps"]):
                    batch = train.pair_batch(config["pairs"], generator)
                    optimizer.zero_grad(set_to_none=True)
                    terms, _ = strategic_loss(model, target, batch)
                    terms["total"].backward()
                    norms.append(torch.nn.utils.clip_grad_norm_(model.active_parameters(), config["grad_clip"]).detach())
                    optimizer.step()
                    losses.append(torch.stack([terms[key].detach() for key in ("total", *LOSS_WEIGHTS)]))
                    sample_stream.append(batch["indices"])
                torch.cuda.synchronize(train.device); seconds = time.perf_counter() - started
                after = evaluate_strategic(model, target, validation)
                train_metrics = evaluate_strategic(model, target, train)
                path = Path(checkpoint_dir) / f"{architecture}-{control}-{seed}.pt"
                checkpoint = save_strategic_checkpoint(model, path, provenance={"configuration": config, "seed": seed,
                                                        "teacher": train_manifest, "split_sha256": split["source_sha256"],
                                                        "fixed_target_state_sha256": target_sha, "strength_evidence": False})
                diagnostics = torch.stack((torch.stack(norms).mean(), torch.stack(norms).max(),
                                           (torch.stack(norms) > config["grad_clip"]).float().mean())).cpu().tolist()
                run = {"architecture": architecture, "control": control, "seed": seed,
                       "active_parameters": sum(p.numel() for p in model.active_parameters()),
                       "before_validation": before, "after_validation": after, "after_training": train_metrics,
                       "training_seconds": seconds, "loss_first": dict(zip(("total", *LOSS_WEIGHTS), losses[0].cpu().tolist())),
                       "loss_last": dict(zip(("total", *LOSS_WEIGHTS), losses[-1].cpu().tolist())),
                       "gradient_norm_mean": diagnostics[0], "gradient_norm_max": diagnostics[1],
                       "gradient_clipped_fraction": diagnostics[2], "checkpoint": checkpoint,
                       "sample_stream_sha256": _tensor_fingerprint({"indices": torch.stack(sample_stream)}),
                       "fixed_target_state_sha256": target_sha}
                report["runs"].append(run); publish()
                print(json.dumps({"stage": "validation", "architecture": architecture, "control": control, "seed": seed,
                                  "cp_mae": after["network"]["ordinary_centipawn_mae"],
                                  "regret_cp": after["network"]["ordering"]["mean_selected_child_regret_cp"]}), flush=True)
                del model, optimizer, losses, norms, sample_stream
            del target
    if len(report["runs"]) != len(config["seeds"]) * len(MODELS) * len(CONTROLS):
        raise RuntimeError("incomplete predeclared factorial")
    for seed in config["seeds"]:
        if len({row["sample_stream_sha256"] for row in report["runs"] if row["seed"] == seed}) != 1:
            raise RuntimeError("control sample streams differ")
    decision = validation_selection(report["runs"])
    decision["configuration"] = config
    selection_path = output.with_name(output.stem + "-selection.json")
    _atomic_json(selection_path, decision)
    report["selection"] = {"record": decision, "path": str(selection_path),
                           "sha256": hashlib.sha256(selection_path.read_bytes()).hexdigest()}
    report["status"] = "validation_selection_sealed_before_test_labels"; publish()
    test, test_manifest = produce_strategic(families["test"], library=library, device=device,
                                            arena_capacity=config["arena_capacity"], tree_depth=config["tree_depth"])
    report["test_corpus"] = test_manifest
    for run in report["runs"]:
        model, _ = load_strategic_checkpoint(run["checkpoint"]["path"], device=device)
        target = comparable_model(run["architecture"], width=config["width"], depth=config["depth"], seed=run["seed"]).to(test.device).eval()
        target.requires_grad_(False)
        run["after_test"] = evaluate_strategic(model, target, test)
        publish(); del model, target
    report["data_fingerprints"] = {name: _tensor_fingerprint({key: tensor[:replay.compact.size] for key, tensor in
                                                             {**replay.compact.storage, **replay.labels}.items()})
                                   for name, replay in (("train", train), ("validation", validation), ("test", test))}
    report["status"] = "completed_once_after_sealed_selection"
    report["source_sha256"] = declared["source_sha256"]
    report["limitations"] = ["finite-depth classical supervision, synthetic legal-game distribution, no outcomes/matches/rating",
                              "whole-root omission excludes forced-terminal/mate-band coverage",
                              "two initialization/sample seeds on one fresh fixed corpus; correlated test families",
                              "exact move/pseudoattack covariates are declared handcrafted priors",
                              "test diagnostics for all frozen controls are reported once, no subsequent selection/tuning",
                              "scalar producer/control synchronization permitted; cold model/checksum downloads explicit"]
    publish()
    return report


def run_timing_pilot(*, library, predeclaration, output, pilot_split, device="cuda"):
    """Disposable lineage timing only; no main validation/test labels touched."""
    declared = json.loads(Path(predeclaration).read_text())
    for path, digest in declared["source_sha256"].items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != digest:
            raise ValueError("predeclared source changed: " + path)
    pilot = declared["timing_pilot"]
    record = json.loads(Path(pilot_split).read_text())
    if hashlib.sha256(Path(pilot_split).read_bytes()).hexdigest() != pilot["split_sha256"]:
        raise ValueError("pilot lineage changed")
    families = tuple(_family(row) for row in record["split"]["source_records"]["train"])
    replay, corpus = produce_strategic(families, library=library, device=device,
                                       arena_capacity=pilot["arena_capacity"])
    results = []
    for control in CONTROLS:
        target = comparable_model("relation", width=16, depth=1, seed=pilot["seed"]).to(replay.device).eval()
        target.requires_grad_(False)
        model = _build("relation", control, 16, 1, pilot["seed"]).to(replay.device)
        optimizer = torch.optim.AdamW(model.active_parameters(), lr=.0003)
        generator = torch.Generator(device=replay.device).manual_seed(pilot["seed"] + 2)
        torch.cuda.synchronize(); started = time.perf_counter()
        for _ in range(pilot["steps"]):
            batch = replay.pair_batch(8, generator)
            optimizer.zero_grad(set_to_none=True)
            terms, _ = strategic_loss(model, target, batch)
            terms["total"].backward()
            torch.nn.utils.clip_grad_norm_(model.active_parameters(), 1.)
            optimizer.step()
        torch.cuda.synchronize()
        results.append({"control": control, "steps": pilot["steps"], "seconds": time.perf_counter() - started,
                        "last_loss": float(terms["total"].detach()), "rows_per_step": batch["indices"].numel()})
        print(json.dumps(results[-1]), flush=True)
        del model, target, optimizer
    report = {"kind": "disposable_strategic_timing_pilot", "strength_evidence": False,
              "configuration": pilot, "corpus": corpus, "runs": results,
              "main_validation_or_test_labels_read": False,
              "estimated_30run_training_seconds": sum(row["seconds"] for row in results) * (30 / 5) * (1024 / pilot["steps"])}
    _atomic_json(output, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("library", "producer-manifest", "predeclaration", "output"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--spent-reports", nargs="+", required=True)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--pilot-split")
    parser.add_argument("--checkpoint-dir", default="results/neural/strategic-study")
    for name, value in (("train-families", 96), ("validation-families", 24), ("test-families", 24),
                        ("corpus-seed", 20261003), ("steps", 1024), ("pairs", 8),
                        ("width", 16), ("depth", 1), ("tree-depth", 3), ("arena-capacity", 250000)):
        parser.add_argument("--" + name, type=int, default=value)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20261003, 20261004])
    options = vars(parser.parse_args())
    pilot = options.pop("pilot"); pilot_split = options.pop("pilot_split")
    if pilot:
        run_timing_pilot(library=options["library"], predeclaration=options["predeclaration"], output=options["output"], pilot_split=pilot_split)
    else:
        run_strategic(**options)


if __name__ == "__main__":
    main()
