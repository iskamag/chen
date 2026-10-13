"""Search label semantics, resident ring, leakage controls and restricted files."""
import copy
import ctypes as c
import os
from pathlib import Path
import tempfile
import unittest

import chess
import torch

from neural.bridge import BridgeError, ExportedTransitions, RuleArena, position_from_fen
from neural.compact import compact_corpus, decode_batch, pack_fens
from neural.data import generate_synthetic
from neural.features import encode_fen, move_to_index, index_to_move
from neural.models import build_model
from neural.teacher import (MATE_BAND, SCALE_CP, SEARCH_LABEL_DTYPES, START_FEN, TeacherLabels,
                            TeacherSink, TeacherResult, SearchReplay, _family_positions,
                            backed_up_tree, canonical_position_key, deduplicate_parents,
                            comparable_model,
                            export_teacher, load_teacher_checkpoint, save_teacher_checkpoint,
                            search_loss, search_step, split_root_families, utility_to_cp)
from neural.tests.test_bridge import PayloadResident, HotResident


def fixture_records():
    """Independent full legal successor set for one parent, CPU schema fixture."""
    board = chess.Board(START_FEN)
    moves = sorted(board.legal_moves, key=lambda move: move.uci())
    legal = sorted(move_to_index(move.uci(), "w") for move in moves)
    successors = []
    for move in moves:
        child = board.copy(); child.push(move)
        successors.append(child.fen(en_passant="fen"))
    count = len(moves)
    pieces, state = pack_fens([START_FEN] * count)
    next_pieces, next_state = pack_fens(successors)
    sparse = torch.full((count, 256), -1, dtype=torch.int16)
    sparse[:, :count] = torch.tensor(legal, dtype=torch.int16)
    tensors = {"pieces": pieces, "state": state, "next_pieces": next_pieces, "next_state": next_state,
               "legal_actions": sparse, "legal_counts": torch.full((count,), count, dtype=torch.int32),
               "actions": torch.tensor([move_to_index(move.uci(), "w") for move in moves]),
               "target_actions": torch.full((count,), legal[0]), "material_proxy": torch.zeros(count),
               "terminal": torch.zeros(count, dtype=torch.bool), "game_ids": torch.zeros(count, dtype=torch.int64),
               "history_handles": torch.full((count,), -1, dtype=torch.int64),
               "next_history_handles": torch.full((count,), -1, dtype=torch.int64)}
    sidecar = {"parent_node": torch.ones(count, dtype=torch.int64),
               "child_node": torch.arange(2, count + 2), "root_epoch": torch.ones(count, dtype=torch.int64),
               "parent_hash": torch.zeros(count, dtype=torch.int64), "child_hash": torch.zeros(count, dtype=torch.int64)}
    manifest = {"context_uuid": "independent-CPU-fixture", "root_epoch": 1, "child_start": 2, "rows": count}
    labels = TeacherLabels({"scores": torch.zeros(count, dtype=torch.int32),
                            "best_actions": torch.full((count,), legal[0]),
                            "parent_ply": torch.ones(count, dtype=torch.int32)}, dict(manifest))
    return ExportedTransitions(tensors, sidecar, manifest), labels


def ring_fixture(count=8):
    source = compact_corpus(generate_synthetic(count, seed=123, max_plies=8)).tensors
    labels = {name: torch.zeros(count, dtype=dtype) for name, dtype in SEARCH_LABEL_DTYPES.items()}
    labels["search_score_cp"] = torch.arange(count, dtype=torch.int32) * 100 - 300
    labels["teacher_parent_ply"].fill_(1)
    labels["material_greedy_action"] = source["actions"].clone()
    labels["root_epoch"].fill_(1)
    return source, labels


class TeacherCPU(unittest.TestCase):
    def test_abi_and_sibling_dedup_has_one_vote_complete_edges_only(self):
        self.assertEqual(c.sizeof(TeacherSink), 48)
        self.assertEqual(c.sizeof(TeacherResult), 32)
        records, labels = fixture_records()
        packed, targets, sidecar = deduplicate_parents(records, labels)
        self.assertEqual(packed["pieces"].shape[0], 1)
        self.assertEqual(int(packed["target_actions"][0]), int(labels.tensors["best_actions"][0]))
        self.assertEqual(int(sidecar["parent_node"][0]), 1)
        bad = TeacherLabels({key: value.clone() for key, value in labels.tensors.items()}, labels.manifest)
        bad.tensors["scores"][-1] = 1
        with self.assertRaises(ValueError):
            deduplicate_parents(records, bad)
        incomplete = ExportedTransitions({key: value[:-1] for key, value in records.tensors.items()},
                                         {key: value[:-1] for key, value in records.sidecar.items()},
                                         {**records.manifest, "rows": 19})
        partial_labels = TeacherLabels({key: value[:-1] for key, value in labels.tensors.items()},
                                       {**labels.manifest, "rows": 19})
        with self.assertRaises(ValueError):
            deduplicate_parents(incomplete, partial_labels)

    def test_ring_label_alignment_overwrite_hot_sampling_and_atomic_validation(self):
        tensors, labels = ring_fixture()
        replay = SearchReplay(3, "cpu")
        replay.append(tensors, labels)
        with HotResident():
            sample = replay.sample(16, torch.Generator().manual_seed(4))
        for sequence, score in zip(sample["sequence_ids"].tolist(), sample["search_score_cp"].tolist()):
            self.assertEqual(score, int(labels["search_score_cp"][sequence]))
            self.assertGreaterEqual(sequence, 5)
        snapshot = sample["search_score_cp"].clone()
        before = replay.compact.total_inserted
        for key, replacement in (("search_score_cp", torch.full((8,), -2147483648, dtype=torch.int32)),
                                 ("teacher_parent_ply", torch.full((8,), 65, dtype=torch.int32)),
                                 ("material_greedy_action", torch.full((8,), -1, dtype=torch.int64))):
            with self.assertRaises(ValueError):
                replay.append(tensors, {**labels, key: replacement})
            self.assertEqual(replay.compact.total_inserted, before)
        replay.append(tensors, labels)
        self.assertTrue(torch.equal(sample["search_score_cp"], snapshot))

    def test_feature_identity_matches_exact_color_clock_ep_equivalence(self):
        board = chess.Board("r3k2r/8/8/3pP3/8/8/8/R3K2R w KQkq d6 101 25")
        fens = [board.fen(en_passant="fen"), board.mirror().fen(en_passant="fen")]
        self.assertEqual(canonical_position_key(fens[0]), canonical_position_key(fens[1]))
        self.assertTrue(torch.equal(encode_fen(fens[0]), encode_fen(fens[1])))
        alias = fens[0].replace("101 25", "65535 999")
        self.assertEqual(canonical_position_key(fens[0]), canonical_position_key(alias))
        changed = fens[0].replace("d6", "-")
        self.assertNotEqual(canonical_position_key(fens[0]), canonical_position_key(changed))
        self.assertFalse(torch.equal(encode_fen(fens[0]), encode_fen(changed)))

    def test_lineage_and_position_split_is_reproducible_and_disjoint(self):
        a, b, manifest = split_root_families(1, 1, seed=7, roots_at=(6, 10))
        c1, d, again = split_root_families(1, 1, seed=7, roots_at=(6, 10))
        self.assertEqual(a, c1); self.assertEqual(b, d)
        self.assertEqual(manifest["source_sha256"], again["source_sha256"])
        self.assertNotEqual(a[0].family_id, b[0].family_id)
        self.assertFalse(_family_positions(a[0]) & _family_positions(b[0]))
        for family in (*a, *b):
            board = chess.Board(family.origin_fen)
            prior = []
            roots = {root.ply: root for root in family.roots}
            for ply, move in enumerate((*family.moves, None)):
                if ply in roots:
                    self.assertEqual(roots[ply].fen, board.fen(en_passant="fen"))
                    self.assertEqual(roots[ply].previous_hashes, tuple(prior))
                if move:
                    prior.append(position_from_fen(board.fen(en_passant="fen")).hash)
                    board.push_uci(move)

    def test_actual_search_loss_finite_update_and_mate_exclusion(self):
        tensors, labels = ring_fixture()
        replay = SearchReplay(8, "cpu"); replay.append(tensors, labels)
        batch = replay.gather(torch.arange(8))
        model = build_model("square", width=8, depth=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
        old = model.wdl.weight.detach().clone()
        with PayloadResident():
            losses = search_step(model, optimizer, batch)
        self.assertTrue(bool(torch.isfinite(losses["total"])))
        self.assertFalse(torch.equal(old, model.wdl.weight))
        mates = {**batch, "search_score_cp": torch.full((8,), 29999, dtype=torch.int32)}
        with HotResident():
            loss = search_loss(model, mates)
        self.assertEqual(float(loss["search_utility"]), 0)
        self.assertTrue(bool(torch.isfinite(loss["total"])))
        cp = torch.tensor([-400.0, 0.0, 400.0])
        self.assertTrue(torch.equal(utility_to_cp(torch.tanh(cp / SCALE_CP)), cp))

    def test_shared_comparison_heads_start_identically(self):
        a, b, c1 = [comparable_model(name, width=8, depth=1, seed=17) for name in ("square", "relation", "plan")]
        for key in ("input.weight", "position", "wdl.weight", "policy_source.weight", "promo_target.bias"):
            self.assertTrue(torch.equal(a.state_dict()[key], b.state_dict()[key]))
            self.assertTrue(torch.equal(a.state_dict()[key], c1.state_dict()[key]))

    def test_search_checkpoint_roundtrip_and_corruption_rejection(self):
        model = build_model("relation", width=8, depth=1)
        teacher = {"teacher_kind": "finite_depth_classical_minimax_search_targets", "scale_cp": SCALE_CP,
                   "tree_depth": 3, "parent_horizon": 2, "evaluator": 1}
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "teacher.pt"
            save_teacher_checkpoint(model, path, teacher_manifest=teacher, training_config={"steps": 1})
            loaded, meta = load_teacher_checkpoint(path)
            self.assertEqual(meta["kind"], "search_teacher_cp_utility_checkpoint")
            for name, value in model.state_dict().items():
                self.assertTrue(torch.equal(value, loaded.state_dict()[name]))
            original = torch.load(path, weights_only=True)
            variants = []
            bad = copy.deepcopy(original); bad["state_dict"]["next_orientation"][0] = 64; variants.append(bad)
            bad = copy.deepcopy(original); bad["state_dict"]["wdl.weight"] = bad["state_dict"]["wdl.weight"].double(); variants.append(bad)
            bad = copy.deepcopy(original); bad["teacher_manifest"]["evaluator"] = True; variants.append(bad)
            bad = copy.deepcopy(original); bad["utility_contract"]["mate_value_loss_excluded"] = 1; variants.append(bad)
            bad = copy.deepcopy(original); bad["schema_version"] = True; variants.append(bad)
            bad = copy.deepcopy(original); bad["kind"] = "synthetic_legal_smoke_checkpoint"; variants.append(bad)
            for bad in variants:
                torch.save(bad, path)
                with self.assertRaises(ValueError):
                    load_teacher_checkpoint(path)


def material_minimax(board, depth, absolute_ply):
    legal = list(board.legal_moves)
    if not legal:
        return -30000 + absolute_ply if board.is_check() else 0
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0
    if not depth:
        return sum((1 if color == board.turn else -1) * len(board.pieces(piece, color)) * weight
                   for color in chess.COLORS for piece, weight in zip(range(1, 7), [100, 320, 330, 500, 900, 0]))
    best = -30001
    for move in legal:
        board.push(move); best = max(best, -material_minimax(board, depth - 1, absolute_ply + 1)); board.pop()
    return best


@unittest.skipUnless(os.environ.get("CHEN_TEACHER_TEST_GPU") == "1", "set CHEN_TEACHER_TEST_GPU=1 for actual teacher tests")
class TeacherGPU(unittest.TestCase):
    def test_complete_bfs_exact_material_parent_labels_and_resident_learning(self):
        fen = "4k3/8/8/8/8/8/4p3/4K2R w - - 0 1"
        packet = position_from_fen(fen)
        with RuleArena(capacity=20000) as arena:
            with PayloadResident():
                records, labels, schedule = backed_up_tree(arena, packet, (), tree_depth=3, evaluator=0)
                packed, targets, sidecar = deduplicate_parents(records, labels)
                targets.update(sidecar, teacher_root_id=torch.zeros(packed["pieces"].shape[0], dtype=torch.int64, device="cuda"))
                replay = SearchReplay(packed["pieces"].shape[0], "cuda")
                replay.append(packed, targets)
            self.assertEqual(schedule["remaining_parent_horizon"], 2)
            self.assertEqual([row["operation"] for row in schedule["schedule"]],
                             ["expand", "expand", "expand", "evaluate", "reduce", "reduce", "reduce"])
            # Explicit independent oracle diagnostic region; not production ingress.
            seen = set()
            current = records.tensors["pieces"].cpu()
            states = records.tensors["state"].cpu()
            exported_scores = labels.tensors["scores"].cpu()
            actions = labels.tensors["best_actions"].cpu()
            # Map every independent legal root child to its exact current bitboards.
            board = chess.Board(fen)
            for move in list(board.legal_moves):
                board.push(move)
                key, state = pack_fens([board.fen(en_passant="fen")])
                indexes = ((current == key[0]).all(-1) & (states == state[0]).all(-1)).nonzero().flatten()
                if indexes.numel():
                    i = int(indexes[0]); seen.add(int(records.sidecar["parent_node"][i]))
                    self.assertEqual(int(exported_scores[i]), material_minimax(board, 2, 1))
                    chosen = chess.Move.from_uci(index_to_move(int(actions[i]), "b"))
                    self.assertIn(chosen, board.legal_moves)
                    board.push(chosen)
                    self.assertEqual(-material_minimax(board, 1, 2), int(exported_scores[i]))
                    board.pop()
                board.pop()
            self.assertEqual(len(seen), replay.compact.size)
            model = build_model("relation", width=8, depth=1).to("cuda")
            optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
            with HotResident():
                batch = replay.sample(8, torch.Generator(device="cuda").manual_seed(4))
                checks = search_loss(model, batch)
                checks["total"].backward()
            with PayloadResident():
                loss = search_step(model, optimizer, batch)
            self.assertTrue(bool(torch.isfinite(loss["total"])))

    def test_unreduced_and_stale_labels_rejected(self):
        with RuleArena(capacity=20000) as arena:
            arena.reset(position_from_fen(START_FEN))
            first = arena.expand()
            with self.assertRaises(BridgeError):
                export_teacher(arena, first["span"])
            second = arena.expand(first["span"])
            arena.evaluate(second["span"], 1)
            with self.assertRaises(BridgeError):
                export_teacher(arena, second["span"])
            arena.reduce(first["span"])
            labels = export_teacher(arena, second["span"])
            self.assertEqual(labels.manifest["evaluator"], 1)
            arena.reset(position_from_fen(START_FEN))
            with self.assertRaises(ValueError):
                export_teacher(arena, second["span"])


if __name__ == "__main__":
    unittest.main()
