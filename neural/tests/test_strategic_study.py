"""Complete-root labels, independent static oracle, causal loss and fresh splits."""
from dataclasses import asdict
import copy
import json
import os
from pathlib import Path
import tempfile
import unittest

import chess
import torch

from neural.bridge import ExportedTransitions
from neural.features import encode_fen, index_to_move
from neural.teacher import TeacherLabels, RootFamily, RootSample, _candidate_family, _family_positions, comparable_model
from neural.tests.test_teacher import fixture_records
from neural.tests.test_bridge import HotResident, PayloadResident
from neural.strategic_transport import StrategicPredictor
from neural.strategic_study import (StrategicReplay, CONTROLS, fresh_split, join_complete_root,
                                   tapered_cp, strategic_loss, validation_selection, produce_strategic, _tensor_fingerprint)


def root_fixture():
    first, _ = fixture_records()
    count = first.tensors["actions"].numel()
    sidecar = {**first.sidecar, "parent_node": torch.zeros(count, dtype=torch.int64),
               "child_node": torch.arange(1, count + 1)}
    manifest = {**first.manifest, "child_start": 1}
    first = ExportedTransitions(first.tensors, sidecar, manifest)
    child_packed = {"pieces": first.tensors["next_pieces"].clone(), "state": first.tensors["next_state"].clone()}
    score = torch.arange(count, dtype=torch.int32) * 20 - 100
    child_labels = {"search_score_cp": score, "teacher_parent_ply": torch.ones(count, dtype=torch.int32)}
    child_sidecar = {"parent_node": sidecar["child_node"].clone()}
    root_labels = TeacherLabels({"scores": torch.full((count,), 100, dtype=torch.int32),
                                "best_actions": first.tensors["actions"][:1].expand(count).clone(),
                                "parent_ply": torch.zeros(count, dtype=torch.int32)}, manifest)
    return first, child_packed, child_labels, child_sidecar, manifest, root_labels


def scalar_tapered(board):
    """Independent raw-color Python loop specified by Rust Tapered source."""
    values, phases = [100, 320, 330, 500, 900, 0], [0, 1, 1, 2, 4, 0]
    mg = eg = phase = 0
    for color in [chess.WHITE, chess.BLACK]:
        sign = 1 if color == chess.WHITE else -1
        for piece in range(1, 7):
            phase += phases[piece - 1] * len(board.pieces(piece, color))
            for square in board.pieces(piece, color):
                file = square % 8
                rank = square // 8 if color == chess.WHITE else 7 - square // 8
                center = 14 - abs(2 * file - 7) - abs(2 * rank - 7)
                if piece == chess.PAWN: m, e = rank * 8 + center * 2, rank * 14 + center
                elif piece == chess.KNIGHT: m, e = center * 6, center * 4
                elif piece == chess.BISHOP: m, e = center * 3 + rank * 2, center * 3
                elif piece == chess.ROOK: m, e = rank * 2 + (18 if rank == 6 else 0), center
                elif piece == chess.QUEEN: m, e = center, center * 2
                else: m, e = -center * 5 - rank * 4 + (35 if rank == 0 and file in [2, 6] else 0), center * 6
                mg += sign * (values[piece - 1] + m); eg += sign * (values[piece - 1] + e)
        if len(board.pieces(chess.BISHOP, color)) >= 2:
            mg += sign * 25; eg += sign * 40
    phase = min(phase, 24)
    score = int((mg * phase + eg * (24 - phase)) / 24)
    return max(-29000, min(29000, score if board.turn else -score))


class StrategicStudyCPU(unittest.TestCase):
    def test_complete_node_join_child_frame_and_full_root_backup(self):
        args = root_fixture()
        labels, reason = join_complete_root(*args)
        self.assertIsNone(reason)
        self.assertEqual(int(labels["root_score_cp"][0]), 100)
        self.assertEqual(int(labels["child_score_cp"][0]), -100)
        corrupt = list(copy.deepcopy(args)); corrupt[4] = {**corrupt[4], "root_epoch": corrupt[4]["root_epoch"] + 1}
        with self.assertRaises(ValueError): join_complete_root(*corrupt)
        corrupt = list(copy.deepcopy(args)); corrupt[1]["state"][0, 3] += 1
        with self.assertRaises(ValueError): join_complete_root(*corrupt)
        corrupt = list(copy.deepcopy(args)); corrupt[5].tensors["scores"].fill_(-100)
        with self.assertRaises(ValueError): join_complete_root(*corrupt)

    def test_missing_terminal_or_mate_child_omits_whole_root(self):
        args = list(root_fixture())
        args[3] = {"parent_node": args[3]["parent_node"][1:]}
        args[1] = {key: value[1:] for key, value in args[1].items()}
        args[2] = {key: value[1:] for key, value in args[2].items()}
        self.assertEqual(join_complete_root(*args)[1], "terminal_or_missing_child_scores")
        args = list(root_fixture()); args[2]["search_score_cp"][4] = 29997
        self.assertEqual(join_complete_root(*args)[1], "mate_band_child_scores")
        args = list(root_fixture()); args[0].tensors["terminal"][3] = True
        self.assertEqual(join_complete_root(*args)[1], "terminal_or_missing_child_scores")

    def test_resident_complete_group_sampling_and_atomic_bad_admission(self):
        args = root_fixture(); labels, _ = join_complete_root(*args)
        count = args[0].tensors["actions"].numel()
        replay = StrategicReplay(count, "cpu")
        replay.append_root(args[0], labels, 3); replay.seal()
        with HotResident(): batch = replay.pair_batch(8, torch.Generator().manual_seed(2))
        self.assertTrue(torch.equal(batch["root_ids"].reshape(-1, 2)[:, 0], batch["root_ids"].reshape(-1, 2)[:, 1]))
        self.assertTrue(bool((batch["actions"].reshape(-1, 2)[:, 0] != batch["actions"].reshape(-1, 2)[:, 1]).all()))
        self.assertGreater(replay.same_material_a.numel(), 0)
        with self.assertRaises(ValueError): replay.append_root(args[0], labels, 4)
        fresh = StrategicReplay(count, "cpu")
        with self.assertRaises(ValueError): fresh.append_root(args[0], {**labels, "child_score_cp": labels["child_score_cp"].float()}, 3)
        self.assertEqual(fresh.compact.total_inserted, 0)
        broken = {**labels, "root_score_cp": torch.zeros(count, dtype=torch.int32)}
        with self.assertRaises(ValueError): fresh.append_root(args[0], broken, 3)
        self.assertEqual(fresh.compact.total_inserted, 0)

    def test_tapered_oracle_colors_special_moves_and_truncation(self):
        boards = [chess.Board()]
        for fen, moves in [
                ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", ["e1g1", "e1c1"]),
                ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", ["e5d6"]),
                ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", ["a7b8q", "a7a8n"]),
                ("4k3/8/8/8/8/2q5/2R5/4K3 w - - 0 1", ["c2c3", "e1f1"])]:
            root = chess.Board(fen); boards += [root, root.mirror()]
            for move in moves:
                child = root.copy(); child.push_uci(move); boards += [child, child.mirror()]
        board = chess.Board()
        for index in range(40):
            if board.is_game_over(): break
            legal = sorted(board.legal_moves, key=lambda move: move.uci())
            board.push(legal[(index * 7 + 3) % len(legal)])
            boards += [board.copy(), board.mirror()]
        features = torch.stack([encode_fen(board.fen(en_passant="fen")) for board in boards])
        with HotResident(): values = tapered_cp(features)
        self.assertEqual(values.tolist(), [scalar_tapered(board) for board in boards])
        self.assertTrue(any(value < 0 for value in values.tolist()))

    def test_three_way_split_excludes_spent_canonical_positions(self):
        old = _candidate_family(0, 718, (4, 8))
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "spent.json"
            path.write_text(json.dumps({"split": {"source_records": {"heldout": [asdict(old)]}}}))
            groups, manifest = fresh_split(1, 1, 1, seed=718, roots_at=(4, 8), spent_reports=[path])
            again, _ = fresh_split(1, 1, 1, seed=718, roots_at=(4, 8), spent_reports=[path])
        self.assertEqual(groups, again)
        keys = {name: _family_positions(families[0]) for name, families in groups.items()}
        spent = _family_positions(old)
        for name in keys:
            self.assertFalse(keys[name] & spent)
            for other in keys:
                if other != name: self.assertFalse(keys[name] & keys[other])
        self.assertGreater(manifest["rejected_candidates"]["past_or_split_canonical_collision"], 0)

    def test_causal_prediction_fixed_target_gradients_and_same_material_blind_ties(self):
        args = root_fixture(); labels, _ = join_complete_root(*args)
        replay = StrategicReplay(20, "cpu"); replay.append_root(args[0], labels, 0); replay.seal()
        batch = replay.pair_batch(4, torch.Generator().manual_seed(8))
        for mode in ("transport", "conditioned", "action_blind"):
            base = comparable_model("square", width=8, depth=1, seed=9)
            model = StrategicPredictor(base, mode=mode)
            target = comparable_model("square", width=8, depth=1, seed=9).eval(); target.requires_grad_(False)
            with HotResident():
                terms, prediction = strategic_loss(model, target, batch)
                terms["total"].backward()
            self.assertTrue(all(parameter.grad is None for parameter in target.parameters()))
            self.assertTrue(bool(torch.isfinite(terms["total"])))
            unchanged = model.predict(batch["root_features"], batch["actions"])["latent"].detach()
            corrupt = {**batch, "successor_features": torch.zeros_like(batch["successor_features"])}
            _, other = strategic_loss(model, target, corrupt)
            self.assertTrue(torch.equal(unchanged, other["latent"].detach()))
            if mode == "action_blind":
                torch.testing.assert_close(prediction["latent"][::2], prediction["latent"][1::2], rtol=0, atol=0)
                torch.testing.assert_close(prediction["value_cp"][::2], prediction["value_cp"][1::2], rtol=0, atol=0)

    def test_cold_fingerprint_supports_scalar_model_parameters_and_all_encoders(self):
        scalar = torch.tensor(1.25)
        self.assertEqual(_tensor_fingerprint({"x": scalar}), _tensor_fingerprint({"x": scalar.reshape(1)}))
        self.assertNotEqual(_tensor_fingerprint({"x": scalar}), _tensor_fingerprint({"x": scalar + 1}))
        self.assertEqual(len(_tensor_fingerprint({"empty": torch.empty(0)})), 64)
        noncontiguous = torch.arange(12, dtype=torch.float32).reshape(3, 4).T
        self.assertFalse(noncontiguous.is_contiguous())
        self.assertEqual(_tensor_fingerprint({"x": noncontiguous}), _tensor_fingerprint({"x": noncontiguous.contiguous()}))
        for architecture in ("square", "relation", "plan"):
            state = comparable_model(architecture, width=16, depth=1, seed=71).state_dict()
            self.assertEqual(len(_tensor_fingerprint(state)), 64)

    def test_validation_selection_cannot_see_test_and_requires_both_baselines(self):
        def record(architecture, control, mae, regret):
            def metric(m, r): return {"ordinary_centipawn_mae": m, "ordering": {"mean_selected_child_regret_cp": r}}
            return {"architecture": architecture, "control": control, "after_validation": {
                "network": metric(mae, regret), "material": metric(100, 50), "static_tapered": metric(90, 40)}}
        rows = [record("square", "transport", 89, 39), record("plan", "transport", 80, 45)]
        selection = validation_selection(rows)
        self.assertEqual(selection["selected"]["architecture"], "square")
        rows[0]["after_test"] = {}
        with self.assertRaises(ValueError): validation_selection(rows)
        self.assertIsNone(validation_selection([record("square", "transport", 99, 39)])["selected"])


@unittest.skipUnless(os.environ.get("CHEN_STRATEGIC_STUDY_GPU") == "1", "GPU slot not granted")
class StrategicStudyGPU(unittest.TestCase):
    def test_gpu_tapered_exact_integer_color_and_special_move_oracle(self):
        fens = [chess.STARTING_FEN, "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
                "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"]
        boards = [chess.Board(fen) for fen in fens]
        boards += [board.mirror() for board in list(boards)]
        features = torch.stack([encode_fen(board.fen(en_passant="fen")) for board in boards]).cuda()
        with HotResident(): scores = tapered_cp(features)
        self.assertEqual(scores.cpu().tolist(), [scalar_tapered(board) for board in boards])

    def test_native_complete_root_child_horizon2_against_independent_tapered_oracle(self):
        fen = "4k3/pp6/8/8/8/8/PP6/4K3 w - - 0 1"
        family = RootFamily(0, fen, (), (RootSample(0, 0, fen, ()),))
        library = os.environ["CHEN_GPU_LIBRARY"]
        with PayloadResident():
            replay, manifest = produce_strategic((family,), library=library, arena_capacity=20000)
        self.assertEqual(manifest["coverage"]["retained_roots"], 1)
        self.assertEqual(replay.compact.size, chess.Board(fen).legal_moves.count())
        def reference(board, depth, ply):
            legal = list(board.legal_moves)
            if not legal: return -30000 + ply if board.is_check() else 0
            if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3): return 0
            if not depth: return scalar_tapered(board)
            scores = []
            for move in legal:
                board.push(move); scores.append(-reference(board, depth - 1, ply + 1)); board.pop()
            return max(scores)
        board = chess.Board(fen)
        # Independent oracle diagnostics explicitly download labels after production.
        actions = replay.compact.storage["actions"][:replay.compact.size].cpu().tolist()
        scores = replay.labels["child_score_cp"][:replay.compact.size].cpu().tolist()
        for action, score in zip(actions, scores):
            board.push_uci(index_to_move(action, "w"))
            self.assertEqual(score, reference(board, 2, 1)); board.pop()
        self.assertEqual(int(replay.labels["root_score_cp"][0]), -min(scores))

    def test_resident_sampler_loss_backward_and_update_on_actual_gpu(self):
        args = list(root_fixture())
        first = args[0]
        first = ExportedTransitions({key: value.cuda() for key, value in first.tensors.items()},
                                    {key: value.cuda() for key, value in first.sidecar.items()}, first.manifest)
        labels = join_complete_root(*args)[0]
        labels = {key: value.cuda() for key, value in labels.items()}
        replay = StrategicReplay(20, "cuda:0")
        with PayloadResident(): replay.append_root(first, labels, 0); replay.seal()
        for mode in ("transport", "conditioned", "action_blind"):
            model = StrategicPredictor(comparable_model("relation", width=8, depth=1, seed=7), mode=mode).cuda()
            target = comparable_model("relation", width=8, depth=1, seed=7).cuda().eval(); target.requires_grad_(False)
            optimizer = torch.optim.AdamW(model.active_parameters(), lr=.003)
            with HotResident():
                batch = replay.pair_batch(4, torch.Generator(device="cuda").manual_seed(77))
                terms, _ = strategic_loss(model, target, batch)
                terms["total"].backward()
            with PayloadResident(): optimizer.step()
            self.assertTrue(bool(torch.isfinite(terms["total"])))
            self.assertEqual(batch["root_features"].device.type, "cuda")


if __name__ == "__main__":
    unittest.main()
