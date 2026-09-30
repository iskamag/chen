import unittest

import chess
import torch

from neural.diamonds import certify_diamond, diamond_losses, feature_identity, generate_diamonds, run_smoke
from neural.features import encode_fen, index_to_move
from neural.models import build_model

PATH = ("g1f3", "g8f6", "b1c3", "b8c6")
ALTERNATE = ("b1c3", "b8c6", "g1f3", "g8f6")


class DiamondTest(unittest.TestCase):
    def test_initial_fixture_certifies_full_fen_and_draw_context(self):
        record = certify_diamond(chess.Board(), PATH, ALTERNATE)
        self.assertEqual(record.path_fens[0][-1], record.path_fens[1][-1])
        self.assertEqual(record.endpoint_fen.split()[1:], ["w", "KQkq", "-", "4", "3"])
        self.assertFalse(any(record.draw_context))

    def test_reject_dependent_moves_pawns_and_draw_boundary(self):
        bad = ("g1f3", "g8f6", "f3g5", "b8c6")
        with self.assertRaises(ValueError):
            certify_diamond(chess.Board(), bad, (bad[2], bad[3], bad[0], bad[1]))
        pawns = ("e2e4", "e7e5", "g1f3", "g8f6")
        with self.assertRaises(ValueError):
            certify_diamond(chess.Board(), pawns, (pawns[2], pawns[3], pawns[0], pawns[1]))
        boundary = chess.Board()
        boundary.halfmove_clock = 90
        with self.assertRaises(ValueError):
            certify_diamond(boundary, PATH, ALTERNATE)
        repeated = chess.Board()
        for move in ("g1f3", "g8f6", "f3g1", "f6g8"):
            repeated.push_uci(move)
        with self.assertRaises(ValueError):
            certify_diamond(repeated, PATH, ALTERNATE)

    def test_raw_ep_schema_and_canonical_split_identity(self):
        root = chess.Board()
        root.push_uci("e2e4")
        path = ("g8f6", "g1f3", "b8c6", "b1c3")
        record = certify_diamond(root, path, (path[2], path[3], path[0], path[1]))
        self.assertEqual(record.root_fen.split()[3], "e3")
        self.assertEqual(encode_fen(record.path_fens[0][0])[16].sum().item(), 1)
        self.assertEqual(record.endpoint_fen.split()[3], "-")
        raw = root.fen(en_passant="fen")
        mirror = root.mirror().fen(en_passant="fen")
        self.assertEqual(feature_identity(raw), feature_identity(mirror))
        root.fullmove_number += 17
        self.assertEqual(feature_identity(raw), feature_identity(root.fen(en_passant="fen")))

    def test_generated_corpus_replays_every_move_and_encodes_canonical_actions(self):
        corpus = generate_diamonds(8, seed=827, max_root_plies=16)
        again = generate_diamonds(8, seed=827, max_root_plies=16)
        self.assertEqual(corpus.manifest["sha256"], again.manifest["sha256"])
        torch.testing.assert_close(corpus.actions, again.actions)
        self.assertGreater(len({record.root_fen for record in corpus.records}), 2)
        for row, record in enumerate(corpus.records):
            root = chess.Board(record.origin_fen)
            for move in record.root_history:
                root.push_uci(move)
            self.assertEqual(root.fen(en_passant="fen"), record.root_fen)
            certified = certify_diamond(root, *record.paths)
            self.assertEqual(certified, record)
            for path_index, path in enumerate(record.paths):
                board = root.copy(stack=True)
                for ply, uci in enumerate(path):
                    side = "w" if board.turn else "b"
                    self.assertEqual(index_to_move(corpus.actions[row, path_index, ply].item(), side), uci)
                    board.push_uci(uci)
            torch.testing.assert_close(corpus.features[row, 0, -1], corpus.features[row, 1, -1])

    def test_composition_refresh_losses_differentiate_dynamics(self):
        corpus = generate_diamonds(2, seed=91, max_root_plies=8)
        torch.manual_seed(3)
        model = build_model("plan", width=16, depth=1)
        losses = diamond_losses(model, corpus.features, corpus.actions)
        for value in losses.values():
            self.assertTrue(torch.isfinite(value))
        (losses["refresh"] + losses["composition"] + losses["endpoint_board"]).backward()
        for name in ("dynamics.3.weight", "action_condition.2.weight", "input.weight", "plans"):
            grad = dict(model.named_parameters())[name].grad
            self.assertIsNotNone(grad)
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        model.zero_grad(set_to_none=True)
        diamond_losses(model, corpus.features, corpus.actions)["composition"].backward()
        grad = dict(model.named_parameters())["dynamics.3.weight"].grad
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_controls_have_identical_initial_metrics(self):
        report = run_smoke(samples=8, steps=1, batch_size=2, width=8, models=("square",), max_root_plies=16, seed=971)
        self.assertFalse(report["strength_evidence"])
        results = report["results"]
        self.assertEqual([row["control"] for row in results], ["ordinary", "refresh", "diamond"])
        self.assertEqual(results[0]["heldout_before"], results[1]["heldout_before"])
        self.assertEqual(results[1]["heldout_before"], results[2]["heldout_before"])
        states = {feature_identity(fen) for record in report["corpus_records"]["train"] for path in record["path_fens"] for fen in path}
        heldout = {feature_identity(fen) for record in report["corpus_records"]["heldout"] for path in record["path_fens"] for fen in path}
        self.assertFalse(states & heldout)


if __name__ == "__main__":
    unittest.main()
