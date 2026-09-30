"""Contract/semantic tests. Run with python -m unittest neural.tests.test_models."""
import unittest
import random

import torch
from neural.features import ACTION_COUNT, encode_fen, index_to_move, legal_probabilities, move_to_index
from neural.models import MODELS, build_model

START = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"

try:
    import chess
except ImportError:
    chess = None


class FeaturesTest(unittest.TestCase):
    def test_move_mapping_is_bijective_and_color_canonical(self):
        for side in ("w", "b"):
            for index in range(ACTION_COUNT):
                self.assertEqual(move_to_index(index_to_move(index, side), side), index)
        self.assertEqual(move_to_index("e2e4", "w"), move_to_index("e7e5", "b"))
        self.assertEqual(move_to_index("a7a8n", "w"), move_to_index("a2a1n", "b"))

    def test_color_rank_flip_preserves_features(self):
        white = "4k3/8/8/3pP3/8/8/8/R3K2R w KQ d6 7 4"
        equivalent_black = "r3k2r/8/8/8/3Pp3/8/8/4K3 b kq d3 7 4"
        torch.testing.assert_close(encode_fen(white), encode_fen(equivalent_black))
        self.assertEqual(encode_fen(START)[:12].sum().item(), 32)
        self.assertAlmostEqual(encode_fen(white)[17, 0, 0].item(), 0.07)

    def test_invalid_fen_and_moves_rejected(self):
        for fen in ("8/8/8/8/8/8/8/7 w - - 0 1", "8/8/8/8/8/8/8/8 w KK - 0 1", "8/8/8/8/8/8/8/8 w - a4 0 1"):
            with self.assertRaises(ValueError):
                encode_fen(fen)
        with self.assertRaises(ValueError):
            move_to_index("e7e8k")

    def test_masks_terminal_rows_and_normalization(self):
        logits = torch.randn(3, ACTION_COUNT, requires_grad=True)
        mask = torch.zeros_like(logits, dtype=torch.bool)
        mask[0, [7, 19, 31]] = True
        mask[2, 42] = True
        probabilities = legal_probabilities(logits, mask)
        torch.testing.assert_close(probabilities.sum(-1), torch.tensor([1., 0., 1.]))
        self.assertEqual(probabilities[~mask].count_nonzero().item(), 0)
        self.assertTrue(torch.isfinite(probabilities).all())
        probabilities[0, 7].backward()
        self.assertTrue(torch.isfinite(logits.grad).all())


class ModelsTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(91)
        self.features = torch.stack([encode_fen(START), encode_fen("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1")])
        self.actions = torch.tensor([move_to_index("e2e4"), move_to_index("e5d6")])

    def test_all_models_shapes_gradients_and_multistep(self):
        for name in MODELS:
            with self.subTest(name=name):
                model = build_model(name, width=16, depth=1)
                output = model(self.features)
                self.assertEqual(output["latent"].shape, (2, 64, 16))
                self.assertEqual(output["wdl_logits"].shape, (2, 3))
                self.assertEqual(output["board_logits"].shape, (2, 12, 8, 8))
                self.assertEqual(output["policy_logits"].shape, (2, ACTION_COUNT))
                successor = model.transition(output["latent"], self.actions)
                successor2 = model.transition(successor, self.actions)
                self.assertFalse(torch.equal(successor, successor2))
                self.assertTrue(torch.isfinite(successor2).all())
                loss = (output["wdl_logits"].square().mean() + output["board_logits"].square().mean()
                        + torch.nn.functional.cross_entropy(output["policy_logits"], self.actions)
                        + output["castling_logits"].square().mean() + output["ep_logits"].square().mean()
                        + output["halfmove"].square().mean()
                        + model.decode(successor2)["board_logits"].square().mean())
                loss.backward()
                for parameter_name, parameter in model.named_parameters():
                    self.assertIsNotNone(parameter.grad, parameter_name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), parameter_name)
                self.assertGreater(output["latent"].var(dim=1).mean().item(), 0.01)

    def test_resident_branch_and_rollout_match_individual_transitions(self):
        model = build_model("relation", width=16, depth=1)
        latent = model.encode(self.features)
        actions = torch.stack((self.actions, torch.tensor([move_to_index("d2d4"), move_to_index("e5e6")])), 1)
        branches = model.branch(latent, actions)
        self.assertEqual(branches.shape, (2, 2, 64, 16))
        for candidate in range(2):
            torch.testing.assert_close(branches[:, candidate], model.transition(latent, actions[:, candidate]))
        path = model.rollout(latent, actions)
        self.assertEqual(path.shape, (2, 3, 64, 16))
        torch.testing.assert_close(path[:, 0], latent)
        torch.testing.assert_close(path[:, 2], model.transition(model.transition(latent, actions[:, 0]), actions[:, 1]))
        torch.testing.assert_close(model.rollout(latent, actions[:, :0])[:, 0], latent)

    def test_pseudoattack_blockers_and_pawn_directions(self):
        model = build_model("relation", width=16, depth=1)
        # White bishop c1: d2 is attacked; e3 is blocked. Enemy pawn f6 attacks e5/g5.
        features = encode_fen("4k3/8/5p2/8/8/8/3P4/2B1K3 w - - 0 1").unsqueeze(0)
        edges = model.attack_edges(features)[0]
        self.assertEqual(edges[2, 11].item(), 1)
        self.assertEqual(edges[2, 20].item(), 0)
        self.assertEqual(edges[45, 36].item(), 1)
        self.assertEqual(edges[45, 38].item(), 1)
        self.assertEqual(edges[45, 52].item(), 0)

    @unittest.skipUnless(chess is not None, "python-chess reference oracle unavailable")
    def test_attacks_and_color_symmetry_against_independent_rules_oracle(self):
        model = build_model("relation", width=16, depth=1)
        board, rng = chess.Board(), random.Random(31415)
        for _ in range(40):
            features = encode_fen(board.fen())
            torch.testing.assert_close(features, encode_fen(board.mirror().fen()))
            edges = model.attack_edges(features.unsqueeze(0))[0]
            expected = torch.zeros(64, 64)
            flip = 56 if board.turn == chess.BLACK else 0
            for source in board.piece_map():
                for target in board.attacks(source):
                    expected[source ^ flip, target ^ flip] = 1
            torch.testing.assert_close(edges, expected)
            moves = list(board.legal_moves)
            if not moves:
                board.reset()
            else:
                board.push(rng.choice(moves))

    def test_move_changes_latent_and_legality_is_external(self):
        model = build_model("plan", width=16, depth=1)
        latent = model.encode(self.features[:1])
        e4 = model.transition(latent, self.actions[:1])
        d4 = model.transition(latent, torch.tensor([move_to_index("d2d4")]))
        self.assertGreater((e4 - d4).abs().max().item(), 0.01)
        mask = torch.zeros(1, ACTION_COUNT, dtype=torch.bool)
        mask[0, self.actions[0]] = True
        policy = legal_probabilities(model.decode(e4, mask)["policy_logits"], mask)
        self.assertEqual(policy[0, self.actions[0]].item(), 1)

    @unittest.skipUnless(torch.cuda.is_available(), "ROCm/CUDA device unavailable")
    def test_gpu_matches_cpu_forward_transition_and_gradients(self):
        for name in MODELS:
            with self.subTest(name=name):
                cpu = build_model(name, width=16, depth=1)
                gpu = build_model(name, width=16, depth=1).cuda()
                gpu.load_state_dict(cpu.state_dict())
                a, b = cpu(self.features), gpu(self.features.cuda())
                for field in ("latent", "wdl_logits", "policy_logits", "board_logits", "castling_logits", "ep_logits", "halfmove"):
                    torch.testing.assert_close(a[field], b[field].cpu(), atol=3e-4, rtol=3e-4)
                torch.testing.assert_close(cpu.transition(a["latent"], self.actions), gpu.transition(b["latent"], self.actions.cuda()).cpu(), atol=3e-4, rtol=3e-4)
                a["wdl_logits"].square().sum().backward()
                b["wdl_logits"].square().sum().backward()
                torch.testing.assert_close(cpu.input.weight.grad, gpu.input.weight.grad.cpu(), atol=3e-4, rtol=3e-4)


if __name__ == "__main__":
    unittest.main()
