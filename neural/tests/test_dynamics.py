import os
import unittest

import chess
import torch

from neural.dynamics import AnchoredState, StrategicTransport, batch_effects, generate_counterfactuals, move_effects, sibling_contrast, tactical_counterexamples, transport_pieces, transport_slots
from neural.features import encode_fen
from neural.models import build_model


class TransportTest(unittest.TestCase):
    def test_source_target_capture_and_castling_slot_identity(self):
        root = chess.Board()
        effects = batch_effects([move_effects(root, chess.Move.from_uci("g1f3"))])
        slots = torch.arange(64.).reshape(1, 64, 1).requires_grad_()
        result = transport_slots(slots, effects)
        self.assertEqual(result[0, 21 ^ 56, 0].item(), 6)
        self.assertEqual(result[0, 6 ^ 56, 0].item(), 0)
        self.assertEqual(result[0, 10 ^ 56, 0].item(), 10)
        result[0, 21 ^ 56].sum().backward()
        self.assertEqual(slots.grad[0, 6].item(), 1)
        castle = chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1")
        result = transport_slots(slots.detach(), batch_effects([move_effects(castle, chess.Move.from_uci("e1g1"))]))
        self.assertEqual(result[0, 6 ^ 56].item(), 4)
        self.assertEqual(result[0, 5 ^ 56].item(), 7)
        self.assertEqual(result[0, 7 ^ 56].item(), 0)

    def test_piece_algebra_exact_all_special_moves_and_colors(self):
        fixtures = [
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1c1"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8g8"),
            ("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1", "e8c8"),
            ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6"),
            ("4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1", "e4d3"),
            ("4k3/8/8/8/8/2p5/3P4/4K3 w - - 0 1", "d2c3"),
        ]
        for promotion in "nbrq":
            fixtures.extend([
                ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8" + promotion),
                ("4k3/8/8/8/8/8/p7/4K3 b - - 0 1", "a2a1" + promotion),
                ("1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7b8" + promotion),
            ])
        for fen, uci in fixtures:
            with self.subTest(move=uci, fen=fen):
                board = chess.Board(fen)
                move = chess.Move.from_uci(uci)
                effects = batch_effects([move_effects(board, move)])
                pieces = transport_pieces(encode_fen(fen)[None], effects)
                board.push(move)
                torch.testing.assert_close(pieces, encode_fen(board.fen(en_passant="fen"))[None, :12])

    def test_random_legal_algebra_and_corpus_replay(self):
        corpus = generate_counterfactuals(16, seed=412, max_plies=64)
        for row, record in enumerate(corpus.records):
            board = chess.Board(record["origin_fen"])
            for move in record["history"]:
                board.push_uci(move)
            self.assertEqual(board.fen(en_passant="fen"), record["root_fen"])
            for candidate, uci in enumerate(record["moves"]):
                effects = {key: value[row:row + 1, candidate] for key, value in corpus.effects.items()}
                predicted = transport_pieces(corpus.roots[row:row + 1], effects)
                torch.testing.assert_close(predicted, corpus.successors[row:row + 1, candidate, :12])
                self.assertEqual(move_effects(board, chess.Move.from_uci(uci)), record["effects"][candidate])

    def test_neural_predictor_does_not_read_exact_successor_and_has_action_gradient(self):
        torch.manual_seed(18)
        corpus = generate_counterfactuals(2, seed=251)
        base = build_model("square", width=16, depth=1)
        transport = StrategicTransport(16)
        latent = base.encode(corpus.roots)
        state = AnchoredState(corpus.roots, latent)
        effect_a = {key: value[:, 0] for key, value in corpus.effects.items()}
        effect_b = {key: value[:, 1] for key, value in corpus.effects.items()}
        a = transport.transition(state, effect_a, corpus.successors[:, 0])
        corrupt_observation = transport.transition(state, effect_a, torch.zeros_like(corpus.successors[:, 0]))
        torch.testing.assert_close(a.strategic, corrupt_observation.strategic)
        self.assertTrue(torch.equal(a.observed, corpus.successors[:, 0]))
        b = transport.predict(latent, corpus.roots, effect_b)
        self.assertGreater((a.strategic - b).abs().max().item(), 0.1)
        target = base.encode(corpus.successors.reshape(-1, 18, 8, 8)).reshape(2, 2, 64, 16)
        prediction = torch.stack((a.strategic, b), 1)
        loss = sibling_contrast(prediction, target) + torch.nn.functional.mse_loss(prediction, target.detach())
        optimizer = torch.optim.AdamW(transport.parameters(), lr=1e-3)
        old = transport.perspective.weight.detach().clone()
        loss.backward()
        self.assertGreater(transport.perspective.weight.grad.abs().sum().item(), 0)
        optimizer.step()
        self.assertFalse(torch.equal(old, transport.perspective.weight))

    def test_contrast_rejects_action_blind_and_swapped_prediction(self):
        target = torch.randn(3, 2, 64, 16)
        correct = sibling_contrast(target.clone(), target)
        blind = sibling_contrast(target.mean(1, keepdim=True).expand_as(target), target)
        swapped = sibling_contrast(target.flip(1), target)
        self.assertAlmostEqual(correct.item(), 0)
        self.assertGreater(blind.item(), correct.item())
        self.assertGreater(swapped.item(), blind.item())

    def test_declared_check_evasion_fixture_is_actually_check_and_both_alternatives_legal(self):
        corpus = tactical_counterexamples()
        record = next(record for record in corpus.records if record["name"] == "check_evasion_capture")
        root = chess.Board(record["root_fen"])
        self.assertTrue(root.is_check())
        for uci in record["moves"]:
            move = chess.Move.from_uci(uci)
            self.assertIn(move, root.legal_moves)
            child = root.copy()
            child.push(move)
            self.assertFalse(child.is_check())

    @unittest.skipUnless(os.environ.get("CHEN_DYNAMICS_GPU") == "1" and torch.cuda.is_available(), "GPU slot not requested")
    def test_gpu_transport_outputs_and_gradients_match_cpu(self):
        corpus = generate_counterfactuals(2, seed=678)
        effects = {key: value[:, 0] for key, value in corpus.effects.items()}
        cpu = StrategicTransport(16)
        gpu = StrategicTransport(16).cuda()
        gpu.load_state_dict(cpu.state_dict())
        latent_cpu = torch.randn(2, 64, 16, requires_grad=True)
        latent_gpu = latent_cpu.detach().cuda().requires_grad_()
        a = cpu.predict(latent_cpu, corpus.roots, effects)
        b = gpu.predict(latent_gpu, corpus.roots.cuda(), {key: value.cuda() for key, value in effects.items()})
        torch.testing.assert_close(a, b.cpu(), atol=3e-4, rtol=3e-4)
        a.square().sum().backward()
        b.square().sum().backward()
        torch.testing.assert_close(latent_cpu.grad, latent_gpu.grad.cpu(), atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(cpu.residual[-1].weight.grad, gpu.residual[-1].weight.grad.cpu(), atol=3e-4, rtol=3e-4)
        torch.testing.assert_close(transport_pieces(corpus.roots, effects), transport_pieces(corpus.roots.cuda(), {key: value.cuda() for key, value in effects.items()}).cpu())


if __name__ == "__main__":
    unittest.main()
