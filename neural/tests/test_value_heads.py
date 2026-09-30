"""Value-only adapters: material oracle, gradient conditioning, artifact scope."""
import copy
from pathlib import Path
import random
import tempfile
import unittest

import chess
import torch
from torch.nn import functional as F

from neural.features import encode_fen
from neural.inference import StaticEvaluator, normalized_scalar
from neural.models import MODELS, build_model
from neural.value_heads import (HEADS, MATERIAL_VALUES, ValueAdapter, material_cp,
                                load_value_checkpoint, save_value_checkpoint)

FENS = (chess.STARTING_FEN, "4k3/8/8/3pP3/8/8/8/R3K2R w KQ d6 7 4",
        "r3k2r/8/8/8/3Pp3/8/8/4K3 b kq d3 7 4",
        "4k3/8/8/8/8/8/8/3QK3 w - - 0 1")


class ValueHeadsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def setUp(self):
        torch.manual_seed(305)
        self.features = torch.stack([encode_fen(fen) for fen in FENS])

    def test_material_matches_independent_piece_count_oracle_and_symmetry(self):
        board, rng = chess.Board(), random.Random(815)
        samples = []
        for _ in range(90):
            raw = sum(value * (len(board.pieces(piece, board.turn)) - len(board.pieces(piece, not board.turn)))
                      for piece, value in enumerate(MATERIAL_VALUES, 1))
            features = encode_fen(board.fen(en_passant="fen"))
            self.assertEqual(material_cp(features[None]).item(), max(-29000, min(29000, raw)))
            self.assertTrue(torch.equal(material_cp(features[None]), material_cp(encode_fen(board.mirror().fen(en_passant="fen"))[None])))
            samples.append(features)
            moves = list(board.legal_moves)
            if not moves:
                board.reset()
            else:
                board.push(rng.choice(moves))
        batch = torch.stack(samples)
        changed = batch.clone(); changed[:, 12:] = torch.rand_like(changed[:, 12:])
        self.assertTrue(torch.equal(material_cp(batch), material_cp(changed)))
        for side in (1, -1):
            excess = torch.zeros(1, 18, 8, 8)
            excess[:, 4 if side > 0 else 10] = 1
            self.assertEqual(material_cp(excess).item(), side * 29000)

    def test_heads_preserve_encoder_and_exact_initial_control(self):
        for name in MODELS:
            for kind in HEADS:
                with self.subTest(model=name, head=kind):
                    base = build_model(name, width=16, depth=1)
                    before = copy.deepcopy(base.state_dict())
                    rng = torch.random.get_rng_state().clone()
                    adapter = ValueAdapter(base, kind)
                    self.assertTrue(torch.equal(torch.random.get_rng_state(), rng))
                    self.assertTrue(all(torch.equal(value, before[key]) for key, value in base.state_dict().items()))
                    output = adapter(self.features)
                    self.assertEqual(set(output), {"value_cp", "utility", "material_cp", "residual_cp"})
                    for tensor in output.values():
                        self.assertEqual(tensor.shape, (4,))
                        self.assertEqual(tensor.dtype, torch.float32)
                        self.assertTrue(bool(torch.isfinite(tensor).all()))
                    if kind == "three_logit":
                        self.assertTrue(torch.equal(output["utility"], normalized_scalar(base, self.features)))
                        self.assertTrue(torch.equal(adapter.static_scores(self.features), StaticEvaluator(base)(self.features)))
                    elif kind == "direct_cp":
                        self.assertEqual(output["value_cp"].count_nonzero().item(), 0)
                    else:
                        self.assertTrue(torch.equal(output["value_cp"], material_cp(self.features)))
                        self.assertEqual(output["residual_cp"].count_nonzero().item(), 0)
                    self.assertTrue(torch.equal(adapter.encode(self.features[1:2]), adapter.encode(self.features[2:3])))

    def test_gradients_update_new_heads_then_encoder_with_same_cp_loss(self):
        targets = torch.tensor([180., -220., 360., 1100.])
        for name in MODELS:
            for kind in HEADS:
                with self.subTest(model=name, head=kind):
                    adapter = ValueAdapter(build_model(name, width=16, depth=1), kind)
                    active = list(adapter.active_parameters())
                    self.assertEqual(len(active), len(set(map(id, active))))
                    self.assertLess(len(active), len(list(adapter.parameters())))
                    optimizer = torch.optim.SGD(active, lr=0.03)
                    losses, encoder_gradients = [], []
                    for _ in range(3):
                        optimizer.zero_grad(set_to_none=True)
                        loss = F.smooth_l1_loss(adapter(self.features)["value_cp"] / 800, targets / 800)
                        loss.backward()
                        for parameter in active:
                            self.assertIsNotNone(parameter.grad)
                            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
                        encoder_gradients.append(adapter.base.input.weight.grad.abs().sum().item())
                        losses.append(loss.item()); optimizer.step()
                    if kind != "three_logit":
                        self.assertEqual(encoder_gradients[0], 0)
                        self.assertGreater(encoder_gradients[1], 0)
                    self.assertLess(losses[-1], losses[0])
                    self.assertIsNone(adapter.base.policy_source.weight.grad)
                    self.assertIsNone(adapter.base.dynamics[1].weight.grad)

    def test_scalar_cp_remains_trainable_beyond_inverse_utility_saturation(self):
        for kind in ("direct_cp", "material_residual"):
            adapter = ValueAdapter(build_model("square", width=8, depth=1), kind)
            with torch.no_grad():
                adapter.head.bias.fill_(100)
            output = adapter(self.features)
            self.assertTrue(bool((output["value_cp"] > 29000).all()))
            self.assertTrue(bool((output["utility"] == 1).all()))
            output["value_cp"].sum().backward()
            self.assertEqual(adapter.head.bias.grad.item(), 4 * 800)
            self.assertTrue(torch.equal(adapter.static_scores(self.features), torch.full((4,), 29000, dtype=torch.int32)))
        control = ValueAdapter(build_model("square", width=8, depth=1), "three_logit")
        with torch.no_grad():
            control.base.wdl.weight.zero_(); control.base.wdl.bias.copy_(torch.tensor([-40., 0., 40.]))
        result = control(self.features)
        self.assertTrue(bool((result["value_cp"] < 6000).all()))
        result["value_cp"].sum().backward()
        self.assertEqual(control.base.wdl.bias.grad.abs().sum().item(), 0)

    def test_material_special_move_effects_and_quiet_action_tie(self):
        positions = (("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 100),
                     ("4k3/P7/8/8/8/8/8/4K3 w - - 0 1", "a7a8q", 800),
                     ("4k3/8/8/8/8/8/8/R3K2R w KQ - 0 1", "e1g1", 0),
                     ("4k2q/8/8/8/8/8/8/4K2R b K - 0 1", "h8h1", 500))
        for fen, uci, gain in positions:
            board = chess.Board(fen)
            before = material_cp(encode_fen(board.fen(en_passant="fen"))[None])
            self.assertIn(chess.Move.from_uci(uci), board.legal_moves)
            board.push_uci(uci)
            after_in_parent_frame = -material_cp(encode_fen(board.fen(en_passant="fen"))[None])
            self.assertEqual((after_in_parent_frame - before).item(), gain)
        board = chess.Board()
        values = []
        for uci in ("g1f3", "b1c3"):
            child = board.copy(); child.push_uci(uci)
            values.append(-material_cp(encode_fen(child.fen())[None]))
        self.assertTrue(torch.equal(*values))

    def test_checkpoint_roundtrip_and_semantic_corruption_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            for name in MODELS:
                for kind in HEADS:
                    adapter = ValueAdapter(build_model(name, width=8, depth=1), kind)
                    path = Path(directory) / f"{name}-{kind}.pt"
                    saved = save_value_checkpoint(adapter, path, {"fixture_only": True, "seed": 305})
                    restored, metadata = load_value_checkpoint(path)
                    self.assertEqual(saved["kind"], "chen_value_adapter_checkpoint")
                    self.assertTrue(metadata["provenance"]["fixture_only"])
                    with torch.inference_mode():
                        self.assertTrue(torch.equal(adapter.static_scores(self.features), restored.static_scores(self.features)))
            original = torch.load(path, weights_only=True)
            for mutate in (lambda p: p.update(checkpoint_version=True),
                           lambda p: p["adapter_config"].update(head_version=True),
                           lambda p: p["adapter_config"].update(scale_cp=400),
                           lambda p: p["adapter_config"]["material_values"].__setitem__(-1, False),
                           lambda p: p["state_dict"]["material_values"].__setitem__(1, 321),
                           lambda p: p["state_dict"]["base.next_orientation"].__setitem__(0, 64),
                           lambda p: p["state_dict"].__setitem__("head.weight", p["state_dict"]["head.weight"].double()),
                           lambda p: p["state_dict"]["head.bias"].fill_(float("nan"))):
                damaged = copy.deepcopy(original); mutate(damaged); torch.save(damaged, path)
                with self.assertRaises(ValueError):
                    load_value_checkpoint(path)
            for malformed in (self.features.double(), self.features[:, :17], self.features[0]):
                with self.assertRaises(ValueError):
                    adapter(malformed)


if __name__ == "__main__":
    unittest.main()
