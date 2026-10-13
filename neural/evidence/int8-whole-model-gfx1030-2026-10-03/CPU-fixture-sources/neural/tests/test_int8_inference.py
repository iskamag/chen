"""CPU arithmetic/causality fixtures; these do not establish GPU deployment."""
import copy
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

import torch

from neural.features import encode_fen, move_to_index
from neural.int8_inference import (MAX_K, NativeInt8Backend, PackedInt8Linear, admit_gfx1030,
                                  symmetric_rows, value_only_copy)
from neural.semantic_strategy import ENCODERS, MODES, SemanticAnchor, SemanticTransition
from neural.symbolic_residual import CHANNELS, SymbolicResidualAnchor, SymbolicState


class Int8Fixtures(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(3191)

    def test_quantization_padding_zero_and_rounding_error_bound(self):
        values = torch.tensor([[0., 0., 0., 0., 0.], [-9., 1., -2., 3., 4.]])
        packed, scale = symmetric_rows(values)
        self.assertEqual(packed.shape, (2, 8))
        self.assertEqual(packed.dtype, torch.int8)
        self.assertEqual(float(scale[0]), 1.)
        self.assertTrue(torch.equal(packed[:, 5:], torch.zeros(2, 3, dtype=torch.int8)))
        error = (packed[:, :5].float() * scale[:, None] - values).abs()
        self.assertTrue(bool((error <= scale[:, None] / 2 + 1e-6).all()))
        self.assertGreaterEqual(int(packed.min()), -127)
        self.assertLessEqual(int(packed.max()), 127)
        self.assertLessEqual(MAX_K * 127 * 127, 2**31 - 1)

    def test_integer_dot_reference_matches_declared_dequantization(self):
        a, sa = symmetric_rows(torch.randn(7, 22))
        w, sw = symmetric_rows(torch.randn(32, 22))
        bias = torch.randn(32)
        actual = (a.int() @ w.int().T).float() * sa[:, None] * sw[None] + bias
        reconstructed = (a.float() * sa[:, None]) @ (w.float() * sw[:, None]).T + bias
        torch.testing.assert_close(actual, reconstructed, rtol=2e-6, atol=2e-6)
        with self.assertRaises(ValueError):
            symmetric_rows(torch.ones(1, 4, dtype=torch.float64))

    def test_zero_tiny_and_half_ties_match_v3_policy(self):
        values = torch.tensor([[-127., -2.5, -1.5, -.5, 0., .5, 1.5, 2.5, 127.],
                               [0.] * 9, [1e-40] * 9])
        packed, scale = symmetric_rows(values)
        self.assertTrue(torch.equal(packed[0, :9], torch.tensor([-127, -2, -2, 0, 0, 0, 2, 2, 127], dtype=torch.int8)))
        self.assertEqual(float(scale[1]), 1.)
        self.assertEqual(float(scale[2]), torch.finfo(torch.float32).tiny)
        self.assertTrue(torch.equal(packed[1:], torch.zeros(2, 12, dtype=torch.int8)))
        with self.assertRaisesRegex(ValueError, 'ROCm'):
            admit_gfx1030('cpu')

    def test_cpu_linear_cannot_become_a_gpu_runtime_fallback(self):
        with self.assertRaisesRegex(ValueError, 'GPU linear'):
            PackedInt8Linear(torch.nn.Linear(18, 32), object())
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'not-a-library'
            path.write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                NativeInt8Backend(path, '0' * 64)

    def test_value_only_anchor_exact_parity_with_nonzero_head_all_encoders(self):
        features = torch.stack([encode_fen(fen) for fen in (
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            '4r1k1/8/8/8/8/8/4N3/4K3 w - - 17 1')])
        for encoder in ENCODERS:
            source = SemanticAnchor(encoder, width=8, depth=1, material_scale=1.3)
            with torch.no_grad():
                source.cp.weight.normal_(0, .01)
            source.eval().requires_grad_(False)
            original = copy.deepcopy(source.state_dict())
            owned = value_only_copy(source)
            with torch.inference_mode():
                expected = source(features)
                # Any accidental decoder invocation would fail this fixture.
                with patch.object(owned.board, 'forward', side_effect=AssertionError('decoder ran')):
                    actual = owned(features)
            for key in ('latent', 'pieces', 'value_cp', 'residual_cp', 'material_cp'):
                torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
            for key, value in source.state_dict().items():
                torch.testing.assert_close(value, original[key], rtol=0, atol=0)
            self.assertTrue(all(not p.requires_grad for p in owned.parameters()))

    def test_value_only_transition_exact_parity_and_action_blind_ties(self):
        features = encode_fen('rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1')[None].repeat(2, 1, 1, 1)
        actions = torch.tensor([move_to_index('e2e4', 'w'), move_to_index('d2d4', 'w')])
        for mode in MODES:
            anchor = SemanticAnchor('threat', width=8)
            with torch.no_grad():
                anchor.cp.weight.normal_(0, .01)
            source = SemanticTransition(anchor, mode)
            source.eval().requires_grad_(False)
            owned = value_only_copy(source)
            with torch.inference_mode():
                actual, expected = owned(features, actions), source(features, actions)
            torch.testing.assert_close(actual['value_cp'], expected['value_cp'], rtol=0, atol=0)
            if mode == 'action_blind':
                torch.testing.assert_close(actual['value_cp'][0], actual['value_cp'][1], rtol=0, atol=0)

    def state_fixture(self):
        f = encode_fen('4r1k1/8/8/8/8/8/4N3/4K3 w - - 17 1')[None]
        return SymbolicState(f, torch.zeros(1, 2, 64, 64, dtype=torch.bool),
                             torch.zeros(1, 2, 64, dtype=torch.bool),
                             torch.zeros(1, 64, dtype=torch.bool), torch.tensor([.5]),
                             torch.tensor([[17, 1, 20, 3]]), 'a' * 64, 'b' * 64, 7)

    def test_symbolic_runtime_rejects_cpu_and_untyped_teacher_fields(self):
        state = self.state_fixture()
        with self.assertRaisesRegex(ValueError, 'GPU'):
            state.validate_metadata()
        with self.assertRaises(TypeError):
            SymbolicState(**{**state.__dict__, 'teacher_cp': torch.tensor([100.])})
        model = SymbolicResidualAnchor(SemanticAnchor('threat', width=8))
        with self.assertRaisesRegex(ValueError, 'typed'):
            model(dict(state=state, future_target=torch.zeros(1)))

    def test_symbolic_channels_feed_actual_learned_residual_not_unused_metadata(self):
        state = self.state_fixture()
        anchor = SemanticAnchor('threat', width=8)
        with torch.no_grad():
            anchor.cp.weight.normal_(0, .03)
        model = SymbolicResidualAnchor(anchor)
        # Explicit CPU-only arithmetic fixture; bypass is never in production.
        with patch.object(SymbolicState, 'validate_metadata', return_value=None):
            baseline = model.anchor(state.features)
            initial = model(state)
            torch.testing.assert_close(initial['value_cp'], baseline['value_cp'], rtol=0, atol=0)
            initial['value_cp'].sum().backward()
            self.assertGreater(float(model.value.weight.grad.abs().sum()), 0)
            self.assertEqual(state.channels().shape, (1, 64, len(CHANNELS)))
            with torch.no_grad():
                model.value.weight.fill_(.1)
            before = model(state)
            state.pinned[:, 0, :] = True
            after = model(state)
            self.assertFalse(torch.equal(before['symbolic_role_residual'], after['symbolic_role_residual']))
            self.assertFalse(torch.equal(before['value_cp'], after['value_cp']))


if __name__ == '__main__':
    unittest.main()
