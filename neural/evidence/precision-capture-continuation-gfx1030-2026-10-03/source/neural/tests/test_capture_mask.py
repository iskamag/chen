"""Mechanical capture fix: old mask arithmetic and checkpoint schema unchanged."""
import importlib.util
from pathlib import Path
import unittest

import torch

from neural.features import encode_fen
from neural.models import ChessLatent


class CaptureMaskFixtures(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(20261003)
        old_path = Path(__file__).resolve().parents[1] / 'evidence/precision-capture-partial-gfx1030-2026-10-03/source/neural-worktree/neural/models.py'
        specification = importlib.util.spec_from_file_location('neural._before_capture_mask', old_path)
        self.old_module = importlib.util.module_from_spec(specification)
        specification.loader.exec_module(self.old_module)

    def fixtures(self, dtype):
        positions = torch.stack([encode_fen(fen) for fen in (
            'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
            '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
            '3r2k1/8/3P4/8/3R4/8/3B4/3K4 w - - 7 1',
            '4k3/8/8/8/8/2q1n3/3N4/4K3 b - - 9 1')])
        # Pure blocked-edge arithmetic also covers empty and densely occupied
        # diagrams. These synthetic rows are not chess ingress certificates.
        artificial = torch.zeros(3, 18, 8, 8)
        artificial[1, 3] = 1
        artificial[2, :12] = torch.randint(0, 2, (12, 8, 8)).float()
        return torch.cat((positions, artificial)).to(dtype)

    def test_bit_exact_edges_and_encoder_float32_float64(self):
        for dtype in (torch.float32, torch.float64):
            features = self.fixtures(dtype)
            for kind in ('square', 'relation', 'plan'):
                old = self.old_module.ChessLatent(kind, width=8, depth=1).eval().to(dtype)
                new = ChessLatent(kind, width=8, depth=1).eval().to(dtype)
                result = new.load_state_dict(old.state_dict(), strict=True)
                self.assertEqual(result.missing_keys, [])
                self.assertEqual(result.unexpected_keys, [])
                with torch.inference_mode():
                    torch.testing.assert_close(new.attack_edges(features), old.attack_edges(features), rtol=0, atol=0)
                    torch.testing.assert_close(new.encode(features), old.encode(features), rtol=0, atol=0)

    def test_nonpersistent_mask_preserves_old_schema_and_device_dtype_movement(self):
        old = self.old_module.ChessLatent('relation', width=8)
        new = ChessLatent('relation', width=8)
        self.assertEqual(set(new.state_dict()), set(old.state_dict()))
        self.assertNotIn('sliding_mask', new.state_dict())
        self.assertIn('sliding_mask', dict(new.named_buffers()))
        new.load_state_dict(old.state_dict(), strict=True)
        old.load_state_dict(new.state_dict(), strict=True)
        new.double()
        self.assertEqual(new.sliding_mask.dtype, torch.float64)
        self.assertEqual(new.sliding_mask.device, new.position.device)


if __name__ == '__main__':
    unittest.main()
