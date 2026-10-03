"""CPU layout/admission checks; actual precision/capture needs GPU evidence."""
import tempfile
import unittest
from pathlib import Path

import torch

from neural.precision_inference import (HalfLinear, NativeHalfBackend,
                                        StaticCapturedInference, pack_half_weight)


class PrecisionFixtures(unittest.TestCase):
    def test_half_pair_layout_restores_odd_width_and_channel_tails(self):
        for n, k in ((1, 1), (33, 45), (32, 20), (96, 32), (1, 256)):
            original = torch.arange(n * k).reshape(n, k).float() / 128
            packed = pack_half_weight(original)
            n_pad, k_pad = (n + 31) // 32 * 32, (k + 3) // 4 * 4
            self.assertEqual(tuple(packed.shape), (k_pad // 2, n_pad * 2))
            self.assertTrue(packed.is_contiguous())
            restored = packed.reshape(k_pad // 2, n_pad, 2).permute(1, 0, 2).reshape(n_pad, k_pad)
            torch.testing.assert_close(restored[:n, :k], original.half(), rtol=0, atol=0)
            self.assertTrue(bool((restored[n:] == 0).all()))
            self.assertTrue(bool((restored[:, k:] == 0).all()))

    def test_cpu_controls_cannot_be_runtime_fallbacks(self):
        source = torch.nn.Linear(18, 32).eval().requires_grad_(False)
        with self.assertRaisesRegex(ValueError, 'GPU affine'):
            HalfLinear(source)
        with self.assertRaisesRegex(ValueError, 'ROCm'):
            StaticCapturedInference(lambda features, actions: features, torch.zeros(1, 18, 8, 8))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'not-a-library'
            path.write_bytes(b'fixture')
            with self.assertRaisesRegex(ValueError, 'library pin'):
                NativeHalfBackend(path, '0' * 64, runtime_path='unused', runtime_sha256='0' * 64)

    def test_pack_rejects_untyped_weights(self):
        for values in (torch.zeros(1, 4, dtype=torch.float64), torch.empty(0, 4), torch.zeros(4)):
            with self.assertRaisesRegex(ValueError, 'weight required'):
                pack_half_weight(values)


if __name__ == '__main__':
    unittest.main()
