from decimal import Decimal, ROUND_HALF_UP
import math
import unittest

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.features import encode_fen
from neural.score_frames import (AS, FRAME_KIND, PHASE_WEIGHTS, REFERENCE_A,
    interval_smooth_l1, score_frame_manifest, sf19_reference_batch, sf19_reference_score)

KIND = 'ordinary_unbounded_cp_no_tablebase'


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError('host scalar extraction')
        return function(*args, **(kwargs or {}))


def phase(board):
    return sum(PHASE_WEIGHTS[piece.piece_type - 1] for piece in board.piece_map().values())


def decimal_a(material):
    m = Decimal(min(max(material, 17), 78)) / Decimal(58)
    coefficients = [Decimal(str(x)) for x in AS]
    return ((coefficients[0] * m + coefficients[1]) * m + coefficients[2]) * m + coefficients[3]


class FrameTests(unittest.TestCase):
    def test_ordinary_reporting_inverse_intervals_and_reference_clamps(self):
        # Independent decimal polynomial + C++ std::round's half-away rule.
        for material in [0, 16, 17, 21, 37, 58, 77, 78, 79, 576]:
            a = decimal_a(material)
            for internal in [-10000, -801, -7, -1, 0, 1, 7, 801, 10000]:
                report = int((100 * Decimal(internal) / a).quantize(Decimal(1), rounding=ROUND_HALF_UP))
                target = float(100 * Decimal(internal) / decimal_a(58))
                record = sf19_reference_score(report, material, score_kind=KIND)
                self.assertLessEqual(record['lower_cp'], target)
                self.assertGreaterEqual(record['upper_cp'], target)
                negative = sf19_reference_score(report, material, score_kind=KIND, negate=True)
                self.assertEqual(negative['value_cp'], -record['value_cp'])
                self.assertEqual(negative['lower_cp'], -record['upper_cp'])
                self.assertEqual(negative['upper_cp'], -record['lower_cp'])
        self.assertEqual(sf19_reference_score(123, 58, score_kind=KIND)['value_cp'], 123.)
        self.assertEqual(sf19_reference_score(123, 0, score_kind=KIND)['value_cp'],
                         sf19_reference_score(123, 17, score_kind=KIND)['value_cp'])
        self.assertEqual(sf19_reference_score(123, 576, score_kind=KIND)['value_cp'],
                         sf19_reference_score(123, 78, score_kind=KIND)['value_cp'])
        self.assertGreater(REFERENCE_A, 0)

    def test_parent_child_normalization_differs_on_capture_and_promotion(self):
        root = chess.Board()
        root.push_uci('e2e4')
        root.push_uci('d7d5')
        fixtures = [(root, 'e4d5'),
                    (chess.Board('1r2k2r/P7/8/8/8/8/8/R3K2R w KQk - 0 1'), 'a7b8q')]
        for board, uci in fixtures:
            move = chess.Move.from_uci(uci)
            self.assertIn(move, board.legal_moves)
            child = board.copy()
            child.push(move)
            parent = sf19_reference_score(100, phase(board), score_kind=KIND, negate=True)
            standalone = sf19_reference_score(-100, phase(child), score_kind=KIND)
            self.assertNotEqual(parent['normalization_root_material'], standalone['normalization_root_material'])
            self.assertNotEqual(parent['value_cp'], standalone['value_cp'])
            mirror = sf19_reference_score(100, phase(board.mirror()), score_kind=KIND, negate=True)
            self.assertEqual(parent, mirror)
        # One parent factor preserves all candidate ordering and ties.
        for material in [17, 40, 58, 78]:
            values = [-250, 0, 0, 100, 801]
            converted = [sf19_reference_score(v, material, score_kind=KIND)['value_cp'] for v in values]
            self.assertEqual(sorted(converted), converted)
            self.assertEqual(converted[1], converted[2])

    def test_resident_batch_agrees_and_outward_intervals_cover_float_cast(self):
        boards = [chess.Board(), chess.Board('4k3/8/8/8/8/8/8/4K3 b - - 0 1'),
                  chess.Board('1r2k2r/P7/8/8/8/8/8/R3K2R w KQk - 0 1')]
        features = torch.stack([encode_fen(b.fen(en_passant='fen')) for b in boards])
        cp = torch.tensor([123., -789., 12345.])
        with NoHostScalar():
            out = sf19_reference_batch(cp, features, score_kind=KIND, negate=True)
        for index, board in enumerate(boards):
            record = sf19_reference_score(int(cp[index]), phase(board), score_kind=KIND, negate=True)
            self.assertEqual(int(out['normalization_root_material'][index]), phase(board))
            self.assertLessEqual(float(out['lower_cp'][index]), record['lower_cp'])
            self.assertGreaterEqual(float(out['upper_cp'][index]), record['upper_cp'])
            self.assertAlmostEqual(float(out['value_cp'][index]), record['value_cp'], delta=.002)
        self.assertTrue(torch.equal(out['normalization_root_material'],
                                    sf19_reference_batch(cp, features.flip(0).flip(0), score_kind=KIND)['normalization_root_material']))
        manifest = score_frame_manifest()
        self.assertEqual(manifest['kind'], FRAME_KIND)
        self.assertEqual(len(manifest['uci_source_sha256']), 64)
        self.assertEqual(manifest['source_rounding'], 'std_round_nearest_half_away_from_zero')

    def test_interval_loss_gradients_and_invalid_frame_admission(self):
        values = torch.tensor([-2., -.5, 0., .5, 2.], requires_grad=True)
        with NoHostScalar():
            loss = interval_smooth_l1(values, torch.zeros(5), torch.ones(5), scale=1.)
            loss.sum().backward()
        torch.testing.assert_close(loss, torch.tensor([.5, 0., 0., 0., .5]))
        torch.testing.assert_close(values.grad, torch.tensor([-1., 0., 0., 0., 1.]))
        for kwargs in [dict(reported_cp=True, root_material=58),
                       dict(reported_cp=2, root_material=True),
                       dict(reported_cp=2, root_material=-1),
                       dict(reported_cp=2.5, root_material=58)]:
            with self.assertRaises(ValueError):
                sf19_reference_score(**kwargs, score_kind=KIND)
        for kind in ['mate', 'tablebase', 'lowerbound', 'cp']:
            with self.assertRaises(ValueError):
                sf19_reference_score(100, 58, score_kind=kind)
        with self.assertRaises(ValueError):
            interval_smooth_l1(values, torch.zeros(1), torch.ones(5))


if __name__ == '__main__':
    unittest.main()
