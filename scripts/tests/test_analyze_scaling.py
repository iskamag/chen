"""Synthetic statistical/admission checks, never chess quality evidence."""
from copy import deepcopy
import math
import unittest

from scripts.analyze_scaling import analyze


def fixture():
    controls = dict(architecture_family='synthetic_transformer', precision='fp32',
                    data_split='development', held_fixed={'data_roots': 2048, 'updates': 10000},
                    **{key: 'a' * 64 for key in ('split_sha256', 'teacher_protocol_sha256',
                        'training_protocol_sha256', 'metric_protocol_sha256', 'axis_design_sha256')})
    rows = [dict(run_id=f'{seed}-{size}', seed=seed, x=size,
                 metric_value=2 + math.exp(.05 * seed) * 30 * size ** -.4,
                 checkpoint_sha256='b' * 64, controls=controls)
            for seed in (1, 2, 3) for size in (1, 2, 4, 8, 16)]
    return dict(schema='chen_scaling_curve_v1', axis='parameters', metric='root_regret_cp',
                declared_floor=2, floor_provenance='synthetic_known_floor', controls=controls,
                observations=rows)


class ScalingAdmission(unittest.TestCase):
    def test_recovers_declared_synthetic_exponent_and_seed_intercepts(self):
        report = analyze(fixture(), bootstrap_samples=100)
        self.assertAlmostEqual(report['exponent'], .4, places=12)
        self.assertAlmostEqual(report['log_r2'], 1, places=12)
        for fit in report['per_seed'].values():
            self.assertAlmostEqual(fit['exponent'], .4, places=12)
        self.assertIn('not a universal', report['fit_scope'])

    def test_resamples_entire_seed_curves_and_keeps_uncertainty_scope(self):
        curve = fixture()
        for row in curve['observations']:
            row['metric_value'] = 2 + 30 * row['x'] ** (-.2 * row['seed'])
        first = analyze(curve, bootstrap_samples=200)
        self.assertEqual(first, analyze(curve, bootstrap_samples=200))
        lower, upper = first['exponent_seed_resampling_95']
        self.assertLess(lower, first['exponent'])
        self.assertGreater(upper, first['exponent'])
        self.assertIn('fixed evaluation families', first['uncertainty_scope'])

    def test_rejects_mixed_protocols_unbalanced_or_insufficient_curves(self):
        variants = []
        curve = fixture()
        curve['observations'][0]['controls'] = dict(curve['controls'], split_sha256='c' * 64)
        variants.append(curve)
        curve = fixture()
        curve['observations'].pop()
        variants.append(curve)
        curve = fixture()
        curve['observations'] = [row for row in curve['observations'] if row['seed'] != 3]
        variants.append(curve)
        curve = fixture()
        curve['observations'] = [row for row in curve['observations'] if row['x'] != 16]
        variants.append(curve)
        curve = fixture()
        curve['controls']['data_split'] = 'test'
        variants.append(curve)
        for malformed in variants:
            with self.subTest(malformed=malformed):
                with self.assertRaises(ValueError):
                    analyze(malformed, bootstrap_samples=100)

    def test_rejects_invalid_numbers_floor_and_elo_fits(self):
        for field, value in (('x', 0), ('x', float('nan')), ('x', True),
                             ('metric_value', 2), ('metric_value', float('inf'))):
            malformed = fixture()
            malformed['observations'][0][field] = value
            with self.assertRaises(ValueError):
                analyze(malformed, bootstrap_samples=100)
        for update in ({'declared_floor': -1}, {'metric': 'elo'}, {'floor_provenance': ''}):
            malformed = deepcopy(fixture())
            malformed.update(update)
            with self.assertRaises(ValueError):
                analyze(malformed, bootstrap_samples=100)


if __name__ == '__main__':
    unittest.main()
