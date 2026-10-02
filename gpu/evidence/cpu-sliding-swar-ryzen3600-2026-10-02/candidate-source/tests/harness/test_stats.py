import unittest
from harness.stats import summarize_pairs


class PairedStatsTests(unittest.TestCase):
    def test_pentanomial_bins_and_score(self):
        summary = summarize_pairs([(0, 0), (0, .5), (0, 1), (.5, 1), (1, 1)])
        self.assertEqual(summary['pentanomial'], [1, 1, 1, 1, 1])
        self.assertEqual(summary['candidate_score'], .5)
        self.assertEqual(summary['logistic_relative_elo_point_estimate'], 0)
        self.assertIsNone(summary['paired_bootstrap_score_interval_95_diagnostic'])

    def test_no_infinite_rating_and_no_small_sample_confidence(self):
        summary = summarize_pairs([(1, 1)])
        self.assertIsNone(summary['logistic_relative_elo_point_estimate'])
        self.assertIsNone(summary['paired_bootstrap_score_interval_95_diagnostic'])

    def test_validation(self):
        with self.assertRaises(ValueError):
            summarize_pairs([])
        with self.assertRaises(ValueError):
            summarize_pairs([(None, 1)])
        for count in [0, -1]:
            with self.assertRaises(ValueError):
                summarize_pairs([(1, 0)] * 30, bootstrap_samples=count)

    def test_bootstrap_reproducible_and_color_pairing_preserved(self):
        pairs = [(0, 1), (1, .5), (.5, .5)] * 10
        a = summarize_pairs(pairs, bootstrap_samples=1000, independent_pairs=True)
        b = summarize_pairs([(y, x) for x, y in pairs], bootstrap_samples=1000, independent_pairs=True)
        self.assertEqual(a, b)
        lo, hi = a['paired_bootstrap_score_interval_95_diagnostic']
        self.assertLessEqual(lo, a['candidate_score'])
        self.assertGreaterEqual(hi, a['candidate_score'])

    def test_many_correlated_openings_do_not_create_confidence(self):
        summary = summarize_pairs([(0, 1)] * 40)
        self.assertEqual(summary['completed_pairs'], 40)
        self.assertIsNone(summary['paired_bootstrap_score_interval_95_diagnostic'])
        self.assertFalse(summary['independent_opening_pairs_asserted'])


if __name__ == '__main__':
    unittest.main()
