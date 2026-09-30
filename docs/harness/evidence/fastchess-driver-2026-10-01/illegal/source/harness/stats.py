"""Pair-aware summaries; no cross-pool conversion to a Lichess rating."""
import math
import random


def summarize_pairs(pairs, seed=20260930, bootstrap_samples=10000, independent_pairs=False):
    if bootstrap_samples <= 0:
        raise ValueError('bootstrap_samples must be positive')
    if not pairs:
        raise ValueError('at least one completed opening pair is required')
    bins = [0] * 5
    for a, b in pairs:
        if a not in (0.0, 0.5, 1.0) or b not in (0.0, 0.5, 1.0):
            raise ValueError('game score must be 0, 0.5, or 1')
        bins[round(2 * (a + b))] += 1
    mean = sum(a + b for a, b in pairs) / (2 * len(pairs))
    delta = None if mean in (0.0, 1.0) else 400 * math.log10(mean / (1 - mean))
    # Resample opening pairs intact. This is diagnostic, not a sequential test.
    # Very small smoke samples cannot support a useful bootstrap estimate.
    interval = None
    if independent_pairs and len(pairs) >= 30:
        rng = random.Random(seed)
        values = [(a + b) / 2 for a, b in pairs]
        samples = sorted(sum(rng.choice(values) for _ in values) / len(values) for _ in range(bootstrap_samples))
        interval = [samples[int(0.025 * bootstrap_samples)], samples[min(bootstrap_samples - 1, int(0.975 * bootstrap_samples))]]
    return {'completed_pairs': len(pairs), 'games': 2 * len(pairs), 'pentanomial': bins, 'candidate_score': mean, 'logistic_relative_elo_point_estimate': delta, 'paired_bootstrap_score_interval_95_diagnostic': interval, 'independent_opening_pairs_asserted': bool(independent_pairs), 'rating_scope': 'relative to the tested engine under this workload; not Lichess Elo', 'bootstrap_assumption': 'opening pairs sampled independently; repeated/correlated openings can invalidate uncertainty estimates' if independent_pairs else 'diagnostic workload; opening-pair independence not asserted'}
