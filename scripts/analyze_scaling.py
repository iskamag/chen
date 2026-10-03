#!/usr/bin/env python3
"""Fit a declared one-axis development curve; never fabricate chess exponents.

The power-law model is metric - declared_floor = amplitude * x ** (-exponent).
Balanced training seeds are resampled as complete curves. Intervals describe
seed variation conditional on the evaluation set, not uncertainty over positions.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import random
import statistics

SCHEMA = 'chen_scaling_curve_v1'
AXES = {'parameters', 'unique_train_roots', 'training_macs'}
PIN_KEYS = ('split_sha256', 'teacher_protocol_sha256', 'training_protocol_sha256',
            'metric_protocol_sha256', 'axis_design_sha256')


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value):
        raise ValueError(f'{name} must be finite numeric data')
    return float(value)


def sha(value):
    return isinstance(value, str) and len(value) == 64 and all(c in '0123456789abcdef' for c in value)


def admit(curve):
    if curve.get('schema') != SCHEMA or curve.get('axis') not in AXES:
        raise ValueError('unknown scaling schema or axis')
    if curve.get('metric') not in {'root_regret_cp', 'value_mae_cp', 'ranking_cross_entropy'}:
        raise ValueError('fit an error/loss metric, not Elo or accuracy as a power law')
    floor = finite(curve.get('declared_floor'), 'declared_floor')
    if floor < 0 or not isinstance(curve.get('floor_provenance'), str) or not curve['floor_provenance']:
        raise ValueError('nonnegative predeclared floor and provenance required')
    controls = curve.get('controls')
    if not isinstance(controls, dict) or controls.get('data_split') != 'development':
        raise ValueError('only declared development curves are admitted')
    if not all(sha(controls.get(key)) for key in PIN_KEYS):
        raise ValueError('split, teacher, training, metric and axis design SHA256 pins required')
    if not controls.get('architecture_family') or not controls.get('precision'):
        raise ValueError('architecture family and precision required')
    if not isinstance(controls.get('held_fixed'), dict) or not controls['held_fixed']:
        raise ValueError('declare held-fixed axes/protocol settings')
    observations = curve.get('observations')
    if not isinstance(observations, list) or not observations:
        raise ValueError('no observations')
    grouped, run_ids = {}, set()
    for row in observations:
        if not isinstance(row, dict) or row.get('controls') != controls:
            raise ValueError('mixed controls: split, teacher, architecture or budgets changed')
        run_id = row.get('run_id')
        if not isinstance(run_id, str) or not run_id or run_id in run_ids:
            raise ValueError('distinct run identities required')
        run_ids.add(run_id)
        if not sha(row.get('checkpoint_sha256')):
            raise ValueError('frozen checkpoint SHA256 required')
        seed = row.get('seed')
        if type(seed) is not int:
            raise ValueError('integer training seed required')
        x = finite(row.get('x'), 'axis value')
        y = finite(row.get('metric_value'), 'metric_value')
        if x <= 0 or y <= floor:
            raise ValueError('positive axis and metric strictly above declared floor required')
        if x in grouped.setdefault(seed, {}):
            raise ValueError('duplicate size/seed observation')
        grouped[seed][x] = math.log(y - floor)
    seeds = sorted(grouped)
    sizes = sorted(grouped[seeds[0]])
    if len(seeds) < 3 or len(sizes) < 5:
        raise ValueError('at least three paired seeds and five axis levels required')
    if any(sorted(grouped[seed]) != sizes for seed in seeds):
        raise ValueError('seed curves must cover identical axis levels')
    if sizes[-1] / sizes[0] < 16:
        raise ValueError('axis range too narrow for the declared local fit')
    return floor, seeds, sizes, grouped


def linear_fit(xs, ys):
    xm, ym = statistics.fmean(xs), statistics.fmean(ys)
    denominator = sum((x - xm) ** 2 for x in xs)
    slope = sum((x - xm) * (y - ym) for x, y in zip(xs, ys)) / denominator
    intercept = ym - slope * xm
    residual = sum((y - (intercept + slope * x)) ** 2 for x, y in zip(xs, ys))
    total = sum((y - ym) ** 2 for y in ys)
    r2 = 1 - residual / total if total else None
    return dict(exponent=-slope, log_amplitude=intercept, log_r2=r2,
                log_rmse=math.sqrt(residual / len(xs)))


def quantile(values, probability):
    ordered = sorted(values)
    index = probability * (len(ordered) - 1)
    lo = int(index)
    return ordered[lo] + (ordered[min(lo + 1, len(ordered) - 1)] - ordered[lo]) * (index - lo)


def analyze(curve, *, bootstrap_samples=2000, random_seed=7249):
    if type(bootstrap_samples) is not int or not 100 <= bootstrap_samples <= 100000:
        raise ValueError('bootstrap_samples must be 100..100000')
    if type(random_seed) is not int:
        raise ValueError('integer resampling seed required')
    floor, seeds, sizes, grouped = admit(curve)
    xs = [math.log(size) for size in sizes]
    def fit(chosen):
        return linear_fit(xs, [statistics.fmean(grouped[s][x] for s in chosen) for x in sizes])
    result = fit(seeds)
    rng = random.Random(random_seed)
    exponents = [fit(rng.choices(seeds, k=len(seeds)))['exponent'] for _ in range(bootstrap_samples)]
    result.update(schema='chen_scaling_fit_v1', axis=curve['axis'], metric=curve['metric'],
                  declared_floor=floor, floor_provenance=curve['floor_provenance'],
                  controls=curve['controls'], seeds=seeds, axis_levels=sizes,
                  per_seed={str(seed): fit([seed]) for seed in seeds},
                  exponent_seed_resampling_95=[quantile(exponents, .025), quantile(exponents, .975)],
                  bootstrap_samples=bootstrap_samples, resampling_seed=random_seed,
                  fit_scope='local log-linear error curve; not a universal scaling law or Elo prediction',
                  uncertainty_scope='training-seed variation conditional on fixed evaluation families',
                  observations_sha256=hashlib.sha256(json.dumps(curve, sort_keys=True,
                      separators=(',', ':'), allow_nan=False).encode()).hexdigest())
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--bootstrap-samples', type=int, default=2000)
    args = parser.parse_args()
    result = analyze(json.loads(args.input.read_text()), bootstrap_samples=args.bootstrap_samples)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True, allow_nan=False) + '\n')
    print(json.dumps({key: result[key] for key in ('axis', 'metric', 'exponent', 'log_r2', 'fit_scope')}))


if __name__ == '__main__':
    main()
