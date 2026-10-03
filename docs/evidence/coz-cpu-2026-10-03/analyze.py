#!/usr/bin/env python3
"""Strict zero-control Coz JSONL throughput analysis; no hot-sample speedup claim."""
import argparse
import collections
import json
import pathlib


def analyze(path):
    experiments = []
    samples = {}
    runtime = 0
    current = None
    for number, line in enumerate(path.read_text().splitlines(), 1):
        row = json.loads(line)
        kind = row['type']
        if kind == 'experiment':
            assert row['duration'] > 0 and 0 <= row['speedup'] <= 1
            current = dict(row, row_number=number, throughput=[], latency=[])
            experiments.append(current)
        elif kind == 'throughput-point':
            assert current is not None and row['delta'] >= 0
            current['throughput'].append(row)
        elif kind == 'latency-point':
            assert current is not None
            current['latency'].append(row)
        elif kind == 'samples':
            # These are cumulative snapshots; adding them double-counts earlier samples.
            assert row['count'] >= samples.get(row['location'], 0)
            samples[row['location']] = row['count']
        elif kind == 'runtime':
            runtime = max(runtime, row['time'])
        else:
            assert kind == 'startup', kind
    groups = {}
    for experiment in experiments:
        for point in experiment['throughput']:
            key = (experiment['selected'], point['name'], experiment['speedup'])
            group = groups.setdefault(key, dict(experiments=0, duration_ns=0, progress=0, selected_samples=0))
            group['experiments'] += 1
            group['duration_ns'] += experiment['duration']
            group['progress'] += point['delta']
            group['selected_samples'] += experiment['selected_samples']
    curves = []
    for selected, name in sorted({key[:2] for key in groups}):
        control = groups.get((selected, name, 0.0))
        points = []
        for line, progress, speedup in sorted(groups):
            if (line, progress) != (selected, name):
                continue
            group = groups[line, progress, speedup]
            admitted = bool(control and control['progress'] >= 5 and group['progress'] >= 5
                            and control['selected_samples'] > 0 and group['selected_samples'] > 0)
            ratio = (control['duration_ns'] / control['progress']) / (group['duration_ns'] / group['progress']) if admitted else None
            points.append(dict(virtual_speedup_fraction=speedup, **group,
                               zero_control_admitted=admitted, predicted_throughput_ratio=ratio,
                               predicted_period_reduction_percent=(1 - 1 / ratio) * 100 if ratio else None))
        curves.append(dict(selected=selected, progress_point=name, points=points))
    return dict(profile=str(path), experiment_count=len(experiments), runtime_ns=runtime,
                cumulative_samples=samples, sampled_source_lines=len(samples), curves=curves,
                latency_records=sum(len(e['latency']) for e in experiments),
                scope='Coz virtual-speedup throughput observations with exact zero controls; '
                      'not an implemented optimization, clean timing, strength, GPU or latency-model estimate')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('profile', type=pathlib.Path)
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    result = analyze(args.profile)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(experiments=result['experiment_count'], sampled_lines=result['sampled_source_lines'],
                         causal_curves=sum(any(p['virtual_speedup_fraction'] > 0 and p['zero_control_admitted']
                                               for p in curve['points']) for curve in result['curves']))))


if __name__ == '__main__':
    main()
