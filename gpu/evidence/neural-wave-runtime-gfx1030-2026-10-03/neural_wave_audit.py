#!/usr/bin/env python3
"""Cold stdlib replay of native wave timings and counter sanity limitations."""
import argparse
import collections
import csv
import gzip
import hashlib
import io
import json
from pathlib import Path
import statistics


def read(path):
    if path.exists():
        return path.read_bytes()
    return gzip.decompress(path.with_name(path.name + '.gz').read_bytes())


def summary(values):
    return dict(min=min(values), median=statistics.median(values), max=max(values), count=len(values))


def audit(root, preparation=None):
    report = json.loads(read(root / 'wave-timing.json'))
    assert report['terminal'] == 'PASS'
    assert len(report['rows']) == 15
    assert sum(row['m'] * row['n'] for row in report['rows']) == 887777
    if preparation:
        manifest_bytes = (preparation / 'manifest.json').read_bytes()
        execution = json.loads(read(root / 'execution.json'))
        assert hashlib.sha256(manifest_bytes).hexdigest() == execution['preparation_manifest_sha256']
        for name, expected in json.loads(manifest_bytes)['payload_sha256'].items():
            assert hashlib.sha256((preparation / name).read_bytes()).hexdigest() == expected, name
        assert hashlib.sha256(read(preparation / 'neural_wave_probe.py')).hexdigest() == report['probe_sha256']
        assert hashlib.sha256(read(preparation / 'source-binding.json')).hexdigest() == report['binding_sha256']
        binding = json.loads(read(preparation / 'source-binding.json'))
        for name, expected in binding['generated'].items():
            assert hashlib.sha256(read(preparation / name)).hexdigest() == expected
    for width in (32, 64):
        admission = report['variants'][str(width)]
        assert admission['runtime']['hip_runtime_get_version'] == 70051831
        assert admission['runtime']['symbol_owner_sha256'] == 'c13a873d6444aa9c32d846feaa888de2108a5ed3c15f8a1b7708c5742a894e81'
        for lib in admission['libraries']:
            assert hashlib.sha256(read(root / Path(lib['path']).name)).hexdigest() == lib['sha256']
        for resource in admission['resources']:
            assert resource['source_wave_width'] == width
            assert resource['lds'] == resource['private_bytes'] == 0
    timings = []
    for row in report['rows']:
        assert len(row['parity']) == 8 and len(row['samples']) == 144
        for method in ('int8_static_residual', 'int8_dynamic_rows_residual', 'fp16', 'fp32'):
            for mode in ('eager', 'graph'):
                values = {width: {x['sample']: x['event_us'] for x in row['samples']
                    if x['method'] == f'wave{width}_{method}' and x['mode'] == mode}
                    for width in (32, 64)}
                assert all(set(v) == set(range(9)) for v in values.values())
                ratios = [values[32][i] / values[64][i] for i in range(9)]
                timings.append(dict(m=row['m'], n=row['n'], source_k=row['source_k'],
                    method=method, mode=mode, wave32_us=summary(list(values[32].values())),
                    wave64_us=summary(list(values[64].values())),
                    paired_speedup32_over64=summary(ratios)))
    counters = []
    for path in sorted(root.glob('counter-wave*/*/*counter_collection.csv*')):
        grouped = collections.defaultdict(list)
        for row in csv.DictReader(io.StringIO(read(path.with_suffix('') if path.suffix == '.gz' else path).decode())):
            name = row['Kernel_Name']
            method = 'dynamic' if 'dynamic_linear' in name else ('fp16' if 'affine<true>' in name else 'fp32')
            grouped[(method, row['Counter_Name'])].append(float(row['Counter_Value']))
        for (method, counter), values in grouped.items():
            assert len(values) == 8
            counters.append(dict(pass_name=path.parents[1].name, method=method,
                                 counter=counter, values=values, summary=summary(values)))
    assert len(counters) == 36
    invalid = dict(wave32_instruction_count_zero=True,
        wave64_instruction_count_zero_in_actual64=True,
        started_waves_disagree_with_exact8192=True,
        mean_occupancy_contains_values_over64=True)
    assert all(x['summary']['max'] == 0 for x in counters if x['counter'] == 'SQ_WAVE32_INSTS')
    assert all(x['summary']['max'] == 0 for x in counters if x['counter'] == 'SQ_WAVE64_INSTS' and 'wave64' in x['pass_name'])
    assert all(x['summary']['median'] != 8192 for x in counters if x['counter'] == 'SQ_WAVES')
    assert any(x['summary']['max'] > 64 for x in counters if x['counter'] == 'MeanOccupancyPerCU')
    return dict(schema=1, terminal='PASS_WITH_COUNTER_LIMITATIONS', shapes=15,
                outputs_per_operator_path=887777, timings=timings, counters=counters,
                counter_sanity_failures=invalid,
                conclusion='retain wave32 default; scalar dynamic wave64 is a diagnostic candidate only; no achieved occupancy/instruction-width conclusions')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--preparation', type=Path, help='verify the linked source/compiler/ISA preparation closure too')
    args = parser.parse_args()
    manifest_path = args.root / 'manifest.json'
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        for path, expected in manifest['payload_sha256'].items():
            assert hashlib.sha256((args.root / path).read_bytes()).hexdigest() == expected, path
    result = audit(args.root, args.preparation)
    if args.output:
        if args.output.exists():
            raise ValueError('refusing to overwrite evidence')
        args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(dict(terminal=result['terminal'], shapes=result['shapes'],
                         outputs=result['outputs_per_operator_path'], counter_sanity_failures=result['counter_sanity_failures'])))


if __name__ == '__main__':
    main()
