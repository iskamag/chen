#!/usr/bin/env python3
"""Reduce frozen serial raw rows; retain each process sample, not just a ratio."""
import argparse
import hashlib
import json
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--matrix', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    manifest_path = args.matrix / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    branch = manifest['branch']
    groups = {}
    for run in manifest['runs']:
        raw = args.matrix / run['file']
        assert hashlib.sha256(raw.read_bytes()).hexdigest() == run['stdout_sha256']
        for line in raw.read_text().splitlines():
            if not line.startswith('{'):
                continue
            row = json.loads(line)
            kind = row.get('kind')
            if branch == 'relation':
                if kind not in ['aggregate', 'kernel_aggregate']:
                    continue
                dimensions = (run['frontend'], kind, row['variant'], row['subset'])
                metrics = {'us_per_root': row['us_per_root']}
            elif branch == 'causal':
                if kind != 'timing' or row.get('family', 'all') not in ['all', 'generated']:
                    continue
                dimensions = (row['mode'], row.get('family', 'all'), row['method'])
                metric = 'ns_per_child' if 'ns_per_child' in row else 'ns_per_root'
                metrics = {metric: row[metric]}
            else:
                if kind not in ['benchmark', 'constraints', 'transitions']:
                    continue
                if row.get('family', 'ALL') != 'ALL':
                    continue
                dimensions = (kind, row.get('variant', kind))
                metrics = {k: v for k, v in row.items() if k.endswith('_ns')}
            for metric, value in metrics.items():
                key = (run['build'],) + dimensions + (metric,)
                groups.setdefault(key, []).append(value)
    summary = []
    for key, values in sorted(groups.items()):
        assert len(values) == manifest['rounds'], ('missing sample', key, values)
        summary.append({'key': key, 'median': statistics.median(values), 'samples': values})
    report = {'branch': branch, 'cpu': manifest['cpu'], 'runs': len(manifest['runs']),
              'rounds': manifest['rounds'], 'corpus_sha256': manifest['corpus_sha256'],
              'binaries': manifest['binaries'],
              'manifest_sha256': hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
              'summary': summary}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'{branch}: {len(summary)} aggregate metrics from {len(manifest["runs"])} process runs')


if __name__ == '__main__':
    main()
