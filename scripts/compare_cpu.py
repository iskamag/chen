#!/usr/bin/env python3
"""Compare CPU search implementations at identical work with alternating order.

No playing-strength inference: this rejects score, move, PV or node changes and
reports search-only wall observations with fresh searchers from `chen bench`.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', type=Path, required=True)
    parser.add_argument('--candidate', type=Path, required=True)
    parser.add_argument('--suite', type=Path, default=Path('benches/positions.txt'))
    parser.add_argument('--rounds', type=int, default=6)
    parser.add_argument('--nodes', type=int, default=100_000)
    parser.add_argument('--depth', type=int, default=64)
    parser.add_argument('--cpu', type=int)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.rounds < 2 or args.rounds % 2 or args.nodes < 1 or not 1 <= args.depth <= 64:
        parser.error('use positive nodes, depth 1..64, and a positive even round count')
    args.base = args.base.resolve()
    args.candidate = args.candidate.resolve()
    args.suite = args.suite.resolve()
    args.out.mkdir(parents=True, exist_ok=False)
    if args.cpu is not None:
        os.sched_setaffinity(0, {args.cpu})
    payload = {
        'source_scope': 'Exact implementation comparison, not a strength test',
        'platform': platform.platform(),
        'affinity': sorted(os.sched_getaffinity(0)),
        'config': {key: str(value) if isinstance(value, Path) else value
                   for key, value in vars(args).items()},
        'sha256': {key: sha(getattr(args, key)) for key in ('base', 'candidate', 'suite')},
        'runner_sha256': sha(__file__),
        'rows': [],
    }
    fields = ('position', 'fen', 'depth', 'score_cp', 'bestmove', 'nodes', 'qnodes', 'stopped', 'pv')
    reference = {}
    totals = {}
    # First round warms both executables; all observations are retained.
    for round_index in range(args.rounds + 2):
        for variant in (('base', 'candidate') if round_index % 2 == 0 else ('candidate', 'base')):
            for pvs in (False, True):
                command = [str(getattr(args, variant)), 'bench', '--suite', str(args.suite),
                           '--depth', str(args.depth), '--nodes', str(args.nodes)]
                if pvs:
                    command.append('--pvs')
                started = time.perf_counter_ns()
                result = subprocess.run(command, text=True, capture_output=True, check=True, timeout=120)
                wall_ns = time.perf_counter_ns() - started
                rows = [json.loads(line) for line in result.stdout.splitlines()]
                if not rows:
                    raise RuntimeError('empty benchmark output')
                for row in rows:
                    signature = {key: row[key] for key in fields}
                    key = (pvs, row['position'])
                    if key in reference and reference[key] != signature:
                        raise RuntimeError(f'changed search trajectory: {variant}/{key}: '
                                           f'{reference[key]!r} != {signature!r}')
                    reference[key] = signature
                    payload['rows'].append({'round': round_index, 'warmup': round_index < 2,
                                            'variant': variant, 'pvs': pvs, **row})
                totals[(round_index, variant, pvs)] = sum(row['elapsed_ns'] for row in rows)
                (args.out / f'{round_index}-{variant}-pvs{int(pvs)}.jsonl').write_text(result.stdout)
                (args.out / f'{round_index}-{variant}-pvs{int(pvs)}.stderr').write_text(result.stderr)
                payload.setdefault('process_wall_ns', []).append(
                    {'round': round_index, 'variant': variant, 'pvs': pvs, 'elapsed_ns': wall_ns})
    summary = {}
    for pvs in (False, True):
        ratios = [totals[(r, 'candidate', pvs)] / totals[(r, 'base', pvs)]
                  for r in range(2, args.rounds + 2)]
        summary[f'pvs_{pvs}'] = {
            'candidate_over_base_round_total_ratios': ratios,
            'median_speedup': 1 / statistics.median(ratios),
            'per_position_median_speedup': {},
        }
        for position in {row['position'] for row in payload['rows']}:
            times = {(row['round'], row['variant']): row['elapsed_ns'] for row in payload['rows']
                     if not row['warmup'] and row['position'] == position and row['pvs'] == pvs}
            paired = [times[(r, 'candidate')] / times[(r, 'base')]
                      for r in range(2, args.rounds + 2)]
            summary[f'pvs_{pvs}']['per_position_median_speedup'][position] = 1 / statistics.median(paired)
    payload['summary'] = summary
    payload['signatures_checked'] = len(reference)
    (args.out / 'report.json').write_text(json.dumps(payload, indent=2) + '\n')
    print(json.dumps({'report': str(args.out / 'report.json'), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
