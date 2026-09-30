#!/usr/bin/env python3
"""Paired host/device resident searches using the same binary/runtime and suite."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import statistics
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', default='./chen')
    parser.add_argument('--suite', default='benches/positions.txt')
    parser.add_argument('--depth', type=int, default=3)
    parser.add_argument('--eval', choices=['material', 'tapered'], default='tapered')
    parser.add_argument('--max-positions', type=int, default=250000)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--output', default='resident-comparison.json')
    args = parser.parse_args()
    if args.repeats < 3 or not 1 <= args.depth <= 64 or args.max_positions < 1:
        parser.error('requires >=3 pairs, depth 1..64 and positive arena capacity')
    binary = str(Path(args.binary).resolve())
    library = os.environ.get('CHEN_GPU_LIBRARY')
    if not library:
        parser.error('CHEN_GPU_LIBRARY must identify the actual HIP shared library')
    commands = {name: [binary, 'bench', '--backend', name, '--depth', str(args.depth),
                      '--eval', args.eval, '--max-positions', str(args.max_positions),
                      '--suite', str(Path(args.suite).resolve())]
                for name in ['cpu-resident', 'hip-resident']}
    signature_fields = ['position', 'fen', 'depth', 'score_cp', 'bestmove', 'nodes',
                        'pv', 'evaluated_positions', 'stopped', 'eval', 'tt',
                        'quiescence', 'ordering', 'max_positions']

    def run(name):
        begin = time.perf_counter_ns()
        completed = subprocess.run(commands[name], text=True, capture_output=True, check=True)
        wall = time.perf_counter_ns() - begin
        rows = [json.loads(line) for line in completed.stdout.splitlines()]
        if not rows or any(row['stopped'] for row in rows):
            raise RuntimeError('empty or capacity-limited search cannot support this comparison')
        for row in rows:
            if row['backend'] != name or row['elapsed_ns'] <= 0:
                raise RuntimeError('wrong backend or invalid timing')
        return rows, wall

    signatures = {}
    for name in commands:
        rows, _ = run(name)
        signatures[name] = [{field: row[field] for field in signature_fields} for row in rows]
    if signatures['cpu-resident'] != signatures['hip-resident']:
        raise RuntimeError('host/device semantic mismatch; timing comparison rejected')
    records = []
    for repeat in range(args.repeats):
        order = list(commands) if repeat % 2 == 0 else list(reversed(commands))
        record = {'pair': repeat, 'order': order, 'runs': {}}
        for name in order:
            rows, wall = run(name)
            if [{f: row[f] for f in signature_fields} for row in rows] != signatures[name]:
                raise RuntimeError(f'non-deterministic semantics in {name}, pair {repeat}')
            record['runs'][name] = {'process_wall_ns': wall, 'positions': rows}
        records.append(record)
    summary = []
    for index, position in enumerate(signatures['cpu-resident']):
        cpu = [record['runs']['cpu-resident']['positions'][index]['elapsed_ns'] for record in records]
        gpu = [record['runs']['hip-resident']['positions'][index]['elapsed_ns'] for record in records]
        ratios = [b / a for a, b in zip(cpu, gpu)]
        summary.append({'position': position['position'], 'depth': position['depth'],
                        'nodes': position['nodes'], 'cpu_median_ms': statistics.median(cpu) / 1e6,
                        'hip_median_ms': statistics.median(gpu) / 1e6,
                        'hip_over_cpu_paired_median': statistics.median(ratios),
                        'paired_ratio_range': [min(ratios), max(ratios)]})
    report = {'kind': 'paired_resident_search_comparison', 'strength_evidence': False,
              'config': vars(args), 'commands': commands, 'platform': platform.platform(),
              'affinity': sorted(os.sched_getaffinity(0)),
              'binary_sha256': digest(binary), 'library_sha256': digest(library),
              'suite_sha256': digest(args.suite), 'runner_sha256': digest(__file__),
              'source_commit': os.environ.get('CHEN_SOURCE_COMMIT'),
              'source_status': os.environ.get('CHEN_SOURCE_STATUS'),
              'gpu_source_sha256': {str(path): digest(path) for path in sorted(Path('gpu').glob('*'))
                                    if path.is_file() and path.suffix in ['.h', '.hip', '.sh']},
              'records': records, 'summary': summary,
              'scope': 'One warmup per backend; alternating paired process runs. Search intervals exclude context construction; process_wall_ns includes it. Same full-width algorithm and exact semantics. No performance acceptance threshold, strength inference or controlled power-state claim.'}
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'output': args.output, 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
