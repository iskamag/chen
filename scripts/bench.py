#!/usr/bin/env python3
"""Run paired baseline/candidate searches and retain metadata plus raw JSONL."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import shlex
import tarfile
import statistics
import subprocess
import time


def command(args):
    return subprocess.check_output(args, text=True).strip()


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base', default='target/release/chen')
    parser.add_argument('--candidate', default='target/release/chen')
    parser.add_argument('--base-args', default='')
    parser.add_argument('--candidate-args', default='--no-tt')
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--suite', default='benches/positions.txt')
    parser.add_argument('--out', default=None)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error('--repeats must be positive')
    out = Path(args.out or f'results/{time.strftime("%Y%m%d-%H%M%S")}')
    out.mkdir(parents=True, exist_ok=False)
    cmds = {label: [str(Path(binary).resolve()), 'bench', '--depth', str(args.depth), '--suite', str(Path(args.suite).resolve())] + shlex.split(extra) for label, binary, extra in [('base', args.base, args.base_args), ('candidate', args.candidate, args.candidate_args)]}
    cpu = next((line.split(':', 1)[1].strip() for line in Path('/proc/cpuinfo').read_text().splitlines() if line.startswith('model name')), platform.processor()) if Path('/proc/cpuinfo').exists() else platform.processor()
    metadata = {'commands': cmds, 'binary_sha256': {k: sha(v[0]) for k, v in cmds.items()}, 'suite_sha256': sha(args.suite), 'rustc': command(['rustc', '--version']), 'platform': platform.platform(), 'cpu': cpu, 'affinity': sorted(os.sched_getaffinity(0)) if hasattr(os, 'sched_getaffinity') else None, 'commit': command(['git', 'rev-parse', '--verify', 'HEAD']) if subprocess.run(['git', 'rev-parse', '--verify', 'HEAD'], capture_output=True).returncode == 0 else None, 'status': command(['git', 'status', '--short']), 'seed': 20260930, 'repeats': args.repeats, 'threads_per_engine': 1, 'node_convention': 'total nodes includes qnodes', 'build_profile': 'record separately if binaries were not cargo --release'}
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    (out / 'dirty.patch').write_text(command(['git', 'diff', 'HEAD']) if metadata['commit'] else command(['git', 'diff']))
    source_files = subprocess.check_output(['git', 'ls-files', '-z', '--cached', '--others', '--exclude-standard']).split(b'\0')
    with tarfile.open(out / 'source.tar.gz', 'w:gz') as archive:
        for name in sorted(set(source_files)):
            if name and Path(os.fsdecode(name)).is_file():
                archive.add(os.fsdecode(name), recursive=False)
    # Warm both binaries, then run alternating AB/BA pairs.
    for cmd in cmds.values():
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
    timings = {'base': {}, 'candidate': {}}
    semantic = {'base': {}, 'candidate': {}}
    for repeat in range(args.repeats):
        order = ['base', 'candidate'] if repeat % 2 == 0 else ['candidate', 'base']
        for label in order:
            raw = command(cmds[label]) + '\n'
            (out / f'{repeat:02}-{label}.jsonl').write_text(raw)
            for row in map(json.loads, raw.splitlines()):
                key = row['position']
                timings[label].setdefault(key, []).append(row['elapsed_ns'])
                signature = (row['depth'], row['score_cp'], row['nodes'], row['bestmove'])
                if key in semantic[label] and semantic[label][key] != signature:
                    raise RuntimeError(f'nondeterministic search: {label}/{key}')
                semantic[label][key] = signature
    if timings['base'].keys() != timings['candidate'].keys():
        raise RuntimeError('suites differ')
    summary = []
    for key in timings['base']:
        ratios = [b / a for a, b in zip(timings['base'][key], timings['candidate'][key])]
        summary.append({'position': key, 'median_candidate_over_base_time': statistics.median(ratios), 'min_ratio': min(ratios), 'max_ratio': max(ratios), 'base': semantic['base'][key], 'candidate': semantic['candidate'][key], 'same_score': semantic['base'][key][1] == semantic['candidate'][key][1]})
    (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({'results': str(out), 'positions': summary}, indent=2))


if __name__ == '__main__':
    main()
