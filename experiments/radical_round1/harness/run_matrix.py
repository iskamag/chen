#!/usr/bin/env python3
"""Replay frozen portable/native executables serially, retaining every raw row.

Run after exploratory agents stop CPU work. This controls ordering and affinity,
not the shared host's frequency, scheduler, thermals or unrelated activity.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--branch', choices=['causal', 'relation', 'bands'], required=True)
    parser.add_argument('--portable', type=Path, required=True)
    parser.add_argument('--native', type=Path, required=True)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--transitions', type=Path)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--rounds', type=int, default=3)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    manifest = {'branch': args.branch, 'cpu': args.cpu, 'rounds': args.rounds,
                'scope': 'Serial frozen replay on a shared host; query-specific comparisons only',
                'corpus_sha256': digest(args.corpus), 'runs': [], 'binaries': {},
                'harness_sha256': digest(Path(__file__))}
    if args.transitions:
        manifest['transitions_sha256'] = digest(args.transitions)
    binaries = {}
    for name, path in [('portable', args.portable), ('native', args.native)]:
        frozen = (args.out / (name + '-frozen')).resolve()
        shutil.copy2(path, frozen)
        binaries[name] = frozen
        manifest['binaries'][name] = {'source_path': str(path), 'sha256': digest(frozen)}
    for pair in range(args.rounds):
        order = ['portable', 'native'] if pair % 2 == 0 else ['native', 'portable']
        for build in order:
            frontends = ['own', 'codec', 'orthodox'] if args.branch == 'relation' else ['none']
            if pair % 2:
                frontends.reverse()
            for frontend in frontends:
                stem = f'{build}-{frontend}-{pair}'
                command = ['taskset', '-c', str(args.cpu), str(binaries[build])]
                if args.branch == 'relation':
                    command += [str(args.corpus.resolve()), '100', f'--frontend={frontend}']
                    if pair % 2:
                        command += ['--reverse']
                else:
                    command += ['--corpus', str(args.corpus.resolve())]
                    if args.branch == 'bands':
                        command += ['--iters', '300', '--all-only']
                    else:
                        assert args.transitions, 'causal requires trajectories'
                        command += ['--transitions', str(args.transitions.resolve()), '--bench-only', '--reps', '20']
                        if pair % 2:
                            command += ['--reverse-methods']
                print(f'{args.branch} {stem}', flush=True)
                started = time.perf_counter()
                with (args.out / (stem + '.jsonl')).open('w') as out, (args.out / (stem + '.stderr')).open('w') as err:
                    subprocess.run(command, stdout=out, stderr=err, check=True)
                manifest['runs'].append({'file': stem + '.jsonl', 'build': build,
                                         'frontend': frontend, 'pair': pair,
                                         'command': command, 'seconds': time.perf_counter() - started,
                                         'stdout_sha256': digest(args.out / (stem + '.jsonl'))})
                (args.out / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
