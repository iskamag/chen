#!/usr/bin/env python3
"""Declared CPU5 Coz run; invoke only after the parent's explicit CPU release."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import signal
import subprocess
import time


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--phase', choices=['pilot', 'scout', 'confirmation'], required=True)
    parser.add_argument('--release', required=True, help='parent release message attribution')
    parser.add_argument('--iterations', type=int, required=True)
    parser.add_argument('--fixed-line')
    args = parser.parse_args()
    assert 1 <= args.iterations <= 1000
    assert args.phase != 'confirmation' or args.fixed_line
    out = pathlib.Path(__file__).resolve().parent
    folder = out / args.phase
    folder.mkdir()
    binding = json.loads((out / 'source-binding.json').read_text())
    binary = out / 'bin/coz_search'
    library = out / 'tools/libcoz.so'
    assert sha(binary) == binding['bindings'][str(out.parent / 'coz-build/causal/examples/coz_search')]['sha256']
    assert sha(library) == binding['bindings']['/usr/lib/libcoz.so']['sha256']
    runtime = {'COZ_PRELOAD': str(library), 'PYTHONDONTWRITEBYTECODE': '1'}
    argv = ['taskset', '-c', '5']
    if args.phase != 'pilot':
        argv += ['python3', str(out / 'tools/coz'), 'run', '--output', str(folder / 'profile.jsonl'),
                 '--source-scope', '%src/core.rs', '--source-scope', '%src/attacks.rs',
                 '--source-scope', '%src/eval.rs']
        if args.fixed_line:
            argv += ['--fixed-line', args.fixed_line]
        # Randomized speedups include the profiler's actual zero-speedup controls.
        argv += ['---']
    argv += [str(binary), str(args.iterations), '200000']
    bound = 20 if args.phase == 'pilot' else 45
    declaration = {'phase': args.phase, 'argv': argv, 'environment_overrides': runtime,
                   'timeout_seconds': bound, 'termination_grace_seconds': 2,
                   'release': args.release, 'iterations': args.iterations, 'nodes_per_search': 200000,
                   'binary_sha256': sha(binary), 'profiler_runtime_sha256': sha(library),
                   'profiler_cli_sha256': sha(out / 'tools/coz'), 'cpu': 5,
                   'fixed_speedup': None, 'fixed_line': args.fixed_line,
                   'timestamp_utc': datetime.datetime.now(datetime.timezone.utc).isoformat()}
    (folder / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')
    started = time.monotonic()
    process = subprocess.Popen(argv, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True, env=dict(os.environ, **runtime), cwd=out)
    timed_out = False
    try:
        stdout, stderr = process.communicate(timeout=bound)
    except subprocess.TimeoutExpired:
        timed_out = True
        os.killpg(process.pid, signal.SIGTERM)
        try:
            stdout, stderr = process.communicate(timeout=2)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGKILL)
            stdout, stderr = process.communicate(timeout=2)
    elapsed = time.monotonic() - started
    (folder / 'stdout.txt').write_bytes(stdout)
    (folder / 'stderr.txt').write_bytes(stderr)
    try:
        os.killpg(process.pid, 0)
        group_live = True
    except ProcessLookupError:
        group_live = False
    if group_live:
        os.killpg(process.pid, signal.SIGKILL)
    expected = f'completed {args.iterations} identical searches, 200000 visited nodes each\n'.encode()
    record = {'phase': args.phase, 'pid': process.pid, 'pgid': process.pid, 'exit_code': process.returncode,
              'elapsed_seconds': elapsed, 'timed_out': timed_out,
              'exact_stationary_assertions_passed': process.returncode == 0 and stdout == expected,
              'process_group_absent_on_retirement': not group_live,
              'binary_hash_unchanged': sha(binary) == declaration['binary_sha256'],
              'runtime_hash_unchanged': sha(library) == declaration['profiler_runtime_sha256'],
              'perf_event_paranoid_after': pathlib.Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip()}
    (folder / 'result.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(record))
    assert record['exact_stationary_assertions_passed'] and not group_live


if __name__ == '__main__':
    main()
