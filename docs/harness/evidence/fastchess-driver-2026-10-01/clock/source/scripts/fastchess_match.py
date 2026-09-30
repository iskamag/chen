#!/usr/bin/env python3
"""Run a frozen paired experiment through pinned Fastchess and audit its PGN."""
import argparse
import json
import math
import os
from pathlib import Path
import platform
import shutil
import signal
import subprocess
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.fastchess import audit_pgn, audit_scoreboard, decision, descriptive, validate_sprt
from scripts.fetch_match_tools import PINS, sha
from scripts.match import opening_rows

RUNNER_SHA = '3d0a8f3c8b837a96366ac838f3ddf6fe87b813abb4ad2852284c0ce409132566'
RUNTIME_KEYS = ('CHEN_GPU_LIBRARY', 'CHEN_NEURAL_MODEL', 'CHEN_NEURAL_LIBRARY',
                'CHEN_NEURAL_HIP_RUNTIME', 'CHEN_NEURAL_EXECUTION', 'CHEN_NEURAL_GRAPH_SHAPES',
                'CHEN_NEURAL_GRAPH_BYTES', 'AMD_DIRECT_DISPATCH', 'LD_PRELOAD',
                'HSA_OVERRIDE_GFX_VERSION', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
                'CUDA_VISIBLE_DEVICES', 'PYTORCH_ROCM_ARCH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def verify_runner(directory):
    directory = Path(directory).resolve()
    record = json.loads((directory / 'manifest.json').read_text())['tools']['fastchess']
    pin = PINS['fastchess']
    binary = directory / pin['binary']
    if (record['pin'] != pin or sha(directory / pin['archive']) != pin['sha256']
            or record['binary_sha256'] != RUNNER_SHA or sha(binary) != RUNNER_SHA):
        raise ValueError('runner archive, manifest or binary differs from pinned release')
    return binary, record


def launcher(entry, name, out):
    if not isinstance(entry.get('command'), list) or not entry['command'] or any(
            not isinstance(x, str) or not x or '\0' in x for x in entry['command']):
        raise ValueError('engine command must be a nonempty argv list')
    cwd = Path(entry.get('cwd') or os.getcwd()).resolve()
    command = list(entry['command'])
    # Remote execution needs independent PID/session ownership; a host process
    # group cannot retire a container/SSH engine. Run the entire harness inside
    # its container with local engine executables instead.
    if any(Path(item).name in ('docker', 'podman', 'ssh', 'rocm_run.sh') for item in command):
        raise ValueError('remote engine wrappers require remote retirement; run the whole harness beside the engine')
    overrides = entry.get('env', {})
    if not isinstance(overrides, dict) or any(not isinstance(k, str) or not k or '=' in k or '\0' in k
           or (v is not None and (not isinstance(v, str) or '\0' in v)) for k, v in overrides.items()):
        raise ValueError('environment overrides must be strings or null for removal')
    env = dict(overrides)
    effective = dict(os.environ)
    for key, value in env.items():
        if value is None:
            effective.pop(key, None)
        else:
            effective[key] = value
    search = effective.get('PATH', os.defpath)
    search = os.pathsep.join(str((cwd / part).resolve()) for part in search.split(os.pathsep))
    executable = shutil.which(command[0], path=search) if '/' not in command[0] else str((cwd / command[0]).resolve())
    if executable is None or not Path(executable).is_file():
        raise ValueError('engine launcher does not exist in configured cwd/PATH')
    command[0] = executable
    if entry.get('capture_rocm_artifacts'):
        env['CHEN_ARTIFACT_DIR'] = str(out / 'artifacts' / name)
    path = out / (name + '-launcher.py')
    # argv and environment are literal Python data, never shell interpolated.
    path.write_text('#!' + sys.executable + '\nimport os\n'
                    + 'os.chdir(' + repr(str(cwd)) + ')\n'
                    + 'env = dict(os.environ)\n'
                    + 'overrides = ' + repr(env) + '\n'
                    + 'for key, value in overrides.items():\n'
                    + '    if value is None: env.pop(key, None)\n'
                    + '    else: env[key] = value\n'
                    + 'command = ' + repr(command) + '\n'
                    + 'os.execvpe(command[0], command, env)\n')
    path.chmod(0o755)
    return path, {'command': command, 'cwd': str(cwd), 'env': env,
                  'launcher_sha256': sha(executable), 'adapter_sha256': sha(path),
                  'declared_artifacts': {key: {'path': str(Path(value).resolve()), 'sha256': sha(value)}
                                         for key, value in entry.get('artifacts', {}).items()}}


def retire(process):
    """Always retire the owned local group, including descendants after exit."""
    report = {'scope': 'owned local process group; launchers must keep engines in this group'}
    for signum, label in ((signal.SIGTERM, 'sigterm_sent'), (signal.SIGKILL, 'sigkill_sent')):
        try:
            os.killpg(process.pid, signum)
            report[label] = True
        except ProcessLookupError:
            report[label] = False
        if signum == signal.SIGTERM:
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                pass
    process.wait(timeout=5)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True, help='base/candidate engine config as for match.py')
    parser.add_argument('--tools', default='results/match-tools')
    parser.add_argument('--openings', required=True, help='project name|FEN rows; sequential paired order')
    parser.add_argument('--max-games', required=True, type=int, help='positive even stopping cap')
    budget = parser.add_mutually_exclusive_group(required=True)
    budget.add_argument('--nodes', type=int)
    budget.add_argument('--depth', type=int)
    budget.add_argument('--tc', help='Fastchess clock, e.g. 10+0.1; forwarded verbatim')
    parser.add_argument('--sprt', help='JSON elo0/elo1/alpha/beta/model/sampling_design predeclaration')
    parser.add_argument('--wall-seconds', required=True, type=float, help='whole-run watchdog; timeout fails run')
    parser.add_argument('--seed', type=int, default=20261001)
    parser.add_argument('--out', required=True)
    args = parser.parse_args()
    if args.max_games <= 0 or args.max_games % 2 or args.seed < 0 or not math.isfinite(args.wall_seconds) or args.wall_seconds <= 0:
        parser.error('positive even game cap, nonnegative seed and positive finite wall watchdog required')
    if any(x is not None and x <= 0 for x in (args.nodes, args.depth)):
        parser.error('node/depth budgets must be positive')
    runner, runner_record = verify_runner(args.tools)
    config = json.loads(Path(args.config).read_text())
    if set(config) != {'base', 'candidate'}:
        raise ValueError('exactly base and candidate engine entries required')
    openings = opening_rows(args.openings)
    sprt = validate_sprt(json.loads(Path(args.sprt).read_text())) if args.sprt else None
    # A finite sequential suite repeated within the run is not an independent
    # sample; reject it for statistical mode before any engines are launched.
    if sprt:
        identities = [' '.join(fen.split()[:4]) for _, fen in openings]
        if args.max_games // 2 > len(openings) or len(set(identities)) != len(identities):
            raise ValueError('SPRT mode requires distinct declared openings and no suite cycling')
    out = Path(args.out).resolve()
    out.mkdir(parents=True, exist_ok=False)
    shutil.copyfile(args.config, out / 'input-config.json')
    shutil.copyfile(args.openings, out / 'input-openings.txt')
    if args.sprt:
        shutil.copyfile(args.sprt, out / 'input-sprt.json')
    (out / 'openings.epd').write_text('\n'.join(fen for _, fen in openings) + '\n')
    command = [str(runner), '-concurrency', '1', '-rounds', str(args.max_games // 2), '-games', '2', '-repeat',
               '-srand', str(args.seed), '-report', 'penta=true', '-strict',
               '-openings', 'file=' + str(out / 'openings.epd'), 'format=epd', 'order=sequential',
               '-pgnout', 'file=' + str(out / 'games.pgn'), 'append=false', 'min=false', 'nodes=true',
               '-config', 'outname=' + str(out / 'runner-state.json'),
               '-log', 'file=' + str(out / 'runner.log'), 'append=false', 'engine=true', 'compress=false']
    engine_records = {}
    for name in ('candidate', 'base'):
        entry = config[name]
        path, engine_records[name] = launcher(entry, name, out)
        command.extend(['-engine', 'cmd=' + str(path), 'name=' + name, 'proto=uci'])
        for key, value in entry.get('options', {}).items():
            if not isinstance(key, str) or not key or '=' in key or '\0' in key or '\n' in key:
                raise ValueError('invalid option name')
            encoded = str(value).lower() if isinstance(value, bool) else str(value)
            if '\0' in encoded or '\n' in encoded:
                raise ValueError('invalid option value')
            command.append('option.' + key + '=' + encoded)
    limit = f'nodes={args.nodes}' if args.nodes else f'depth={args.depth}' if args.depth else 'tc=' + args.tc
    command.extend(['-each', limit])
    if sprt:
        command.extend(['-sprt'] + [f'{key}={sprt[key]}' for key in ('elo0', 'elo1', 'alpha', 'beta', 'model')])
    metadata = {'kind': 'pinned_fastchess_paired_experiment', 'argv': command,
                'parameters': vars(args), 'engines': engine_records, 'runner': runner_record,
                'input_sha256': {'config': sha(args.config), 'openings': sha(args.openings)},
                'sprt_declaration': sprt, 'chess_version': __import__('chess').__version__,
                'platform': platform.platform(), 'script_sha256': sha(__file__),
                'audit_sha256': sha(Path(__file__).resolve().parents[1] / 'harness/fastchess.py'),
                'selected_inherited_environment': {k: os.environ[k] for k in RUNTIME_KEYS if k in os.environ},
                'rating_scope': 'Relative offline experiment; no Lichess rating certification',
                'sampling_scope': 'Caller-declared design, not independently proven' if sprt else 'Workload diagnostic; no statistical acceptance'}
    source_root = Path(__file__).resolve().parents[1]
    for relative in ('scripts/fastchess_match.py', 'scripts/fetch_match_tools.py',
                     'scripts/match.py', 'harness/fastchess.py', 'harness/stats.py'):
        destination = out / 'source' / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_root / relative, destination)
    write_json(out / 'metadata.json', metadata)
    start = time.monotonic()
    timed_out = False
    wait_error = None
    with (out / 'stdout.txt').open('w') as stdout, (out / 'stderr.txt').open('w') as stderr:
        process = subprocess.Popen(command, cwd=out, stdout=stdout, stderr=stderr, start_new_session=True)
        try:
            process.wait(timeout=args.wall_seconds)
        except subprocess.TimeoutExpired:
            timed_out = True
        except BaseException as error:
            wait_error = f'{type(error).__name__}: {error}'
        finally:
            retirement = retire(process)
    summary = {'returncode': process.returncode, 'timed_out': timed_out,
               'wall_seconds': time.monotonic() - start, 'decision': 'technical_failure',
               'retirement': retirement, 'wait_error': wait_error}
    try:
        if timed_out or process.returncode or wait_error:
            raise ValueError('runner failed or whole-run watchdog expired')
        rows, pairs = audit_pgn(out / 'games.pgn', openings, args.max_games)
        state = json.loads((out / 'runner-state.json').read_text())
        scoreboard = audit_scoreboard(state, pairs)
        summary.update(descriptive(pairs))
        summary.update({'games': len(rows), 'scoreboard': scoreboard,
                        'decision': decision((out / 'stdout.txt').read_text(), len(rows), args.max_games, sprt, process.returncode)})
        write_json(out / 'replayed-games.json', rows)
    except Exception as error:
        summary['audit_error'] = f'{type(error).__name__}: {error}'
    write_json(out / 'summary.json', summary)
    payloads = {str(path.relative_to(out)): sha(path) for path in sorted(out.rglob('*')) if path.is_file()}
    write_json(out / 'payload-sha256.json', payloads)
    print(json.dumps({'results': str(out), **summary}, indent=2))
    if summary['decision'] in ('technical_failure', 'unexpected_early_stop'):
        raise SystemExit(1)


if __name__ == '__main__':
    main()
