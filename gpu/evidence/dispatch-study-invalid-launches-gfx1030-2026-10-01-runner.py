#!/usr/bin/env python3
"""One predeclared dispatch trial; delegate unchanged protocol to frozen probes."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(record):
    print(json.dumps(record), flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True)
    p.add_argument('--trial', type=int, required=True)
    args = p.parse_args()
    config = json.loads(Path(args.config).read_text())
    trial = config['schedule'][args.trial - 1]
    assert trial['trial'] == args.trial
    expected = str(trial['amd_direct_dispatch'])
    assert os.environ.get('AMD_DIRECT_DISPATCH') == expected
    assert not os.environ.get('LD_PRELOAD')
    for path, digest in config['artifact_sha256'].items():
        assert sha(path) == digest, path
    for name, digest in config['model_sha256'].items():
        assert sha(Path(config['fixed']['models']) / name) == digest, name
    began = time.monotonic()
    emit({'record': 'dispatch_study_trial_begin', 'trial': trial,
          'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
          'config_sha256': sha(args.config), 'runner_sha256': sha(__file__),
          'parent_pid': os.getpid(), 'requested_amd_direct_dispatch': expected,
          'parent_environment': {key: os.environ.get(key) for key in
                                 ('AMD_DIRECT_DISPATCH', 'LD_PRELOAD', 'CHEN_NEURAL_EXECUTION')},
          'load_average': os.getloadavg(),
          'runtime_flag_scope': 'Observed process environment, not an internal HIP flag-variable read.'})
    # Import Torch only after validating the flag supplied at process launch.
    import eager_game_probe as probe
    original = subprocess.Popen

    class ObservedProcess(original):
        def __init__(self, command, *arguments, **keywords):
            super().__init__(command, *arguments, **keywords)
            env = keywords.get('env') or os.environ
            actual_engine = config['fixed']['engine']
            executable = command[0] if isinstance(command, (list, tuple)) else None
            if executable != actual_engine and env.get('CHEN_DEBUG_ENGINE') != actual_engine:
                return
            # The child is already exec'd when Popen returns. Observe only the
            # selected public configuration fields, never unrelated environment.
            try:
                fields = Path(f'/proc/{self.pid}/environ').read_bytes().split(b'\0')
                fields = dict(item.decode().split('=', 1) for item in fields if b'=' in item)
                selected = {key: fields.get(key) for key in
                            ('AMD_DIRECT_DISPATCH', 'CHEN_NEURAL_EXECUTION', 'OMP_NUM_THREADS',
                             'OPENBLAS_NUM_THREADS', 'CHEN_NEURAL_MODEL', 'CHEN_NEURAL_LIBRARY',
                             'CHEN_GPU_LIBRARY', 'LD_PRELOAD')}
                observed = {'actual_environment': selected}
            except OSError as exc:
                observed = {'actual_environment_error': str(exc)}
            emit({'record': 'dispatch_study_engine_environment', 'trial': args.trial,
                  'pid': self.pid, 'command': list(command),
                  'requested_amd_direct_dispatch': env.get('AMD_DIRECT_DISPATCH'), **observed})
            assert env.get('AMD_DIRECT_DISPATCH') == expected
            if 'actual_environment' in observed:
                assert observed['actual_environment']['AMD_DIRECT_DISPATCH'] == expected

    subprocess.Popen = ObservedProcess
    fixed = config['fixed']
    sys.argv = [probe.__file__, '--case', 'full', '--engine', fixed['engine'],
                '--plugin', fixed['plugin'], '--rules', fixed['rules'], '--models', fixed['models'],
                '--output-dir', f'/tmp/chen-dispatch-study/trial{args.trial:02d}']
    try:
        probe.main()
    except BaseException as exc:
        emit({'record': 'dispatch_study_trial_end', 'trial': args.trial, 'status': 'failed',
              'exception': repr(exc), 'elapsed_wall_seconds': time.monotonic() - began,
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})
        raise
    else:
        emit({'record': 'dispatch_study_trial_end', 'trial': args.trial, 'status': 'passed',
              'elapsed_wall_seconds': time.monotonic() - began,
              'utc': datetime.datetime.now(datetime.timezone.utc).isoformat()})


if __name__ == '__main__':
    main()
