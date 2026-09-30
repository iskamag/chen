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


def observe_environment(pid):
    selected_keys = ('AMD_DIRECT_DISPATCH', 'CHEN_NEURAL_EXECUTION', 'OMP_NUM_THREADS',
                     'OPENBLAS_NUM_THREADS', 'CHEN_NEURAL_MODEL', 'CHEN_NEURAL_LIBRARY',
                     'CHEN_GPU_LIBRARY', 'LD_PRELOAD')
    for attempt in range(3):
        try:
            raw = Path(f'/proc/{pid}/environ').read_bytes()
            fields = dict(item.decode().split('=', 1) for item in raw.split(b'\0') if b'=' in item)
            if fields.get('AMD_DIRECT_DISPATCH') is not None:
                return {'observation_status': 'available', 'attempts': attempt + 1,
                        'actual_environment': {key: fields.get(key) for key in selected_keys}}
            reason = 'empty environment' if not raw else 'requested flag absent'
        except (OSError, UnicodeError) as exc:
            reason = str(exc)
        if attempt < 2:
            time.sleep(.005)
    return {'observation_status': 'unavailable', 'attempts': 3, 'reason': reason}


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
    reference = probe.reference
    original_environment = reference.environment

    def requested_environment(*arguments, **keywords):
        env = original_environment(*arguments, **keywords)
        assert env.get('AMD_DIRECT_DISPATCH') == expected
        emit({'record': 'dispatch_study_requested_engine_environment', 'trial': args.trial,
              'amd_direct_dispatch': env['AMD_DIRECT_DISPATCH'],
              'model': env['CHEN_NEURAL_MODEL'], 'plugin': env['CHEN_NEURAL_LIBRARY'],
              'rules': env['CHEN_GPU_LIBRARY'], 'execution': env['CHEN_NEURAL_EXECUTION']})
        return env

    class ObservedUCI(reference.UCI):
        def __init__(self, *arguments, **keywords):
            super().__init__(*arguments, **keywords)
            # Base construction has received readyok. Observation is bounded;
            # absent /proc data is reported, never a constructor exception.
            observed = observe_environment(self.process.pid)
            emit({'record': 'dispatch_study_engine_environment', 'trial': args.trial,
                  'pid': self.process.pid, 'observed_after': 'readyok',
                  'requested_amd_direct_dispatch': expected, **observed})
            if observed['observation_status'] == 'available':
                if observed['actual_environment']['AMD_DIRECT_DISPATCH'] != expected:
                    self.close()
                    raise AssertionError(('ready child environment mismatch', observed))

    reference.environment = requested_environment
    reference.UCI = ObservedUCI
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
