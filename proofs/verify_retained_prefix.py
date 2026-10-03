#!/usr/bin/env python3
"""Freeze/import the existing certificate; audit only new retained-path theorems."""
import argparse
import datetime
import hashlib
import json
import os
import pathlib
import re
import shutil
import subprocess


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=pathlib.Path, default=pathlib.Path('proofs/RetainedPrefix.lean'))
    parser.add_argument('--dependency', type=pathlib.Path, default=pathlib.Path('proofs/CacheAndBackup.lean'))
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # A fresh output protects earlier observations from accidental overwrites.
    assert not (out / 'verification.json').exists(), 'choose a fresh output directory'
    modules = out / 'proofs'
    modules.mkdir(exist_ok=True)
    sources = {}
    for source, name in [(args.dependency, 'CacheAndBackup'), (args.source, 'RetainedPrefix')]:
        text = source.read_text()
        declarations = re.sub(r'/\-.*?\-/', '', text, flags=re.S)
        assert not re.search(r'\b(sorry|admit|axiom)\b', declarations), name
        destination = modules / (name + '.lean')
        if source.resolve() != destination.resolve():
            destination.write_text(text)
        sources[str(destination.relative_to(out))] = sha(destination)
    text = (modules / 'RetainedPrefix.lean').read_text()
    names = re.findall(r'^theorem\s+(\w+)', text, flags=re.M)
    assert names and len(names) == len(set(names))
    audit = out / 'axioms-audit.lean'
    audit.write_text('import proofs.RetainedPrefix\n' + '\n'.join(
        '#print axioms Chen.Retained.' + name for name in names) + '\n')
    environment = dict(os.environ, LEAN_PATH=str(out))
    commands = []

    def run(phase, argv):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=60,
                                cwd=out, env=environment)
        (out / (phase + '.stdout')).write_text(result.stdout)
        (out / (phase + '.stderr')).write_text(result.stderr)
        commands.append(dict(phase=phase, argv=argv, cwd=str(out), exit_code=result.returncode,
                             timeout_seconds=60, environment_overrides={'LEAN_PATH': str(out)}))
        (out / 'commands.json').write_text(json.dumps(commands, indent=2) + '\n')
        assert result.returncode == 0, (phase, result.stdout, result.stderr)
        return result.stdout

    run('version', ['lean', '--version'])
    run('lake-version', ['lake', '--version'])
    prefix = pathlib.Path(run('toolchain-prefix', ['lean', '--print-prefix']).strip())
    for name, phase in [('CacheAndBackup', 'dependency-compile'), ('RetainedPrefix', 'compile')]:
        run(phase, ['lean', '-DwarningAsError=true', '-o', 'proofs/' + name + '.olean',
                    'proofs/' + name + '.lean'])
    actual = run('axioms-audit', ['lean', '-DwarningAsError=true', 'axioms-audit.lean'])
    assert 'sorryAx' not in actual
    per_theorem = {}
    for name in names:
        line = next(line for line in actual.splitlines() if "'Chen.Retained." + name + "'" in line)
        match = re.search(r'depends on axioms: \[([^]]*)\]', line)
        assert match or 'does not depend on any axioms' in line, line
        per_theorem[name] = match[1].split(', ') if match and match[1] else []
    record = dict(status='passed', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=sources, verifier_sha256=sha(__file__),
        lean_launcher=str(pathlib.Path(shutil.which('lean')).resolve()),
        lean_launcher_sha256=sha(shutil.which('lean')), toolchain_prefix=str(prefix),
        toolchain_lean_sha256=sha(prefix / 'bin/lean'), std_olean_sha256=sha(prefix / 'lib/lean/Std.olean'),
        commands=commands, warnings_as_errors=True, theorems=names, theorem_count=len(names),
        theorem_axioms=per_theorem, trusted_axioms=sorted(set(sum(per_theorem.values(), []))),
        sorryAx_absent_from_actual_print_axioms=True, previous_theorems_reaudited=False,
        scope='Finite abstract path/cost certificates; no HIP refinement, bitvector/memory-model proof, '
              'GPU latency optimum, neural batching equivalence across batch sizes, or chess-strength theorem')
    (out / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(status='passed', theorem_count=len(names), source_sha256=sources,
                         trusted_axioms=record['trusted_axioms'])))


if __name__ == '__main__':
    main()
