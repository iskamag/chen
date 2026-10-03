#!/usr/bin/env python3
"""Compile the imported bound model; freeze and audit only new root certificates."""
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
    parser.add_argument('--source', type=pathlib.Path, default=pathlib.Path('proofs/RootCertificates.lean'))
    parser.add_argument('--dependency', type=pathlib.Path, default=pathlib.Path('proofs/SearchBounds.lean'))
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=False)
    modules = out / 'proofs'
    modules.mkdir()
    source_hashes = {}
    for path, name in [(args.dependency, 'SearchBounds'), (args.source, 'RootCertificates')]:
        source = path.read_text()
        declarations = re.sub(r'/\-.*?\-/', '', source, flags=re.S)
        assert not re.search(r'\b(sorry|admit|axiom)\b', declarations), name
        destination = modules / (name + '.lean')
        destination.write_text(source)
        source_hashes[str(destination.relative_to(out))] = sha(destination)
    names = re.findall(r'^theorem\s+(\w+)', args.source.read_text(), flags=re.M)
    assert names and len(names) == len(set(names))
    audit = out / 'axioms-audit.lean'
    audit.write_text('import proofs.RootCertificates\n' + '\n'.join(
        '#print axioms Chen.RootCertificates.' + name for name in names) + '\n')
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
    prefix = pathlib.Path(run('toolchain-prefix', ['lean', '--print-prefix']).strip())
    for module, phase in [('SearchBounds', 'dependency-compile'), ('RootCertificates', 'compile')]:
        run(phase, ['lean', '-DwarningAsError=true', '-o', 'proofs/' + module + '.olean',
                    'proofs/' + module + '.lean'])
    actual = run('axioms-audit', ['lean', '-DwarningAsError=true', 'axioms-audit.lean'])
    per_theorem = {}
    for name in names:
        lines = [line for line in actual.splitlines() if "'Chen.RootCertificates." + name + "'" in line]
        assert len(lines) == 1, (name, lines)
        match = re.search(r'depends on axioms: \[([^]]*)\]', lines[0])
        assert match or 'does not depend on any axioms' in lines[0], lines[0]
        axioms = match[1].split(', ') if match and match[1] else []
        assert set(axioms) <= {'propext', 'Quot.sound'}, (name, axioms)
        per_theorem[name] = axioms
    assert len(actual.splitlines()) == len(names), actual
    record = dict(status='passed', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        source_sha256=source_hashes, verifier_sha256=sha(__file__),
        lean_launcher=str(pathlib.Path(shutil.which('lean')).resolve()),
        lean_launcher_sha256=sha(shutil.which('lean')), toolchain_prefix=str(prefix),
        toolchain_lean_sha256=sha(prefix / 'bin/lean'), std_olean_sha256=sha(prefix / 'lib/lean/Std.olean'),
        commands=commands, warnings_as_errors=True, theorems=names, theorem_count=len(names),
        theorem_axioms=per_theorem, trusted_axioms=sorted(set(sum(per_theorem.values(), []))),
        sorryAx_absent_from_actual_print_axioms=True,
        dependency_recompiled=True, historical_archive_reaudited=False,
        scope='Finite packet and interval information model; worker soundness, complete legal IDs, '
              'identity checking and atomic publication are premises. No HIP refinement, '
              'global chess optimum, minimal-query theorem or GPU latency optimum.')
    (out / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(status='passed', theorem_count=len(names), source_sha256=source_hashes,
                         trusted_axioms=record['trusted_axioms'])))


if __name__ == '__main__':
    main()
