#!/usr/bin/env python3
"""Compile the abstract Chen certificates and audit every theorem's axioms."""
import argparse, datetime, hashlib, json, pathlib, re, shutil, subprocess


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=pathlib.Path, default=pathlib.Path('proofs/CacheAndBackup.lean'))
    parser.add_argument('--output', type=pathlib.Path, required=True)
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    source = args.source.resolve()
    text = source.read_text()
    # Ignore prose comments when checking for proof admissions/user axioms.
    declarations = re.sub(r'/\-.*?\-/', '', text, flags=re.S)
    assert not re.search(r'\b(sorry|admit|axiom)\b', declarations)
    names = re.findall(r'^theorem\s+(\w+)', text, flags=re.M)
    assert names and len(names) == len(set(names))
    frozen = out / 'CacheAndBackup.lean'
    if source != frozen:
        frozen.write_text(text)
    audit = out / 'axioms-audit.lean'
    audit.write_text(text + '\n' + '\n'.join('#print axioms Chen.' + name for name in names) + '\n')
    commands = []
    def run(phase, argv):
        result = subprocess.run(argv, capture_output=True, text=True, timeout=30, cwd=out)
        (out / (phase + '.stdout')).write_text(result.stdout)
        (out / (phase + '.stderr')).write_text(result.stderr)
        commands.append(dict(phase=phase, argv=argv, exit_code=result.returncode, timeout_seconds=30, cwd=str(out)))
        assert result.returncode == 0, (phase, result.stderr)
        return result.stdout
    run('version', ['lean', '--version'])
    run('lake-version', ['lake', '--version'])
    prefix = pathlib.Path(run('toolchain-prefix', ['lean', '--print-prefix']).strip())
    run('compile', ['lean', '-DwarningAsError=true', '-o', str(out / 'CacheAndBackup.olean'), str(frozen)])
    actual = run('axioms-audit', ['lean', '-DwarningAsError=true', str(audit)])
    assert 'sorryAx' not in actual
    per_theorem = {}
    for name in names:
        line = next(line for line in actual.splitlines() if "'Chen." + name + "'" in line)
        per_theorem[name] = re.findall(r'\[([^]]*)\]', line)[0].split(', ') if 'depends on axioms:' in line else []
    record = dict(status='passed', timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
        proof_source_sha256=sha(frozen), lean_launcher=str(pathlib.Path(shutil.which('lean')).resolve()),
        lean_launcher_sha256=sha(shutil.which('lean')), toolchain_prefix=str(prefix),
        toolchain_lean_sha256=sha(prefix / 'bin/lean'), std_olean_sha256=sha(prefix / 'lib/lean/Std.olean'),
        theorems=names, theorem_count=len(names), commands=commands, warnings_as_errors=True,
        theorem_axioms=per_theorem, trusted_axioms=sorted(set(sum(per_theorem.values(), []))),
        sorryAx_absent_from_actual_print_axioms=True,
        scope='Abstract Lean/Std mechanical certificates; no HIP refinement, global chess optimum, oracle calibration, stochastic PAC or GPU latency optimum')
    (out / 'verification.json').write_text(json.dumps(record, indent=2) + '\n')
    print(json.dumps(dict(status='passed', theorems=len(names), source_sha256=record['proof_source_sha256'], trusted_axioms=record['trusted_axioms'])))


if __name__ == '__main__':
    main()
