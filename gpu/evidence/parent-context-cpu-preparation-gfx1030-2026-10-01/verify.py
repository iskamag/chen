#!/usr/bin/env python3
"""CPU-only controls; no engine, Torch, neural module or GPU execution."""
import argparse
import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[3]
PROBE = ROOT / 'gpu/parent_context_probe.py'
QA = ROOT / 'results/parent-context-transport-qa'
QA.mkdir(parents=True, exist_ok=False)
spec = importlib.util.spec_from_file_location('parent_control', PROBE)
module = importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
source = PROBE.read_text()
for node in ast.parse(source).body:
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in module.HELPER_SOURCE_SHA256:
        assert hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest() == module.HELPER_SOURCE_SHA256[node.name]
old = (ROOT / 'gpu/neural_search_probe.py').read_text()
start = '    # UCI option-order behavior is observed separately, with no silent fallback.\n'
old_body = old[old.index(start):old.index('    unchanged = all(')]
new_body = source[source.index(start):source.index('\n\n\nFIXED_ARTIFACTS')].rstrip() + '\n'
assert new_body == old_body and hashlib.sha256(new_body.encode()).hexdigest() == module.GAME_PROTOCOL_SHA256
assert module.digest(ROOT / 'benches/positions.txt') == module.SUITE_SHA256
module.check_cpu_parent()

controls = []
def run_mirror(name, program, seconds=10, expected=0):
    fixture = QA / (name + '.py');fixture.write_text(program)
    journal, stdout, stderr = [QA / (name + suffix) for suffix in ('.journal.jsonl', '.stdout', '.stderr')]
    command = [sys.executable, str(PROBE), 'mirror', '--journal', str(journal),
               '--mirror-stdout', str(stdout), '--mirror-stderr', str(stderr),
               '--mirror-wall-seconds', str(seconds), '--exec', sys.executable, str(fixture)]
    began = time.monotonic()
    completed = subprocess.run(command, cwd=ROOT, capture_output=True, timeout=15)
    elapsed = time.monotonic() - began
    (QA / (name + '.driver.stdout')).write_bytes(completed.stdout)
    (QA / (name + '.driver.stderr')).write_bytes(completed.stderr)
    records = [json.loads(line) for line in journal.read_text().splitlines()]
    assert completed.returncode == expected, (name, completed.stderr)
    controls.append({'name': name, 'returncode': completed.returncode, 'elapsed_seconds': elapsed})
    return records, stdout, stderr

records, out, err = run_mirror('success', 'import os\nos.write(1,b"stdout\\npartial")\nos.write(2,b"stderr\\n")\n')
assert out.read_bytes() == b'stdout\npartial' and err.read_bytes() == b'stderr\n'
assert any(r['record'] == 'parent_context_command_complete' for r in records)

records, _, _ = run_mirror('wall', 'import signal,time\nsignal.signal(signal.SIGTERM,signal.SIG_IGN)\ntime.sleep(60)\n', .15, 1)
failed = next(r for r in records if r['record'] == 'parent_context_mirror_transport_failed')
assert 'wall deadline' in failed['exception'] and failed['local_transport_reaped']
assert failed['remote_retirement'] == 'unknown' and failed['study_end_required'] and not failed['automatic_replay']
assert controls[-1]['elapsed_seconds'] < 8

records, out, _ = run_mirror('inherited_pipe', '''import json,os,signal,time
pid=os.fork()
if pid:
 print(json.dumps({'descendant_pid':pid}),flush=True)
 os._exit(0)
signal.signal(signal.SIGTERM,signal.SIG_IGN)
time.sleep(60)
''', 10, 1)
failed = next(r for r in records if r['record'] == 'parent_context_mirror_transport_failed')
assert 'pipes still inherited' in failed['exception'] and failed['local_transport_reaped']
assert failed['remote_retirement'] == 'unknown' and failed['study_end_required'] and not failed['automatic_replay']
assert controls[-1]['elapsed_seconds'] < 4
pid = json.loads(out.read_text())['descendant_pid']
for _ in range(100):
    try:
        state = Path(f'/proc/{pid}/stat').read_text().split(') ',1)[1].split()[0]
    except FileNotFoundError:
        state = 'absent'
    if state in ('Z', 'absent'):
        break
    time.sleep(.01)
assert state in ('Z', 'absent'), (pid, state)
controls[-1].update(descendant_pid=pid, descendant_final_state=state)

records, _, _ = run_mirror('nonzero_exit', 'raise SystemExit(7)\n', expected=1)
assert any(r['record'] == 'parent_context_mirror_end' and r['returncode'] == 7 for r in records)
assert not any(r['record'] == 'parent_context_mirror_transport_failed' for r in records)

fixture = QA / 'toy-uci'
fixture.write_text('''#!/usr/bin/env python3
import sys
for line in sys.stdin:
 if line.strip() == 'uci': print('uciok',flush=True)
 elif line.strip() == 'isready': print('readyok',flush=True)
 elif line.strip() == 'quit': break
''');fixture.chmod(0o755)
args = argparse.Namespace(journal=str(QA / 'owned-child.jsonl'), gdb='/absent-CPU-fixture-only-debugger')
with Path(args.journal).open('x') as stream:
    module.JOURNAL=stream;module.install_owned_uci(args)
    env=os.environ.copy();env.update(AMD_DIRECT_DISPATCH='1', CHEN_NEURAL_EXECUTION='eager');env.pop('LD_PRELOAD', None)
    client=module.UCI(str(fixture),env,'neural');client.send('go depth 2')
    try:
        client.wait('bestmove ', timeout=.02)
    except AssertionError:
        pass
    else:
        raise AssertionError('timeout fixture did not time out')
    client.close()
    assert client.process.poll() is not None and not module.OWNED_CHILDREN
    module.emit({'record':'CPU_owned_control_pass','source_sha256':module.digest(PROBE)})
module.JOURNAL=None
owned=[json.loads(line) for line in Path(args.journal).read_text().splitlines()]
assert sum(r['record']=='parent_context_timeout_snapshot' for r in owned)==2
assert next(r for r in owned if r['record']=='parent_context_UCI_ready')['child_environment']['AMD_DIRECT_DISPATCH']=='1'
summary={'scope':'CPU-only source, transport and ownership controls; GPU producer/consumers unexecuted',
         'probe_sha256':module.digest(PROBE),'exact_reference_helpers_and_game_protocol':True,
         'CPU_parent_guard':module.check_cpu_parent(),'controls':controls,
         'owned_child_reaped':True,'owned_children_remaining':len(module.OWNED_CHILDREN),
         'test_only_UCI_timeout_seconds':.02,'declared_UCI_timeout_seconds':30}
(QA / 'verification.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(summary,sort_keys=True))
