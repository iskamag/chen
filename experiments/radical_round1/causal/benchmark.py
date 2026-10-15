#!/usr/bin/env python3
"""Alternating CPU-pinned portable/native replay; no builds in measured interval."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import re
import statistics
import subprocess

p = argparse.ArgumentParser()
p.add_argument('--corpus', required=True)
p.add_argument('--transitions', required=True)
p.add_argument('--out', required=True)
p.add_argument('--cpu', type=int, default=1)
p.add_argument('--reps', type=int, default=20)
p.add_argument('--rounds', type=int, default=4)
a = p.parse_args()
base = Path(__file__).resolve().parent
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
binaries = {name: base / target / 'release/chen-causal-round1'
            for name, target in [('portable', 'target'), ('znver2', 'target-znver2')]}
sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
meta = {'platform': platform.platform(), 'rustc': subprocess.check_output(['rustc', '-Vv'], text=True),
        'lscpu': subprocess.check_output(['lscpu'], text=True), 'cpu': a.cpu,
        'corpus_sha256': sha(a.corpus), 'transitions_sha256': sha(a.transitions),
        'binaries': {k: {'path': str(v), 'sha256': sha(v)} for k, v in binaries.items()},
        'sources': {str(f.relative_to(base)): sha(f) for f in sorted(base.rglob('*'))
                    if f.is_file() and ('target' not in f.relative_to(base).parts[0])
                    and 'evidence' not in f.relative_to(base).parts and f.suffix in ['.rs', '.toml', '.py']}}
for name, binary in binaries.items():
    dis = subprocess.check_output(['objdump', '-d', '-C', str(binary)], text=True)
    match = re.search(r'^\w+ <chen_causal_round1::dense_union>:\n.*?(?=\n\n)', dis, re.M | re.S)
    assert match, 'dense_union symbol missing'
    (out / f'dense-union-{name}.asm').write_text(match.group())
records = []
for round_no in range(a.rounds):
    for name in (['portable', 'znver2'] if round_no % 2 == 0 else ['znver2', 'portable']):
        path = out / f'{name}-{round_no}.jsonl'
        cmd = ['taskset', '-c', str(a.cpu), str(binaries[name]), '--corpus', a.corpus,
               '--transitions', a.transitions, '--bench-only', '--reps', str(a.reps), '--out', str(path)]
        if round_no % 2:
            cmd += ['--reverse-methods']
        subprocess.run(cmd, stdout=subprocess.DEVNULL, check=True)
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        for row in rows:
            row.update(build=name, round=round_no)
            records.append(row)
        print(name, round_no, flush=True)
summary = []
keys = {(r['build'], r['mode'], r.get('family', 'all'), r['method']) for r in records if r['kind'] == 'timing'}
for build, mode, family, method in sorted(keys):
    rows = [r for r in records if r['kind'] == 'timing' and
            (r['build'], r['mode'], r.get('family', 'all'), r['method']) == (build, mode, family, method)]
    field = 'ns_per_root' if mode == 'fresh_ingress' else 'ns_per_child'
    checksums = {r['checksum'] for r in rows}
    assert len(checksums) == 1
    summary.append({'build': build, 'mode': mode, 'family': family, 'method': method,
                    'median_ns': statistics.median(r[field] for r in rows),
                    'runs_ns': [r[field] for r in rows], 'checksum': rows[0]['checksum'],
                    'allocations': [r.get('allocations') for r in rows]})
(out / 'metadata.json').write_text(json.dumps(meta, indent=2) + '\n')
(out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
