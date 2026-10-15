#!/usr/bin/env python3
"""Assemble compact round evidence from hashed audits and serial replay summaries."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    files = {
        'causal_portable': 'causal-cert-independent.json',
        'causal_native': 'causal-cert-native-independent.json',
        'bands_portable': 'bands/cert-final-portable-independent.json',
        'bands_native': 'bands/cert-final-native-independent.json',
        'relation_portable': 'relation/cert-independent.json',
        'relation_native': 'relation/cert-native-independent.json',
        'old_policy_choices': 'quality/cert-oldpolicy-choices.json',
        'depth1_choices': 'quality/cert-depth1-choices.json',
        'auditor_negative_checks': 'auditor-negative/report.json',
    }
    report = {
        'scope': 'CPU-only architecture round; challenge-corpus exactness and query cost, no Elo or general search replacement claim',
        'code_revision': subprocess.check_output(['git', 'rev-parse', 'HEAD'], text=True).strip(),
        'rustc': subprocess.check_output(['rustc', '-Vv'], text=True).strip(),
        'cpu': 'AMD Ryzen 5 3600; serial replay CPU 2; shared host',
        'release_profile': 'thin LTO, codegen-units=1; native adds -C target-cpu=znver2; matched candidate/control flags',
        'corpus': json.loads((args.root / 'corpus/manifest.json').read_text()),
        'audits': {name: json.loads((args.root / path).read_text()) for name, path in files.items()},
        'matrices': {branch: json.loads((args.root / 'root-serial' / branch / 'summary.json').read_text())
                     for branch in ['causal', 'relation', 'bands']},
    }
    source = Path('experiments/radical_round1')
    report['source_sha256'] = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                               for p in sorted(source.rglob('*')) if p.is_file()
                               and not any(part.startswith('target') or part == '__pycache__' for part in p.parts)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(f'assembled {len(report["source_sha256"])} source/evidence hashes, {len(files)} audit records, 30 process runs')


if __name__ == '__main__':
    main()
