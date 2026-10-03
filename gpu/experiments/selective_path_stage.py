#!/usr/bin/env python3
"""Stage a retained-prefix selector behind the existing classical ABI."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--baseline', default='5a2c82b', help='frozen root-restart baseline commit')
    args = p.parse_args()
    root = Path(__file__).resolve().parents[2]
    commit = subprocess.check_output(['git', 'rev-parse', args.baseline], cwd=root).decode().strip()
    # Follow actual quoted includes, rather than copy unrelated benchmarks.
    sources, pending = {}, ['build.sh', 'chen_gpu.hip', 'chen_rules.hip']
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        if Path(name).name != name:
            raise ValueError('runtime closure requires local GPU includes')
        content = subprocess.check_output(['git', 'show', commit + ':gpu/' + name], cwd=root)
        sources[name] = content
        pending.extend(re.findall(r'^\s*#include "([^"]+)"', content.decode(), re.MULTILINE))
    args.output.mkdir(parents=True, exist_ok=False)
    baseline, candidate = args.output / 'baseline', args.output / 'candidate'
    baseline.mkdir();candidate.mkdir()
    binding = dict(commit=commit, baseline_sha256={})
    for name, content in sorted(sources.items()):
        (baseline / name).write_bytes(content);(candidate / name).write_bytes(content)
        binding['baseline_sha256'][name] = hashlib.sha256(content).hexdigest()
    source = (candidate / 'chen_selective.inc').read_text()
    needle = 'template<bool Cached> __device__ void selective_choose_device('
    assert source.count(needle) == 1
    source = source.replace(needle, needle.replace('selective_choose_device', 'selective_choose_device_reference'), 1)
    marker = '__global__ void selective_choose('
    assert source.count(marker) == 1
    source = source.replace(marker, '#include "selective_retained_path.inc"\n' + marker, 1)
    (candidate / 'chen_selective.inc').write_text(source)
    (candidate / 'selective_retained_path.inc').write_bytes(Path(__file__).with_name('selective_retained_path.inc').read_bytes())
    binding['candidate_sha256'] = {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in candidate.iterdir()}
    binding['scope'] = 'only classical choose replaced; external uncached selector delegates to exact reference'
    (args.output / 'source-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    print(json.dumps(dict(output=str(args.output), baseline_files=len(sources))))


if __name__ == '__main__':
    main()
