#!/usr/bin/env python3
"""Create an isolated prototype from exact committed 4e5782a GPU sources."""
import argparse
import hashlib
import json
import pathlib
import shutil
import subprocess

COMMIT = '4e5782a'


def function(source, name):
    start = source.index('__global__ void ' + name + '(')
    opening = source.index('{', start)
    depth = 1
    end = opening + 1
    while depth:
        depth += (source[end] == '{') - (source[end] == '}')
        end += 1
    return source[start:end]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', required=True, type=pathlib.Path)
    args = parser.parse_args()
    root = pathlib.Path(__file__).resolve().parents[2]
    output = args.output.resolve()
    if output.exists():
        raise RuntimeError('refusing to overwrite a source closure')
    archive = root / 'gpu/evidence/selective-cache-gfx1030-2026-10-02/sources/candidate'
    baseline, candidate = output / 'baseline', output / 'candidate'
    baseline.mkdir(parents=True)
    candidate.mkdir()
    binding = dict(commit=subprocess.check_output(['git', 'rev-parse', COMMIT], cwd=root).decode().strip(),
                   baseline_sha256={}, archival_differences={})
    for original in sorted(archive.iterdir()):
        if not original.is_file():
            continue
        data = subprocess.check_output(['git', 'show', COMMIT + ':gpu/' + original.name], cwd=root)
        (baseline / original.name).write_bytes(data)
        (candidate / original.name).write_bytes(data)
        binding['baseline_sha256'][original.name] = hashlib.sha256(data).hexdigest()
        if original.read_bytes() != data:
            binding['archival_differences'][original.name] = dict(
                archive_sha256=hashlib.sha256(original.read_bytes()).hexdigest(),
                commit_sha256=hashlib.sha256(data).hexdigest())
    rules = (baseline / 'chen_rules.hip').read_text()
    selective = (baseline / 'chen_selective.inc').read_text()
    helpers = []
    for source, name, extra, removed in [
        (rules, 'classify_kernel', 'uint64_t offset',
         '    const uint64_t offset=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;\n'),
        (selective, 'selective_children_classical', 'uint64_t row',
         '    const uint64_t row=uint64_t(blockIdx.x)*blockDim.x+threadIdx.x;\n'),
        (selective, 'selective_choose_classical', None, None),
        (selective, 'selective_complete_classical', None, None),
    ]:
        helper = function(source, name).replace('__global__ void ' + name,
                                                '__device__ void epoch_' + name, 1)
        if extra:
            opening = helper.index('{')
            closing = helper.rfind(')', 0, opening)
            helper = helper[:closing] + ',' + extra + helper[closing:]
        if removed:
            assert helper.count(removed) == 1, name
            helper = helper.replace(removed, '', 1)
        helpers.append(helper)
    rules += '\nnamespace {\n' + '\n'.join(helpers) + '\n}\n#include "device_epoch.inc"\n'
    (candidate / 'chen_rules.hip').write_text(rules)
    for name in ['device_epoch.inc', 'device_epoch.h']:
        shutil.copyfile(root / 'gpu/experiments' / name, candidate / name)
    binding['candidate_sha256'] = {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                                    for p in sorted(candidate.iterdir()) if p.is_file()}
    binding['transform'] = 'original kernels unchanged; append exact device-helper copies with explicit logical row, then experiment include'
    (output / 'source-binding.json').write_text(json.dumps(binding, indent=2) + '\n')
    print(json.dumps(dict(output=str(output), baseline_files=len(binding['baseline_sha256']),
                         archival_differences=binding['archival_differences'])))


if __name__ == '__main__':
    main()
