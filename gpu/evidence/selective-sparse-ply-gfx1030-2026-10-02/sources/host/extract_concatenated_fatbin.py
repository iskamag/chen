#!/usr/bin/env python3
"""Read-only input extraction of every gfx1030 ELF in concatenated HIP bundles.

The timed .so must first be dumped using llvm-objcopy with an explicit separate
output .so. This utility accepts the dumped .hip_fatbin, not an input .so.
"""
import argparse
import hashlib
import json
from pathlib import Path
import struct

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('fatbin', type=Path)
p.add_argument('output', type=Path)
a = p.parse_args()
data = a.fatbin.read_bytes()
a.output.mkdir(parents=True, exist_ok=False)
magic = b'__CLANG_OFFLOAD_BUNDLE__'
cursor = 0
records = []
for bundle in range(64):
    start = data.find(magic, cursor)
    if start < 0:
        break
    pos = start + len(magic)
    count = struct.unpack_from('<Q', data, pos)[0]
    pos += 8
    assert 1 <= count <= 16
    descriptors = []
    for _ in range(count):
        offset, size, length = struct.unpack_from('<QQQ', data, pos)
        pos += 24
        assert length < 1024 and pos + length <= len(data)
        target = data[pos:pos + length].decode()
        pos += length
        assert start + offset + size <= len(data)
        descriptors.append((offset, size, target))
    for offset, size, target in descriptors:
        if 'gfx1030' not in target:
            continue
        payload = data[start + offset:start + offset + size]
        assert payload.startswith(b'\x7fELF')
        filename = 'bundle' + str(bundle) + '.gfx1030.elf'
        (a.output / filename).write_bytes(payload)
        records.append({'bundle': bundle, 'bundle_start': start, 'relative_offset': offset,
                        'size': size, 'target': target, 'elf': filename,
                        'sha256': hashlib.sha256(payload).hexdigest()})
    cursor = max(start + offset + size for offset, size, _ in descriptors)
else:
    raise RuntimeError('bundle bound exceeded')
assert records
print(json.dumps(records, indent=2))
