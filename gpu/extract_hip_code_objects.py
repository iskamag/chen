#!/usr/bin/env python3
"""Read-only ELF64 .hip_fatbin and concatenated gfx1030 bundle extraction."""
import argparse
import hashlib
import json
from pathlib import Path
import struct


def extract(path, output):
    data = path.read_bytes()
    assert data[:6] == b'\x7fELF\x02\x01', 'requires little-endian ELF64'
    section_offset = struct.unpack_from('<Q', data, 40)[0]
    section_size, section_count, strings_index = struct.unpack_from('<HHH', data, 58)
    assert section_size == 64 and 0 < section_count < 65535 and strings_index < section_count
    assert section_offset + section_count * section_size <= len(data)
    sections = [struct.unpack_from('<IIQQQQIIQQ', data, section_offset + i*section_size) for i in range(section_count)]
    string_section = sections[strings_index]
    strings = data[string_section[4]:string_section[4]+string_section[5]]
    matches = [section for section in sections if strings[section[0]:].split(b'\0',1)[0] == b'.hip_fatbin']
    assert len(matches) == 1
    section = matches[0]
    assert section[4]+section[5] <= len(data)
    fatbin = data[section[4]:section[4]+section[5]]
    output.mkdir(parents=True, exist_ok=False)
    (output/'input.fatbin').write_bytes(fatbin)
    magic = b'__CLANG_OFFLOAD_BUNDLE__'
    cursor = 0
    records = []
    for bundle in range(64):
        start = fatbin.find(magic,cursor)
        if start < 0:
            break
        at = start+len(magic)
        count = struct.unpack_from('<Q',fatbin,at)[0]
        at += 8
        assert 1 <= count <= 16
        descriptors = []
        for _ in range(count):
            offset,size,length = struct.unpack_from('<QQQ',fatbin,at)
            at += 24
            assert length < 1024 and at+length <= len(fatbin)
            target = fatbin[at:at+length].decode()
            at += length
            assert start+offset+size <= len(fatbin)
            descriptors.append((offset,size,target))
        for offset,size,target in descriptors:
            if 'gfx1030' not in target:
                continue
            payload = fatbin[start+offset:start+offset+size]
            assert payload.startswith(b'\x7fELF')
            filename = 'bundle'+str(bundle)+'.gfx1030.elf'
            (output/filename).write_bytes(payload)
            records.append(dict(bundle=bundle,bundle_start=start,relative_offset=offset,size=size,
                target=target,elf=filename,sha256=hashlib.sha256(payload).hexdigest()))
        cursor = max(start+offset+size for offset,size,_ in descriptors)
    else:
        raise ValueError('bundle count exceeded')
    assert records
    result = dict(status='passed',input_path=str(path),input_sha256=hashlib.sha256(data).hexdigest(),
        fatbin_sha256=hashlib.sha256(fatbin).hexdigest(),records=records,input_read_only=True)
    (output/'extraction.json').write_text(json.dumps(result,indent=2)+'\n')
    return result


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('library',type=Path)
    parser.add_argument('output',type=Path)
    args=parser.parse_args()
    result=extract(args.library,args.output)
    print(json.dumps(dict(status=result['status'],input_sha256=result['input_sha256'],bundles=len(result['records']))))
