#!/usr/bin/env python3
"""Cold stdlib-only archive integrity and arithmetic-report audit; no GPU call."""
import argparse
import gzip
import hashlib
import json
import pathlib
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('archive', type=pathlib.Path)
    args = parser.parse_args()
    root = args.archive.resolve(strict=True)
    manifest = json.loads((root / 'manifest.json').read_text())
    payloads = manifest['payload_sha256']
    actual = {str(p.relative_to(root)) for p in root.rglob('*')
              if p.is_file() and p.name != 'manifest.json'}
    assert actual == set(payloads), (actual - set(payloads), set(payloads) - actual)
    for name, expected in payloads.items():
        path = (root / name).resolve(strict=True)
        assert path.is_relative_to(root)
        assert hashlib.sha256(path.read_bytes()).hexdigest() == expected, name
    for name, binding in json.loads((root / 'compressed-payloads.json').read_text()).items():
        raw = gzip.decompress((root / name).read_bytes())
        assert len(raw) == binding['uncompressed_bytes'], name
        assert hashlib.sha256(raw).hexdigest() == binding['uncompressed_sha256'], name
    report = json.loads((root / 'affine.json').read_text())
    assert report['terminal'] == 'PASS'
    assert report['library_sha256'] == manifest['library_sha256']
    assert report['probe_sha256'] == payloads['frozen/int8_linear_probe.py']
    assert report['float_library_sha256'] == manifest['float_library_sha256']
    assert report['quantization_policy'] == manifest['quantization_policy']
    assert report['runtime_admission']['symbol_owner_sha256'] == manifest['loaded_hip_sha256']
    assert report['torch_int_mm']['supported'] is False
    assert report['dynamic_edge_cases']['maximum_error'] == 0
    assert report['dynamic_edge_cases']['subnormal_input'] is True
    assert len(report['rows']) == 16
    assert sum(r['outputs'] for r in report['rows']) == 674337
    medians = json.loads((root / 'medians.json').read_text())
    for row, aggregate in zip(report['rows'], medians, strict=True):
        assert row['kpad'] == (row['source_k'] + 3) & ~3
        assert row['npad'] == (row['n'] + 31) & ~31
        assert row['outputs'] == row['m'] * row['n']
        assert len(row['reference_sha256']) == 64
        methods = {s['method'] for s in row['samples']}
        assert len(methods) == 12
        for method in methods:
            samples = [s for s in row['samples'] if s['method'] == method]
            assert len(samples) == 5 and {s['sample'] for s in samples} == set(range(5))
            for s in samples:
                assert s['event_us'] > 0 and s['synchronized_wall_us'] > 0
            assert aggregate['median_event_us'][method] == statistics.median(s['event_us'] for s in samples)
            assert aggregate['median_wall_us'][method] == statistics.median(s['synchronized_wall_us'] for s in samples)
        for name, error in row['errors'].items():
            assert error >= 0
            if name != 'dynamic_quantization_vs_fp32':
                assert error <= 2e-5, (name, error)
    # Only the previously tested variant is failed; the final same-policy source
    # and shared library never overwrite or relabel that failure.
    assert 'Terminal exit1' in (root / 'failed-strict-scale/failure.txt').read_text()
    assert 'NOT neural symmetric_rows' in (root / 'floor1e8-variant/scope.txt').read_text()
    print(json.dumps(dict(terminal='PASS', payloads=len(payloads),
                         affine_shapes=16, outputs_per_path=674337,
                         samples=sum(len(r['samples']) for r in report['rows']),
                         scope='integrity, source/runtime binding and report aggregation; no new hardware execution')))


if __name__ == '__main__':
    main()
