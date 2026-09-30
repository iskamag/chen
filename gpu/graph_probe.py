#!/usr/bin/env python3
"""Native exact-shape graph parity, staging ownership and lifecycle checks."""
import argparse
import ctypes as c
import io
import json
import subprocess
import sys
from pathlib import Path

import torch
from inference_bench import FENS, emit, sha
from neural.bridge import RulePosition, position_from_fen
from neural.features import encode_fen
from neural.inference import load_inference


class SyncEvaluator(torch.nn.Module):
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        value = int(features.sum().item())
        return (features[:, 0, 0, 0] * 0 + value).to(torch.int32)


class Config(c.Structure):
    _fields_ = [('abi_version', c.c_uint32), ('mode', c.c_uint32),
                ('max_cached_shapes', c.c_uint32), ('reserved', c.c_uint32),
                ('retained_budget_bytes', c.c_uint64)]


class Stats(c.Structure):
    _fields_ = [('abi_version', c.c_uint32), ('mode', c.c_uint32),
                ('max_cached_shapes', c.c_uint32), ('first_used', c.c_uint32)] + [
        (name, c.c_uint64) for name in ('cached_shapes', 'hits', 'misses', 'captures', 'evictions',
                                      'retained_pool_bytes', 'staging_logical_bytes', 'peak_observed_pool_bytes')
    ] + [('capture_wall_ms', c.c_double)]


def bind(path):
    runtime = c.CDLL(str(Path(torch.__file__).parent / 'lib/libamdhip64.so'), mode=c.RTLD_GLOBAL)
    lib = c.CDLL(path, mode=c.RTLD_GLOBAL)
    lib.chen_neural_create.restype = c.c_void_p
    lib.chen_neural_create.argtypes = [c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64,
                                     c.c_int32, c.c_uint64, c.c_void_p, c.c_uint64]
    lib.chen_neural_destroy.argtypes = [c.c_void_p]
    lib.chen_neural_evaluate.argtypes = [c.c_void_p, c.c_int32, c.c_void_p, c.c_uint64,
                                       c.c_void_p, c.c_void_p, c.c_uint64]
    lib.chen_neural_set_execution.argtypes = [c.c_void_p, c.POINTER(Config), c.c_void_p, c.c_uint64]
    lib.chen_neural_execution_stats.argtypes = [c.c_void_p, c.POINTER(Stats), c.c_void_p, c.c_uint64]
    lib.chen_neural_test_inject_failure.argtypes = [c.c_void_p, c.c_uint32]
    lib.chen_neural_runtime_identity.argtypes = [c.c_uint32]
    lib.chen_neural_runtime_identity.restype = c.c_uint64
    for index, name in enumerate(('hipPointerGetAttributes', 'hipMemGetAddressRange',
                                 'hipMalloc', 'hipFree', 'hipStreamSynchronize')):
        assert lib.chen_neural_runtime_identity(index) == c.cast(getattr(runtime, name), c.c_void_p).value
    assert lib.chen_neural_execution_abi_version() == 1
    assert c.sizeof(Config) == 24 and c.sizeof(Stats) == 88
    return lib


def packets(batch, offset=0):
    fens = [FENS[(i+offset) % len(FENS)] for i in range(batch)]
    objects = (RulePosition * batch)(*(position_from_fen(fen) for fen in fens))
    raw = c.string_at(c.addressof(objects), c.sizeof(objects))
    return torch.tensor(list(raw), dtype=torch.uint8, device='cuda'), torch.stack([encode_fen(f) for f in fens]).cuda()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('library')
    p.add_argument('--models', default='/tmp/chen-neural-inference/trained')
    p.add_argument('--injections', action='store_true')
    p.add_argument('--fatal-injection', action='store_true')
    p.add_argument('--unsupported-capture', action='store_true')
    args = p.parse_args()
    assert torch.version.hip and torch.cuda.is_available()
    assert 'gfx1030' in torch.cuda.get_device_properties(0).gcnArchName
    torch.set_num_threads(1)
    lib = bind(args.library)
    error = c.create_string_buffer(4096)
    contexts = []
    emit({'record': 'native_graph_provenance', 'library_sha256': sha(args.library),
          'probe_sha256': sha(__file__), 'torch': str(torch.__version__), 'hip': torch.version.hip,
          'injections': args.injections})

    def create(blob, manifest, graph=True, budget=256*1024*1024):
        owner = lib.chen_neural_create(blob, len(blob)-1, manifest, len(manifest), 0, 256, error, len(error))
        assert owner, error.value.decode()
        contexts.append(owner)
        config = Config(1, int(graph), 2 if graph else 0, 0, budget if graph else 0)
        assert lib.chen_neural_set_execution(owner, config, error, len(error)) == 0, error.value.decode()
        return owner

    def stats(owner):
        out = Stats()
        assert lib.chen_neural_execution_stats(owner, out, error, len(error)) == 0, error.value.decode()
        return {name: getattr(out, name) for name, _ in out._fields_}

    def evaluate(owner, packet, batch, output):
        torch.cuda.synchronize()
        return lib.chen_neural_evaluate(owner, 0, packet.data_ptr(), batch, output.data_ptr(), error, len(error))

    def destroy(owner):
        lib.chen_neural_destroy(owner)
        contexts.remove(owner)

    try:
        with torch.inference_mode():
            if args.fatal_injection or args.unsupported_capture:
                archive = Path(args.models) / 'square-inference.pt'
                metadata = json.loads(Path(str(archive)+'.json').read_text())
                blob = c.create_string_buffer(archive.read_bytes());manifest = metadata['manifest_json'].encode()
                packet, features = packets(1)
                output = torch.full((1,), -31000, dtype=torch.int32, device='cuda')
                if args.fatal_injection:
                    owner = create(blob, manifest)
                    assert lib.chen_neural_test_inject_failure(owner, 5) == 0
                    evaluate(owner, packet, 1, output)
                    raise AssertionError('fatal retirement injection unexpectedly returned')
                eager = create(blob, manifest, graph=False)
                unsupported = torch.jit.freeze(torch.jit.script(SyncEvaluator().eval()))
                storage = io.BytesIO()
                torch.jit.save(unsupported, storage, _extra_files={'chen_manifest.json': metadata['manifest_json']})
                owner = create(c.create_string_buffer(storage.getvalue()), manifest)
                assert evaluate(owner, packet, 1, output) != 0
                failure = error.value.decode()
                assert output.item() == -31000
                assert evaluate(eager, packet, 1, output) != 0 and b'process restart required' in error.value
                assert lib.chen_neural_set_execution(eager, Config(1, 0, 0, 0, 0), error, len(error)) != 0
                assert b'process restart required' in error.value
                created = lib.chen_neural_create(blob, len(blob)-1, manifest, len(manifest), 0, 256, error, len(error))
                assert not created and b'process restart required' in error.value
                emit({'record': 'native_graph_actual_capture_failure', 'error': failure,
                      'existing_eager_denied': True, 'new_context_denied': True, 'config_denied': True,
                      'output_unchanged': True, 'policy': 'Proved retirement, process poison; process restart required.'})
                return
            for name in ('square', 'relation', 'plan'):
                archive = Path(args.models) / (name+'-inference.pt')
                metadata = json.loads(Path(str(archive)+'.json').read_text())
                blob = c.create_string_buffer(archive.read_bytes())
                manifest = metadata['manifest_json'].encode()
                reference = load_inference(archive, expected_sha256=metadata['archive_sha256'], device='cuda:0')
                graph, eager = create(blob, manifest), create(blob, manifest, graph=False)
                assert lib.chen_neural_evaluate(graph, 0, None, 0, None, error, len(error)) == 0
                assert stats(graph)['first_used'] == 0
                for batch in (1, 3, 8, 17, 32, 127, 128, 255, 256):
                    packet, features = packets(batch)
                    # A different allocation and input contents on the cache hit.
                    other_packet, other_features = packets(batch, 1)
                    output = torch.full((batch,), -31000, dtype=torch.int32, device='cuda')
                    for current, encoded in ((packet, features), (other_packet, other_features)):
                        expected = reference(encoded)
                        assert evaluate(graph, current, batch, output) == 0, error.value.decode()
                        assert torch.equal(output, expected), (name, batch, 'graph')
                        assert evaluate(eager, current, batch, output) == 0, error.value.decode()
                        assert torch.equal(output, expected), (name, batch, 'eager')
                    emit({'record': 'native_graph_exact_parity', 'model': name, 'batch': batch,
                          'different_borrowed_input_addresses': packet.data_ptr() != other_packet.data_ptr(),
                          'exact_integer_scores': True, 'archive_sha256': sha(archive)})
                observed = stats(graph)
                assert observed['cached_shapes'] == 2 and observed['captures'] == 9
                assert observed['misses'] == 9 and observed['hits'] == 9 and observed['evictions'] == 7
                assert observed['retained_pool_bytes'] <= 256*1024*1024 and observed['staging_logical_bytes'] == 32768
                assert lib.chen_neural_set_execution(graph, Config(1, 0, 0, 0, 0), error, len(error)) != 0
                emit({'record': 'native_graph_cache_lifecycle', 'model': name, 'stats': observed,
                      'configuration_locked_after_use': True})
                destroy(graph);destroy(eager)
            # Budget overflow is an admission error after an observed candidate
            # allocation, never a promise of bounded transient capture memory.
            archive = Path(args.models) / 'square-inference.pt'
            metadata = json.loads(Path(str(archive)+'.json').read_text())
            blob = c.create_string_buffer(archive.read_bytes());manifest = metadata['manifest_json'].encode()
            reference = load_inference(archive, expected_sha256=metadata['archive_sha256'], device='cuda:0')
            packet, features = packets(1)
            output = torch.full((1,), -31000, dtype=torch.int32, device='cuda')
            owner = create(blob, manifest, budget=1)
            assert evaluate(owner, packet, 1, output) != 0
            failure = error.value.decode()
            assert 'retained pool budget' in failure and output.item() == -31000
            observed = stats(owner)
            assert observed['cached_shapes'] == 0 and observed['retained_pool_bytes'] == 0
            assert observed['peak_observed_pool_bytes'] > 1
            emit({'record': 'native_graph_budget_rejection', 'error': failure, 'stats': observed,
                  'output_unchanged': True})
            destroy(owner)
            if args.injections:
                for stage in range(1, 5):
                    owner = create(blob, manifest)
                    assert lib.chen_neural_test_inject_failure(owner, stage) == 0
                    output.fill_(-31000)
                    assert evaluate(owner, packet, 1, output) != 0
                    failure = error.value.decode()
                    assert 'injected graph failure stage' in failure and output.item() == -31000
                    assert stats(owner)['cached_shapes'] == 0 and stats(owner)['retained_pool_bytes'] == 0
                    assert evaluate(owner, packet, 1, output) == 0, error.value.decode()
                    assert torch.equal(output, reference(features)), 'injection recovery parity'
                    emit({'record': 'native_graph_failure_injection', 'stage': stage, 'error': failure,
                          'same_context_recovered': True, 'output_unchanged_on_error': True, 'stats': stats(owner)})
                    destroy(owner)
                fatal = subprocess.run([sys.executable, __file__, args.library, '--models', args.models,
                                        '--fatal-injection'], capture_output=True, text=True, timeout=30)
                assert fatal.returncode == 70 and 'injected unproven completion fence' in fatal.stderr
                emit({'record': 'native_graph_retirement_failstop_child', 'returncode': fatal.returncode,
                      'stdout': fatal.stdout, 'stderr': fatal.stderr, 'storage_reclamation_skipped': True})
            else:
                owner = create(blob, manifest)
                assert lib.chen_neural_test_inject_failure(owner, 1) != 0
                destroy(owner)
            unsupported = subprocess.run([sys.executable, __file__, args.library, '--models', args.models,
                                          '--unsupported-capture'], capture_output=True, text=True, timeout=30)
            assert unsupported.returncode == 0, (unsupported.stdout, unsupported.stderr)
            emit({'record': 'native_graph_unsupported_capture_child', 'returncode': unsupported.returncode,
                  'stdout': unsupported.stdout, 'stderr': unsupported.stderr})
    finally:
        for owner in reversed(contexts):
            lib.chen_neural_destroy(owner)
    emit({'record': 'native_graph_complete', 'exact_model_batch_pairs': 27,
          'scope': 'Actual device rule packets and unchanged trained integer scores; not a strength measurement.'})


if __name__ == '__main__':
    main()
