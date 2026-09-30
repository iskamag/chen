#!/usr/bin/env python3
"""Bounded gfx1030 same-checkpoint inference experiment; no strength claim.

CPU packet preparation and integer-score inspection are diagnostic boundaries.
Graph replay is an experimental Python path, not an installed native backend.
"""
import argparse
import ctypes as c
import hashlib
import json
import os
import statistics
import subprocess
import time
from pathlib import Path

import torch
from neural.bridge import RulePosition, position_from_fen
from neural.features import encode_fen
from neural.inference import StaticEvaluator, load_inference
from neural.teacher import load_teacher_checkpoint

FENS = (
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1',
    'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 1 2',
    'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 99 11',
    'k3r3/8/8/3pP3/8/8/8/4K3 w - d6 7 12',
    '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 41 1',
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(row):
    print(json.dumps(row, allow_nan=False), flush=True)


def measure(fn, warmup, samples):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    wall, event = [], []
    for _ in range(samples):
        begin.record()
        started = time.perf_counter_ns()
        output = fn()
        end.record()
        end.synchronize()
        wall.append((time.perf_counter_ns() - started) / 1e6)
        event.append(begin.elapsed_time(end))
    return output, {'wall_ms_median': statistics.median(wall), 'event_ms_median': statistics.median(event),
                    'wall_ms_min': min(wall), 'wall_ms_max': max(wall),
                    'wall_ms_samples': wall, 'event_ms_samples': event}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--library', default='/tmp/chen-native-plugin/libchen_neural.so')
    parser.add_argument('--models', default='/tmp/chen-neural-inference/trained')
    parser.add_argument('--checkpoints', default='/tmp/chen-neural-methods/project/results/neural/search-teacher')
    parser.add_argument('--source-dir', default='/tmp/chen-native-plugin/gpu')
    parser.add_argument('--warmup', type=int, default=5)
    parser.add_argument('--samples', type=int, default=12)
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 8, 32, 128, 256])
    args = parser.parse_args()
    assert args.warmup >= 3 and 3 <= args.samples <= 100
    assert torch.cuda.is_available() and torch.version.hip
    props = torch.cuda.get_device_properties(0)
    assert 'gfx1030' in props.gcnArchName
    torch.set_num_threads(1)
    runtime_path = Path(torch.__file__).parent / 'lib/libamdhip64.so'
    runtime = c.CDLL(str(runtime_path), mode=c.RTLD_GLOBAL)
    lib = c.CDLL(args.library, mode=c.RTLD_GLOBAL)
    lib.chen_neural_create.restype = c.c_void_p
    lib.chen_neural_create.argtypes = [c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64,
                                     c.c_int32, c.c_uint64, c.c_void_p, c.c_uint64]
    lib.chen_neural_destroy.argtypes = [c.c_void_p]
    lib.chen_neural_evaluate.argtypes = [c.c_void_p, c.c_int32, c.c_void_p, c.c_uint64,
                                       c.c_void_p, c.c_void_p, c.c_uint64]
    lib.chen_neural_decode.argtypes = [c.c_void_p, c.c_uint64, c.c_void_p, c.c_void_p]
    lib.chen_neural_runtime_identity.argtypes = [c.c_uint32]
    lib.chen_neural_runtime_identity.restype = c.c_uint64
    for index, name in enumerate(('hipPointerGetAttributes', 'hipMemGetAddressRange',
                                 'hipMalloc', 'hipFree', 'hipStreamSynchronize')):
        assert lib.chen_neural_runtime_identity(index) == c.cast(getattr(runtime, name), c.c_void_p).value
    source_paths = [Path(__file__), Path(args.library), Path(args.library + '.runtime'), runtime_path]
    source_paths += [Path(args.source_dir) / name for name in ('chen_neural.cpp', 'chen_neural.h', 'chen_neural_decode.hip')]
    source_paths += sorted(Path('/tmp/chen-native-plugin/neural').glob('*.py'))
    provenance = {str(path): sha(path) for path in source_paths if path.is_file()}
    cpu = {'load_average': list(os.getloadavg()), 'affinity_cpus': sorted(os.sched_getaffinity(0)),
           'threads': torch.get_num_threads(), 'environment': {key: os.environ.get(key) for key in
           ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'AMD_DIRECT_DISPATCH', 'HIP_VISIBLE_DEVICES', 'LD_PRELOAD')},
           'process_snapshot': subprocess.run(['ps', '-eo', 'pid,ppid,pcpu,stat,args'], capture_output=True,
                                               text=True, check=True).stdout.splitlines()}
    emit({'record': 'inference_perf_provenance', 'torch': str(torch.__version__), 'hip': torch.version.hip,
          'device': props.name, 'arch': props.gcnArchName, 'warmup': args.warmup, 'samples': args.samples,
          'artifacts_sha256': provenance, 'cpu': cpu,
          'event_scope': 'Stream event interval includes host dispatch gaps; not pure GPU active time.',
          'wall_scope': 'Synchronous per-call latency; event recording before call excluded, end event synchronization included.'})
    errors = c.create_string_buffer(2048)
    with torch.inference_mode():
        for name in ('square', 'relation', 'plan'):
            archive = Path(args.models) / (name + '-inference.pt')
            checkpoint = Path(args.checkpoints) / (name + '.pt')
            sidecar = json.loads(Path(str(archive) + '.json').read_text())
            assert sha(archive) == sidecar['archive_sha256'] and sha(checkpoint) == sidecar['manifest']['checkpoint_sha256']
            assert sidecar['manifest']['target_kind'] == 'finite_depth_minimax_cp_utility'
            model, _ = load_teacher_checkpoint(checkpoint, device='cuda:0')
            eager = StaticEvaluator(model.eval()).eval()
            scripted = load_inference(archive, expected_sha256=sidecar['archive_sha256'], device='cuda:0').module
            blob = c.create_string_buffer(archive.read_bytes())
            manifest = sidecar['manifest_json'].encode()
            create_started = time.perf_counter_ns()
            context = lib.chen_neural_create(blob, len(blob)-1, manifest, len(manifest), 0,
                                             sidecar['manifest']['max_batch'], errors, len(errors))
            create_ms = (time.perf_counter_ns() - create_started) / 1e6
            assert context, errors.value.decode()
            emit({'record': 'inference_perf_model', 'name': name, 'archive_sha256': sha(archive),
                  'checkpoint_sha256': sha(checkpoint), 'sidecar_sha256': sha(str(archive)+'.json'),
                  'manifest': sidecar['manifest'], 'native_context_create_wall_ms': create_ms})
            try:
                for batch in args.batches:
                    assert 1 <= batch <= sidecar['manifest']['max_batch']
                    fens = [FENS[i % len(FENS)] for i in range(batch)]
                    packets = (RulePosition * batch)(*(position_from_fen(fen) for fen in fens))
                    raw = c.string_at(c.addressof(packets), c.sizeof(packets))
                    device_packets = torch.tensor(list(raw), dtype=torch.uint8, device='cuda')
                    features = torch.stack([encode_fen(fen) for fen in fens]).cuda()
                    decoded = torch.empty_like(features)
                    scores = torch.empty(batch, dtype=torch.int32, device='cuda')

                    def decode():
                        rc = lib.chen_neural_decode(device_packets.data_ptr(), batch, decoded.data_ptr(),
                                                     torch.cuda.current_stream().cuda_stream)
                        assert rc == 0
                        return decoded

                    def native():
                        rc = lib.chen_neural_evaluate(context, 0, device_packets.data_ptr(), batch,
                                                      scores.data_ptr(), errors, len(errors))
                        assert rc == 0, errors.value.decode()
                        return scores

                    def guarded():
                        decode()
                        output = scripted(decoded)
                        assert output.dtype == torch.int32 and output.shape == (batch,) and output.is_contiguous()
                        assert bool(((output >= -29000) & (output <= 29000)).all().item())
                        scores.copy_(output)
                        torch.cuda.synchronize()
                        return scores

                    expected = eager(features)
                    decode()
                    torch.cuda.synchronize()
                    assert torch.equal(features, decoded)
                    for path, fn in (('eager_model_only', lambda: eager(features)),
                                     ('scripted_model_only', lambda: scripted(features)),
                                     ('scripted_full_guarded', guarded), ('native_callback', native)):
                        first_output, first = measure(fn, 0, 1)
                        assert torch.equal(first_output, expected)
                        output, timing = measure(fn, args.warmup, args.samples)
                        assert torch.equal(output, expected), (name, batch, path, output.cpu().tolist(), expected.cpu().tolist())
                        emit({'record': 'inference_perf_timing', 'model': name, 'batch': batch, 'path': path,
                              'exact_integer_parity': True, 'first_path_invocation_wall_ms': first['wall_ms_median'],
                              'first_path_invocation_event_ms': first['event_ms_median'],
                              'first_call_scope': 'First call in this batch after fixture/eager reference preparation; native B1 is first callback in fresh native context. Process/runtime is shared and not cold.',
                              'positions_per_second': batch * 1000 / timing['wall_ms_median'], **timing})
                    # Fixed-address, exact-count graph includes GPU feature decode,
                    # unchanged frozen model, range predicate and D2D score copy.
                    # The bounded scalar guard and synchronous return stay outside.
                    graph = None
                    try:
                        stream = torch.cuda.Stream()
                        stream.wait_stream(torch.cuda.current_stream())
                        with torch.cuda.stream(stream):
                            for _ in range(args.warmup):
                                decode()
                                graph_output = scripted(decoded)
                                graph_valid = ((graph_output >= -29000) & (graph_output <= 29000)).all()
                                scores.copy_(graph_output)
                        torch.cuda.current_stream().wait_stream(stream)
                        torch.cuda.synchronize()
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph, stream=stream):
                            decode()
                            graph_output = scripted(decoded)
                            graph_valid = ((graph_output >= -29000) & (graph_output <= 29000)).all()
                            scores.copy_(graph_output)

                        def replay():
                            graph.replay()
                            assert bool(graph_valid.item())
                            torch.cuda.synchronize()
                            return scores

                        output, timing = measure(replay, args.warmup, args.samples)
                        assert torch.equal(output, expected), (name, batch, 'graph')
                        # Change every row's actual packet contents at identical
                        # addresses to prove replay is not an output cache.
                        changed = (RulePosition * batch)(*(position_from_fen(FENS[(i+1) % len(FENS)]) for i in range(batch)))
                        changed_raw = c.string_at(c.addressof(changed), c.sizeof(changed))
                        device_packets.copy_(torch.tensor(list(changed_raw), dtype=torch.uint8, device='cuda'))
                        changed_expected = eager(torch.stack([encode_fen(FENS[(i+1) % len(FENS)]) for i in range(batch)]).cuda())
                        assert torch.equal(replay(), changed_expected)
                        changed_score_rows = int((changed_expected != expected).sum().item())
                        emit({'record': 'inference_perf_timing', 'model': name, 'batch': batch, 'path': 'graph_full_guarded',
                              'exact_integer_parity': True, 'changed_packet_parity': True,
                              'changed_score_rows': changed_score_rows,
                              'positions_per_second': batch * 1000 / timing['wall_ms_median'], **timing})
                        device_packets.copy_(torch.tensor(list(raw), dtype=torch.uint8, device='cuda'))
                        torch.cuda.synchronize()
                        paired_native, paired_graph = [], []
                        for index in range(args.samples):
                            # Alternate which path is first in each pair.
                            for paired_path, fn in ((('native', native), ('graph', replay)) if index % 2 == 0
                                                    else (('graph', replay), ('native', native))):
                                output, paired_timing = measure(fn, 0, 1)
                                assert torch.equal(output, expected)
                                (paired_native if paired_path == 'native' else paired_graph).append(paired_timing['wall_ms_median'])
                        emit({'record': 'inference_perf_paired', 'model': name, 'batch': batch,
                              'native_wall_ms_samples': paired_native, 'graph_wall_ms_samples': paired_graph,
                              'native_wall_ms_median': statistics.median(paired_native),
                              'graph_wall_ms_median': statistics.median(paired_graph),
                              'native_over_graph_speedup': statistics.median(paired_native) / statistics.median(paired_graph),
                              'exact_integer_parity': True, 'ordering': 'Alternating native-first/graph-first pairs.'})
                    except Exception as exc:
                        torch.cuda.synchronize()
                        emit({'record': 'inference_perf_graph_failure', 'model': name, 'batch': batch,
                              'error': str(exc), 'fallback': False})
                    finally:
                        del graph
                    emit({'record': 'inference_perf_batch_complete', 'model': name, 'batch': batch})
            finally:
                lib.chen_neural_destroy(context)
            del eager, scripted, model
            torch.cuda.empty_cache()
    assert all(sha(path) == value for path, value in provenance.items())
    emit({'record': 'inference_perf_complete', 'inputs_unchanged': True,
          'load_average_end': list(os.getloadavg()),
          'scope': 'Warm synchronous latency, repeated fixed fixture packets; no strength or search throughput inference.'})


if __name__ == '__main__':
    main()
