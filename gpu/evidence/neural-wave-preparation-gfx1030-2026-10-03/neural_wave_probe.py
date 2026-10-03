#!/usr/bin/env python3
"""Bounded, source-pinned native wave32/64 parity and alternating timing.

Only run after exclusive device reservation. No model, Elo or whole-forward
claim. Inputs stay resident; static input/weight preparation is excluded.
Dynamic rows includes quantization; graph replay removes repeated ctypes calls.
"""
import argparse
import ctypes as C
import hashlib
import json
from pathlib import Path
import random
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from int8_linear_probe import runtime_admission


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked(rc):
    if rc:
        raise RuntimeError(f'HIP call failed: {rc}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--closure', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--repeats', type=int, default=24)
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--shape', help='one M,N,sourceK triple for counter collection')
    parser.add_argument('--widths', default='32,64', help='32,64 by default; one width for separate counter passes')
    args = parser.parse_args()
    if args.output.exists() or not 1 <= args.samples <= 99 or not 1 <= args.repeats <= 1000:
        raise ValueError('invalid bounds or existing evidence')
    closure = args.closure.resolve()
    widths = [int(x) for x in args.widths.split(',')]
    if not widths or len(set(widths)) != len(widths) or any(x not in (32, 64) for x in widths):
        raise ValueError('widths must be distinct32/64 entries')
    manifest = json.loads((closure / 'source-binding.json').read_text())
    for name, sha in manifest['generated'].items():
        assert digest(closure / name) == sha, name
    torch.set_num_threads(1)
    torch.manual_seed(61003)
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=device)
    libs = {}
    pointer, size = C.c_void_p, C.c_uint64
    report = dict(schema=1, argv=sys.argv, probe_sha256=digest(__file__),
                  binding_sha256=digest(closure / 'source-binding.json'),
                  torch=torch.__version__, torch_hip=torch.version.hip,
                  scope='native affine only, resident tensors, weight/input preparation excluded',
                  quantization='zero1_nonzero_max_times_fp32_reciprocal127_clamp_tiny_round_even',
                  mapping='one physical wave per row/output tile; packed N stride equals wave-rounded N',
                  variants={}, rows=[])
    for width in widths:
        paths = [closure / f'libwave{width}-{stem}.so'
                 for stem in ('chen_int8_linear', 'float_linear_reference')]
        integer, floating = [C.CDLL(str(p)) for p in paths]
        integer.chen_int8_pack_weights.argtypes = [pointer] * 2 + [size] * 2 + [pointer]
        integer.chen_int8_linear_packed_residual.argtypes = [pointer] * 7 + [size] * 3 + [pointer]
        integer.chen_int8_linear_dynamic_packed_rows.argtypes = [pointer] * 6 + [size] * 4 + [pointer]
        floating.chen_float_linear.argtypes = [pointer] * 4 + [size] * 4 + [pointer]
        for lib in (integer, floating):
            lib.chen_wave_resources.argtypes = [size, size, C.POINTER(size)]
            lib.chen_wave_resources.restype = C.c_int32
        resources = []
        for label, lib, count in [('integer', integer, 4), ('floating', floating, 2)]:
            for kind in range(count):
                out = (size * 6)()
                checked(lib.chen_wave_resources(kind, 128, out))
                resources.append(dict(library=label, kind=kind, vgpr=out[0], lds=out[1],
                    private_bytes=out[2], max_threads=out[3], active_blocks_per_reported_mp=out[4],
                    source_wave_width=out[5], queried_block_size=128))
        report['variants'][str(width)] = dict(libraries=[dict(path=str(p), sha256=digest(p)) for p in paths],
                                            runtime=runtime_admission(integer), resources=resources)
        libs[width] = integer, floating
    report['device'] = str(torch.cuda.get_device_properties(device))
    shapes = [(1, 1, 4), (5, 31, 18), (5, 33, 20), (5, 63, 22), (5, 65, 45),
              (64, 32, 20), (64, 64, 32), (64, 96, 32), (64, 32, 64),
              (64, 64, 160), (32, 1, 256), (512, 32, 45), (2048, 32, 32),
              (8192, 32, 22), (16384, 32, 32)]
    if args.shape:
        shapes = [tuple(map(int, args.shape.split(',')))]
        if len(shapes[0]) != 3:
            raise ValueError('shape requires M,N,K')
    with torch.cuda.stream(stream):
        for m, n, source_k in shapes:
            k = (source_k + 3) & ~3
            ac = torch.randint(-128, 128, (m, k), dtype=torch.int8)
            wc = torch.randint(-128, 128, (n, k), dtype=torch.int8)
            ac[:, source_k:] = 0
            wc[:, source_k:] = 0
            # Powers of two keep static integer/half references exactly representable.
            sa_cpu = torch.full((m,), 1 / 128, dtype=torch.float32)
            sw_cpu = torch.full((n,), 1 / 128, dtype=torch.float32)
            bias_cpu = torch.arange(n, dtype=torch.float32) / 256
            residual_cpu = torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(31) / 64
            reference = (ac.long() @ wc.long().T).float() * (sa_cpu[:, None] * sw_cpu[None, :]) + bias_cpu
            a, w, sa, sw, bias, residual = [x.to(device) for x in (ac, wc, sa_cpu, sw_cpu, bias_cpu, residual_cpu)]
            source_cpu = ac[:, :source_k].float() / 128
            # Include exact half ties, zero, tiny and subnormal policy fixtures.
            if m >= 5 and source_k >= 7:
                source_cpu[0] = 0
                source_cpu[0, :7] = torch.tensor([127, .5, 1.5, -.5, -1.5, 2.5, -2.5])
                source_cpu[1] = source_cpu[0] * (2 ** -20)
                source_cpu[2] = 0
                source_cpu[3] = 0
                source_cpu[3, 0] = 1e-12
                source_cpu[4] = 0
                source_cpu[4, 0] = 1e-38
            max_cpu = source_cpu.abs().amax(dim=1)
            scale_cpu = torch.where(max_cpu == 0, torch.ones_like(max_cpu), max_cpu * torch.tensor(1 / 127, dtype=torch.float32)).clamp_min(torch.finfo(torch.float32).tiny)
            qcpu = (source_cpu / scale_cpu[:, None]).round().clamp(-127, 127).to(torch.int8)
            qcpu = F.pad(qcpu, (0, k - source_k))
            dynamic_ref = (qcpu.long() @ wc.long().T).float() * (scale_cpu[:, None] * sw_cpu[None, :]) + bias_cpu + residual_cpu
            source = source_cpu.to(device)
            af, wf = a.float() * sa[:, None], w.float() * sw[:, None]
            ah = af.half().contiguous()
            keepalive = [a, w, sa, sw, bias, residual, source, af, wf, ah]
            methods, outputs = {}, {}
            for width, (integer, floating) in libs.items():
                npad = (n + width - 1) & ~(width - 1)
                packed = torch.empty((k // 4, npad), device=device, dtype=torch.int32)
                checked(integer.chen_int8_pack_weights(w.data_ptr(), packed.data_ptr(), n, k, stream.cuda_stream))
                wp = F.pad(wf.T.contiguous(), (0, npad - n)).contiguous()
                wh = F.pad(wf.half().reshape(n, k // 2, 2).permute(1, 0, 2).contiguous().view(torch.int32).squeeze(-1), (0, npad - n)).contiguous()
                y = torch.full((m, n), float('nan'), device=device)
                keepalive += [packed, wp, wh, y]
                calls = {
                    'int8_static_residual': (integer.chen_int8_linear_packed_residual,
                        [a.data_ptr(), packed.data_ptr(), sa.data_ptr(), sw.data_ptr(), bias.data_ptr(), residual.data_ptr(), y.data_ptr(), m, n, k, stream.cuda_stream]),
                    'int8_dynamic_rows_residual': (integer.chen_int8_linear_dynamic_packed_rows,
                        [source.data_ptr(), packed.data_ptr(), sw.data_ptr(), bias.data_ptr(), residual.data_ptr(), y.data_ptr(), m, n, source_k, k, stream.cuda_stream]),
                    'fp16': (floating.chen_float_linear, [ah.data_ptr(), wh.data_ptr(), bias.data_ptr(), y.data_ptr(), m, n, k, 1, stream.cuda_stream]),
                    'fp32': (floating.chen_float_linear, [af.data_ptr(), wp.data_ptr(), bias.data_ptr(), y.data_ptr(), m, n, k, 0, stream.cuda_stream]),
                }
                for label, (fn, call_args) in calls.items():
                    name = f'wave{width}_{label}'
                    methods[name] = lambda fn=fn, call_args=call_args: checked(fn(*call_args))
                    y.fill_(float('nan'))
                    methods[name]()
                    stream.synchronize()
                    actual = y.cpu()
                    expected = dynamic_ref if 'dynamic' in label else (reference + residual_cpu if 'residual' in label else reference)
                    torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
                    outputs[name] = dict(max_error=float((actual - expected).abs().max()), sha256=hashlib.sha256(actual.numpy().tobytes()).hexdigest())
                expected_pack = F.pad(wc.view(torch.int32).T.contiguous(), (0, npad - n))
                assert torch.equal(packed.cpu(), expected_pack)
            row = dict(m=m, n=n, source_k=source_k, kpad=k, parity=outputs, samples=[])
            if not args.parity_only:
                graphs = {}
                for name, fn in methods.items():
                    for _ in range(5):
                        fn()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph, stream=stream):
                        for _ in range(args.repeats):
                            fn()
                    graphs[name] = graph
                stream.synchronize()
                for sample in range(args.samples):
                    order = [(name, mode) for name in methods for mode in ('eager', 'graph')]
                    random.Random(61003 + sample).shuffle(order)
                    for name, mode in order:
                        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                        before = time.perf_counter_ns()
                        start.record(stream)
                        if mode == 'graph':
                            graphs[name].replay()
                        else:
                            for _ in range(args.repeats):
                                methods[name]()
                        end.record(stream)
                        end.synchronize()
                        row['samples'].append(dict(sample=sample, method=name, mode=mode,
                            event_us=start.elapsed_time(end) * 1000 / args.repeats,
                            synchronized_wall_us=(time.perf_counter_ns() - before) / 1000 / args.repeats))
            report['rows'].append(row)
            args.output.with_suffix('.partial.json').write_text(json.dumps(report, indent=2) + '\n')
    report['terminal'] = 'PASS'
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(terminal='PASS', shapes=len(shapes), outputs=sum(m*n for m,n,k in shapes), parity_only=args.parity_only)))


if __name__ == '__main__':
    main()
