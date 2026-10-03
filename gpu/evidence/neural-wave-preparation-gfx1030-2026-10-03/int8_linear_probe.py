#!/usr/bin/env python3
"""Standalone real gfx1030 affine probe; no model, training or Elo claims.

Run inside cpp-rocm after reserving the GPU. Libraries, source hashes, timing
samples and independent CPU integer outputs are bound into the JSON report.
"""
import argparse
import ctypes as C
import hashlib
import json
import pathlib
import random
import time

import torch
import torch.nn.functional as F


def digest(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def runtime_admission(library):
    """Resolve the HIP provider actually used by this ctypes/Torch process."""
    function = library.hipRuntimeGetVersion
    function.argtypes = [C.POINTER(C.c_int)]
    function.restype = C.c_int
    version = C.c_int()
    if function(C.byref(version)):
        raise RuntimeError('HIP runtime version admission failed')

    class DlInfo(C.Structure):
        _fields_ = [('filename', C.c_char_p), ('base', C.c_void_p),
                    ('symbol', C.c_char_p), ('address', C.c_void_p)]
    dl = C.CDLL(None)
    dl.dladdr.argtypes = [C.c_void_p, C.POINTER(DlInfo)]
    dl.dladdr.restype = C.c_int
    owner = DlInfo()
    if not dl.dladdr(C.cast(function, C.c_void_p), C.byref(owner)):
        raise RuntimeError('HIP runtime symbol ownership admission failed')
    path = pathlib.Path(owner.filename.decode()).resolve(strict=True)
    mappings = sorted({line.split(maxsplit=5)[5].strip()
                       for line in pathlib.Path('/proc/self/maps').read_text().splitlines()
                       if 'libamdhip64' in line and len(line.split(maxsplit=5)) == 6})
    return dict(hip_runtime_get_version=version.value,
                symbol_owner_path=str(path), symbol_owner_sha256=digest(path),
                mapped_hip_libraries=[dict(path=p, sha256=digest(p)) for p in mappings])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--float-library', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--samples', type=int, default=9)
    parser.add_argument('--repeats', type=int, default=40)
    parser.add_argument('--shape-indices', help='comma separated indices; default all')
    args = parser.parse_args()
    output = pathlib.Path(args.output)
    if output.exists():
        raise RuntimeError('refusing to overwrite evidence')
    if not 1 <= args.samples <= 99 or not 1 <= args.repeats <= 1000:
        raise RuntimeError('invalid bounded benchmark size')
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    lib, floats = C.CDLL(args.library), C.CDLL(args.float_library)
    pointer, size = C.c_void_p, C.c_uint64
    for name in ('chen_int8_linear', 'chen_int8_linear_packed'):
        fn = getattr(lib, name)
        fn.argtypes = [pointer] * 6 + [size] * 3 + [pointer]
        fn.restype = C.c_int32
    lib.chen_int8_linear_packed_residual.argtypes = [pointer] * 7 + [size] * 3 + [pointer]
    lib.chen_int8_linear_packed_residual.restype = C.c_int32
    lib.chen_int8_pack_weights.argtypes = [pointer, pointer, size, size, pointer]
    lib.chen_int8_pack_weights.restype = C.c_int32
    lib.chen_int8_linear_dynamic_packed.argtypes = [pointer] * 6 + [size] * 3 + [pointer]
    lib.chen_int8_linear_dynamic_packed.restype = C.c_int32
    lib.chen_int8_linear_dynamic_packed_rows.argtypes = [pointer] * 6 + [size] * 4 + [pointer]
    lib.chen_int8_linear_dynamic_packed_rows.restype = C.c_int32
    floats.chen_float_linear.argtypes = [pointer] * 4 + [size] * 4 + [pointer]
    floats.chen_float_linear.restype = C.c_int32
    # These rejected requests do not dereference pointers or launch kernels.
    invalid = []
    for m, n, k in [(0, 1, 4), (0, 0, 4), (1, 32, 3), (1048577, 1, 4), (1, 4097, 4)]:
        rc = lib.chen_int8_linear(*([None] * 6), m, n, k, None)
        expected = 0 if (m, n, k) == (0, 1, 4) else 1
        assert rc == expected, (m, n, k, rc)
        invalid.append(dict(m=m, n=n, k=k, rc=rc))
    torch.manual_seed(26061002)
    device = torch.device('cuda:0')
    stream = torch.cuda.Stream(device=device)
    report = dict(schema=1, torch=torch.__version__, hip=torch.version.hip,
                  device=torch.cuda.get_device_name(device),
                  library_sha256=digest(args.library),
                  runtime_admission=runtime_admission(lib),
                  float_library_sha256=digest(args.float_library),
                  probe_sha256=digest(__file__), invalid_abi=invalid,
                  precision='integer exact dot; FP32 scales/output; FP16 FDOT2 with FP32 accumulation',
                  quantization_policy='zero_scale1_nonzero_max_times_fp32_reciprocal127_clamp_fp32tiny_round_even_symmetric127',
                  timings_scope='resident tensors; one-time weight/input packing excluded; dynamic variant includes GPU row absmax, quantization and padding; Python/ctypes launch gaps included',
                  rows=[])
    properties = torch.cuda.get_device_properties(device)
    report['torch_device_properties'] = dict(representation=str(properties),
        name=properties.name, total_memory=properties.total_memory,
        multi_processor_count=properties.multi_processor_count,
        major=properties.major, minor=properties.minor)
    shapes = [(1, 1, 4), (4, 31, 20), (5, 33, 48), (3, 63, 160), (2, 1, 4096),
              (64, 32, 20), (64, 64, 32), (64, 32, 64), (64, 96, 32),
              (64, 64, 160), (64, 1, 256), (512, 32, 22), (512, 32, 45),
              (512, 64, 96), (2048, 32, 32), (16384, 32, 32)]
    if args.shape_indices:
        shapes = [shapes[int(i)] for i in args.shape_indices.split(',')]

    def checked(rc):
        if rc:
            raise RuntimeError(f'HIP launch returned {rc}')

    with torch.cuda.stream(stream):
        report['stream'] = 'owned nondefault stream, no global/default-stream operations'
        for index, (m, n, actual_k) in enumerate(shapes):
            k = (actual_k + 3) & ~3
            npad = (n + 31) & ~31
            ac = torch.randint(-128, 128, (m, k), dtype=torch.int8)
            wc = torch.randint(-128, 128, (n, k), dtype=torch.int8)
            ac[0, :actual_k] = 127
            if m > 1:
                ac[1, :actual_k] = -128
            if m > 3:
                ac[3] = 0
            ac[:, actual_k:] = 0
            wc[:, actual_k:] = 0
            sa_cpu = torch.tensor([2.0 ** (-7 - r % 3) for r in range(m)], dtype=torch.float32)
            sw_cpu = torch.tensor([2.0 ** (-7 - c % 3) for c in range(n)], dtype=torch.float32)
            bias_cpu = torch.arange(n, dtype=torch.float32) / 256
            residual_cpu = torch.arange(m * n, dtype=torch.float32).reshape(m, n).remainder(31) / 64
            # Exact independent CPU int64 dot, then explicit FP32 dequantization.
            reference = (ac.to(torch.int64) @ wc.to(torch.int64).T).float()
            reference *= sa_cpu[:, None] * sw_cpu[None, :]
            reference += bias_cpu
            a, w, sa, sw, bias, residual = [t.to(device) for t in (ac, wc, sa_cpu, sw_cpu, bias_cpu, residual_cpu)]
            packed = torch.empty((k // 4, npad), device=device, dtype=torch.int32)
            y = torch.empty((m, n), device=device, dtype=torch.float32)
            checked(lib.chen_int8_pack_weights(w.data_ptr(), packed.data_ptr(), n, k, stream.cuda_stream))
            af = a.float() * sa[:, None]
            wf = w.float() * sw[:, None]
            # Coalesced [K,Npad] FP32 and packed-half2 [K/2,Npad] weights.
            wp = F.pad(wf.T.contiguous(), (0, npad - n)).contiguous()
            ah = af.half().contiguous()
            wh = F.pad(wf.half().reshape(n, k // 2, 2).permute(1, 0, 2).contiguous().view(torch.int32).squeeze(-1), (0, npad - n)).contiguous()

            def launch_int(packed_mode=True, res=False, aa=a, scales=sa):
                pointers = [aa.data_ptr(), (packed if packed_mode else w).data_ptr(), scales.data_ptr(), sw.data_ptr(), bias.data_ptr()]
                if res:
                    pointers.append(residual.data_ptr())
                pointers.append(y.data_ptr())
                fn = lib.chen_int8_linear_packed_residual if res else (lib.chen_int8_linear_packed if packed_mode else lib.chen_int8_linear)
                checked(fn(*pointers, m, n, k, stream.cuda_stream))

            errors = {}
            for name, fn, expected in [
                ('int8_rowmajor', lambda: launch_int(False), reference),
                ('int8_packed', launch_int, reference),
                ('int8_residual', lambda: launch_int(res=True), reference + residual_cpu),
                ('fp32_control', lambda: checked(floats.chen_float_linear(af.data_ptr(), wp.data_ptr(), bias.data_ptr(), y.data_ptr(), m, n, k, 0, stream.cuda_stream)), reference),
                ('fp16_control', lambda: checked(floats.chen_float_linear(ah.data_ptr(), wh.data_ptr(), bias.data_ptr(), y.data_ptr(), m, n, k, 1, stream.cuda_stream)), reference),
            ]:
                fn()
                stream.synchronize()
                actual = y.cpu()
                torch.testing.assert_close(actual, expected, rtol=2e-6, atol=2e-5)
                errors[name] = float((actual - expected).abs().max())
            # Packed weight layout independently reconstructed on CPU, including
            # unsigned bit patterns and channel padding.
            expected_pack = F.pad(wc.view(torch.int32).T.contiguous(), (0, npad - n)).contiguous()
            assert torch.equal(packed.cpu(), expected_pack)
            # Activation quantization remains GPU work in the measured closure.
            dynamic_keepalive = []
            def dynamic():
                maximum = af.abs().amax(dim=1)
                scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum / 127).clamp_min(torch.finfo(torch.float32).tiny)
                aq = (af / scale[:, None]).round().clamp(-127, 127).to(torch.int8).contiguous()
                launch_int(aa=aq, scales=scale)
                dynamic_keepalive[:] = [aq, scale]

            dynamic()
            stream.synchronize()
            aq_cpu, scale_cpu = [t.cpu() for t in dynamic_keepalive]
            dynamic_reference = (aq_cpu.long() @ wc.long().T).float() * (scale_cpu[:, None] * sw_cpu[None, :]) + bias_cpu
            torch.testing.assert_close(y.cpu(), dynamic_reference, rtol=2e-6, atol=2e-5)
            errors['dynamic_quantization_vs_fp32'] = float((y.cpu() - reference).abs().max())
            def dynamic_fused(res=False):
                checked(lib.chen_int8_linear_dynamic_packed(af.data_ptr(), packed.data_ptr(), sw.data_ptr(), bias.data_ptr(), residual.data_ptr() if res else None, y.data_ptr(), m, n, k, stream.cuda_stream))

            for name, use_residual in [('int8_dynamic_fused', False), ('int8_dynamic_fused_residual', True)]:
                dynamic_fused(use_residual)
                stream.synchronize()
                expected = dynamic_reference + residual_cpu if use_residual else dynamic_reference
                torch.testing.assert_close(y.cpu(), expected, rtol=2e-6, atol=2e-5)
                errors[name] = float((y.cpu() - expected).abs().max())
            unpadded = af[:, :actual_k].contiguous()
            def dynamic_rows(res=False):
                checked(lib.chen_int8_linear_dynamic_packed_rows(unpadded.data_ptr(), packed.data_ptr(), sw.data_ptr(), bias.data_ptr(), residual.data_ptr() if res else None, y.data_ptr(), m, n, actual_k, k, stream.cuda_stream))
            for name, use_residual in [('int8_dynamic_rows', False), ('int8_dynamic_rows_residual', True)]:
                dynamic_rows(use_residual)
                stream.synchronize()
                expected = dynamic_reference + residual_cpu if use_residual else dynamic_reference
                torch.testing.assert_close(y.cpu(), expected, rtol=2e-6, atol=2e-5)
                errors[name] = float((y.cpu() - expected).abs().max())
            methods = {
                'int8_rowmajor': lambda: launch_int(False), 'int8_packed': launch_int,
                'int8_packed_residual': lambda: launch_int(res=True),
                'int8_dynamic': dynamic,
                'int8_dynamic_fused': dynamic_fused,
                'int8_dynamic_fused_residual': lambda: dynamic_fused(True),
                'int8_dynamic_rows': dynamic_rows,
                'int8_dynamic_rows_residual': lambda: dynamic_rows(True),
                'fp32_control': lambda: checked(floats.chen_float_linear(af.data_ptr(), wp.data_ptr(), bias.data_ptr(), y.data_ptr(), m, n, k, 0, stream.cuda_stream)),
                'fp16_control': lambda: checked(floats.chen_float_linear(ah.data_ptr(), wh.data_ptr(), bias.data_ptr(), y.data_ptr(), m, n, k, 1, stream.cuda_stream)),
                'torch_fp32': lambda: F.linear(af, wf, bias),
                'torch_fp16': lambda: F.linear(ah, wf.half(), None).float() + bias,
            }
            # Cache static half weight outside forward timing.
            wh_torch = wf.half()
            methods['torch_fp16'] = lambda: F.linear(ah, wh_torch, None).float() + bias
            samples = []
            for fn in methods.values():
                for _ in range(5):
                    fn()
            stream.synchronize()
            for sample in range(args.samples):
                order = list(methods)
                random.Random(61002 + sample).shuffle(order)
                for name in order:
                    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    before = time.perf_counter_ns()
                    start.record(stream)
                    for _ in range(args.repeats):
                        methods[name]()
                    end.record(stream)
                    end.synchronize()
                    wall = (time.perf_counter_ns() - before) / 1000 / args.repeats
                    samples.append(dict(sample=sample, method=name, event_us=start.elapsed_time(end) * 1000 / args.repeats, synchronized_wall_us=wall))
            report['rows'].append(dict(m=m, n=n, source_k=actual_k, kpad=k, npad=npad,
                                       outputs=m * n, errors=errors,
                                       reference_sha256=hashlib.sha256(reference.numpy().tobytes()).hexdigest(), samples=samples))
            # Save progress on failure without making a partial report terminal.
            output.with_suffix('.partial.json').write_text(json.dumps(report, indent=2) + '\n')
        # Explicit ties-to-even, zero row and tiny scales are separate from
        # static integer input tests. Finite inputs only are
        # the operator contract; no NaN/Inf arithmetic behavior is claimed.
        edge_a = torch.zeros((5, 20), dtype=torch.float32)
        edge_a[0, :7] = torch.tensor([127, 0.5, 1.5, -0.5, -1.5, 2.5, -2.5])
        edge_a[1] = edge_a[0] * (2 ** -20)
        edge_a[3, 0] = 1e-12
        edge_a[4, 0] = 1e-38
        edge_w = torch.randint(-128, 128, (33, 20), dtype=torch.int8)
        edge_max = edge_a.abs().amax(dim=1)
        edge_scale = torch.where(edge_max == 0, torch.ones_like(edge_max), edge_max * torch.tensor(1 / 127, dtype=torch.float32)).clamp_min(torch.finfo(torch.float32).tiny)
        edge_q = (edge_a / edge_scale[:, None]).round().clamp(-127, 127).to(torch.int8)
        edge_ref = (edge_q.long() @ edge_w.long().T).float() * edge_scale[:, None]
        ea, ew = edge_a.to(device), edge_w.to(device)
        ep = torch.empty((5, 64), device=device, dtype=torch.int32)
        esw = torch.ones(33, device=device)
        ey = torch.empty((5, 33), device=device)
        checked(lib.chen_int8_pack_weights(ew.data_ptr(), ep.data_ptr(), 33, 20, stream.cuda_stream))
        checked(lib.chen_int8_linear_dynamic_packed(ea.data_ptr(), ep.data_ptr(), esw.data_ptr(), None, None, ey.data_ptr(), 5, 33, 20, stream.cuda_stream))
        stream.synchronize()
        torch.testing.assert_close(ey.cpu(), edge_ref, rtol=2e-6, atol=0)
        report['dynamic_edge_cases'] = dict(rows=5, channels=33, ties_to_even=True, zero_row=True, tiny_scale=True, subnormal_input=True, maximum_error=float((ey.cpu() - edge_ref).abs().max()))
        try:
            ii = torch._int_mm(torch.ones((64, 32), device=device, dtype=torch.int8), torch.ones((32, 32), device=device, dtype=torch.int8))
            stream.synchronize()
            assert bool((ii == 32).all())
            report['torch_int_mm'] = dict(supported=True, output_dtype=str(ii.dtype))
        except Exception as exc:
            report['torch_int_mm'] = dict(supported=False, error=repr(exc))
    report['terminal'] = 'PASS'
    output.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(dict(terminal='PASS', shapes=len(shapes), outputs=sum(r['outputs'] for r in report['rows']), library_sha256=report['library_sha256'], torch_int_mm=report['torch_int_mm'])))


if __name__ == '__main__':
    main()
