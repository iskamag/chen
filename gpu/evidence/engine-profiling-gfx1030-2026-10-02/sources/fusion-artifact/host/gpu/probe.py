#!/usr/bin/env python3
"""Standalone HIP ABI oracle + timing sweep; requires a real gfx1030 device."""
import argparse
import ctypes as c
import json
import random
import statistics
import time


class Position(c.Structure):
    _fields_ = [("pieces", c.c_uint64 * 12), ("side", c.c_uint32), ("reserved", c.c_uint32)]


class Device(c.Structure):
    _fields_ = [("name", c.c_char * 128), ("architecture", c.c_char * 64),
                ("total_memory", c.c_uint64), ("compute_units", c.c_uint32),
                ("wave_size", c.c_uint32), ("async_engines", c.c_uint32),
                ("runtime_version", c.c_uint32)]


class Timings(c.Structure):
    _fields_ = [("h2d_ms", c.c_double), ("kernel_ms", c.c_double),
                ("d2h_ms", c.c_double), ("count", c.c_uint64), ("capacity", c.c_uint64)]


def reference(p, mode):
    values, phases = [100, 320, 330, 500, 900, 0], [0, 1, 1, 2, 4, 0]
    mg = eg = phase = 0
    for color in range(2):
        sign = 1 if color == 0 else -1
        for piece in range(6):
            bb = p.pieces[color * 6 + piece]
            if mode == 0:
                mg += sign * values[piece] * bb.bit_count()
                continue
            phase += phases[piece] * bb.bit_count()
            while bb:
                bit = bb & -bb
                sq = bit.bit_length() - 1
                bb ^= bit
                file, rank = sq % 8, sq // 8 if color == 0 else 7 - sq // 8
                center = 14 - abs(2 * file - 7) - abs(2 * rank - 7)
                m, e = [(rank * 8 + center * 2, rank * 14 + center),
                        (center * 6, center * 4),
                        (center * 3 + rank * 2, center * 3),
                        (rank * 2 + (18 if rank == 6 else 0), center),
                        (center, center * 2),
                        (-center * 5 - rank * 4 + (35 if rank == 0 and file in (2, 6) else 0),
                         center * 6)][piece]
                mg += sign * (values[piece] + m)
                eg += sign * (values[piece] + e)
        if mode == 1 and p.pieces[color * 6 + 2].bit_count() >= 2:
            mg += sign * 25
            eg += sign * 40
    phase = min(24, phase)
    numerator = mg * phase + eg * (24 - phase)
    score = mg if mode == 0 else (abs(numerator) // 24) * (1 if numerator >= 0 else -1)
    return score if p.side == 0 else -score


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--max-batch", type=int, default=65536)
    parser.add_argument("--repetitions", type=int, default=7)
    args = parser.parse_args()
    assert args.max_batch >= 1 and args.repetitions >= 1
    assert c.sizeof(Position) == 104 and c.sizeof(Device) == 216 and c.sizeof(Timings) == 40
    lib = c.CDLL(args.library)
    error = c.create_string_buffer(1024)
    lib.chen_gpu_abi_version.restype = c.c_uint32
    lib.chen_gpu_create.argtypes = [c.c_int32, c.c_uint64, c.c_char_p, c.c_uint64]
    lib.chen_gpu_create.restype = c.c_void_p
    lib.chen_gpu_device_info.argtypes = [c.c_void_p, c.POINTER(Device), c.c_char_p, c.c_uint64]
    lib.chen_gpu_evaluate.argtypes = [c.c_void_p, c.POINTER(Position), c.POINTER(c.c_int32),
                                    c.c_uint64, c.c_uint32, c.POINTER(Timings), c.c_char_p, c.c_uint64]
    lib.chen_gpu_destroy.argtypes = [c.c_void_p]
    lib.chen_gpu_destroy.restype = None
    assert lib.chen_gpu_abi_version() == 1
    assert not lib.chen_gpu_create(0, 0, error, len(error))
    ctx = lib.chen_gpu_create(0, args.max_batch, error, len(error))
    assert ctx, error.value.decode()
    try:
        device = Device()
        assert lib.chen_gpu_device_info(ctx, c.byref(device), error, len(error)) == 0
        print(json.dumps({"record": "device", **{
            field: getattr(device, field).decode() if field in ("name", "architecture")
            else getattr(device, field) for field, _ in Device._fields_}}, sort_keys=True))
        rng = random.Random(0xC4E1030)
        inputs = (Position * args.max_batch)()
        outputs = (c.c_int32 * args.max_batch)()
        # Broad arithmetic coverage; this corpus is deliberately not a legality corpus.
        for p in inputs:
            p.side = rng.randrange(2)
            for sq in rng.sample(range(64), rng.randrange(0, 65)):
                p.pieces[rng.randrange(12)] |= 1 << sq
        timings = Timings()
        def evaluate(count, mode):
            result = lib.chen_gpu_evaluate(ctx, inputs, outputs, count, mode,
                                           c.byref(timings), error, len(error))
            assert result == 0, error.value.decode()
        oracle_count = min(args.max_batch, 4096)
        for mode in (0, 1):
            evaluate(oracle_count, mode)
            for i in range(oracle_count):
                expected = reference(inputs[i], mode)
                assert outputs[i] == expected, (i, mode, outputs[i], expected)
        assert lib.chen_gpu_evaluate(ctx, None, None, 0, 1, c.byref(timings), error, len(error)) == 0
        assert lib.chen_gpu_evaluate(ctx, inputs, outputs, args.max_batch + 1, 1,
                                    c.byref(timings), error, len(error)) != 0
        assert lib.chen_gpu_evaluate(ctx, inputs, outputs, 1, 2,
                                    c.byref(timings), error, len(error)) != 0
        inputs[0].reserved = 1
        assert lib.chen_gpu_evaluate(ctx, inputs, outputs, 1, 1,
                                    c.byref(timings), error, len(error)) != 0
        inputs[0].reserved = 0
        old_side, inputs[0].side = inputs[0].side, 2
        assert lib.chen_gpu_evaluate(ctx, inputs, outputs, 1, 1,
                                    c.byref(timings), error, len(error)) != 0
        inputs[0].side = old_side
        print(json.dumps({"record": "oracle", "positions_per_evaluator": oracle_count,
                          "evaluators": 2, "seed": 0xC4E1030, "boundary_checks": 5}))
        sizes = sorted(set([1, *[n for n in (8, 32, 128, 512, 2048, 8192, 32768)
                                 if n <= args.max_batch], args.max_batch]))
        for mode in (0, 1):
            for count in sizes:
                evaluate(count, mode)  # untimed warmup
                samples = []
                for _ in range(args.repetitions):
                    start = time.perf_counter_ns()
                    evaluate(count, mode)
                    wall_ms = (time.perf_counter_ns() - start) / 1e6
                    samples.append((wall_ms, timings.h2d_ms, timings.kernel_ms, timings.d2h_ms))
                median = [statistics.median(x[i] for x in samples) for i in range(4)]
                print(json.dumps({"record": "benchmark", "evaluator": ["material", "tapered"][mode],
                                  "batch": count, "repetitions": args.repetitions,
                                  "wall_ms": median[0], "h2d_ms": median[1],
                                  "kernel_ms": median[2], "d2h_ms": median[3],
                                  "positions_per_second": count * 1000 / median[0]}, sort_keys=True))
    finally:
        lib.chen_gpu_destroy(ctx)


if __name__ == "__main__":
    main()
