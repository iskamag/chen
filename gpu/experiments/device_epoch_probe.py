#!/usr/bin/env python3
"""Bounded actual-device epoch trajectory, admission, equal-work/deadline probe.

No Torch import, board download or engine-strength claim. Run only in a reserved
GPU window with an external process timeout; every successful native call fences.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import pathlib
import random
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from rules_probe import FIXTURES, Position, Move, Result, pack
import chess


class Config(C.Structure):
    _fields_ = [('abi_version', C.c_uint32), ('struct_size', C.c_uint32),
                ('max_ply', C.c_uint32), ('policy', C.c_uint32),
                ('batch_leaves', C.c_uint32), ('candidate_limit', C.c_uint32),
                ('reserved', C.c_uint64)]


class Stats(C.Structure):
    _fields_ = [('abi_version', C.c_uint32), ('struct_size', C.c_uint32)] + [
        (n, C.c_uint64) for n in ['arena_positions', 'classified_positions', 'evaluated_positions',
                                  'completed_steps', 'expanded_leaves', 'open_frontier_leaves']] + [
        ('last_selected', C.c_uint32), ('last_added', C.c_uint32), ('kernel_ms', C.c_double)] + [
        (n, C.c_uint32) for n in ['minimum_frontier_depth', 'maximum_frontier_depth',
                                  'pv_depth', 'root_flags', 'stop_reason', 'reserved']]


class Trace(C.Structure):
    _fields_ = [('stats', Stats), ('result', Result), ('pv', Move * 64)]


class Report(C.Structure):
    _fields_ = [(n, C.c_uint32) for n in ['abi_version', 'struct_size', 'attempts', 'mode',
                                         'advance_bound', 'error', 'resident_blocks_per_multiprocessor', 'grid_blocks']] + [
        ('setup_ms', C.c_double), ('trace', Trace * 4)]


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def normal(trace):
    stats = {n: getattr(trace.stats, n) for n, _ in Stats._fields_ if n != 'kernel_ms'}
    return dict(stats=stats, score=trace.result.score if stats['root_flags'] & 1 else None,
                pv=[trace.pv[i].uci() for i in range(trace.result.pv_length)] if stats['root_flags'] & 1 else [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--library-sha256', required=True)
    parser.add_argument('--output', required=True, type=pathlib.Path)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--modes', default='0,1,2')
    parser.add_argument('--parity-only', action='store_true')
    parser.add_argument('--skip-parity', action='store_true', help='timing-only after separately preserved parity')
    parser.add_argument('--deadline-ms', type=int, default=100)
    args = parser.parse_args()
    if sha(args.library) != args.library_sha256:
        raise RuntimeError('library fingerprint mismatch')
    if args.cpu not in os.sched_getaffinity(0):
        raise RuntimeError('CPU outside allowed affinity')
    if not 1 <= args.deadline_ms <= 1000:
        raise RuntimeError('unbounded deadline fixture')
    for key in ['HSA_OVERRIDE_GFX_VERSION', 'LD_PRELOAD', 'CHEN_NEURAL_MODEL', 'CHEN_NEURAL_LIBRARY']:
        if os.environ.get(key):
            raise RuntimeError('unexpected inherited ' + key)
    args.output.mkdir(parents=True, exist_ok=False)
    os.sched_setaffinity(0, {args.cpu})
    assert C.sizeof(Config) == 32 and C.sizeof(Stats) == 96
    assert C.sizeof(Trace) == 360 and C.sizeof(Report) == 1480
    lib = C.CDLL(args.library)
    ptr, size, u32, error = C.c_void_p, C.c_uint64, C.c_uint32, C.c_char_p
    for name, arguments, returns in [
        ('chen_selective_create', [C.c_int32, size, error, size], ptr),
        ('chen_selective_destroy', [ptr], None),
        ('chen_selective_reset', [ptr, ptr, ptr, size, ptr, ptr, error, size], C.c_int32),
        ('chen_selective_advance', [ptr, size, u32, ptr, ptr, error, size], C.c_int32),
        ('chen_selective_finish', [ptr, ptr, ptr, size, ptr, error, size], C.c_int32),
        ('chen_epoch_create', [C.c_int32, size, u32, u32, error, size], ptr),
        ('chen_epoch_destroy', [ptr], None),
        ('chen_epoch_reset', [ptr, ptr, ptr, size, ptr, error, size], C.c_int32),
        ('chen_epoch_advance', [ptr, size, u32, u32, ptr, error, size], C.c_int32),
    ]:
        function = getattr(lib, name)
        function.argtypes, function.restype = arguments, returns
    ev = C.c_int()
    lib.hipRuntimeGetVersion.argtypes = [C.POINTER(C.c_int)]
    assert lib.hipRuntimeGetVersion(C.byref(ev)) == 0
    mapped = sorted({l.split(maxsplit=5)[5].strip() for l in pathlib.Path('/proc/self/maps').read_text().splitlines()
                     if 'libamdhip64' in l and len(l.split(maxsplit=5)) == 6})
    declaration = dict(probe_sha256=sha(__file__), helper_sha256=sha(pathlib.Path(__file__).resolve().parents[1] / 'rules_probe.py'),
                       library_sha256=args.library_sha256, cpu_affinity=[args.cpu], modes=args.modes,
                       hip_runtime_get_version=ev.value, mapped_hip_libraries=[dict(path=p, sha256=sha(p)) for p in mapped],
                       deadline_ms=args.deadline_ms, parity_only=args.parity_only, skip_parity=args.skip_parity,
                       argv=sys.argv, scope='exact per-advance trajectory plus bounded epoch host cancellation; no model, training or Elo claim')
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')
    err = C.create_string_buffer(1024)
    modes = [int(m) for m in args.modes.split(',')]
    assert set(modes) <= {0, 1, 2} and len(modes) == len(set(modes))
    records, timings, deadlines = [], [], []

    def checked(rc, name):
        if rc:
            raise RuntimeError(f'{name}: {rc} {err.value.decode()}')

    def reset(context, candidate, board, config, history):
        position = pack(board)
        past = (C.c_uint64 * len(history))(*history) if history else None
        if candidate:
            checked(lib.chen_epoch_reset(context, C.byref(position), past, len(history), C.byref(config), err, len(err)), 'epoch reset')
        else:
            initial = Stats()
            checked(lib.chen_selective_reset(context, C.byref(position), past, len(history), C.byref(config), C.byref(initial), err, len(err)), 'baseline reset')

    def advance_one(context, limit, evaluator, publish=True):
        trace = Trace()
        checked(lib.chen_selective_advance(context, limit, evaluator, None, C.byref(trace.stats), err, len(err)), 'baseline advance')
        if publish and trace.stats.root_flags & 1:
            ignored = Stats()
            checked(lib.chen_selective_finish(context, C.byref(trace.result), trace.pv, 64, C.byref(ignored), err, len(err)), 'baseline finish')
        return trace

    def advance_epoch(context, limit, evaluator, bound):
        report = Report()
        checked(lib.chen_epoch_advance(context, limit, evaluator, bound, C.byref(report), err, len(err)), 'epoch advance')
        assert report.abi_version == 1 and report.struct_size == C.sizeof(Report)
        assert 1 <= report.attempts <= bound and report.advance_bound == bound and not report.error
        return report

    def legal_pv(board, trajectory):
        copy = board.copy()
        for text in trajectory['pv']:
            move = chess.Move.from_uci(text)
            assert move in copy.legal_moves
            copy.push(move)

    try:
        # All fixtures have independent python-chess position/hash preparation.
        cases = [(f'fixture{i}', chess.Board(fen), 5000, 2000, 4, 8, 1024, []) for i, fen in enumerate(FIXTURES)]
        cases += [('black-start', chess.Board(FIXTURES[0].replace(' w ', ' b ')), 5000, 2000, 4, 8, 1024, [])]
        root = chess.Board(FIXTURES[0]);root.halfmove_clock = 8
        cases += [('repetition-history', root, 5000, 2000, 4, 8, 1024, [pack(root).hash] * 2)]
        for cap, limit in [(1, 0), (1, 1), (1, 2000), (20, 2000), (21, 21), (5000, 1), (5000, 20), (5000, 21)]:
            cases.append((f'cap{cap}-limit{limit}', chess.Board(), cap, limit, 4, 8, 1024, []))
        cases += [('max-ply1', chess.Board(), 5000, 2000, 1, 8, 1024, []),
                  ('max-ply64', chess.Board(FIXTURES[2]), 5000, 2000, 64, 8, 1024, []),
                  ('candidate-scan1', chess.Board(), 5000, 2000, 4, 1, 1, []),
                  ('zero-bootstrap-recovery', chess.Board(), 5000, 2000, 4, 8, 1024, []),
                  ('cap-bootstrap-recovery', chess.Board(), 5000, 21, 4, 8, 1024, [])]
        for mode in modes:
            for bound in [2, 4]:
                for name, board, capacity, limit, ply, batch, candidates, history in ([] if args.skip_parity else cases):
                    baseline = lib.chen_selective_create(0, capacity, err, len(err))
                    candidate = lib.chen_epoch_create(0, capacity, mode, 64, err, len(err))
                    assert baseline and candidate, err.value.decode()
                    config = Config(1, C.sizeof(Config), ply, 0, batch, candidates, 0)
                    trajectory = []
                    try:
                        reset(baseline, False, board, config, history)
                        reset(candidate, True, board, config, history)
                        if name in ['zero-bootstrap-recovery', 'cap-bootstrap-recovery']:
                            initial_cap = 0 if name == 'zero-bootstrap-recovery' else 20
                            initial = advance_epoch(candidate, initial_cap, 1, bound)
                            assert initial.attempts == 1
                            assert normal(initial.trace[0]) == normal(advance_one(baseline, initial_cap, 1))
                            assert not initial.trace[0].stats.root_flags & 1
                            assert initial.trace[0].stats.classified_positions == (0 if initial_cap == 0 else 1)
                        for epoch in range(1000):
                            report = advance_epoch(candidate, limit, 1, bound)
                            for step in range(report.attempts):
                                expected = normal(advance_one(baseline, limit, 1))
                                actual = normal(report.trace[step])
                                assert actual == expected, (mode, bound, name, epoch, step, expected, actual)
                                legal_pv(board, actual)
                                trajectory.append(actual)
                            if report.trace[report.attempts - 1].stats.stop_reason:
                                break
                        else:
                            raise RuntimeError('parity fixture exceeded epoch bound')
                        # A stopped tree may be called again without losing its
                        # complete publication or counters; it emits one clean stop.
                        report = advance_epoch(candidate, limit, 1, bound)
                        expected = normal(advance_one(baseline, limit, 1))
                        assert report.attempts == 1 and normal(report.trace[0]) == expected
                        # Wrong evaluator invalidates; report remains unchanged.
                        sentinel = Report();C.memset(C.byref(sentinel), 0xA5, C.sizeof(sentinel))
                        original = bytes(sentinel)
                        assert lib.chen_epoch_advance(candidate, limit, 0, bound, C.byref(sentinel), err, len(err)) < 0
                        assert bytes(sentinel) == original
                        assert lib.chen_epoch_advance(candidate, limit, 1, bound, C.byref(sentinel), err, len(err)) < 0
                        assert bytes(sentinel) == original
                        # Device root/hash validation failure also invalidates
                        # and preserves the caller's complete output packet.
                        damaged = pack(board);damaged.hash ^= 1
                        past = (C.c_uint64 * len(history))(*history) if history else None
                        assert lib.chen_epoch_reset(candidate, C.byref(damaged), past, len(history), C.byref(config), err, len(err)) < 0
                        assert lib.chen_epoch_advance(candidate, limit, 1, bound, C.byref(sentinel), err, len(err)) < 0
                        assert bytes(sentinel) == original
                        reset(candidate, True, board, config, history)
                        recovered = advance_epoch(candidate, limit, 1, bound)
                        assert normal(recovered.trace[0]) == trajectory[0]
                        records.append(dict(mode=mode, bound=bound, name=name, capacity=capacity, limit=limit,
                                            ply=ply, batch=batch, candidates=candidates,
                                            attempts=len(trajectory), trajectory=trajectory,
                                            resident_blocks_per_multiprocessor=report.resident_blocks_per_multiprocessor,
                                            grid_blocks=report.grid_blocks, setup_ms=report.setup_ms,
                                            repeated_stop=True, sticky_invalid_admission=True,
                                            sticky_device_hash_failure=True, reset_recovery=True,
                                            bootstrap_continuation=name in ['zero-bootstrap-recovery', 'cap-bootstrap-recovery']))
                    finally:
                        lib.chen_selective_destroy(baseline)
                        lib.chen_epoch_destroy(candidate)
                    (args.output / 'parity.partial.json').write_text(json.dumps(records, indent=2) + '\n')
        if not args.parity_only:
            arms = [('baseline', None, 1)] + [(f'mode{m}-bound{b}', m, b) for m in modes for b in [2, 4]]
            config = Config(1, C.sizeof(Config), 64, 0, 32, 1024, 0)
            for work_limit in [2000, 50000]:
                reference = None
                for sample in range(4):
                    order = arms.copy();random.Random(20261002 + sample).shuffle(order)
                    for arm, mode, bound in order:
                        context = (lib.chen_selective_create(0, 100000, err, len(err)) if mode is None
                                   else lib.chen_epoch_create(0, 100000, mode, 64, err, len(err)))
                        assert context, err.value.decode()
                        try:
                            reset(context, mode is not None, chess.Board(FIXTURES[1]), config, [])
                            calls = 0;before = time.perf_counter_ns()
                            while calls < 10000:
                                if mode is None:
                                    trace = advance_one(context, work_limit, 1, False)
                                else:
                                    report = advance_epoch(context, work_limit, 1, bound)
                                    trace = report.trace[report.attempts - 1]
                                calls += 1
                                if trace.stats.stop_reason:
                                    break
                            else:
                                raise RuntimeError('timing fixture exceeded call cap')
                            elapsed = time.perf_counter_ns() - before
                            end = {n: getattr(trace.stats, n) for n, _ in Stats._fields_ if n != 'kernel_ms'}
                            if reference is None:
                                reference = end
                            assert reference == end, (arm, work_limit, reference, end)
                            timings.append(dict(arm=arm, sample=sample, limit=work_limit, elapsed_ns=elapsed,
                                                calls=calls, stats=end, setup_excluded=True))
                        finally:
                            (lib.chen_selective_destroy if mode is None else lib.chen_epoch_destroy)(context)
            # Batch1 sustains a long cancellation fixture rather than stopping
            # at the fixed-work budget before the declared one-second deadline.
            deadline_config = Config(1, C.sizeof(Config), 64, 0, 1, 1024, 0)
            for arm, mode, bound in arms:
                context = (lib.chen_selective_create(0, 250000, err, len(err)) if mode is None
                           else lib.chen_epoch_create(0, 250000, mode, 64, err, len(err)))
                assert context, err.value.decode()
                try:
                    reset(context, mode is not None, chess.Board(FIXTURES[1]), deadline_config, [])
                    before = time.perf_counter_ns();deadline = before + args.deadline_ms * 1000000
                    calls = 0;last_begin = before
                    while time.perf_counter_ns() < deadline:
                        last_begin = time.perf_counter_ns()
                        if mode is None:
                            trace = advance_one(context, 250000, 1, False)
                        else:
                            report = advance_epoch(context, 250000, 1, bound)
                            trace = report.trace[report.attempts - 1]
                        calls += 1
                        if trace.stats.stop_reason:
                            break
                    after = time.perf_counter_ns()
                    deadlines.append(dict(arm=arm, requested_ms=args.deadline_ms, elapsed_ns=after-before,
                                          deadline_overshoot_ns=max(0, after-deadline), last_epoch_ns=after-last_begin,
                                          calls=calls, completed_steps=trace.stats.completed_steps,
                                          classified=trace.stats.classified_positions, node_cap=250000,
                                          batch_leaves=1, stop_reason=trace.stats.stop_reason,
                                          deadline_reached=after >= deadline,
                                          cancellation='host deadline checked between bounded calls, admitted siblings always complete'))
                finally:
                    (lib.chen_selective_destroy if mode is None else lib.chen_epoch_destroy)(context)
        (args.output / 'result.json').write_text(json.dumps(dict(terminal='PASS', parity=records, timings=timings, deadlines=deadlines), indent=2) + '\n')
        print(json.dumps(dict(terminal='PASS', cases=len(records), trajectory_attempts=sum(x['attempts'] for x in records), timing_rows=len(timings), deadline_rows=len(deadlines))))
    except Exception as exc:
        (args.output / 'result.json').write_text(json.dumps(dict(terminal='FAIL_OR_PARTIAL', error=repr(exc), parity=records, timings=timings, deadlines=deadlines), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
