#!/usr/bin/env python3
"""Matched real dispatch intervention; not Coz virtual-speedup predictions.

Graph/queued arms execute identical kernels, dimensions, ordering, copies,
publication and fence. Production is a contextual arm with differing control
and publication. No timing instrumentation or repeated broad correctness suite.
"""
import argparse
import ctypes as C
import hashlib
import json
import os
import pathlib
import random
import time

from device_epoch_probe import Config, Stats, Trace, Report, normal, pack, chess, FIXTURES


def sha(path):
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--library', required=True)
    p.add_argument('--library-sha256', required=True)
    p.add_argument('--output', type=pathlib.Path, required=True)
    args = p.parse_args()
    assert sha(args.library) == args.library_sha256
    assert 2 in os.sched_getaffinity(0)
    os.sched_setaffinity(0, {2})
    for key in ['LD_PRELOAD', 'HSA_OVERRIDE_GFX_VERSION', 'CHEN_NEURAL_MODEL', 'CHEN_NEURAL_LIBRARY']:
        assert not os.environ.get(key), key
    args.output.mkdir(parents=True, exist_ok=False)
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
    arms = {'production': None, 'graph4': 0, 'queued4': 3}
    # One wide batch and one latency-heavy batch: limit search equally, not time.
    fixtures = [('batch32', 32, 50000), ('batch1', 1, 5000)]
    declaration = dict(source_sha256=sha(__file__), library_sha256=args.library_sha256,
                       fixtures=fixtures, samples=8, warmups=1, seed=20261003,
                       cpu_affinity=[2], bound=4, capacity=100000,
                       compared='graph4 versus queued4; production contextual only',
                       progress='same completed advance trajectory and classified positions',
                       scope='real launch intervention, no virtual-speedup curve or strength claim')
    (args.output / 'declaration.json').write_text(json.dumps(declaration, indent=2) + '\n')
    err = C.create_string_buffer(1024)
    rows = []

    def checked(rc):
        if rc:
            raise RuntimeError((rc, err.value.decode()))

    try:
        for name, batch, limit in fixtures:
            config = Config(1, C.sizeof(Config), 64, 0, batch, 1024, 0)
            root = pack(chess.Board(FIXTURES[1]))
            expected = None
            contexts = {}
            try:
                for arm, mode in arms.items():
                    ctx = (lib.chen_selective_create(0, 100000, err, len(err)) if mode is None
                           else lib.chen_epoch_create(0, 100000, mode, 64, err, len(err)))
                    assert ctx, err.value.decode()
                    contexts[arm] = ctx
                for sample in range(-1, 8):
                    order = list(arms)
                    random.Random(20261003 + sample + batch).shuffle(order)
                    trajectories = {}
                    for arm in order:
                        ctx, mode = contexts[arm], arms[arm]
                        initial = Stats()
                        if mode is None:
                            checked(lib.chen_selective_reset(ctx, C.byref(root), None, 0,
                                                            C.byref(config), C.byref(initial), err, len(err)))
                        else:
                            checked(lib.chen_epoch_reset(ctx, C.byref(root), None, 0,
                                                        C.byref(config), err, len(err)))
                        # Preallocate outputs; analysis/normalization outside wall timing.
                        outputs = [(Stats() if mode is None else Report()) for _ in range(2000)]
                        before = time.perf_counter_ns()
                        for calls, output in enumerate(outputs, 1):
                            if mode is None:
                                checked(lib.chen_selective_advance(ctx, limit, 1, None,
                                                                  C.byref(output), err, len(err)))
                                stats = output
                            else:
                                checked(lib.chen_epoch_advance(ctx, limit, 1, 4,
                                                              C.byref(output), err, len(err)))
                                assert 1 <= output.attempts <= 4
                                stats = output.trace[output.attempts - 1].stats
                            if stats.stop_reason:
                                break
                        else:
                            raise RuntimeError('advance bound exceeded')
                        elapsed = time.perf_counter_ns() - before
                        finish = Trace()
                        if mode is None:
                            checked(lib.chen_selective_finish(ctx, C.byref(finish.result), finish.pv, 64,
                                                             C.byref(finish.stats), err, len(err)))
                            sequence = [{k: getattr(o, k) for k, _ in Stats._fields_ if k != 'kernel_ms'}
                                        for o in outputs[:calls]]
                            final = normal(finish)
                            assert sequence[-1] == final['stats']
                        else:
                            sequence = [normal(o.trace[i]) for o in outputs[:calls] for i in range(o.attempts)]
                            final = sequence[-1]
                        trajectories[arm] = sequence
                        if expected is None:
                            expected = final
                        assert final == expected, (name, arm, final, expected)
                        if sample >= 0:
                            rows.append(dict(fixture=name, sample=sample, arm=arm, elapsed_ns=elapsed,
                                             calls=calls, attempts=len(sequence), final=final))
                    assert trajectories['graph4'] == trajectories['queued4'], name
                    assert trajectories['production'] == [t['stats'] for t in trajectories['graph4']], name
            finally:
                for arm, ctx in contexts.items():
                    (lib.chen_selective_destroy if arms[arm] is None else lib.chen_epoch_destroy)(ctx)
        summary = []
        for name, _, _ in fixtures:
            totals = {arm: sum(r['elapsed_ns'] for r in rows if r['fixture'] == name and r['arm'] == arm)
                      for arm in arms}
            ratios = []
            for sample in range(8):
                pair = {r['arm']: r['elapsed_ns'] for r in rows if r['fixture'] == name and r['sample'] == sample}
                ratios.append(pair['queued4'] / pair['graph4'])
            summary.append(dict(fixture=name, totals_ns=totals,
                                queued_over_graph=totals['queued4'] / totals['graph4'],
                                production_over_graph=totals['production'] / totals['graph4'],
                                paired_queued_over_graph=ratios))
        result = dict(terminal='PASS', rows=rows, summary=summary,
                      exact_all_advance_graph_queued_score_pv_stats=True,
                      exact_all_advance_production_stats_and_final_score_pv=True)
        (args.output / 'result.json').write_text(json.dumps(result, indent=2) + '\n')
        print(json.dumps(dict(terminal='PASS', rows=len(rows), summary=summary)), flush=True)
    except BaseException as exc:
        (args.output / 'partial.json').write_text(json.dumps(dict(error=repr(exc), rows=rows), indent=2) + '\n')
        raise


if __name__ == '__main__':
    main()
