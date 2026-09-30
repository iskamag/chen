#!/usr/bin/env python3
"""Bounded eager UCI stall reproduction with stacks before child retirement.

Imports the frozen search oracle for its protocol/position checks. Production
engine/native sources and watchdog remain unchanged. All debugger artifacts are
written under the explicitly selected /tmp/chen-* directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time

import chess
import torch
import neural_search_probe as reference

LAUNCHER = r'''
#define _GNU_SOURCE
#include <sys/prctl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc,char **argv) {
    const char *engine=getenv("CHEN_DEBUG_ENGINE");
    if(argc<2 || !engine) return 71;
    if(prctl(PR_SET_PTRACER,PR_SET_PTRACER_ANY,0,0,0)) {perror("PR_SET_PTRACER");return 71;}
    argv[0]=(char *)engine;execv(engine,argv);perror("execv");return 71;
}
'''


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(row):
    print(json.dumps(row), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', default='/tmp/chen-graph-plugin/chen')
    parser.add_argument('--plugin', default='/tmp/chen-graph-plugin/libchen_neural.so')
    parser.add_argument('--rules', default='/tmp/chen-gpu/build/libchen_gpu.so')
    parser.add_argument('--models', default='/tmp/chen-neural-inference/trained')
    parser.add_argument('--suite', default='/tmp/chen-native-plugin/benches/positions.txt')
    parser.add_argument('--rust-source', default='/tmp/chen-neural-graph-source')
    parser.add_argument('--gdb', default='/tmp/chen-debugger/root/usr/bin/gdb')
    parser.add_argument('--output-dir', default='/tmp/chen-eager-diagnosis')
    parser.add_argument('--case', choices=['fresh-pair', 'warm-pair', 'full'], default='fresh-pair')
    parser.add_argument('--rounds', type=int, default=3)
    args = parser.parse_args()
    os.environ['CHEN_NEURAL_EXECUTION'] = 'eager'
    destination = Path(args.output_dir)
    assert str(destination).startswith('/tmp/chen-')
    destination.mkdir(parents=True, exist_ok=True)
    launcher_source = destination / 'launcher.c'
    launcher_source.write_text(LAUNCHER)
    launcher = destination / 'chen-ptrace'
    subprocess.run(['gcc', '-O2', '-Wall', '-Wextra', '-Werror', str(launcher_source), '-o', str(launcher)], check=True)
    gdb_env = os.environ.copy()
    gdb_env['LD_LIBRARY_PATH'] = '/tmp/chen-debugger/root/usr/lib/x86_64-linux-gnu'
    assert not os.environ.get('LD_PRELOAD')
    version = subprocess.run([args.gdb, '--version'], env=gdb_env, capture_output=True, text=True, check=True)
    emit({'record': 'eager_stall_provenance', 'case': args.case, 'watchdog_seconds': 30,
          'gdb_version': version.stdout.splitlines()[0], 'artifacts_sha256': {
              str(path): sha(path) for path in (Path(__file__), Path(reference.__file__), Path(args.engine),
                                               Path(args.plugin), Path(args.rules), launcher_source, launcher)},
          'environment': {key: os.environ.get(key) for key in ('CHEN_NEURAL_EXECUTION', 'OMP_NUM_THREADS',
                         'OPENBLAS_NUM_THREADS', 'AMD_DIRECT_DISPATCH', 'LD_PRELOAD')}})

    def snapshot(pid, label):
        proc = Path('/proc') / str(pid)
        report = {'pid': pid, 'label': label, 'threads': []}
        for name in ('status', 'maps', 'wchan', 'sched'):
            try:
                path = destination / f'{label}-{pid}-{name}.txt'
                path.write_text((proc / name).read_text())
                report[name] = str(path)
            except OSError as exc:
                report[name] = str(exc)
        try:
            for task in sorted((proc / 'task').iterdir()):
                item = {'tid': int(task.name)}
                for name in ('comm', 'wchan', 'stat'):
                    try:
                        item[name] = (task / name).read_text().strip()
                    except OSError as exc:
                        item[name] = str(exc)
                report['threads'].append(item)
        except OSError as exc:
            report['task_error'] = str(exc)
        command = [args.gdb, '--nx', '--batch', '-ex', 'set pagination off', '-ex', 'set debuginfod enabled off',
                   '-ex', 'set print thread-events off', '-ex', 'thread apply all bt 25',
                   '-ex', 'info sharedlibrary', '-ex', 'detach', '-p', str(pid)]
        try:
            traced = subprocess.run(command, env=gdb_env, capture_output=True, text=True, timeout=15)
            path = destination / f'{label}-{pid}-gdb.txt'
            path.write_text(traced.stdout + '\n--- debugger stderr ---\n' + traced.stderr)
            report.update(gdb_output=str(path), gdb_returncode=traced.returncode,
                          gdb_sha256=sha(path), gdb_attached='ptrace: Operation not permitted' not in traced.stderr)
        except subprocess.TimeoutExpired as exc:
            report['gdb_error'] = str(exc)
        emit({'record': 'eager_stall_thread_snapshot', **report})
        return report

    # Verify ptrace admission before any GPU diagnostic case, so a stalled
    # engine is not the first time debugger availability is checked.
    sleeping_env = os.environ.copy();sleeping_env['CHEN_DEBUG_ENGINE'] = '/bin/sleep'
    sleeping = subprocess.Popen([str(launcher), '20'], env=sleeping_env)
    try:
        time.sleep(.05)
        checked = snapshot(sleeping.pid, 'ptrace-control')
        assert checked.get('gdb_returncode') == 0 and checked.get('gdb_attached'), checked
    finally:
        sleeping.terminate();sleeping.wait(timeout=3)

    class DiagnosticUCI(reference.UCI):
        def __init__(self, binary, environment, evaluator, backend='hip-resident', order='neural-first'):
            self.timed_out = False
            self.actual_engine = binary
            marked = environment.copy();marked['CHEN_DEBUG_ENGINE'] = binary
            super().__init__(str(launcher), marked, evaluator, backend, order)

        def wait(self, prefix, timeout=30):
            lines, deadline = [], time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    line = self.lines.get(timeout=max(.01, deadline-time.monotonic()))
                except queue.Empty:
                    break
                lines.append(line)
                if line.startswith(prefix):
                    return lines
            self.timed_out = True
            emit({'record': 'eager_stall_timeout', 'waiting_for': prefix, 'returncode': self.process.poll(),
                  'diagnostic': self.diagnostic.copy(), 'stdout': lines, 'stderr': list(self.errors)})
            snapshot(self.process.pid, 'timeout-first')
            # Two host stack snapshots distinguish a static wait from ordinary
            # compilation/progress; no watchdog extension or extra search.
            time.sleep(.25)
            snapshot(self.process.pid, 'timeout-second')
            raise AssertionError(('eager UCI timeout', prefix, self.process.pid))

        def close(self):
            if self.timed_out:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait(timeout=3)
                emit({'record': 'eager_stall_owned_child_retired', 'pid': self.process.pid,
                      'returncode': self.process.returncode, 'stderr': list(self.errors)})
            else:
                super().close()

    reference.UCI = DiagnosticUCI
    assert torch.version.hip and torch.cuda.is_available()
    assert 'gfx1030' in torch.cuda.get_device_properties(0).gcnArchName
    if args.case == 'full':
        sys.argv = [reference.__file__, '--engine', args.engine, '--plugin', args.plugin, '--rules', args.rules,
                    '--models', args.models, '--suite', args.suite, '--rust-source', args.rust_source]
        reference.main()
        return
    if args.case == 'warm-pair':
        torch.set_num_threads(1)
        from neural.features import encode_fen
        from neural.inference import load_inference
        from inference_bench import FENS
        with torch.inference_mode():
            for name in ('square', 'relation', 'plan'):
                archive = Path(args.models) / (name+'-inference.pt')
                module = load_inference(archive, expected_sha256=sha(archive), device='cuda:0')
                for count in (1, 8, 32, 128, 256):
                    features = torch.stack([encode_fen(FENS[i % len(FENS)]) for i in range(count)]).cuda()
                    for _ in range(3):
                        module(features)
                torch.cuda.synchronize()
                del module
                torch.cuda.empty_cache()
                emit({'record': 'eager_stall_parent_warmed', 'model': name})
    board = chess.Board()
    for move in ('e2e4', 'e7e5', 'g1f3', 'b8c6'):
        board.push_uci(move)
    model = Path(args.models) / 'square-inference.pt'
    env = reference.environment(args, model)
    env['CHEN_NEURAL_EXECUTION'] = 'eager'
    for iteration in range(args.rounds):
        clients = []
        order = ('tapered', 'neural') if iteration % 2 == 0 else ('neural', 'tapered')
        try:
            for evaluator in order:
                clients.append(DiagnosticUCI(args.engine, env, evaluator))
            neural = clients[order.index('neural')]
            neural.diagnostic.update(case=args.case, iteration=iteration, creation_order=order)
            emit({'record': 'eager_stall_search_begin', 'iteration': iteration, 'creation_order': order,
                  'engine_pids': [client.process.pid for client in clients], 'fen': board.fen(en_passant='fen')})
            started = time.monotonic()
            move, info = neural.search(board, 'go depth 2 nodes 20000')
            emit({'record': 'eager_stall_search_pass', 'iteration': iteration, 'move': move.uci(),
                  'elapsed_seconds': time.monotonic()-started, 'search': info})
        finally:
            for client in reversed(clients):
                client.close()
    emit({'record': 'eager_stall_complete', 'case': args.case, 'rounds': args.rounds,
          'scope': 'Bounded first-position reproduction; no reliability guarantee from passing cases.'})


if __name__ == '__main__':
    main()
