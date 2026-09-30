#!/usr/bin/env python3
"""Predeclared CPU-parent versus HIP-parent control; no Torch import at module load."""
import argparse
import ast
import datetime
import hashlib
import json
import os
from pathlib import Path
import queue
import selectors
import signal
import subprocess
import sys
import threading
import time

import chess

MATE = 30000
REFERENCE_SCRIPT_SHA256 = "d8c580b796e694ce5f80eb91c7d93d0b4706bc15797a73eaabd61c216d7ada9c"
HELPER_SOURCE_SHA256 = {'terminal': 'c396842ffb3b765b755c2ecee4f8e9c05f5b7ecce7476747edc3cf413d9dd908', 'oracle': 'ee67d51570924501b8b99e4de538ee9fc94898ed3d007220f160de980d885aa8', 'check_pv': '5b9e3f3c1d6e25ab59ea56e4150c97a82d5a83c710d0e28897716da9c1794aef', 'UCI': 'e115ab8b7f24f9341541e9e60f43179923f1cac949a96c002311ed620c469b05', 'environment': '47a9bb1d325514ed2870fbe23b5d61d1d6e916d4754b3bc59e7732169e59fd76'}
JOURNAL = None
OWNED_CHILDREN = []


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(record):
    record = {**record, 'observed_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'monotonic_seconds': time.monotonic()}
    line = json.dumps(record, sort_keys=True)
    if JOURNAL is not None:
        JOURNAL.write(line + "\n")
        JOURNAL.flush()
        os.fsync(JOURNAL.fileno())
    print(line, flush=True)


def terminal(board, ply):
    moves = list(board.legal_moves)
    if not moves:
        return (-MATE + ply if board.is_check() else 0), moves
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0, moves
    return None, moves

def oracle(board, depth, module, max_batch):
    nodes, leaves = [], []
    def visit(position, ply):
        index = len(nodes)
        value, moves = terminal(position, ply)
        node = {"board": position, "ply": ply, "score": value, "terminal": value is not None, "children": []}
        nodes.append(node)
        if value is not None:
            return index
        if ply == depth:
            leaves.append(index)
            return index
        for move in moves:
            successor = position.copy(stack=True)
            successor.push(move)
            node["children"].append((move, visit(successor, ply + 1)))
        return index
    visit(board.copy(stack=True), 0)
    for start in range(0, len(leaves), max_batch):
        selected = leaves[start:start + max_batch]
        features = torch.stack([encode_fen(nodes[index]["board"].fen(en_passant="fen")) for index in selected]).to("cuda:0")
        scores = module(features)
        assert scores.dtype == torch.int32 and scores.device.type == "cuda" and tuple(scores.shape) == (len(selected),)
        assert bool(((scores >= -29000) & (scores <= 29000)).all())
        for index, value in zip(selected, scores.cpu().tolist()):
            nodes[index]["score"] = value
    for node in reversed(nodes):
        if node["children"]:
            node["score"] = max(-nodes[index]["score"] for _, index in node["children"])
    best = [move.uci() for move, index in nodes[0]["children"] if -nodes[index]["score"] == nodes[0]["score"]]
    return {"score_cp": nodes[0]["score"], "optimal_moves": best, "visited_positions": len(nodes),
            "evaluated_leaves": len(leaves), "terminal_positions": sum(node["terminal"] for node in nodes)}

def check_pv(board, pv, depth):
    moves = pv.split()
    assert len(moves) <= depth
    position = board.copy(stack=True)
    for text in moves:
        move = chess.Move.from_uci(text)
        assert move in position.legal_moves, (board.fen(), text, pv)
        position.push(move)
    return moves

class UCI:
    def __init__(self, binary, environment, evaluator, backend="hip-resident", order="neural-first"):
        self.process = subprocess.Popen([binary, "uci"], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, text=True, bufsize=1, env=environment)
        self.lines, self.errors = queue.Queue(), []
        self.diagnostic = {'evaluator': evaluator, 'backend': backend, 'option_order': order,
                           'engine_pid': self.process.pid, 'last_command': None}
        threading.Thread(target=lambda: [self.lines.put(line.rstrip()) for line in self.process.stdout], daemon=True).start()
        threading.Thread(target=lambda: [self.errors.append(line.rstrip()) for line in self.process.stderr], daemon=True).start()
        self.send("uci")
        self.wait("uciok")
        self.send("setoption name MaxPositions value 50000")
        if order == "backend-first":
            self.send(f"setoption name Backend value {backend}")
            self.send(f"setoption name Eval value {evaluator}")
        else:
            self.send(f"setoption name Eval value {evaluator}")
            self.send(f"setoption name Backend value {backend}")
        self.send("isready")
        self.configuration = self.wait("readyok")

    def send(self, command):
        assert self.process.poll() is None, (command, self.errors)
        self.diagnostic['last_command'] = command
        self.process.stdin.write(command + "\n")
        self.process.stdin.flush()

    def wait(self, prefix, timeout=30):
        collected, deadline = [], time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                line = self.lines.get(timeout=max(.01, deadline - time.monotonic()))
            except queue.Empty:
                break
            collected.append(line)
            if line.startswith(prefix):
                return collected
        processes = subprocess.run(['ps', '-eo', 'pid,ppid,pgid,stat,args'], capture_output=True, text=True, check=True).stdout
        active = [line for line in processes.splitlines() if '/tmp/chen-' in line or str(self.process.pid) in line]
        emit({'record': 'neural_UCI_failure', 'waiting_for': prefix, 'returncode': self.process.poll(),
              'diagnostic': self.diagnostic, 'stdout': collected, 'stderr': self.errors, 'active_processes': active})
        raise AssertionError(("UCI timeout", prefix, self.process.poll(), self.diagnostic, collected, self.errors))

    def position(self, board):
        self.diagnostic['board_fen'] = board.fen(en_passant='fen')
        self.diagnostic['history'] = [move.uci() for move in board.move_stack]
        root = board.root()
        suffix = " ".join(move.uci() for move in board.move_stack)
        command = "position fen " + root.fen(en_passant="fen")
        if suffix:
            command += " moves " + suffix
        self.send(command)

    def search(self, board, go="go depth 2", stop=False):
        self.position(board)
        self.send(go)
        if stop:
            self.send("stop")
        lines = self.wait("bestmove ")
        assert not any("search backend failed:" in line for line in lines), lines
        text = lines[-1].split()[1]
        move = chess.Move.from_uci(text)
        assert move in board.legal_moves, (board.fen(), lines, self.configuration)
        infos = [line for line in lines if line.startswith("info depth ")]
        depths = [int(line.split()[2]) for line in infos]
        node_counts = [int(line.split()[line.split().index("nodes") + 1]) for line in infos]
        for line in infos:
            if " pv " in line:
                check_pv(board, line.split(" pv ", 1)[1], int(line.split()[2]))
        return move, {"bestmove": text, "depths": depths, "node_counts": node_counts, "lines": lines}

    def close(self):
        if self.process.poll() is None:
            self.send("quit")
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait(timeout=5)
        assert self.process.returncode == 0, (self.process.returncode, self.errors)

def environment(args, model):
    result = os.environ.copy()
    result.update(CHEN_GPU_LIBRARY=args.rules, CHEN_NEURAL_LIBRARY=args.plugin,
                  CHEN_NEURAL_MODEL=str(model), OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    return result


GAME_PROTOCOL_SHA256 = '6926480f7adb0395709b52ae2ec809169e7b7657699d34de00589603f198bf3b'

def protocol_games(args, model_paths):
    # UCI option-order behavior is observed separately, with no silent fallback.
    uci_model = model_paths[0]
    env = environment(args, uci_model)
    for order in ('neural-first', 'backend-first'):
        ordered = UCI(args.engine, env, 'neural', order=order)
        try:
            configuration_failures = [line for line in ordered.configuration if 'backend creation failed:' in line]
            if order == 'neural-first':
                assert configuration_failures == ['info string backend creation failed: unknown evaluator: neural']
            else:
                assert not configuration_failures, ordered.configuration
            _, search = ordered.search(chess.Board(), 'go depth 2')
            assert search['depths'][-1] == 2
            # Invalid syntax retains working config; unsupported valid config
            # becomes unavailable, then an explicit restore recovers inference.
            ordered.send('setoption name Eval value unsupported')
            ordered.send('isready')
            syntax_failure = ordered.wait('readyok')
            assert 'info string invalid or unsupported option' in syntax_failure
            _, after_syntax = ordered.search(chess.Board(), 'go depth 1')
            assert after_syntax['depths'] == [1]
            ordered.send('setoption name Backend value cpu-resident')
            ordered.send('isready')
            unavailable = ordered.wait('readyok')
            assert any('neural evaluator requires hip-resident' in line for line in unavailable), unavailable
            ordered.send('setoption name Backend value hip-resident')
            ordered.send('isready')
            restored = ordered.wait('readyok')
            assert not any('backend creation failed:' in line for line in restored), restored
            _, recovered = ordered.search(chess.Board(), 'go depth 2')
            assert recovered['depths'][-1] == 2
            emit({'record': 'neural_UCI_option_order', 'order': order, 'configuration': ordered.configuration,
                  'search': search, 'invalid_syntax': syntax_failure, 'unsupported_configuration': unavailable,
                  'restored_configuration': restored, 'recovered_search': recovered,
                  'without_LD_PRELOAD': True, 'scope': 'Startup binding/order and rejected-configuration recovery, no CPU substitution.'})
        finally:
            ordered.close()
    client = UCI(args.engine, env, 'neural')
    try:
        board = chess.Board()
        _, low = client.search(board, 'go depth 2 nodes 20')
        _, exactcap = client.search(board, 'go depth 2 nodes 21')
        assert not low['depths'] and exactcap['depths'] == [1]
        client.send('setoption name Level value 1')
        _, level = client.search(board, 'go depth 5')
        assert level['depths'] == [1] and max(level['node_counts']) <= 128
        client.send('setoption name Level value 20')
        _, stopped = client.search(board, 'go depth 64', stop=True)
        _, recovery = client.search(board)
        assert recovery['depths'][-1] == 2
        emit({"record": "neural_UCI_limits", "model_sha256": digest(uci_model), "configuration": client.configuration,
              "nodes20": low, "nodes21": exactcap, "level1": level, "immediate_stop": stopped,
              "recovery": recovery, "scope": "Legal fallback, budget, stop and subsequent-search protocol proof."})
    finally:
        client.close()
    # Ordinary openings, alternating neural color, no rating interpretation.
    opponent = UCI(args.engine, env, 'tapered')
    neural = UCI(args.engine, env, 'neural')
    try:
        for game_index, opening in enumerate((('e2e4', 'e7e5', 'g1f3', 'b8c6'), ('d2d4', 'd7d5', 'c2c4', 'e7e6'))):
            board = chess.Board()
            for text in opening:
                board.push_uci(text)
            played = []
            neural_color = chess.WHITE if game_index == 0 else chess.BLACK
            for ply in range(args.game_plies):
                score, _ = terminal(board, len(board.move_stack))
                if score is not None:
                    break
                chooser = neural if board.turn == neural_color else opponent
                chooser.diagnostic.update(game_index=game_index, ply_after_opening=ply,
                                          chooser='neural' if chooser is neural else 'resident-tapered')
                emit({'record': 'neural_UCI_game_search_begin', 'game': game_index, 'ply_after_opening': ply,
                      'chooser': chooser.diagnostic['chooser'], 'fen': board.fen(en_passant='fen')})
                move, info = chooser.search(board, 'go depth 2 nodes 20000')
                board.push(move)
                played.append({"engine": 'neural' if chooser is neural else 'resident-tapered',
                               "move": move.uci(), "completed_depth": max(info['depths'], default=0)})
            emit({"record": "neural_UCI_legal_game", "opening": list(opening), "neural_color": 'white' if neural_color else 'black',
                  "plies_after_opening": len(played), "moves": played, "final_fen": board.fen(en_passant='fen'),
                  "outcome": str(board.outcome(claim_draw=False)), "scope": "Legal nonterminal protocol exercise, no strength inference."})
    finally:
        neural.close()
        opponent.close()


FIXED_ARTIFACTS = {
    '/tmp/chen-graph-plugin/chen': '62eb2c7f665a6389d84df14c2543cc855ab1a6467fdb21855c9547a3ade2982b',
    '/tmp/chen-graph-plugin/libchen_neural.so': 'cc673437ffc412cbee9755a51c1cd28b7e907d0ae08f99d4f0fec6213a786505',
    '/tmp/chen-graph-plugin/libchen_neural.so.runtime': '3feb2f4c6f834b5ea0064769c2c068a7386809c3780abae79411a850f3c8700d',
    '/usr/local/lib/python3.12/dist-packages/torch/lib/libamdhip64.so': 'c13a873d6444aa9c32d846feaa888de2108a5ed3c15f8a1b7708c5742a894e81',
    '/tmp/chen-gpu/build/libchen_gpu.so': '2ff8dfc0a6d6a2ef418fae2f0b858d81ea12087fd0af8f30cc629a963462d271',
}
FIXED_MODELS = {
    'square-inference.pt': '55aa95cf458b81e71fa04ca9cd9dc5c0a1f36365c554dcd86e6253c6f9a8eac3',
    'relation-inference.pt': 'a90c1c3138c9ae0fec96ca93ec8e0150ac814121cf2ee1027071e377621fd0d8',
    'plan-inference.pt': '3e137749f2c6888c62015b634310573e575302e5194396f57c64dcbb038b864d',
}
FEATURE_SOURCE = '/tmp/chen-native-plugin/neural/features.py'
FEATURE_SHA256 = '3aecc21de78d1c586543117aca78b3e8920b84b3694377d700228425a5e67cfc'
SUITE_SHA256 = '76d7d7d7df2011c9dd9f6be45fbb173c1d2266d4dc826f644ab6066cf5c84d74'


def files(args):
    values = {p: digest(p) for p in FIXED_ARTIFACTS}
    assert values == FIXED_ARTIFACTS, values
    for name, expected in FIXED_MODELS.items():
        path = str(Path(args.models) / name)
        values[path] = digest(path)
        assert values[path] == expected, path
        values[path + '.json'] = digest(path + '.json')
    values[args.suite] = digest(args.suite)
    assert values[args.suite] == SUITE_SHA256, 'benchmark fixtures changed'
    values[FEATURE_SOURCE] = digest(FEATURE_SOURCE)
    assert values[FEATURE_SOURCE] == FEATURE_SHA256
    values[str(Path(__file__).resolve())] = digest(__file__)
    return values


def fixtures(args):
    result = [(name.strip(), chess.Board(fen.strip()))
              for line in Path(args.suite).read_text().splitlines()
              if line.strip() and not line.lstrip().startswith('#')
              for name, fen in [line.split('|', 1)]]
    assert len(result) == 6
    return result


def torch_runtime():
    # This function is reached only in the GPU producer or HIP-parent arm.
    import torch as runtime
    from neural.features import encode_fen as encoder
    assert Path(encoder.__code__.co_filename).resolve() == Path(FEATURE_SOURCE).resolve()
    globals()['torch'], globals()['encode_fen'] = runtime, encoder
    runtime.set_num_threads(1)
    assert runtime.version.hip and runtime.cuda.is_available()
    props = runtime.cuda.get_device_properties(0)
    assert 'gfx1030' in props.gcnArchName
    return runtime


def process_identity(pid):
    try:
        data = Path(f'/proc/{pid}/stat').read_text()
    except FileNotFoundError:
        return None
    return {'pid': pid, 'start_ticks': data[data.rfind(')') + 2:].split()[19]}


def producer(args):
    assert os.environ.get('AMD_DIRECT_DISPATCH') == '1'
    assert not os.environ.get('LD_PRELOAD')
    provenance = files(args)
    runtime = torch_runtime()
    payload = {'schema_version': 1, 'source_sha256': digest(__file__),
               'reference_script_sha256': REFERENCE_SCRIPT_SHA256,
               'helper_source_sha256': HELPER_SOURCE_SHA256,
               'artifact_sha256': provenance, 'producer': process_identity(os.getpid()),
               'torch_version': str(runtime.__version__), 'torch_hip_version': runtime.version.hip,
               'fixtures': [{'name': name, 'fen': board.fen(en_passant='fen')} for name, board in fixtures(args)],
               'models': {}}
    with runtime.inference_mode():
        for name, expected in FIXED_MODELS.items():
            archive = Path(args.models) / name
            manifest = json.loads(Path(str(archive) + '.json').read_text())
            assert manifest['archive_sha256'] == expected
            assert 1 <= manifest['manifest']['max_batch'] <= 256
            module = runtime.jit.load(str(archive), map_location='cuda:0').eval()
            rows = {}
            for fixture, board in fixtures(args):
                rows[fixture] = {str(depth): oracle(board, depth, module, manifest['manifest']['max_batch'])
                                 for depth in (1, 2)}
                emit({'record': 'parent_context_oracle_fixture', 'model': name,
                      'fixture': fixture, 'answers': rows[fixture]})
            runtime.cuda.synchronize()
            payload['models'][name] = {'archive_sha256': expected,
                                       'max_batch': manifest['manifest']['max_batch'], 'rows': rows}
            del module
            runtime.cuda.empty_cache()
    assert all(digest(path) == value for path, value in provenance.items())
    path = Path(args.oracle)
    path.parent.mkdir(parents=True, exist_ok=True)
    assert not path.exists(), 'refusing to replace a frozen oracle cache'
    temporary = path.with_suffix('.writing')
    with temporary.open('w') as stream:
        json.dump(payload, stream, indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    temporary.replace(path)
    emit({'record': 'parent_context_oracle_complete', 'path': str(path), 'sha256': digest(path),
          'producer_identity': payload['producer'], 'producer_must_exit_before_trials': True})


def check_cpu_parent():
    imported = sorted(name for name in sys.modules if name == 'torch' or name.startswith('torch.')
                      or name == 'neural' or name.startswith('neural.'))
    maps = Path('/proc/self/maps').read_text()
    loaded = [line for line in maps.splitlines() if any(text in line for text in
              ('libamdhip64', 'libhsa-runtime', 'libtorch', 'libchen_neural'))]
    assert not imported and not loaded, (imported, loaded)
    return {'torch_neural_modules': imported, 'HIP_Torch_DSO_maps': loaded}


def snapshot(pid, args, label):
    # Diagnostics only; include raw bounded stacks/proc snapshots directly in
    # the fsynced journal and stdout mirror, avoiding host /tmp-only evidence.
    record = {'record': 'parent_context_timeout_snapshot', 'pid': pid, 'label': label}
    for field in ('status', 'maps', 'wchan'):
        try:
            record[field] = Path(f'/proc/{pid}/{field}').read_text()
        except OSError as exc:
            record[field] = str(exc)
    record['threads'] = []
    try:
        for task in Path(f'/proc/{pid}/task').iterdir():
            record['threads'].append({'tid': task.name, 'wchan': (task / 'wchan').read_text()})
    except OSError as exc:
        record['thread_error'] = str(exc)
    if Path(args.gdb).is_file():
        env = os.environ.copy()
        env['LD_LIBRARY_PATH'] = '/tmp/chen-debugger/root/usr/lib/x86_64-linux-gnu'
        try:
            traced = subprocess.run([args.gdb, '--nx', '--batch', '-ex', 'set pagination off',
                '-ex', 'set debuginfod enabled off', '-ex', 'thread apply all bt 25',
                '-ex', 'info sharedlibrary', '-ex', 'detach', '-p', str(pid)],
                env=env, capture_output=True, text=True, timeout=15)
            record.update(gdb_stdout=traced.stdout, gdb_stderr=traced.stderr,
                          gdb_returncode=traced.returncode)
        except subprocess.TimeoutExpired as exc:
            record['gdb_error'] = str(exc)
    else:
        record['gdb_error'] = 'declared debugger absent; proc snapshots retained'
    emit(record)


def install_owned_uci(args):
    global UCI
    base = UCI
    root = Path(args.journal).resolve().parent
    source, launcher = root / 'launcher.c', root / 'chen-ptrace'
    source.write_text('''#define _GNU_SOURCE
#include <sys/prctl.h>
#include <stdlib.h>
#include <unistd.h>
int main(int argc,char **argv){const char *e=getenv("CHEN_DEBUG_ENGINE");
if(argc<2||!e)return 71;if(prctl(PR_SET_PTRACER,PR_SET_PTRACER_ANY,0,0,0))return 71;
argv[0]=(char*)e;execv(e,argv);return 71;}
''')
    subprocess.run(['gcc', '-O2', str(source), '-o', str(launcher)], check=True)

    class OwnedUCI(base):
        def __init__(self, binary, environment, evaluator, backend='hip-resident', order='neural-first'):
            self.timed_out = False
            marked = environment.copy();marked['CHEN_DEBUG_ENGINE'] = binary
            try:
                super().__init__(str(launcher), marked, evaluator, backend, order)
            except BaseException:
                if hasattr(self, 'process'):
                    self.retire()
                raise
            OWNED_CHILDREN.append(self)
            observed = None
            for _ in range(3):
                try:
                    entries = Path(f'/proc/{self.process.pid}/environ').read_bytes().split(b'\0')
                    values = dict(entry.decode(errors='replace').split('=', 1)
                                  for entry in entries if b'=' in entry)
                    if 'AMD_DIRECT_DISPATCH' in values:
                        observed = {key: values.get(key) for key in
                                    ('AMD_DIRECT_DISPATCH', 'CHEN_NEURAL_EXECUTION', 'LD_PRELOAD')}
                        break
                except OSError:
                    pass
                time.sleep(.01)
            emit({'record': 'parent_context_UCI_ready', 'pid': self.process.pid,
                  'evaluator': evaluator, 'configuration': self.configuration,
                  'child_environment': observed,
                  'environment_observation': 'available' if observed is not None else 'unavailable'})
            if observed is not None:
                assert observed['AMD_DIRECT_DISPATCH'] == '1'
                assert observed['CHEN_NEURAL_EXECUTION'] == 'eager'
                assert not observed['LD_PRELOAD']

        def retire(self):
            if self.process.poll() is None:
                self.process.terminate()
                try:
                    self.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    self.process.kill();self.process.wait(timeout=3)
            emit({'record': 'parent_context_owned_child_retired', 'pid': self.process.pid,
                  'returncode': self.process.returncode, 'stderr': list(getattr(self, 'errors', []))})
            if self in OWNED_CHILDREN:
                OWNED_CHILDREN.remove(self)

        def wait(self, prefix, timeout=30):
            lines, deadline = [], time.monotonic() + timeout
            while time.monotonic() < deadline:
                try:
                    line = self.lines.get(timeout=max(.01, deadline - time.monotonic()))
                except queue.Empty:
                    break
                lines.append(line)
                if line.startswith(prefix):
                    return lines
            self.timed_out = True
            emit({'record': 'parent_context_UCI_timeout', 'waiting_for': prefix,
                  'diagnostic': dict(self.diagnostic), 'stdout': lines, 'stderr': list(self.errors)})
            snapshot(self.process.pid, args, 'first')
            time.sleep(.25)
            snapshot(self.process.pid, args, 'second')
            raise AssertionError(('UCI timeout', prefix, self.process.pid))

        def close(self):
            try:
                if self.timed_out:
                    self.retire()
                else:
                    try:
                        super().close()
                    finally:
                        if self.process.poll() is None:
                            self.retire()
                        emit({'record': 'parent_context_owned_child_closed', 'pid': self.process.pid,
                              'returncode': self.process.returncode, 'stderr': list(self.errors)})
            finally:
                if self in OWNED_CHILDREN:
                    OWNED_CHILDREN.remove(self)
    UCI = OwnedUCI


def declare(args):
    declaration = {
        'schema_version': 1, 'declared_utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
        'probe_sha256': digest(__file__), 'reference_script_sha256': REFERENCE_SCRIPT_SHA256,
        'helper_source_sha256': HELPER_SOURCE_SHA256, 'game_protocol_sha256': GAME_PROTOCOL_SHA256,
        'artifact_sha256': FIXED_ARTIFACTS, 'model_sha256': FIXED_MODELS,
        'feature_source_sha256': {FEATURE_SOURCE: FEATURE_SHA256},
        'suite_sha256': SUITE_SHA256,
        'schedule': [{'trial': i + 1, 'pair': i // 2 + 1, 'parent': name}
                     for i, name in enumerate(('cpu', 'hip', 'hip', 'cpu', 'cpu', 'hip'))],
        'dispatch': '1', 'watchdog_seconds': 30, 'execution': 'eager', 'game_plies_each': 20,
        'oracle_binding': 'One producer exits before any consumer; one emitted cache SHA is fixed for all six consumers.',
        'durability': 'fsynced local journal plus fsynced host repository results stdout/stderr mirror at every phase',
        'host_transport': {'wall_seconds': 600, 'post_exit_pipe_drain_seconds': 1,
                           'local_TERM_wait_seconds': 3, 'local_KILL_wait_seconds': 3,
                           'transport_failure': 'remote retirement unknown; end study without automatic replay'},
        'stop': 'Exactly six declared consumers; external interruption ends partial study without automatic replay.',
        'causal_limitations': ['HIP arm repeats original GPU oracle work at the same prelude points; CPU arm reads cached answers.',
                              'Parent context, GPU warming/history, prior device state, and elapsed prelude time are not separable by this control.',
                              'Finite exploratory trials do not estimate failure rate, strength, or a production workaround.'],
    }
    path = Path(args.declaration)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('x') as stream:
        json.dump(declaration, stream, indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    emit({'record': 'parent_context_declaration', 'path': str(path), 'sha256': digest(path),
          'declaration': declaration})


def consumer(args):
    assert os.environ.get('AMD_DIRECT_DISPATCH') == '1'
    assert not os.environ.get('LD_PRELOAD')
    os.environ['CHEN_NEURAL_EXECUTION'] = 'eager'
    declaration = json.loads(Path(args.declaration).read_text())
    assert declaration['probe_sha256'] == digest(__file__)
    scheduled = declaration['schedule'][args.trial - 1]
    assert scheduled['trial'] == args.trial and scheduled['parent'] == args.parent
    assert declaration['dispatch'] == '1' and declaration['watchdog_seconds'] == 30
    assert digest(args.oracle) == args.oracle_sha
    cache = json.loads(Path(args.oracle).read_text())
    assert cache['source_sha256'] == digest(__file__)
    assert cache['helper_source_sha256'] == HELPER_SOURCE_SHA256
    assert process_identity(cache['producer']['pid']) != cache['producer'], 'producer still alive'
    provenance = files(args)
    assert cache['artifact_sha256'] == provenance
    observed = [{'name': name, 'fen': board.fen(en_passant='fen')} for name, board in fixtures(args)]
    assert observed == cache['fixtures']
    runtime = torch_runtime() if args.parent == 'hip' else None
    cpu_guard = check_cpu_parent() if runtime is None else None
    emit({'record': 'parent_context_trial_start', 'trial': scheduled, 'oracle_sha256': args.oracle_sha,
          'artifact_sha256': provenance, 'parent_pid': os.getpid(), 'producer_retired': True,
          'cpu_parent_guard': cpu_guard, 'requested_dispatch': '1',
          'parent_environment': {key: os.environ.get(key) for key in
              ('AMD_DIRECT_DISPATCH', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES',
               'CUDA_VISIBLE_DEVICES', 'PYTORCH_HIP_ALLOC_CONF', 'PYTORCH_CUDA_ALLOC_CONF',
               'LD_PRELOAD', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS')},
          'parent_torch_version': str(runtime.__version__) if runtime is not None else None,
          'parent_torch_hip_version': runtime.version.hip if runtime is not None else None,
          'mode': 'eager', 'cached_oracle_role': 'comparison only; engine still loads unchanged GPU model',
          'load_average': os.getloadavg()})
    install_owned_uci(args)
    comparisons, leaves = 0, 0
    model_paths = [Path(args.models) / f'{name}-inference.pt' for name in ('square', 'relation', 'plan')]
    for archive in model_paths:
        cached_model = cache['models'][archive.name]
        module = runtime.jit.load(str(archive), map_location='cuda:0').eval() if runtime is not None else None
        command = [args.engine, 'bench', '--backend', 'hip-resident', '--eval', 'neural', '--depth', '2',
                   '--max-positions', '50000', '--suite', args.suite]
        emit({'record': 'parent_context_benchmark_begin', 'model': archive.name, 'command': command})
        started = time.monotonic()
        try:
            completed = subprocess.run(command, env=environment(args, archive), capture_output=True,
                                       text=True, timeout=90)
        except subprocess.TimeoutExpired as exc:
            def decode(value):
                return value.decode(errors='replace') if isinstance(value, bytes) else value
            emit({'record': 'parent_context_benchmark_timeout', 'model': archive.name,
                  'timeout_seconds': 90, 'stdout': decode(exc.stdout), 'stderr': decode(exc.stderr),
                  'child_retired_by_subprocess_run': True})
            raise
        emit({'record': 'parent_context_benchmark_raw', 'model': archive.name,
              'returncode': completed.returncode, 'elapsed_wall_seconds': time.monotonic() - started,
              'stdout': completed.stdout, 'stderr': completed.stderr})
        assert completed.returncode == 0
        actual_rows = [json.loads(line) for line in completed.stdout.splitlines()]
        assert len(actual_rows) == 6
        for (name, board), actual in zip(fixtures(args), actual_rows):
            assert actual['backend'] == 'hip-resident' and actual['eval'] == 'neural' and not actual['stopped']
            assert actual['evaluator_artifact_sha256'] == cached_model['archive_sha256']
            assert actual['neural_execution']['mode'] == 'eager'
            assert actual['position'] == name
            depth = actual['depth']
            assert depth == 2 or (depth == 1 and abs(actual['score_cp']) >= MATE - 128)
            expected = [cached_model['rows'][name][str(d)] for d in range(1, depth + 1)]
            if runtime is not None:
                # Same original prelude work; context remains live but idle in games.
                with runtime.inference_mode():
                    fresh = [oracle(board, d, module, cached_model['max_batch']) for d in range(1, depth + 1)]
                assert fresh == expected, (archive.name, name, 'cache/GPU oracle drift')
            last = expected[-1]
            assert actual['score_cp'] == last['score_cp'] and actual['bestmove'] in last['optimal_moves']
            assert actual['nodes'] == sum(row['visited_positions'] for row in expected)
            assert actual['evaluated_positions'] == sum(row['evaluated_leaves'] for row in expected)
            pv = check_pv(board, actual['pv'], depth)
            assert pv and pv[0] == actual['bestmove']
            comparisons += 1;leaves += sum(row['evaluated_leaves'] for row in expected)
            emit({'record': 'parent_context_exact_comparison', 'model': archive.name, 'position': name,
                  'actual': actual, 'expected': last, 'exact': True, 'parent': args.parent})
        if runtime is not None:
            runtime.cuda.synchronize();del module;runtime.cuda.empty_cache()
        else:
            check_cpu_parent()
    emit({'record': 'parent_context_prelude_complete', 'comparisons': comparisons,
          'iterative_evaluated_leaves': leaves, 'parent': args.parent,
          'parent_maps': Path('/proc/self/maps').read_text()})
    protocol_games(args, model_paths)
    if runtime is None:
        check_cpu_parent()
    assert digest(args.oracle) == args.oracle_sha
    assert all(digest(path) == value for path, value in provenance.items())
    emit({'record': 'parent_context_trial_complete', 'trial': args.trial, 'comparisons': comparisons,
          'game_protocol_plies_requested': 40, 'inputs_unchanged': True, 'parent': args.parent,
          'scope': 'Finite oracle/protocol control, no reliability-rate or strength claim.'})


def mirror(args):
    # These bounds cover process waiting and pipe draining, assuming filesystem
    # writes remain responsive. Local process groups do not own remote Docker
    # exec children: any transport failure ends the study, with retirement unknown.
    assert args.exec_args and args.mirror_stdout and args.mirror_stderr
    assert 0 < args.mirror_wall_seconds <= 600
    check_cpu_parent()
    outputs = [Path(args.mirror_stdout), Path(args.mirror_stderr)]
    for path in outputs:
        assert path.resolve().is_relative_to(Path(args.host_results_root).resolve())
        path.parent.mkdir(parents=True, exist_ok=True)
    with outputs[0].open('xb') as stdout, outputs[1].open('xb') as stderr:
        process = subprocess.Popen(args.exec_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                                   start_new_session=True)
        selector = selectors.DefaultSelector()
        started = time.monotonic()
        deadline = started + args.mirror_wall_seconds
        drain_deadline = None
        try:
            for source, destination in ((process.stdout, stdout), (process.stderr, stderr)):
                os.set_blocking(source.fileno(), False)
                selector.register(source, selectors.EVENT_READ, destination)
            emit({'record': 'parent_context_mirror_start', 'pid': process.pid,
                  'command': args.exec_args, 'stdout_path': str(outputs[0]), 'stderr_path': str(outputs[1]),
                  'wall_deadline_seconds': args.mirror_wall_seconds, 'exit_pipe_drain_seconds': 1,
                  'remote_retirement_guarantee': False})
            while selector.get_map() or process.poll() is None:
                now = time.monotonic()
                if now >= deadline:
                    raise TimeoutError('host transport wall deadline exceeded')
                if process.poll() is not None and selector.get_map():
                    if drain_deadline is None:
                        drain_deadline = now + 1
                    if now >= drain_deadline:
                        raise TimeoutError('output pipes still inherited after transport exit')
                limit = min(deadline, drain_deadline) if drain_deadline is not None else deadline
                for key, _ in selector.select(timeout=min(.1, max(0, limit - now))):
                    try:
                        data = os.read(key.fileobj.fileno(), 65536)
                    except BlockingIOError:
                        continue
                    if data:
                        key.data.write(data);key.data.flush();os.fsync(key.data.fileno())
                    else:
                        selector.unregister(key.fileobj)
                        key.fileobj.close()
            code = process.poll()
        except BaseException as exc:
            # Kill the local group even if its original leader has already
            # exited: a descendant may still own a pipe. No unbounded wait/join.
            for action in (signal.SIGTERM, signal.SIGKILL):
                try:
                    os.killpg(process.pid, action)
                except ProcessLookupError:
                    pass
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    continue
            emit({'record': 'parent_context_mirror_transport_failed', 'exception': repr(exc),
                  'elapsed_wall_seconds': time.monotonic() - started,
                  'local_transport_returncode': process.poll(), 'remote_retirement': 'unknown',
                  'local_transport_reaped': process.poll() is not None,
                  'open_pipe_count_at_failure': len(selector.get_map()),
                  'remaining_output_may_be_discarded': True,
                  'study_end_required': True, 'automatic_replay': False,
                  'local_group_TERM_KILL_sent': True})
            raise
        finally:
            selector.close()
            process.stdout.close();process.stderr.close()
        emit({'record': 'parent_context_mirror_end', 'returncode': code,
              'elapsed_wall_seconds': time.monotonic() - started,
              'remote_retirement_guarantee': False})
        assert code == 0, ('remote command failed; inspect durable consumer retirement journal', code)


def main():
    global JOURNAL
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('command', choices=('declare', 'produce', 'consume', 'mirror'))
    p.add_argument('--engine', default='/tmp/chen-graph-plugin/chen')
    p.add_argument('--plugin', default='/tmp/chen-graph-plugin/libchen_neural.so')
    p.add_argument('--rules', default='/tmp/chen-gpu/build/libchen_gpu.so')
    p.add_argument('--models', default='/tmp/chen-neural-inference/trained')
    p.add_argument('--suite', default='/tmp/chen-native-plugin/benches/positions.txt')
    p.add_argument('--journal', required=True)
    p.add_argument('--declaration')
    p.add_argument('--oracle')
    p.add_argument('--oracle-sha')
    p.add_argument('--trial', type=int)
    p.add_argument('--parent', choices=('cpu', 'hip'))
    p.add_argument('--game-plies', type=int, default=20)
    p.add_argument('--gdb', default='/tmp/chen-debugger/root/usr/bin/gdb')
    p.add_argument('--mirror-stdout')
    p.add_argument('--mirror-stderr')
    p.add_argument('--host-results-root', default='results')
    p.add_argument('--mirror-wall-seconds', type=float, default=600)
    p.add_argument('--exec', dest='exec_args', nargs=argparse.REMAINDER)
    args = p.parse_args()
    assert args.engine == '/tmp/chen-graph-plugin/chen'
    assert args.plugin == '/tmp/chen-graph-plugin/libchen_neural.so'
    assert args.rules == '/tmp/chen-gpu/build/libchen_gpu.so'
    assert args.game_plies == 20
    if args.command == 'declare':
        assert args.declaration
    elif args.command in ('produce', 'consume'):
        assert args.oracle
    if args.command == 'consume':
        assert args.declaration and args.oracle_sha and args.trial and args.parent
    path = Path(args.journal)
    path.parent.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    with path.open('x') as stream:
        JOURNAL = stream
        try:
            {'declare': declare, 'produce': producer, 'consume': consumer, 'mirror': mirror}[args.command](args)
        except BaseException as exc:
            emit({'record': 'parent_context_command_failed', 'command': args.command,
                  'exception': repr(exc), 'elapsed_wall_seconds': time.monotonic() - started})
            raise
        else:
            emit({'record': 'parent_context_command_complete', 'command': args.command,
                  'elapsed_wall_seconds': time.monotonic() - started})
        finally:
            for child in list(reversed(OWNED_CHILDREN)):
                try:
                    child.close()
                except BaseException as exc:
                    emit({'record': 'parent_context_cleanup_error', 'exception': repr(exc)})
                    child.retire()
            JOURNAL = None


if __name__ == '__main__':
    main()
