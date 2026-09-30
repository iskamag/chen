#!/usr/bin/env python3
"""Independent python-chess minimax + same Torch GPU artifact versus Rust search.

CPU board expansion/feature ingestion and score inspection are oracle boundaries,
not the deployed neural search path. Legal games prove protocol behavior only.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import tempfile
import threading
import time

import chess
import torch
from neural.features import encode_fen

MATE = 30000


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def emit(record):
    print(json.dumps(record), flush=True)


def terminal(board, ply):
    moves = list(board.legal_moves)
    if not moves:
        return (-MATE + ply if board.is_check() else 0), moves
    if board.halfmove_clock >= 100 or board.is_insufficient_material() or board.is_repetition(3):
        return 0, moves
    return None, moves


@torch.inference_mode()
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", default="/tmp/chen-native-plugin/chen")
    parser.add_argument("--plugin", default="/tmp/chen-native-plugin/libchen_neural.so")
    parser.add_argument("--rules", default="/tmp/chen-gpu/build/libchen_gpu.so")
    parser.add_argument("--models", default="/tmp/chen-neural-inference/trained")
    parser.add_argument("--suite", default="/tmp/chen-native-plugin/benches/positions.txt")
    parser.add_argument("--game-plies", type=int, default=20)
    parser.add_argument("--rust-source", default="/tmp/chen-neural-search-source")
    parser.add_argument("--isolate-games", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.version.hip
    assert not os.environ.get('LD_PRELOAD'), 'Final startup check must not rely on LD_PRELOAD'
    assert "gfx1030" in torch.cuda.get_device_properties(0).gcnArchName
    torch.set_num_threads(1)
    if args.isolate_games:
        model = Path(args.models) / 'square-inference.pt'
        env = environment(args, model)
        board = chess.Board()
        for text in ('e2e4', 'e7e5', 'g1f3', 'b8c6'):
            board.push_uci(text)
        for label, first, second in (('neural-alone', 'neural', None), ('tapered-alone', 'tapered', None),
                                     ('tapered-then-neural', 'tapered', 'neural'), ('neural-then-tapered', 'neural', 'tapered')):
            clients = []
            emit({'record': 'neural_game_isolation_begin', 'case': label, 'fen': board.fen(en_passant='fen'),
                  'engine_sha256': digest(args.engine), 'model_sha256': digest(model)})
            try:
                clients.append(UCI(args.engine, env, first))
                if second:
                    clients.append(UCI(args.engine, env, second))
                for client in clients:
                    client.diagnostic['isolation_case'] = label
                    move, search = client.search(board, 'go depth 2 nodes 20000')
                    emit({'record': 'neural_game_isolation_search', 'case': label, 'evaluator': client.diagnostic['evaluator'],
                          'configuration': client.configuration, 'move': move.uci(), 'search': search})
            finally:
                for client in reversed(clients):
                    client.close()
        return
    sources = [Path(__file__), Path(args.engine), Path(args.plugin), Path(args.plugin + '.runtime'), Path(args.rules), Path(args.suite)]
    sources += sorted(Path('/tmp/chen-native-plugin/neural').glob('*.py'))
    sources += sorted(Path('/tmp/chen-native-plugin/gpu').glob('chen_neural.*'))
    sources += [Path('/tmp/chen-native-plugin/gpu/chen_neural_decode.hip'), Path('/tmp/chen-gpu/chen_rules.hip')]
    sources += [Path(args.rust_source) / 'main.rs', Path(args.rust_source) / 'neural.rs',
                Path(torch.__file__).parent / 'lib/libamdhip64.so']
    provenance = {str(path): digest(path) for path in sources if path.is_file()}
    emit({"record": "neural_search_provenance", "target": "gfx1030", "torch_version": torch.__version__,
          "torch_hip_version": torch.version.hip, "artifacts_sha256": provenance,
          "inspection_scope": "CPU rules/feature ingestion and Tensor payload reads are independent oracle diagnostics only."})
    fixtures = [(name.strip(), chess.Board(fen.strip())) for line in Path(args.suite).read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#') for name, fen in [line.split('|', 1)]]
    assert len(fixtures) == 6
    total, exact, evaluated = 0, 0, 0
    model_paths = [Path(args.models) / f'{name}-inference.pt' for name in ('square', 'relation', 'plan')]
    for model in model_paths:
        envelope = json.loads(Path(str(model) + '.json').read_text())
        artifact_sha = digest(model)
        assert artifact_sha == envelope['archive_sha256']
        module = torch.jit.load(str(model), map_location='cuda:0').eval()
        max_batch = envelope['manifest']['max_batch']
        assert 1 <= max_batch <= 256
        emit({"record": "neural_search_model", "model": model.stem, "archive_sha256": artifact_sha,
              "sidecar_sha256": digest(str(model) + '.json'), "max_batch": max_batch,
              "manifest": envelope['manifest']})
        command = [args.engine, 'bench', '--backend', 'hip-resident', '--eval', 'neural', '--depth', '2',
                   '--max-positions', '50000', '--suite', args.suite]
        completed = subprocess.run(command, env=environment(args, model), check=True, capture_output=True, text=True, timeout=90)
        native = [json.loads(line) for line in completed.stdout.splitlines()]
        assert len(native) == len(fixtures)
        for (name, board), actual in zip(fixtures, native):
            assert actual['backend'] == 'hip-resident' and actual['eval'] == 'neural' and not actual['stopped']
            assert actual['evaluator_artifact_sha256'] == artifact_sha and actual['position'] == name
            depth = actual['depth']
            assert depth == 2 or (depth == 1 and abs(actual['score_cp']) >= MATE - 128)
            answers = [oracle(board, current, module, max_batch) for current in range(1, depth + 1)]
            expected = answers[-1]
            equal = actual['score_cp'] == expected['score_cp']
            optimal = actual['bestmove'] in expected['optimal_moves']
            expected_nodes = sum(answer['visited_positions'] for answer in answers)
            expected_evals = sum(answer['evaluated_leaves'] for answer in answers)
            pv = check_pv(board, actual['pv'], depth)
            assert pv and pv[0] == actual['bestmove']
            emit({"record": "neural_search_comparison", "model": model.stem, "position": name,
                  "requested_depth": 2, "completed_depth": depth, "Rust": actual, "oracle": expected,
                  "exact_score": equal, "best_in_optimal_tie_set": optimal,
                  "oracle_single_iteration_nodes": expected['visited_positions'],
                  "oracle_iterative_nodes": expected_nodes, "oracle_iterative_evaluated_leaves": expected_evals,
                  "iterative_nodes_match": actual['nodes'] == expected_nodes,
                  "evaluated_leaves_match": actual['evaluated_positions'] == expected_evals,
                  "PV_legal": True})
            assert equal and optimal, (model, name, actual, expected)
            assert actual['nodes'] == expected_nodes and actual['evaluated_positions'] == expected_evals
            total += 1
            exact += int(equal)
            evaluated += expected_evals
        del module
        torch.cuda.empty_cache()
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
    unchanged = all(digest(path) == sha for path, sha in provenance.items())
    assert unchanged, 'Inputs changed during oracle execution'
    emit({"record": "neural_search_summary", "model_position_comparisons": total, "exact_scores": exact,
          "iterative_oracle_evaluated_leaves": evaluated, "inputs_unchanged": unchanged,
          "limitations": ['Finite six-position/two-ply oracle, not exhaustive search verification.',
                          'Optimal tie set compared; python-chess and Rust move orders differ.',
                          'Benchmark times include cold model/runtime work and are not steady-state speed claims.',
                          'Inherited resident h2d_ms/d2h_ms are unmeasured zero placeholders.',
                          'Legal games and UCI limits do not establish playing strength or 2500 Lichess.']})


if __name__ == '__main__':
    main()
