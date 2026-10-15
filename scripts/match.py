#!/usr/bin/env python3
"""Sequential paired offline UCI matches with exact configs, PGNs and failure logs."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time

import chess
import chess.engine
import chess.pgn
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from harness.stats import summarize_pairs
from harness.opening_book import SharedOpeningBook


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def opening_rows(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, fen = line.split('|', 1)
        board = chess.Board(fen.strip())
        if not board.is_valid() or board.is_game_over(claim_draw=True):
            raise ValueError(f'invalid or terminal opening: {name}')
        rows.append((name.strip(), board.fen(en_passant='fen')))
    if not rows:
        raise ValueError('opening suite is empty')
    return rows


def play(engines, names, opening, candidate_color, limit, max_plies, book=None, decision_info=False):
    board = chess.Board(opening[1])
    game = chess.pgn.Game()
    game.setup(board)
    game.headers.update({'Event': 'Chen offline paired experiment', 'Site': 'local', 'Date': time.strftime('%Y.%m.%d'), 'White': names[0], 'Black': names[1], 'Opening': opening[0]})
    node = game
    failure = None
    completed = False
    move_trace = []
    for ply in range(max_plies):
        outcome = board.outcome(claim_draw=True)
        if outcome is not None:
            completed = True
            break
        try:
            engine_index = 0 if board.turn == chess.WHITE else 1
            initial_fen = board.fen(en_passant='fen')
            started = time.perf_counter_ns()
            book_move = book.choose(board) if book is not None else None
            if book_move is not None:
                result = chess.engine.PlayResult(book_move, None, info={'depth': 0, 'nodes': 0, 'time': 0.0})
            else:
                result = engines[engine_index].play(board, limit, game=game,
                                                    info=chess.engine.INFO_BASIC | (chess.engine.INFO_PV if decision_info else 0))
            elapsed_ns = time.perf_counter_ns() - started
            if result.move not in board.legal_moves:
                raise ValueError(f'illegal or missing move: {result.move}')
            move_trace.append({'ply': ply, 'engine': names[engine_index],
                               'fen': initial_fen, 'move': result.move.uci(),
                               'wall_ns': elapsed_ns,
                               'book': book_move is not None,
                               'reported': {key: result.info[key] for key in
                                            ('depth', 'seldepth', 'nodes', 'nps', 'time', 'hashfull', 'tbhits')
                                            if key in result.info}})
            if decision_info and book_move is None:
                pv = result.info.get('pv', [])
                move_trace[-1]['decision'] = {
                    'searched_move': pv[0].uci() if pv else None,
                    'diagnostic': result.info.get('string'),
                }
            board.push(result.move)
            node = node.add_variation(result.move)
            if book_move is not None:
                node.comment = 'Shared opening book'
        except Exception as error:
            failure = f'{type(error).__name__}: {error}'
            break
    outcome = board.outcome(claim_draw=True)
    if outcome is not None:
        completed = True
    if completed:
        result = outcome.result()
        score = 0.5 if outcome.winner is None else float(outcome.winner == candidate_color)
        reason = outcome.termination.name
    else:
        result = '*'
        score = None
        reason = 'technical_failure' if failure else 'ply_cap_inconclusive'
    game.headers['Result'] = result
    game.headers['Termination'] = reason
    return game, {'opening': opening[0], 'candidate_color': 'white' if candidate_color else 'black', 'result': result, 'candidate_score': score, 'plies': board.ply() - chess.Board(opening[1]).ply(), 'termination': reason, 'failure': failure, 'final_fen': board.fen(en_passant='fen'), 'move_trace': move_trace}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True, help='JSON engines base/candidate: command(list), options(dict), env(dict optional), cwd(optional)')
    p.add_argument('--openings', default='benches/openings.txt')
    p.add_argument('--rounds', type=int, default=1, help='repeat each opening with swapped colors per round')
    p.add_argument('--depth', type=int)
    p.add_argument('--nodes', type=int)
    p.add_argument('--movetime', type=float, help='seconds per move')
    p.add_argument('--max-plies', type=int, default=300)
    p.add_argument('--independent-opening-pairs', action='store_true',
                   help='assert an independent opening-pair sampling design before emitting a bootstrap interval')
    p.add_argument('--out', required=True)
    p.add_argument('--book-dir', help='use the same controller-supplied ECO TSV book for both engines')
    p.add_argument('--book-plies', type=int, default=12)
    p.add_argument('--book-seed', type=int, default=0)
    p.add_argument('--decision-info', action='store_true', help='retain the searched root move and final UCI diagnostic')
    args = p.parse_args()
    if sum(v is not None for v in (args.depth, args.nodes, args.movetime)) != 1:
        p.error('choose exactly one depth, nodes, or movetime budget')
    if args.rounds < 1 or args.max_plies < 1 or any(v is not None and v <= 0 for v in (args.depth, args.nodes, args.movetime)):
        p.error('budgets, rounds and max-plies must be positive')
    config = json.loads(Path(args.config).read_text())
    book = SharedOpeningBook(args.book_dir, args.book_plies, args.book_seed) if args.book_dir else None
    openings = opening_rows(args.openings)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    limit = chess.engine.Limit(depth=args.depth, nodes=args.nodes, time=args.movetime)
    metadata = {'config': config, 'budgets': vars(args), 'opening_sha256': sha(args.openings), 'chess_version': chess.__version__, 'platform': platform.platform(), 'concurrency': 1, 'launcher_sha256': {name: sha(entry['command'][0]) for name, entry in config.items()}, 'declared_artifact_sha256': {name: {label: sha(path) for label, path in entry.get('artifacts', {}).items()} for name, entry in config.items()}, 'artifact_scope': {name: 'declared-files' if entry.get('artifacts') else 'launcher-only' for name, entry in config.items()}, 'rating_target': '2500 Lichess in a declared time-control pool; this offline match does not certify it'}
    metadata['inherited_runtime_environment'] = {key: os.environ[key] for key in ['CHEN_GPU_LIBRARY', 'CHEN_NEURAL_MODEL', 'CHEN_NEURAL_LIBRARY', 'CHEN_NEURAL_HIP_RUNTIME', 'CHEN_NEURAL_EXECUTION', 'CHEN_NEURAL_GRAPH_SHAPES', 'CHEN_NEURAL_GRAPH_BYTES', 'HSA_OVERRIDE_GFX_VERSION', 'HIP_VISIBLE_DEVICES', 'ROCR_VISIBLE_DEVICES', 'CUDA_VISIBLE_DEVICES', 'PYTORCH_ROCM_ARCH', 'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS'] if key in os.environ}
    metadata['match_script_sha256'] = sha(__file__)
    metadata['opening_book'] = book.metadata() if book is not None else None
    (out / 'metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    engines = {}
    try:
        for name in ['base', 'candidate']:
            entry = config[name]
            environment = {**os.environ, **entry.get('env', {})}
            if entry.get('capture_rocm_artifacts', False):
                environment['CHEN_ARTIFACT_DIR'] = str((out / 'artifacts' / name).resolve())
            engines[name] = chess.engine.SimpleEngine.popen_uci(entry['command'], timeout=30, cwd=entry.get('cwd'), env=environment)
            engines[name].configure(entry.get('options', {}))
        pairs = []
        failures = 0
        incomplete_pairs = 0
        with (out / 'games.pgn').open('w') as pgn, (out / 'games.jsonl').open('w') as log:
            for round_index in range(args.rounds):
                for opening in openings:
                    pair = []
                    for color in [chess.WHITE, chess.BLACK]:
                        order = ['candidate', 'base'] if color == chess.WHITE else ['base', 'candidate']
                        game, row = play([engines[name] for name in order], order, opening, color, limit, args.max_plies, book, args.decision_info)
                        row['round'] = round_index
                        pgn.write(str(game) + '\n\n'); pgn.flush()
                        log.write(json.dumps(row) + '\n'); log.flush()
                        pair.append(row['candidate_score'])
                        failures += row['failure'] is not None
                        if row['failure']:
                            raise RuntimeError(row['failure'])
                    if all(score is not None for score in pair):
                        pairs.append(tuple(pair))
                    else:
                        incomplete_pairs += 1
        summary = summarize_pairs(pairs, independent_pairs=args.independent_opening_pairs) if pairs else {'completed_pairs': 0, 'rating_scope': 'no strength inference: no complete pairs'}
        summary['independent_opening_pairs_asserted'] = args.independent_opening_pairs
        summary.update({'incomplete_pairs': incomplete_pairs, 'technical_failures': failures})
        (out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        print(json.dumps({'results': str(out), **summary}, indent=2))
    finally:
        for engine in engines.values():
            try:
                engine.quit()
            except Exception:
                engine.close()


if __name__ == '__main__':
    main()
