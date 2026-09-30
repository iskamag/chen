#!/usr/bin/env python3
"""Independent replay of the archived 2026-10-01 offline reference evidence."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import chess
import chess.pgn


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def verify(directory):
    directory = Path(directory)
    read = lambda name: json.loads((directory / name).read_text())
    metadata = read('reference-metadata.json')
    old = read('reference-summary-original.json')
    openings = read('reference-openings.json')
    rows = [json.loads(line) for line in
            (directory / 'reference-games.jsonl').read_text().splitlines() if line.strip()]
    require(len(openings) == 32 and len(rows) == 64, 'expected 32 openings / 64 games')
    require(metadata['budgets']['nodes'] == 2000 and
            metadata['budgets']['depth'] is None and metadata['budgets']['movetime'] is None and
            metadata['budgets']['rounds'] == 1 and metadata['budgets']['max_plies'] == 600,
            'unexpected recorded workload')
    require(metadata['opening_sha256'] == sha(directory / 'reference-openings.txt'),
            'opening hash mismatch')
    names = set()
    fen_by_name = {}
    for opening in openings:
        name = opening['id']
        require(name not in names, 'duplicate opening ID')
        names.add(name)
        board = chess.Board()
        for uci in opening['uci_moves']:
            move = chess.Move.from_uci(uci)
            require(move in board.legal_moves, f'illegal opening move {name}: {uci}')
            board.push(move)
        require(board.fen(en_passant='fen') == opening['fen'], f'opening FEN mismatch {name}')
        require(board.is_valid() and board.outcome(claim_draw=True) is None,
                f'invalid/terminal opening {name}')
        fen_by_name[name] = opening['fen']
    txt_rows = [line.split('|', 1) for line in
                (directory / 'reference-openings.txt').read_text().splitlines()
                if line.strip() and not line.lstrip().startswith('#')]
    require([(name.strip(), fen.strip()) for name, fen in txt_rows] ==
            [(o['id'], o['fen']) for o in openings], 'opening text/JSON disagreement')
    records = []
    outcomes = Counter()
    terminations = Counter()
    pairs = {}
    total_plies = 0
    with (directory / 'reference-games.pgn').open() as stream:
        for index, row in enumerate(rows):
            game = chess.pgn.read_game(stream)
            require(game is not None and not game.errors, f'PGN missing/parser errors {index}')
            require(row['opening'] == openings[index // 2]['id'], f'opening order mismatch {index}')
            color = 'white' if index % 2 == 0 else 'black'
            require(row['candidate_color'] == color and row['round'] == 0,
                    f'color/round schedule mismatch {index}')
            white, black = ('candidate', 'base') if color == 'white' else ('base', 'candidate')
            for field, expected in {'White': white, 'Black': black,
                                    'Opening': row['opening'], 'Result': row['result'],
                                    'Termination': row['termination'],
                                    'Date': '2026.10.01', 'SetUp': '1',
                                    'Event': 'Chen offline paired experiment', 'Site': 'local'}.items():
                require(game.headers.get(field) == expected, f'PGN {field} mismatch {index}')
            board = game.board()
            # Game.setup serializes legally effective EP; the JSON opening stores
            # raw EP. Only EP may differ, and both must reconstruct the same rules.
            expected = chess.Board(fen_by_name[row['opening']])
            require(board.fen(en_passant='legal') == expected.fen(en_passant='legal'),
                    f'initial PGN/opening state mismatch {index}')
            require(board.is_valid(), f'invalid initial board {index}')
            require(row['failure'] is None, f'technical failure {index}')
            moves = list(game.mainline_moves())
            for ply, move in enumerate(moves):
                require(board.outcome(claim_draw=True) is None,
                        f'move after available termination game {index} ply {ply}')
                require(move in board.legal_moves, f'illegal move game {index} ply {ply}')
                board.push(move)
                require(board.is_valid(), f'invalid resulting board game {index} ply {ply}')
            outcome = board.outcome(claim_draw=True)
            require(outcome is not None, f'incomplete game {index}')
            require(outcome.result() == row['result'] and
                    outcome.termination.name == row['termination'],
                    f'final result/termination mismatch {index}')
            require(board.fen(en_passant='fen') == row['final_fen'], f'final FEN mismatch {index}')
            require(len(moves) == row['plies'] and len(moves) <= 600, f'ply count mismatch {index}')
            candidate_won = outcome.winner == (color == 'white') if outcome.winner is not None else False
            score = .5 if outcome.winner is None else float(candidate_won)
            require(score == row['candidate_score'], f'candidate score mismatch {index}')
            key = (row['round'], row['opening'])
            pair = pairs.setdefault(key, {})
            require(color not in pair, f'duplicate pair color {index}')
            pair[color] = score
            outcomes['draws' if score == .5 else 'wins' if score == 1 else 'losses'] += 1
            terminations[outcome.termination.name] += 1
            total_plies += len(moves)
            records.append({'index': index, 'opening': row['opening'], 'candidate_color': color,
                            'plies': len(moves), 'result': outcome.result(),
                            'termination': outcome.termination.name,
                            'final_fen': board.fen(en_passant='fen')})
        require(chess.pgn.read_game(stream) is None, 'extra PGN games')
    require(len(pairs) == 32 and all(set(pair) == {'white', 'black'} for pair in pairs.values()),
            'incomplete color-swapped pairs')
    pentanomial = [0] * 5
    for pair in pairs.values():
        pentanomial[round(2 * (pair['white'] + pair['black']))] += 1
    score = sum(row['candidate_score'] for row in rows) / len(rows)
    for field, expected in {'completed_pairs': 32, 'games': 64, 'pentanomial': pentanomial,
                            'candidate_score': score, 'incomplete_pairs': 0,
                            'technical_failures': 0}.items():
        require(old[field] == expected, f'original summary {field} mismatch')
    require(outcomes == Counter(losses=61, draws=3), 'unexpected W/D/L total')
    return {'kind': 'independent_python_chess_reference_replay', 'date': '2026-10-01',
            'chess_version': chess.__version__, 'games': 64, 'completed_pairs': 32,
            'legal_replayed_plies': total_plies, 'candidate_wins': outcomes['wins'],
            'candidate_draws': outcomes['draws'], 'candidate_losses': outcomes['losses'],
            'candidate_score': score, 'pentanomial': pentanomial,
            'termination_counts': dict(terminations), 'technical_failures': 0,
            'incomplete_pairs': 0, 'all_headers_results_final_fens_match': True,
            'no_moves_after_available_draw_or_terminal_outcome': True,
            'bootstrap_interval_accepted_for_inference': None,
            'elo_inference': None, 'lichess_rating_inference': None, 'sprt_completed': False,
            'scope': 'Replay and fixed-workload descriptive outcomes; authored correlated openings',
            'input_sha256': {name: sha(directory / name) for name in
                ['reference-games.pgn', 'reference-games.jsonl', 'reference-metadata.json',
                 'reference-summary-original.json', 'reference-openings.json', 'reference-openings.txt']},
            'verifier_sha256': sha(__file__), 'games_verified': records}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = verify(args.directory)
    text = json.dumps(report, indent=2) + '\n'
    if args.output:
        args.output.write_text(text)
    print(json.dumps({k: v for k, v in report.items() if k != 'games_verified'}, indent=2))


if __name__ == '__main__':
    main()
