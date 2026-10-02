#!/usr/bin/env python3
"""Independently replay a closed paired match and its per-move timing witnesses."""
import argparse
import hashlib
import json
import math
from pathlib import Path

import chess
import chess.pgn


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(run, declaration_path):
    declaration = json.loads(declaration_path.read_bytes())
    require(declaration['kind'] == 'fixed_1second_UCI_reference_diagnostic', 'unknown declaration')
    require(declaration['requested_seconds_per_move'] == 1.0, 'unexpected budget')
    for name, expected in declaration['source_and_binary_sha256'].items():
        require(hashlib.sha256(Path(name).read_bytes()).hexdigest() == expected, 'changed source/artifact: ' + name)
    metadata = json.loads((run / 'metadata.json').read_bytes())
    require(metadata['budgets']['movetime'] == 1.0 and metadata['concurrency'] == 1, 'collector budget differs')
    require(metadata['budgets']['depth'] is None and metadata['budgets']['nodes'] is None, 'mixed budgets')
    require(metadata['config']['base']['options']['UCI_Elo'] == 1320 and
            metadata['config']['base']['options']['UCI_LimitStrength'] is True, 'opponent setting differs')
    openings = []
    for line in Path(metadata['budgets']['openings']).read_text().splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            name, fen = line.split('|', 1)
            openings.append((name.strip(), fen.strip()))
    require(len(openings) == declaration['opening_pairs'], 'opening count differs')
    rows = [json.loads(line) for line in (run / 'games.jsonl').read_text().splitlines()]
    require(len(rows) == 2 * len(openings), 'closed match must contain both colors for every opening')
    games = []
    with (run / 'games.pgn').open() as handle:
        while (game := chess.pgn.read_game(handle)) is not None:
            require(not game.errors, 'PGN parse error')
            games.append(game)
    require(len(games) == len(rows), 'PGN/log inventory differs')
    walls = {'candidate': [], 'base': []}
    scores = []
    total_plies = 0
    for index, (row, game) in enumerate(zip(rows, games)):
        name, fen = openings[index // 2]
        candidate_color = chess.WHITE if index % 2 == 0 else chess.BLACK
        require(row['opening'] == name and row['candidate_color'] == ('white' if candidate_color else 'black'), 'pair order differs')
        require(row['failure'] is None and row['round'] == 0, 'technical failure or unexpected round')
        require(game.headers['White'] == ('candidate' if candidate_color else 'base') and
                game.headers['Black'] == ('base' if candidate_color else 'candidate'), 'engine colors differ')
        board = chess.Board(fen)
        require(board.fen(en_passant='legal') == game.board().fen(en_passant='legal'), 'initial legal rule state differs')
        moves = list(game.mainline_moves())
        require(len(moves) == row['plies'] == len(row['move_trace']), 'missing move witness')
        for ply, (move, trace) in enumerate(zip(moves, row['move_trace'])):
            require(board.outcome(claim_draw=True) is None, 'moves after rule endpoint')
            engine = 'candidate' if board.turn == candidate_color else 'base'
            require(trace['ply'] == ply and trace['engine'] == engine and
                    trace['fen'] == board.fen(en_passant='fen') and trace['move'] == move.uci(), 'move trace identity differs')
            require(type(trace['wall_ns']) is int and trace['wall_ns'] >= 0, 'invalid move wall timing')
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in trace['reported'].values()), 'invalid UCI telemetry')
            require(move in board.legal_moves, 'illegal move')
            walls[engine].append(trace['wall_ns'] / 1e6)
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        require(outcome is not None, 'inconclusive game')
        result = outcome.result()
        score = .5 if outcome.winner is None else float(outcome.winner == candidate_color)
        require(row['result'] == game.headers['Result'] == result and row['candidate_score'] == score, 'result/score differs')
        require(row['termination'] == game.headers['Termination'] == outcome.termination.name, 'endpoint differs')
        require(row['final_fen'] == board.fen(en_passant='fen'), 'final exact FEN differs')
        scores.append(score)
        total_plies += len(moves)
    mean = sum(scores) / len(scores)
    delta = None if mean in (0., 1.) else 400 * math.log10(mean / (1 - mean))
    summary = json.loads((run / 'summary.json').read_bytes())
    require(summary['candidate_score'] == mean and summary['technical_failures'] == 0 and
            summary['incomplete_pairs'] == 0 and summary['games'] == len(rows), 'summary differs')
    bins = [0] * 5
    for i in range(0, len(scores), 2):
        bins[round(2 * (scores[i] + scores[i + 1]))] += 1
    require(summary['pentanomial'] == bins, 'pair histogram differs')
    return dict(kind='independent_closed_timed_match_audit', status='pass',
                games=len(rows), complete_opening_pairs=len(openings), legal_plies=total_plies,
                wins=scores.count(1.), draws=scores.count(.5), losses=scores.count(0.),
                candidate_score=mean, relative_logistic_point_estimate=delta,
                configured_opponent_proxy_rating=None if delta is None else 1320 + delta,
                proxy_scope='Stockfish19 configured1320 plus relative logistic point estimate; uncalibrated to actual rating pools',
                lichess_rating=None, confidence_interval=None,
                independence_asserted=False, technical_failures=0,
                move_timing_ms={name: dict(moves=len(values), maximum=max(values),
                                           median=sorted(values)[len(values)//2]) for name, values in walls.items()},
                requested_seconds_per_move=1., timing_scope='wall time includes UCI transport and scheduling; early completion allowed',
                declaration_sha256=hashlib.sha256(declaration_path.read_bytes()).hexdigest(),
                auditor_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                run_sha256={p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(run.iterdir()) if p.is_file()})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', required=True, type=Path)
    parser.add_argument('--declaration', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.run, args.declaration)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
