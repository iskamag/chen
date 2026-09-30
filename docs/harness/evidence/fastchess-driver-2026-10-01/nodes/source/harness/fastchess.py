"""Independent result audit for the pinned external paired-match runner.

SPRT mathematics and boundary decisions belong to Fastchess. This module checks
the experiment's actual games and refuses inference from incomplete evidence.
"""
import math
from pathlib import Path
import re

import chess
import chess.pgn

from harness.stats import summarize_pairs


class ExactResultBuilder(chess.pgn.GameBuilder):
    """Check the body result which the default builder can silently discard."""
    def __init__(self):
        super().__init__()
        self.result_tokens = []

    def visit_result(self, result):
        self.result_tokens.append(result)
        if self.game.headers.get('Result') != result:
            raise ValueError('movetext result disagrees with Result header')
        super().visit_result(result)

    def end_game(self):
        if len(self.result_tokens) != 1:
            raise ValueError('exactly one movetext result token required')
        super().end_game()


def validate_sprt(value):
    keys = {'elo0', 'elo1', 'alpha', 'beta', 'model', 'sampling_design'}
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError('SPRT declaration must contain exactly ' + ', '.join(sorted(keys)))
    for key in ('elo0', 'elo1', 'alpha', 'beta'):
        if isinstance(value[key], bool) or not isinstance(value[key], (int, float)) or not math.isfinite(value[key]):
            raise ValueError('finite numeric SPRT parameters required')
    if value['elo0'] >= value['elo1'] or not 0 < value['alpha'] < .5 or not 0 < value['beta'] < .5:
        raise ValueError('ordered hypotheses and alpha/beta in (0, .5) required')
    if value['model'] not in ('normalized', 'logistic'):
        raise ValueError('paired SPRT supports normalized or logistic model')
    if not isinstance(value['sampling_design'], str) or not value['sampling_design'].strip():
        raise ValueError('describe the independent opening-pair sampling design')
    return value


def audit_pgn(path, openings, max_games):
    """Replay legal normal endpoints, verify paired FENs/colors and all results."""
    rows = []
    with Path(path).open() as stream:
        while (game := chess.pgn.read_game(stream, Visitor=ExactResultBuilder)) is not None:
            if game.errors:
                raise ValueError(f'PGN parsing failed: {game.errors}')
            headers = game.headers
            if headers.get('Termination') != 'normal':
                raise ValueError('technical failure, interruption or adjudication: ' + headers.get('Termination', 'missing'))
            if {headers.get('White'), headers.get('Black')} != {'candidate', 'base'}:
                raise ValueError('unexpected engine identities')
            board = game.board()
            initial = board.fen(en_passant='fen')
            if not board.is_valid():
                raise ValueError('invalid initial position')
            plies = 0
            for move in game.mainline_moves():
                if move not in board.legal_moves:
                    raise ValueError('illegal PGN move')
                board.push(move)
                plies += 1
            outcome = board.outcome(claim_draw=True)
            if outcome is None or outcome.result() != headers.get('Result'):
                raise ValueError('result does not agree with independent terminal replay')
            if int(headers.get('PlyCount', -1)) != plies:
                raise ValueError('PGN ply count disagrees with replay')
            result = outcome.result()
            white_score = {'1-0': 1., '0-1': 0., '1/2-1/2': .5}[result]
            color = headers['White'] == 'candidate'
            rows.append({'round': int(headers['Round']), 'candidate_white': color,
                         'initial_fen': initial, 'result': result,
                         'candidate_score': white_score if color else 1 - white_score,
                         'plies': plies, 'termination': outcome.termination.name,
                         'final_fen': board.fen(en_passant='fen')})
    if not rows or len(rows) % 2 or len(rows) > max_games:
        raise ValueError('nonempty complete pairs within declared cap required')
    pairs = []
    for i in range(0, len(rows), 2):
        first, second = rows[i:i + 2]
        pair_index = i // 2
        expected = openings[pair_index % len(openings)][1]
        if (first['round'] != pair_index + 1 or second['round'] != pair_index + 1
                or first['candidate_white'] == second['candidate_white']
                or first['initial_fen'] != expected or second['initial_fen'] != expected):
            raise ValueError('opening order, round, FEN or color pairing disagrees')
        pairs.append((first['candidate_score'], second['candidate_score']))
    return rows, pairs


def audit_scoreboard(state, pairs):
    """Require the external scoreboard to match independently replayed scores."""
    stats = state.get('stats', {})
    if set(stats) != {'candidate vs base'}:
        raise ValueError('unexpected external scoreboard identities')
    actual = stats['candidate vs base']
    scores = [score for pair in pairs for score in pair]
    expected = {'wins': scores.count(1.), 'draws': scores.count(.5), 'losses': scores.count(0.),
                'penta_WW': pairs.count((1., 1.)), 'penta_LL': pairs.count((0., 0.)),
                'penta_DD': pairs.count((.5, .5)),
                'penta_WL': sum(set(pair) == {0., 1.} for pair in pairs),
                'penta_WD': sum(set(pair) == {.5, 1.} for pair in pairs),
                'penta_LD': sum(set(pair) == {0., .5} for pair in pairs)}
    if actual != expected:
        raise ValueError('external scoreboard disagrees with independently replayed pairs')
    return expected


def decision(console, games, max_games, declaration, returncode, timed_out=False):
    """Accept only an audited, predeclared external decision before the game cap.

    Rounded console LLRs are diagnostic, never recomputed as a boundary proof.
    Reaching the cap is conservatively inconclusive even if a boundary occurs
    on the final pair. Caller must complete both game/scoreboard audits first.
    """
    if timed_out or returncode != 0:
        return 'technical_failure'
    if declaration is None:
        return 'diagnostic_complete' if games == max_games else 'unexpected_early_stop'
    if games == max_games:
        return 'inconclusive_game_cap'
    labels = re.findall(r'SPRT \([^\n]*\) completed - (H[01]) was accepted', console)
    if labels and len(set(labels)) == 1:
        return 'external_sprt_' + labels[0]
    return 'unexpected_early_stop'


def descriptive(pairs):
    return summarize_pairs(pairs, independent_pairs=False)
