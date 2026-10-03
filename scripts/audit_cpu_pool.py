#!/usr/bin/env python3
"""Independent replay/audit of a fresh four-lane CPU configured-strength pool.

No engines are launched. Only complete swapped-color pairs contribute points.
The logistic fit is descriptive against Stockfish settings, never Lichess Elo.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re

import chess
import chess.pgn

KIND = 'fixed_1second_CPU_PVS_configured_strength_pool_v1'
LANES = (('off1800', False, 1800, 1), ('on1600', True, 1600, 2),
         ('on1800', True, 1800, 3), ('on2000', True, 2000, 4))
GPU_ENV = ('CHEN_GPU_LIBRARY', 'CHEN_NEURAL_MODEL', 'CHEN_NEURAL_LIBRARY',
           'CHEN_NEURAL_HIP_RUNTIME', 'CHEN_NEURAL_EXECUTION')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1048576), b''):
            value.update(block)
    return value.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def equal_json(left, right):
    # JSON control values distinguish bool/int and int/float, unlike dict ==.
    return json.dumps(left, sort_keys=True) == json.dumps(right, sort_keys=True)


def expected_score(rating, setting):
    z = (setting - rating) * math.log(10.) / 400.
    if z >= 0:
        inverse = math.exp(-z)
        return inverse / (1. + inverse)
    return 1. / (1. + math.exp(z))


def configured_fit(groups):
    """Solve sum observed points = sum per-game logistic expectations."""
    require(all(type(g['games']) is int and g['games'] >= 0 and
                type(g['setting']) is int and type(g['points']) in (int, float) and
                math.isfinite(g['points']) and 0 <= g['points'] <= g['games'] for g in groups),
            'invalid configured-fit group')
    groups = [g for g in groups if g['games']]
    games = sum(g['games'] for g in groups)
    points = sum(g['points'] for g in groups)
    result = dict(included_games=games, observed_points=points, groups=groups,
                  configured_setting_logistic_point_estimate=None,
                  expected_points_at_estimate=None, score_equation_residual=None,
                  lichess_rating=None, confidence_interval=None,
                  independence_asserted=False, calibrated_rating=False)
    if not games:
        return dict(result, status='no_complete_paired_games')
    if points in (0., float(games)):
        return dict(result, status='unbounded_score_boundary',
                    direction='negative_infinity' if points == 0 else 'positive_infinity')
    expectation = lambda rating: sum(g['games'] * expected_score(rating, g['setting']) for g in groups)
    low, high = min(g['setting'] for g in groups) - 10000., max(g['setting'] for g in groups) + 10000.
    require(expectation(low) < points < expectation(high), 'score fit bracket failed')
    for _ in range(160):
        middle = (low + high) / 2
        if expectation(middle) < points:
            low = middle
        else:
            high = middle
    estimate = (low + high) / 2
    predicted = expectation(estimate)
    return dict(result, status='finite_descriptive_fit',
                configured_setting_logistic_point_estimate=estimate,
                expected_points_at_estimate=predicted, score_equation_residual=predicted - points,
                equation='sum(points) = sum(n_setting/(1+10**((setting-rating)/400)))')


def openings(path):
    result = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.lstrip().startswith('#'):
            continue
        name, fen = line.split('|', 1)
        name, fen = name.strip(), fen.strip()
        board = chess.Board(fen)
        require(board.is_valid() and board.outcome(claim_draw=True) is None, 'invalid/terminal opening')
        require(name and name not in [old[0] for old in result], 'duplicate/empty opening name')
        result.append((name, fen))
    return result


def timing(values):
    if not values:
        return dict(moves=0, minimum_ms=None, median_ms=None, maximum_ms=None,
                    above_requested_1second=0, above_1point1seconds=0)
    ordered = sorted(values)
    n = len(ordered)
    median = ordered[n // 2] if n % 2 else (ordered[n // 2 - 1] + ordered[n // 2]) / 2
    return dict(moves=n, minimum_ms=ordered[0], median_ms=median, maximum_ms=ordered[-1],
                above_requested_1second=sum(v > 1000. for v in values),
                above_1point1seconds=sum(v > 1100. for v in values))


def check_summary(summary, pairs, incomplete_pairs, failures):
    count = len(pairs)
    scores = [score for pair in pairs for score in pair]
    mean = sum(scores) / len(scores) if scores else None
    bins = [sum(round(2 * sum(pair)) == index for pair in pairs) for index in range(5)]
    require(summary.get('completed_pairs') == count and summary.get('incomplete_pairs') == incomplete_pairs and
            summary.get('technical_failures') == failures, 'summary pair/failure inventory differs')
    require(summary.get('games', 0) == len(scores) and summary.get('candidate_score') == mean and
            summary.get('pentanomial', [0] * 5) == bins, 'summary points/pentanomial differs')
    relative = None if mean in (None, 0., 1.) else 400. * math.log10(mean / (1 - mean))
    actual = summary.get('logistic_relative_elo_point_estimate')
    require(actual is None if relative is None else type(actual) in (int, float) and
            math.isfinite(actual) and math.isclose(actual, relative, rel_tol=1e-12, abs_tol=1e-9),
            'summary relative logistic estimate differs')
    require(summary.get('independent_opening_pairs_asserted') is False and
            summary.get('paired_bootstrap_score_interval_95_diagnostic') is None,
            'unsupported summary independence/interval')


def audit_lane(lane, declaration, suite, incomplete=False):
    run = Path(lane['run'])
    require(sha(lane['config_path']) == lane['config_sha256'], 'config hash changed: ' + lane['id'])
    config = read_json(lane['config_path'])
    require(set(config) == {'base', 'candidate'}, 'unexpected engine roles')
    common = dict(Backend='alphabeta', Eval='tapered', Level=20, Hash=16,
                  UseTT=True, Quiescence=True, MoveOrdering=True, AspirationWindow=0)
    require(equal_json(config['candidate']['options'], dict(common, PVS=lane['pvs'])), 'CPU candidate policy differs')
    require(equal_json(config['base']['options'], dict(Threads=1, Hash=16, NumaPolicy='none',
            UCI_LimitStrength=True, UCI_Elo=lane['stockfish_elo'], SyzygyProbeLimit=0)), 'Stockfish policy differs')
    for role, expected in (('candidate', declaration['engine_sha256']), ('base', declaration['stockfish_sha256'])):
        entry = config[role]
        require(isinstance(entry['command'], list) and len(entry['command']) == 1 and
                sha(entry['command'][0]) == expected and
                declaration['source_and_binary_sha256'].get(entry['command'][0]) == expected,
                'unbound/different engine command')
        require(entry.get('capture_rocm_artifacts', False) is False and
                not any(key in entry.get('env', {}) for key in GPU_ENV), 'CPU-only config contains neural/GPU selector')
        require(entry.get('artifacts') == {'engine': entry['command'][0]}, 'unbound engine artifact')
    metadata = read_json(run / 'metadata.json')
    require(equal_json(metadata['config'], config), 'metadata/config differs')
    require(metadata.get('chess_version') == chess.__version__ == '1.11.2', 'rules provider version differs')
    require(metadata['concurrency'] == 1, 'lane must be sequential')
    require(metadata['opening_sha256'] == declaration['opening_sha256'] and
            metadata['match_script_sha256'] == declaration['match_script_sha256'], 'collector/opening pin differs')
    require(metadata['launcher_sha256'] == dict(base=declaration['stockfish_sha256'], candidate=declaration['engine_sha256']), 'launcher hashes differ')
    require(metadata.get('declared_artifact_sha256') == dict(base=dict(engine=declaration['stockfish_sha256']),
            candidate=dict(engine=declaration['engine_sha256'])), 'declared executable hashes differ')
    require(metadata.get('artifact_scope') == dict(base='declared-files', candidate='declared-files'), 'artifact scope differs')
    require(not any(key in metadata.get('inherited_runtime_environment', {}) for key in GPU_ENV), 'inherited neural/GPU selector')
    budgets = metadata['budgets']
    require(budgets['movetime'] == declaration['requested_seconds_per_move'] and
            budgets['depth'] is None and budgets['nodes'] is None and budgets['rounds'] == 1 and
            budgets['max_plies'] == declaration['max_plies'] and budgets['independent_opening_pairs'] is False,
            'collector budgets differ')
    for field, expected in (('openings', declaration['opening_path']), ('config', lane['config_path']), ('out', lane['run'])):
        require(Path(budgets[field]).resolve() == Path(expected).resolve(), 'collector path differs: ' + field)
    jsonl_path, pgn_path = run / 'games.jsonl', run / 'games.pgn'
    require(incomplete or jsonl_path.is_file() and pgn_path.is_file(), 'terminal collector missing saved game artifacts')
    rows = [json.loads(line) for line in jsonl_path.read_text().splitlines() if line.strip()] if jsonl_path.is_file() else []
    games = []
    if pgn_path.is_file():
        with pgn_path.open() as handle:
            while (game := chess.pgn.read_game(handle)) is not None:
                require(not game.errors, 'PGN parse error')
                games.append(game)
    planned = 2 * len(suite)
    require(len(rows) == len(games) and len(rows) <= planned, 'PGN/JSONL inventory differs')
    require(incomplete or len(rows) == planned, 'terminal complete lane requires exact16 game inventory')
    statuses, scores, walls = [], [], {'candidate': [], 'base': []}
    telemetry = {'candidate': [], 'base': []}
    total_plies = 0
    for index, (row, game) in enumerate(zip(rows, games)):
        name, fen = suite[index // 2]
        color = chess.WHITE if index % 2 == 0 else chess.BLACK
        require(row['opening'] == name and row['round'] == 0 and
                row['candidate_color'] == ('white' if color else 'black'), 'pair order/opening/color differs')
        require(game.headers.get('Opening') == name and
                game.headers['White'] == ('candidate' if color else 'base') and
                game.headers['Black'] == ('base' if color else 'candidate'), 'PGN engine/opening roles differ')
        board = chess.Board(fen)
        require(board.fen(en_passant='legal') == game.board().fen(en_passant='legal'), 'PGN initial rule state differs')
        moves = list(game.mainline_moves())
        require(type(row['plies']) is int and len(moves) == row['plies'] == len(row['move_trace']) and
                len(moves) <= declaration['max_plies'], 'missing move trace or invalid ply count')
        for ply, (move, trace) in enumerate(zip(moves, row['move_trace'])):
            require(board.outcome(claim_draw=True) is None, 'move after rule endpoint')
            engine = 'candidate' if board.turn == color else 'base'
            require(trace['ply'] == ply and trace['engine'] == engine and trace['move'] == move.uci() and
                    trace['fen'] == board.fen(en_passant='fen'), 'trace move/state/actor differs')
            require(type(trace['wall_ns']) is int and trace['wall_ns'] >= 0, 'invalid wall timing')
            reported = trace['reported']
            require(isinstance(reported, dict) and {'depth', 'nodes', 'time'} <= reported.keys(), 'missing UCI telemetry')
            require(all(type(v) in (int, float) and math.isfinite(v) and v >= 0 for v in reported.values()), 'nonfinite/negative UCI telemetry')
            require(type(reported['depth']) is int and type(reported['nodes']) is int, 'noninteger depth/node telemetry')
            require(move in board.legal_moves, 'illegal PGN move')
            walls[engine].append(trace['wall_ns'] / 1e6)
            telemetry[engine].append(reported)
            board.push(move)
        outcome = board.outcome(claim_draw=True)
        failure = row['failure']
        if failure is not None:
            require(incomplete and index == len(rows) - 1 and isinstance(failure, str) and failure,
                    'technical failure needs explicit final-prefix incomplete admission')
            require(outcome is None, 'failure row at finished rule state')
            result, score, termination, status = '*', None, 'technical_failure', 'technical_failure'
        elif outcome is not None:
            result = outcome.result()
            score = .5 if outcome.winner is None else float(outcome.winner == color)
            termination, status = outcome.termination.name, 'rule_finished'
        else:
            require(len(moves) == declaration['max_plies'], 'unfinished game before declared cap')
            result, score, termination, status = '*', None, 'ply_cap_inconclusive', 'capped_unscored'
        require(row['result'] == game.headers['Result'] == result and
                row['candidate_score'] == score and type(row['candidate_score']) is not bool, 'result/score differs')
        require(row['termination'] == game.headers.get('Termination') == termination, 'endpoint differs')
        require(row['final_fen'] == board.fen(en_passant='fen'), 'final exact FEN differs')
        scores.append(score);statuses.append(status);total_plies += len(moves)
    included, excluded = [], []
    for index, (name, _) in enumerate(suite):
        pair = scores[index * 2:index * 2 + 2]
        if len(pair) == 2 and all(score is not None for score in pair):
            included.append(dict(opening_index=index, opening=name, row_indices=[index * 2, index * 2 + 1], scores=pair))
        else:
            excluded.append(dict(opening_index=index, opening=name,
                observed_statuses=statuses[index * 2:index * 2 + 2], missing_games=2-len(pair)))
    paired_scores = [score for pair in included for score in pair['scores']]
    unpaired = [index for index, score in enumerate(scores) if score is not None and
                not any(index in pair['row_indices'] for pair in included)]
    summary_path = run / 'summary.json'
    if not incomplete:
        require(summary_path.is_file(), 'terminal complete collector has no summary')
    if summary_path.is_file():
        check_summary(read_json(summary_path), [p['scores'] for p in included], len(excluded), statuses.count('technical_failure'))
    return dict(id=lane['id'], collector_state='INCOMPLETE' if incomplete else 'COMPLETE',
        pvs=lane['pvs'], stockfish_configured_setting=lane['stockfish_elo'], declared_cpu=lane['cpu'],
        observed_games=len(rows), planned_games=planned, missing_games=planned-len(rows), legal_plies=total_plies,
        complete_pairs=included, excluded_pairs=excluded, included_games=len(paired_scores),
        candidate_points=sum(paired_scores), wins=paired_scores.count(1.), draws=paired_scores.count(.5), losses=paired_scores.count(0.),
        capped_unscored_games=statuses.count('capped_unscored'), technical_failures=statuses.count('technical_failure'),
        unpaired_finished_game_indices=unpaired, unpaired_finished_games=len(unpaired),
        move_wall_ms={role: timing(values) for role, values in walls.items()},
        reported_telemetry={role: {key: dict(count=len(values), minimum=min((r[key] for r in values), default=None),
            maximum=max((r[key] for r in values), default=None)) for key in ('depth', 'nodes', 'time')} for role, values in telemetry.items()},
        timing_scope='requested movetime1s; witnessed wall includes transport/scheduling, early finishes and overruns reported, no asserted hard deadline',
        affinity_scope='declared CPU only; saved games do not independently certify core affinity/isolation',
        artifact_sha256={name: sha(run/name) if (run/name).is_file() else None
                         for name in ('metadata.json', 'games.jsonl', 'games.pgn')},
        summary_sha256=sha(summary_path) if summary_path.is_file() else None)


def audit(declaration_path, incomplete_lanes=()):
    declaration_path = Path(declaration_path)
    declaration = read_json(declaration_path)
    require(declaration['kind'] == KIND and declaration['requested_seconds_per_move'] == 1.0 and
            type(declaration['requested_seconds_per_move']) in (int, float) and
            declaration['rounds'] == 1 and declaration['opening_pairs'] == 8 and declaration['max_plies'] == 512 and
            declaration['authored_pairs_independent'] is False and
            isinstance(declaration['source_commit'], str) and
            re.fullmatch(r'[0-9a-fA-F]{40}', declaration['source_commit']) is not None,
            'unexpected fresh CPU pool declaration')
    require([(l['id'], l['pvs'], l['stockfish_elo'], l['cpu']) for l in declaration['lanes']] == list(LANES), 'lane/control allocation differs')
    require(all(type(l['pvs']) is bool and type(l['stockfish_elo']) is int and type(l['cpu']) is int
                for l in declaration['lanes']), 'lane control types differ')
    require(set(incomplete_lanes) <= {l[0] for l in LANES}, 'unknown incomplete lane')
    for path, pin in declaration['source_and_binary_sha256'].items():
        require(sha(path) == pin, 'source/artifact hash changed: ' + path)
    require(sha(declaration['opening_path']) == declaration['opening_sha256'], 'opening hash changed')
    require(declaration['match_script_sha256'] in declaration['source_and_binary_sha256'].values(), 'unbound collector source')
    suite = openings(declaration['opening_path'])
    require(len(suite) == 8, 'opening count differs')
    lanes = [audit_lane(lane, declaration, suite, lane['id'] in incomplete_lanes) for lane in declaration['lanes']]
    on = [dict(setting=lane['stockfish_configured_setting'], games=lane['included_games'], points=lane['candidate_points']) for lane in lanes if lane['pvs']]
    off = lanes[0]
    return dict(kind='independent_CPU_configured_strength_pool_audit',
        status='COMPLETE' if all(l['collector_state'] == 'COMPLETE' for l in lanes) else 'INCOMPLETE',
        declaration_sha256=sha(declaration_path), auditor_sha256=sha(__file__), source_commit=declaration['source_commit'],
        lanes=lanes, pvs_on_configured_pool_fit=configured_fit(on),
        default_pvs_off_1800_fit=configured_fit([dict(setting=1800, games=off['included_games'], points=off['candidate_points'])]),
        fit_scope='descriptive Stockfish19 configured-setting score equation over complete pairs only; authored openings correlated and completion selection explicit',
        shared_cache_power_clocks='uncontrolled', lichess_rating=None, confidence_interval=None,
        independence_asserted=False, architecture_winner=None,
        auditor_chess_version=chess.__version__,
        auditor_rules_provider_sha256=sha(chess.__file__),
        auditor_pgn_provider_sha256=sha(chess.pgn.__file__),
        default_off_in_on_fit=False, no_cap_or_failure_imputation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--declaration', required=True, type=Path)
    parser.add_argument('--out', required=True, type=Path)
    parser.add_argument('--incomplete-lane', action='append', default=[], choices=[lane[0] for lane in LANES])
    args = parser.parse_args()
    result = audit(args.declaration, args.incomplete_lane)
    with args.out.open('x') as handle:
        handle.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
