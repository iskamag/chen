"""Cold independent result admission and chess replay; performs no GPU calls."""
import argparse
import hashlib
import json
from pathlib import Path

import chess


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def one(rows, kind):
    found = [row for row in rows if row['record'] == kind]
    assert len(found) == 1, kind
    return found[0]


def audit(evidence, declaration):
    declared = json.loads(declaration.read_text())
    cache_path = evidence / 'oracle.json'
    cache = json.loads(cache_path.read_text())
    assert cache['source_sha256'] == declared['probe_sha256']
    assert cache['helper_source_sha256'] == declared['helper_source_sha256']
    assert all(cache['artifact_sha256'][p] == value
               for p, value in declared['artifact_sha256'].items())
    assert {name: row['archive_sha256'] for name, row in cache['models'].items()} == declared['model_sha256']
    fixtures = {row['name']: row['fen'] for row in cache['fixtures']}
    producer = records(evidence / 'producer.stdout.jsonl')
    assert (evidence / 'producer.stdout.jsonl').read_bytes() == (evidence / 'producer.remote.jsonl').read_bytes()
    assert one(producer, 'parent_context_oracle_complete')['sha256'] == sha(cache_path)
    assert one(records(evidence / 'producer.mirror.jsonl'), 'parent_context_mirror_end')['returncode'] == 0
    producer_end = one(producer, 'parent_context_command_complete')['observed_utc']
    trials, total_comparisons, total_plies = [], 0, 0
    for scheduled in declared['schedule']:
        trial = scheduled['trial']
        prefix = f'trial-{trial}'
        rows = records(evidence / f'{prefix}.stdout.jsonl')
        assert (evidence / f'{prefix}.stdout.jsonl').read_bytes() == (evidence / f'{prefix}.remote.jsonl').read_bytes()
        assert not any('timeout' in row['record'] or row['record'].endswith('_failed') for row in rows)
        start = one(rows, 'parent_context_trial_start')
        assert start['trial'] == scheduled
        assert start['producer_retired'] and start['observed_utc'] > producer_end
        assert start['oracle_sha256'] == sha(cache_path)
        assert start['artifact_sha256'] == cache['artifact_sha256']
        if scheduled['parent'] == 'cpu':
            assert start['cpu_parent_guard'] == {'torch_neural_modules': [], 'HIP_Torch_DSO_maps': []}
        mirror = records(evidence / f'{prefix}.mirror.jsonl')
        assert one(mirror, 'parent_context_mirror_end')['returncode'] == 0
        comparisons = [row for row in rows if row['record'] == 'parent_context_exact_comparison']
        assert len(comparisons) == 18
        assert {(row['model'], row['position']) for row in comparisons} == {
            (model, position) for model in cache['models'] for position in fixtures}
        leaves = 0
        for row in comparisons:
            actual = row['actual']
            expected = cache['models'][row['model']]['rows'][row['position']]
            depth = actual['depth']
            assert depth in (1, 2)
            assert depth == 2 or abs(actual['score_cp']) >= 29872
            target = expected[str(depth)]
            assert actual['score_cp'] == target['score_cp']
            assert actual['bestmove'] in target['optimal_moves']
            assert actual['evaluator_artifact_sha256'] == declared['model_sha256'][row['model']]
            assert actual['backend'] == 'hip-resident' and actual['eval'] == 'neural'
            assert actual['neural_execution']['mode'] == 'eager' and not actual['stopped']
            assert actual['nodes'] == sum(expected[str(d)]['visited_positions'] for d in range(1, depth + 1))
            evaluated = sum(expected[str(d)]['evaluated_leaves'] for d in range(1, depth + 1))
            assert actual['evaluated_positions'] == evaluated
            leaves += evaluated
            board = chess.Board(fixtures[row['position']])
            assert actual['fen'] == board.fen(en_passant='fen')
            pv = actual['pv'].split()
            assert 1 <= len(pv) <= depth and pv[0] == actual['bestmove']
            for move in pv:
                board.push_uci(move)
        ready = [row for row in rows if row['record'] == 'parent_context_UCI_ready']
        assert len(ready) == 5
        for row in ready:
            env = row['child_environment']
            assert env and env['AMD_DIRECT_DISPATCH'] == '1'
            assert env['CHEN_NEURAL_EXECUTION'] == 'eager' and not env['LD_PRELOAD']
        games = [row for row in rows if row['record'] == 'neural_UCI_legal_game']
        assert len(games) == 2
        expected_openings = [('e2e4', 'e7e5', 'g1f3', 'b8c6'), ('d2d4', 'd7d5', 'c2c4', 'e7e6')]
        plies = 0
        for game, opening, color in zip(games, expected_openings, ('white', 'black')):
            assert tuple(game['opening']) == opening and game['neural_color'] == color
            board = chess.Board()
            for move in opening:
                board.push_uci(move)
            assert len(game['moves']) == game['plies_after_opening'] == 20
            for entry in game['moves']:
                neural_turn = board.turn == (chess.WHITE if color == 'white' else chess.BLACK)
                assert entry['engine'] == ('neural' if neural_turn else 'resident-tapered')
                board.push_uci(entry['move'])
                plies += 1
            assert game['final_fen'] == board.fen(en_passant='fen')
        end = one(rows, 'parent_context_trial_complete')
        assert end['inputs_unchanged'] and end['comparisons'] == 18
        retirement = json.loads((evidence / f'{prefix}.retirement.json').read_text())
        assert retirement['complete_record'] and not retirement['timeouts']
        assert set(retirement['recorded_retirement_pids']) == {row['pid'] for row in ready}
        assert not retirement['actual_inspection']['live_engine_executables']
        assert all(value is None for value in retirement['actual_inspection']['observed_owned_processes'].values())
        trials.append({'trial': trial, 'parent': scheduled['parent'], 'comparisons': 18,
                       'iterative_evaluated_leaves': leaves, 'independently_replayed_game_plies': plies,
                       'ready_environment_observations': len(ready), 'retirement_records_pass': True})
        total_comparisons += 18
        total_plies += plies
    return {'kind': 'root_independent_parent_context_cold_audit', 'passed': True,
            'audit_source_sha256': sha(Path(__file__)),
            'declaration_sha256': sha(declaration), 'oracle_sha256': sha(cache_path),
            'trials': trials, 'comparisons': total_comparisons, 'game_plies': total_plies,
            'chess_version': chess.__version__,
            'scope': 'Cold comparison to frozen oracle cache, independent PV/game legality replay, '
                     'environment and retirement record admission. Does not regenerate GPU scores, '
                     'establish causation, a reliability rate, or playing strength.',
            'payload_sha256': {p.name: sha(p) for p in sorted(evidence.iterdir()) if p.is_file()}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--evidence', type=Path, required=True)
    parser.add_argument('--declaration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.evidence, args.declaration)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in ('passed', 'comparisons', 'game_plies', 'scope')}))
