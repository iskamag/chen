"""Synthetic protocol fixtures; no engine or hardware execution."""
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import chess
import chess.pgn

spec = importlib.util.spec_from_file_location('timed_auditor', Path(__file__).parents[1] / 'audit_timed_match.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class DirectMatchAudit(unittest.TestCase):
    def fixture(self, directory, capped=False):
        root = Path(directory)
        fen = chess.STARTING_FEN if capped else '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'
        opening = root / 'openings.txt'
        opening.write_text('fixture | ' + fen + '\n')
        common = dict(Backend='alphabeta', Eval='tapered', Level=20, Hash=16,
                      UseTT=True, Quiescence=True, MoveOrdering=True, AspirationWindow=0)
        config = {name: dict(command=['/synthetic/chen'], options=dict(common, PVS=enabled))
                  for name, enabled in [('base', False), ('candidate', True)]}
        metadata = dict(config=config, concurrency=1, launcher_sha256=dict(base='synthetic', candidate='synthetic'),
                        budgets=dict(movetime=1.0, depth=None, nodes=None, rounds=1, max_plies=2,
                                     openings=str(opening), independent_opening_pairs=False))
        (root / 'metadata.json').write_text(json.dumps(metadata))
        declaration = dict(kind='fixed_1second_UCI_direct_pvs_diagnostic', requested_seconds_per_move=1.0,
                           source_and_binary_sha256={}, engine_sha256='synthetic', max_plies=2, opening_pairs=1)
        declaration_path = root / 'declaration.json'
        declaration_path.write_text(json.dumps(declaration))
        games, rows = [], []
        for candidate_color in [chess.WHITE, chess.BLACK]:
            board = chess.Board(fen)
            game = chess.pgn.Game()
            game.setup(board)
            game.headers.update(White='candidate' if candidate_color else 'base',
                                Black='base' if candidate_color else 'candidate')
            node, trace = game, []
            for ply, uci in enumerate(['e2e4', 'e7e5'] if capped else ['f7g7']):
                move = chess.Move.from_uci(uci)
                trace.append(dict(ply=ply, engine='candidate' if board.turn == candidate_color else 'base',
                                  fen=board.fen(en_passant='fen'), move=uci, wall_ns=1,
                                  reported=dict(depth=1, nodes=1, time=0.001)))
                board.push(move)
                node = node.add_variation(move)
            outcome = board.outcome(claim_draw=True)
            result = '*' if capped else outcome.result()
            termination = 'ply_cap_inconclusive' if capped else outcome.termination.name
            score = None if capped else float(outcome.winner == candidate_color)
            game.headers.update(Result=result, Termination=termination)
            games.append(str(game))
            rows.append(dict(opening='fixture', candidate_color='white' if candidate_color else 'black',
                             failure=None, round=0, plies=len(trace), move_trace=trace, result=result,
                             candidate_score=score, termination=termination, final_fen=board.fen(en_passant='fen')))
        (root / 'games.pgn').write_text('\n\n'.join(games) + '\n')
        (root / 'games.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
        summary = dict(completed_pairs=0 if capped else 1, incomplete_pairs=1 if capped else 0,
                       technical_failures=0, independent_opening_pairs_asserted=False)
        if not capped:
            summary.update(candidate_score=0.5, games=2, pentanomial=[0, 0, 1, 0, 0],
                           paired_bootstrap_score_interval_95_diagnostic=None)
        (root / 'summary.json').write_text(json.dumps(summary))
        return root, declaration_path

    def test_complete_direct_pair_has_no_external_rating_anchor(self):
        with tempfile.TemporaryDirectory() as directory:
            root, declaration = self.fixture(directory)
            result = module.audit(root, declaration)
            self.assertEqual((result['wins'], result['draws'], result['losses']), (1, 0, 1))
            self.assertEqual(result['relative_logistic_point_estimate'], 0.0)
            self.assertIsNone(result['configured_opponent_proxy_rating'])
            self.assertIsNone(result['confidence_interval'])

    def test_all_capped_pairs_are_unscored(self):
        with tempfile.TemporaryDirectory() as directory:
            root, declaration = self.fixture(directory, capped=True)
            result = module.audit(root, declaration)
            self.assertEqual(result['completed_paired_games'], 0)
            self.assertEqual(result['capped_unscored_games'], 2)
            self.assertIsNone(result['candidate_score'])
            self.assertIsNone(result['relative_logistic_point_estimate'])

    def test_other_option_difference_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, declaration = self.fixture(directory)
            metadata = json.loads((root / 'metadata.json').read_text())
            metadata['config']['candidate']['options']['AspirationWindow'] = 25
            (root / 'metadata.json').write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, 'direct policy differs'):
                module.audit(root, declaration)

    def test_scored_cap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root, declaration = self.fixture(directory, capped=True)
            rows = [json.loads(line) for line in (root / 'games.jsonl').read_text().splitlines()]
            rows[0]['candidate_score'] = 0.5
            (root / 'games.jsonl').write_text(''.join(json.dumps(row) + '\n' for row in rows))
            with self.assertRaisesRegex(ValueError, 'result/score differs'):
                module.audit(root, declaration)


if __name__ == '__main__':
    unittest.main()
