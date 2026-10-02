import importlib.util
from pathlib import Path
import unittest
try:
    import chess
    import chess.engine
except ImportError:
    chess = None


@unittest.skipIf(chess is None, 'optional python-chess QA dependency unavailable')
class MatchTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        spec = importlib.util.spec_from_file_location('chen_match', Path('scripts/match.py'))
        cls.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cls.module)

    def test_ply_cap_is_inconclusive(self):
        class Legal:
            def play(self, board, *args, **kwargs):
                return chess.engine.PlayResult(next(iter(board.legal_moves)), None,
                                               info={'nodes': 17, 'depth': 1})
        game, row = self.module.play([Legal(), Legal()], ['a', 'b'], ('start', chess.STARTING_FEN), chess.WHITE, chess.engine.Limit(depth=1), 1)
        self.assertIsNone(row['candidate_score'])
        self.assertEqual(row['termination'], 'ply_cap_inconclusive')
        self.assertEqual(game.headers['Result'], '*')
        self.assertEqual(len(row['move_trace']), 1)
        trace = row['move_trace'][0]
        self.assertEqual(trace['fen'], chess.STARTING_FEN)
        self.assertEqual(trace['engine'], 'a')
        self.assertEqual(trace['reported'], {'nodes': 17, 'depth': 1})
        self.assertGreaterEqual(trace['wall_ns'], 0)
        self.assertIn(chess.Move.from_uci(trace['move']), chess.Board().legal_moves)

    def test_illegal_move_is_a_technical_failure(self):
        class Bad:
            def play(self, *args, **kwargs):
                return chess.engine.PlayResult(chess.Move.from_uci('e2e5'), None)
        _, row = self.module.play([Bad(), Bad()], ['a', 'b'], ('start', chess.STARTING_FEN), chess.WHITE, chess.engine.Limit(depth=1), 1)
        self.assertIsNone(row['candidate_score'])
        self.assertEqual(row['termination'], 'technical_failure')
        self.assertIn('illegal', row['failure'])
        self.assertEqual(row['move_trace'], [])

    def test_mate_at_ply_cap_is_completed(self):
        class Mate:
            def play(self, *args, **kwargs):
                return chess.engine.PlayResult(chess.Move.from_uci('f7g7'), None)
        _, row = self.module.play([Mate(), Mate()], ['a', 'b'], ('mate', '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'), chess.WHITE, chess.engine.Limit(depth=1), 1)
        self.assertEqual(row['candidate_score'], 1)
        self.assertEqual(row['termination'], 'CHECKMATE')


if __name__ == '__main__':
    unittest.main()
