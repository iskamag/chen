import importlib.util
from pathlib import Path
import unittest
import tempfile
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

    def test_decision_info_distinguishes_search_proposal_from_played_move(self):
        class Override:
            def play(self, board, *args, **kwargs):
                self_test.assertTrue(kwargs['info'] & chess.engine.INFO_PV)
                return chess.engine.PlayResult(chess.Move.from_uci('e2e4'), None,
                    info={'pv': [chess.Move.from_uci('d2d4')], 'string': 'guard selected e2e4'})
        self_test = self
        _, row = self.module.play([Override(), Override()], ['candidate', 'base'],
            ('start', chess.STARTING_FEN), chess.WHITE, chess.engine.Limit(time=.5), 1,
            decision_info=True)
        self.assertEqual(row['move_trace'][0]['move'], 'e2e4')
        self.assertEqual(row['move_trace'][0]['decision'],
            {'searched_move': 'd2d4', 'diagnostic': 'guard selected e2e4'})

    def test_mate_at_ply_cap_is_completed(self):
        class Mate:
            def play(self, *args, **kwargs):
                return chess.engine.PlayResult(chess.Move.from_uci('f7g7'), None)
        _, row = self.module.play([Mate(), Mate()], ['a', 'b'], ('mate', '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'), chess.WHITE, chess.engine.Limit(depth=1), 1)
        self.assertEqual(row['candidate_score'], 1)
        self.assertEqual(row['termination'], 'CHECKMATE')

    def test_shared_book_applies_to_both_sides_then_preserves_search_history(self):
        class Engine:
            def __init__(self):
                self.histories = []
            def play(self, board, *args, **kwargs):
                self.histories.append(len(board.move_stack))
                return chess.engine.PlayResult(next(iter(board.legal_moves)), None)
        with tempfile.TemporaryDirectory() as directory:
            for c in 'abcde':
                Path(directory, c + '.tsv').write_text('eco\tname\tpgn\n')
            Path(directory, 'a.tsv').write_text('eco\tname\tpgn\nC20\tTest\t1. e4 e5 2. Nf3 Nc6\n')
            book = self.module.SharedOpeningBook(directory, plies=2, seed=42)
            engines = [Engine(), Engine()]
            _, row = self.module.play(engines, ['candidate', 'base'], ('start', chess.STARTING_FEN),
                                     chess.WHITE, chess.engine.Limit(time=.5), 3, book)
            self.assertEqual([r['book'] for r in row['move_trace']], [True, True, False])
            self.assertEqual([r['move'] for r in row['move_trace'][:2]], ['e2e4', 'e7e5'])
            self.assertEqual(engines[0].histories, [2])
            self.assertEqual(engines[1].histories, [])
            # Reversing player identities retains the same declared book line.
            _, second = self.module.play([Engine(), Engine()], ['base', 'candidate'],
                ('start', chess.STARTING_FEN), chess.BLACK, chess.engine.Limit(time=.5), 2, book)
            self.assertEqual([r['move'] for r in second['move_trace']], ['e2e4', 'e7e5'])


if __name__ == '__main__':
    unittest.main()
