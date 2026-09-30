import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest

import chess
import chess.pgn

from harness.fastchess import audit_configuration, audit_pgn, audit_scoreboard, decision, validate_sprt, normalized_opening_fen, opening_identity
from scripts.fastchess_match import launcher, retire

FEN = '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'
DECLARATION = {'elo0': 0, 'elo1': 5, 'alpha': .05, 'beta': .05,
               'model': 'normalized', 'sampling_design': 'Independent held-out opening-pair sample'}


def mate_game(candidate_white, round_id=1, fen=FEN):
    board = chess.Board(fen)
    game = chess.pgn.Game()
    game.setup(board)
    game.headers.update({'White': 'candidate' if candidate_white else 'base',
                         'Black': 'base' if candidate_white else 'candidate',
                         'Round': str(round_id), 'Termination': 'normal', 'PlyCount': '1'})
    move = next(move for move in board.legal_moves if board.gives_check(move)
                and _is_mate(board, move))
    board.push(move)
    game.add_variation(move)
    game.headers['Result'] = board.result()
    return game


def _is_mate(board, move):
    child = board.copy()
    child.push(move)
    return child.is_checkmate()


class FastchessAuditTests(unittest.TestCase):
    def audit(self, games, cap=2):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'games.pgn'
            path.write_text('\n\n'.join(str(game) for game in games) + '\n\n')
            return audit_pgn(path, [('mate', FEN)], cap)

    def test_valid_pair_and_independent_scoreboard(self):
        rows, pairs = self.audit([mate_game(True), mate_game(False)])
        self.assertEqual(pairs, [(1., 0.)])
        self.assertEqual([row['termination'] for row in rows], ['CHECKMATE', 'CHECKMATE'])
        stats = {'wins': 1, 'losses': 1, 'draws': 0, 'penta_WW': 0, 'penta_LL': 0,
                 'penta_WD': 0, 'penta_LD': 0, 'penta_DD': 0, 'penta_WL': 1}
        state = {'stats': {'candidate vs base': stats}}
        self.assertEqual(audit_scoreboard(state, pairs), stats)
        state['stats']['candidate vs base']['wins'] = 2
        with self.assertRaises(ValueError):
            audit_scoreboard(state, pairs)

    def test_ineffective_ep_normalization_preserves_declared_opening(self):
        raw = '7k/5Q2/6K1/p7/8/8/8/8 w - a6 0 12'
        games = [mate_game(True, fen=raw), mate_game(False, fen=raw)]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'games.pgn'
            path.write_text('\n\n'.join(str(game) for game in games) + '\n\n')
            rows, pairs = audit_pgn(path, [('ineffective-ep', raw)], 2)
        self.assertEqual(pairs, [(1., 0.)])
        self.assertEqual([row['declared_opening_fen'] for row in rows], [raw, raw])
        self.assertTrue(all(row['initial_fen'].split()[3] == '-' for row in rows))
        self.assertTrue(all(row['initial_fen'].endswith('0 12') for row in rows))

    def test_effective_ep_cannot_be_dropped_from_game(self):
        raw = '7k/5Q2/6K1/pP6/8/8/8/8 w - a6 0 12'
        self.assertTrue(chess.Board(raw).has_legal_en_passant())
        games = [mate_game(True, fen=raw), mate_game(False, fen=raw)]
        games[0].headers['FEN'] = raw.replace('a6', '-')
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'games.pgn'
            path.write_text('\n\n'.join(str(game) for game in games) + '\n\n')
            with self.assertRaises(ValueError):
                audit_pgn(path, [('effective-ep', raw)], 2)

    def test_ep_identity_uses_legal_capture_and_preserves_clock_rights(self):
        ineffective = '7k/5Q2/6K1/p7/8/8/8/8 w - a6 0 12'
        legal = '7k/5Q2/6K1/pP6/8/8/8/8 w - a6 0 12'
        pinned = '4k3/8/8/r4pPK/8/8/8/8 w - f6 0 12'
        self.assertTrue(chess.Board(pinned).has_pseudo_legal_en_passant())
        self.assertFalse(chess.Board(pinned).has_legal_en_passant())
        for fen in (ineffective, pinned):
            target = fen.split()[3]
            without = fen.replace(' ' + target + ' ', ' - ')
            self.assertEqual(opening_identity(fen), opening_identity(without))
            self.assertEqual(normalized_opening_fen(fen), without)
        self.assertEqual(normalized_opening_fen(legal), legal)
        self.assertNotEqual(opening_identity(legal), opening_identity(legal.replace('a6', '-')))
        clocks = chess.Board().fen().replace('0 1', '17 42')
        self.assertEqual(normalized_opening_fen(clocks), clocks)

    def test_forfeit_is_failure_even_with_legal_mate_moves(self):
        for termination in ('time forfeit', 'abandoned', 'illegal move', 'adjudication', 'unterminated'):
            game = mate_game(True)
            game.headers['Termination'] = termination
            with self.subTest(termination=termination), self.assertRaises(ValueError):
                self.audit([game, mate_game(False)])

    def test_wrong_endpoint_pair_color_round_and_clock_rejected(self):
        for field, value in [('Result', '1/2-1/2'), ('PlyCount', '2'), ('Round', '2'),
                             ('White', 'base'), ('FEN', FEN.replace('0 1', '1 1'))]:
            game = mate_game(True)
            game.headers[field] = value
            with self.subTest(field=field), self.assertRaises(ValueError):
                self.audit([game, mate_game(False)])
        with self.assertRaises(ValueError):
            self.audit([mate_game(True)])
        with self.assertRaises(ValueError):
            self.audit([mate_game(True), mate_game(True)])

    def test_stop_classifier_never_promotes_game_cap_or_failed_run(self):
        console = 'SPRT (0, 5) completed - H1 was accepted'
        self.assertEqual(decision(console, 100, 100, DECLARATION, 0), 'inconclusive_game_cap')
        self.assertEqual(decision(console, 98, 100, DECLARATION, 0), 'external_sprt_H1')
        self.assertEqual(decision(console, 98, 100, DECLARATION, 1), 'technical_failure')
        self.assertEqual(decision(console, 98, 100, DECLARATION, 0, True), 'technical_failure')
        self.assertEqual(decision('', 98, 100, DECLARATION, 0), 'unexpected_early_stop')
        self.assertEqual(decision(console + '\nSPRT (0, 5) completed - H0 was accepted',
                                  98, 100, DECLARATION, 0), 'unexpected_early_stop')
        self.assertEqual(decision(console, 100, 100, None, 0), 'diagnostic_complete')

    def test_predeclaration_validation(self):
        self.assertEqual(validate_sprt(DECLARATION), DECLARATION)
        for key, value in [('elo0', 5), ('elo1', float('nan')), ('alpha', 0),
                           ('beta', True), ('sampling_design', ''), ('model', 'bayesian')]:
            declaration = {**DECLARATION, key: value}
            with self.subTest(key=key), self.assertRaises(ValueError):
                validate_sprt(declaration)

    def test_body_result_corruption_or_missing_token_rejected(self):
        original = str(mate_game(True)) + '\n\n' + str(mate_game(False)) + '\n\n'
        for changed in (original.replace('# 1-0', '# 0-1', 1), original.replace('# 1-0', '#', 1)):
            with tempfile.TemporaryDirectory() as folder:
                path = Path(folder) / 'games.pgn'
                path.write_text(changed)
                with self.assertRaises(ValueError):
                    audit_pgn(path, [('mate', FEN)], 2)

    def test_native_configuration_must_match_frozen_experiment(self):
        state = {'concurrency': 1, 'games': 2, 'rounds': 50, 'seed': 1,
                 'report_penta': True, 'noswap': False, 'reverse': False,
                 'recover': False, 'variant': 0, 'type': 0,
                 **{name: {'enabled': False} for name in ('draw', 'resign', 'maxmoves', 'tb_adjudication')},
                 'sprt': {**DECLARATION, 'enabled': True}}
        audit_configuration(state, 100, 1, DECLARATION)
        for key, value in [('rounds', 49), ('concurrency', 2), ('noswap', True), ('recover', True)]:
            with self.subTest(key=key), self.assertRaises(ValueError):
                audit_configuration({**state, key: value}, 100, 1, DECLARATION)
        with self.assertRaises(ValueError):
            audit_configuration({**state, 'sprt': {**state['sprt'], 'elo1': 10}}, 100, 1, DECLARATION)
        with self.assertRaises(ValueError):
            audit_configuration({**state, 'draw': {'enabled': True}}, 100, 1, DECLARATION)

    def test_launcher_preserves_literal_arguments_cwd_and_environment(self):
        with tempfile.TemporaryDirectory(prefix='chen launcher ') as folder:
            root = Path(folder)
            program = root / 'capture.py'
            result = root / 'result.txt'
            program.write_text('import os,sys,json\nopen(sys.argv[1],"w").write(json.dumps([sys.argv[2:],os.getcwd(),os.environ.get("CHEN_TEST_LITERAL"),os.environ.get("CHEN_TEST_REMOVE")]))\n')
            literal = 'spaces `echo no` $(echo no) "quotes"\nnewline'
            entry = {'command': [os.sys.executable, str(program), str(result), literal],
                     'cwd': str(root), 'env': {'CHEN_TEST_LITERAL': literal, 'CHEN_TEST_REMOVE': None}}
            adapter, _ = launcher(entry, 'candidate', root)
            subprocess.run([str(adapter)], env={**os.environ, 'CHEN_TEST_REMOVE': 'remove'}, check=True)
            import json
            self.assertEqual(json.loads(result.read_text()), [[literal], str(root), literal, None])

    def test_configured_path_relative_to_engine_cwd(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            (root / 'bin').mkdir()
            executable = root / 'bin' / 'chen-fixture'
            executable.write_text('#!/bin/sh\nexit 0\n')
            executable.chmod(0o755)
            adapter, record = launcher({'command': ['chen-fixture'], 'cwd': str(root),
                                        'env': {'PATH': 'bin'}}, 'candidate', root)
            self.assertEqual(record['command'][0], str(executable))
            subprocess.run([str(adapter)], check=True)
            for command in (['docker', 'exec', 'container', 'engine'], ['./scripts/rocm_run.sh', 'uci']):
                with self.assertRaises(ValueError):
                    launcher({'command': command}, 'candidate', root)

    def test_normal_parent_exit_does_not_leave_live_owned_child(self):
        code = ('import subprocess,sys; p=subprocess.Popen([sys.executable,"-c",'
                '"import time; time.sleep(60)"],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL);'
                'print(p.pid,flush=True)')
        process = subprocess.Popen([os.sys.executable, '-c', code], stdout=subprocess.PIPE,
                                   text=True, start_new_session=True)
        try:
            child_pid = int(process.stdout.readline())
            process.wait(timeout=5)
            report = retire(process)
            self.assertTrue(report['sigterm_sent'])
            for _ in range(50):
                status = Path(f'/proc/{child_pid}/status')
                if not status.exists() or '\nState:\tZ' in status.read_text():
                    break
                time.sleep(.01)
            else:
                self.fail('owned child still live after normal parent retirement')
        finally:
            retire(process)
            process.stdout.close()


if __name__ == '__main__':
    unittest.main()
