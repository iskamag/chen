"""CPU admission/parser/lifecycle tests; no real teacher scores or GPU work."""
import dataclasses
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

import chess
import chess.pgn

from neural.features import move_to_index
from neural.semantic_data import (GamePosition, StockfishSession, PARENT_ACTION_CP,
    STANDALONE_CP, admit_families, common_cp_labels, family_at, family_positions,
    index_games, parse_pv, sha256_file)


def position(moves=()):
    board = chess.Board()
    for uci in moves:
        board.push_uci(uci)
    return GamePosition('fixture', len(moves), board.fen(en_passant='fen'), tuple(moves), ())


def packet(pos, depth=7, *, score=200, all_legal=True):
    board = chess.Board(pos.fen)
    moves = sorted(move.uci() for move in board.legal_moves)
    if not all_legal:
        moves = moves[:1]
    return [f'info depth {depth} seldepth {depth+3} multipv {rank} score cp {score-rank} tbhits 0 nodes 32000 pv {move}'
            for rank, move in enumerate(moves, 1)]


def pgn_game(seed, site):
    rng, board = random.Random(seed), chess.Board()
    game = chess.pgn.Game()
    game.headers.update(Event='Rated classical game', Site='https://lichess.org/' + site,
                        WhiteElo='2000', BlackElo='2100', TimeControl='600+5')
    node = game
    for _ in range(24):
        moves = list(board.legal_moves)
        if not moves or board.is_game_over():
            break
        move = rng.choice(moves)
        board.push(move); node = node.add_variation(move)
    game.comment = 'Unicode café; [%eval 99.9] not a label'
    return str(game) + '\n\n'


class AdmissionTests(unittest.TestCase):
    def test_family_history_source_and_split(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.pgn'
            path.write_text(''.join(pgn_game(seed, f'{seed:08d}') for seed in range(12)), encoding='utf8')
            index, stats = index_games(path)
            self.assertEqual(stats['games_scanned'], 12)
            family, reason = family_at(path, index[0], plies=(8, 12), action_root_ply=12)
            self.assertIsNone(reason)
            for observed in family.positions:
                self.assertEqual(len(observed.moves), observed.ply)
                self.assertEqual(len(observed.previous_hashes), observed.ply)
                board = chess.Board()
                for move in observed.moves: board.push_uci(move)
                self.assertEqual(board.fen(en_passant='fen'), observed.fen)
            raw = path.read_bytes()[family.source_offset:family.source_end_offset]
            self.assertEqual(hashlib.sha256(raw).hexdigest(), family.game_sha256)
            boundary, reason = family_at(path, index[0], plies=(8, 24), action_root_ply=8, min_game_plies=24)
            self.assertIsNone(reason)
            self.assertEqual(boundary.positions[-1].ply, 24)
            groups, manifest = admit_families(path, counts={'train': 1, 'validation': 1, 'test': 1},
                                            plies=(8, 12), action_root_ply=8)
            keys = {name: set().union(*(family_positions(f) for f in families)) for name, families in groups.items()}
            self.assertFalse(keys['train'] & keys['validation'])
            self.assertFalse(keys['test'] & (keys['train'] | keys['validation']))
            self.assertFalse(manifest['outcomes_comments_embedded_evaluations_used_as_labels'])
            for families in groups.values():
                self.assertEqual([chess.Board(p.fen).turn for p in families[0].positions].count(chess.WHITE), 1)
                self.assertEqual([chess.Board(p.fen).turn for p in families[0].positions].count(chess.BLACK), 1)
                self.assertIn(families[0].positions[families[0].action_root_index].ply, (8, 9))
            groups2, manifest2 = admit_families(path, counts={'train': 1, 'validation': 1, 'test': 1},
                                              plies=(8, 12), action_root_ply=8)
            self.assertEqual(groups, groups2)
            self.assertEqual(manifest['source_sha256'], manifest2['source_sha256'])
            spent = Path(directory) / 'spent.json'
            spent.write_text(json.dumps({'canonical_position_fens': [groups['train'][0].positions[0].fen]}))
            fresh, fresh_manifest = admit_families(path, counts={'train': 1, 'validation': 1, 'test': 1},
                                                  plies=(8, 12), action_root_ply=8, spent_reports=[spent])
            self.assertNotEqual(fresh['train'][0].game_id, groups['train'][0].game_id)
            self.assertGreater(fresh_manifest['rejections'].get('spent_input_collision', 0), 0)

    def test_header_caps_and_illegal_or_short_family(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'data.pgn'
            base = pgn_game(0, 'abcDEF01')
            path.write_text(base + base.replace('abcDEF01', 'abcDEF02').replace('600+5', '30+0') +
                            base.replace('abcDEF01', 'abcDEF03').replace('2000', '1700'))
            index, stats = index_games(path)
            self.assertEqual(len(index), 1)
            self.assertEqual(stats['rating_filter'], 1)
            self.assertEqual(stats['estimated_bullet_or_unbounded_control'], 1)
            with self.assertRaisesRegex(ValueError, 'cap'): index_games(path, max_games=1)
            _, reason = family_at(path, index[0])
            self.assertEqual(reason, 'game_length')


class LabelTests(unittest.TestCase):
    def test_complete_packets_not_mixed_updates_and_frames(self):
        root = position()
        lines = packet(root)
        result, error = common_cp_labels(lines + packet(root, depth=8)[:3], root, all_legal=True)
        self.assertIsNone(error)
        self.assertEqual(result['common_reported_depth'], 7)
        # An incomplete same-depth newer iteration cannot alter old rank1 only.
        changed = packet(root, score=800)[:1]
        result2, _ = common_cp_labels(lines + changed, root, all_legal=True)
        self.assertEqual(result2['rows'], result['rows'])
        self.assertEqual(result['target_kind'], PARENT_ACTION_CP)
        row = result['rows'][0]
        self.assertEqual(row['converted_score']['reported_uci_cp'], row['root_score_cp'])
        self.assertTrue(row['converted_score']['negated'])
        self.assertLess(row['converted_score']['value_cp'], 0)
        self.assertEqual(result['normalization_root_material'], 78)
        self.assertFalse(result['exact_minimax_or_uniform_horizon_claim'])
        standalone, _ = common_cp_labels(packet(root, all_legal=False), root)
        self.assertEqual(standalone['target_kind'], STANDALONE_CP)
        self.assertFalse(standalone['rows'][0]['converted_score']['negated'])

    def test_black_canonical_action_capture_material_and_raw_ep(self):
        root = position(('e2e4',))
        self.assertEqual(root.fen.split()[3], 'e3')
        result, error = common_cp_labels(packet(root), root, all_legal=True)
        self.assertIsNone(error)
        self.assertEqual(result['root_side_to_move'], 'b')
        for row in result['rows']:
            self.assertEqual(row['canonical_action'], move_to_index(row['uci'], 'b'))
        capture = position(('e2e4', 'd7d5', 'e4d5'))
        result, _ = common_cp_labels(packet(capture), capture, all_legal=True)
        self.assertEqual(result['normalization_root_material'], 77)
        child = chess.Board(capture.fen)
        # Parent normalizer stays77 even if the evaluated action captures.
        capturing = next(m for m in child.legal_moves if child.is_capture(m))
        row = next(r for r in result['rows'] if r['uci'] == capturing.uci())
        self.assertEqual(row['converted_score']['normalization_root_material'], 77)
        with self.assertRaisesRegex(ValueError, 'mismatch'):
            common_cp_labels(packet(root), dataclasses.replace(root, moves=()), all_legal=True)

    def test_incomplete_duplicate_mate_bound_extreme_and_bad_pv(self):
        root = position(); lines = packet(root)
        _, error = common_cp_labels(lines[:-1], root, all_legal=True)
        self.assertEqual(error, 'no_eligible_ordinary_complete_packet')
        for replace, expected in [('score cp 199', 'decisive_mate_tablebase_or_extreme_evidence'),
                                  ('score cp 199 lowerbound', 'no_eligible_ordinary_complete_packet'),
                                  ('score cp 20000', 'decisive_mate_tablebase_or_extreme_evidence')]:
            changed = lines[:]
            changed[0] = changed[0].replace('score cp 199', replace if replace != 'score cp 199' else 'score mate 3')
            _, error = common_cp_labels(changed, root, all_legal=True)
            self.assertEqual(error, expected)
        changed = lines[:]; changed[-1] = changed[-1].rsplit(' pv ', 1)[0] + ' pv ' + lines[0].split(' pv ')[1]
        _, error = common_cp_labels(changed, root, all_legal=True)
        self.assertEqual(error, 'nonunique_or_incomplete_legal_actions')
        changed = lines[:]; changed[0] += ' e2e4'
        _, error = common_cp_labels(changed, root, all_legal=True)
        self.assertEqual(error, 'illegal_packet_PV')
        self.assertIsNone(parse_pv('info string score cp 1 pv a2a3'))
        changed = lines[:]; changed[0] = changed[0].replace('tbhits 0', 'tbhits 1')
        _, error = common_cp_labels(changed, root, all_legal=True)
        self.assertEqual(error, 'decisive_mate_tablebase_or_extreme_evidence')


FAKE = '''#!/usr/bin/env python3
import sys, time
for line in sys.stdin:
 c=line.strip()
 if c=='uci': print('id name Stockfish 19 fixture'); print('uciok',flush=True)
 elif c=='isready': print('readyok',flush=True)
 elif c.startswith('go '):
  if MODE=='timeout': time.sleep(5)
  elif MODE=='eof': sys.exit(0)
  else: print('info depth 7 seldepth 9 multipv 1 score cp 10 tbhits 0 nodes 32000 pv e2e4'); print('bestmove e2e4',flush=True)
 elif c=='quit': break
'''


class SessionTests(unittest.TestCase):
    def executable(self, directory, mode):
        path = Path(directory) / 'fake-stockfish'
        path.write_text(FAKE.replace("import sys, time", "import sys, time\nMODE=" + repr(mode)))
        path.chmod(0o755)
        return path

    def test_no_implicit_search_exact_commands_and_clean_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = self.executable(directory, 'ok')
            with StockfishSession(binary, expected_sha256=sha256_file(binary), timeout=.5) as session:
                self.assertFalse(any('go nodes' in x for x in session.setup_trace))
                self.assertIn('> setoption name SyzygyPath value', session.setup_trace)
                with self.assertRaises(ValueError): session.analyze(position(), nodes=32000, declaration_sha256='bad')
                with self.assertRaises(ValueError): session.send('go nodes 2\nquit')
                lines = session.analyze(position(), nodes=32000, declaration_sha256='a'*64)
                trace = session.take_trace()
                self.assertIn('> ucinewgame', trace)
                self.assertIn('> position startpos', trace)
                self.assertIn('> go nodes 32000', trace)
                self.assertTrue(lines[-1].startswith('bestmove'))
            self.assertIsNotNone(session.process.poll())
            self.assertTrue(session.process.stdout.closed)

    def test_timeout_eof_kills_and_prevents_reuse(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode, exception in [('timeout', TimeoutError), ('eof', RuntimeError)]:
                binary = self.executable(directory, mode)
                session = StockfishSession(binary, expected_sha256=sha256_file(binary), timeout=.1)
                with self.assertRaises(exception):
                    session.analyze(position(), nodes=32, declaration_sha256='a'*64)
                self.assertIsNotNone(session.process.poll())
                self.assertTrue(session.process.stdout.closed)
                session.close()
            with self.assertRaisesRegex(ValueError, 'SHA'):
                StockfishSession(binary, expected_sha256='0'*64)


if __name__ == '__main__': unittest.main()
