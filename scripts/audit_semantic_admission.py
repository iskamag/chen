#!/usr/bin/env python3
"""Independent CPU replay of licensed semantic admissions; no teacher or model.

Requires the optional python-chess QA dependency. Reads only game/input fields
of spent reports, never their loss, score, outcome or selection metrics.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import io
import json
from pathlib import Path
import re

import chess
import chess.pgn


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1048576), b''):
            digest.update(block)
    return digest.hexdigest()


def feature_key(board):
    # Construct planes square-by-square rather than using the producer's
    # bitboard byte-flip implementation. Match actual schema1 input identity.
    planes = [0] * 12
    for square, piece in board.piece_map().items():
        relative = square ^ (56 if board.turn == chess.BLACK else 0)
        plane = piece.piece_type - 1 + (0 if piece.color == board.turn else 6)
        planes[plane] |= 1 << relative
    rights = [board.has_kingside_castling_rights(color) if kingside else
              board.has_queenside_castling_rights(color)
              for color in (board.turn, not board.turn) for kingside in (True, False)]
    ep = -1 if board.ep_square is None else board.ep_square ^ (56 if not board.turn else 0)
    return hashlib.sha256(json.dumps([*planes, *rights, ep, min(board.halfmove_clock, 100)],
                                    separators=(',', ':')).encode()).hexdigest()


def hash_key(offset):
    mask = (1 << 64) - 1
    value = (0x4348454E00000000 + offset + 0x9E3779B97F4A7C15) & mask
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & mask
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & mask
    return value ^ (value >> 31)


def repetition_hash(board):
    rights = sum(1 << i for i, (color, kingside) in enumerate(
        ((chess.WHITE, True), (chess.WHITE, False), (chess.BLACK, True), (chess.BLACK, False)))
        if (board.has_kingside_castling_rights(color) if kingside else
            board.has_queenside_castling_rights(color)))
    value = hash_key(768 + rights) ^ (hash_key(792) if board.turn == chess.BLACK else 0)
    if board.has_legal_en_passant():
        value ^= hash_key(784 + chess.square_file(board.ep_square))
    for square, piece in board.piece_map().items():
        plane = piece.piece_type - 1 + (0 if piece.color == chess.WHITE else 6)
        value ^= hash_key(plane * 64 + square)
    return value


def keys_at(board, depth):
    yield feature_key(board)
    if depth:
        for move in list(board.legal_moves):
            board.push(move)
            yield from keys_at(board, depth - 1)
            board.pop()


def audit(admission_path, pgn_path, spent_repo):
    if not __debug__:
        raise RuntimeError('admission auditor requires Python assertion checks enabled')
    path = Path(admission_path)
    raw = gzip.decompress(path.read_bytes()) if path.suffix == '.gz' else path.read_bytes()
    admission = json.loads(raw)
    assert admission['admission_revision'] == 2
    assert admission['teacher_labels_generated'] is False
    assert sha(pgn_path) == admission['pgn_sha256']
    assert hashlib.sha256(json.dumps(admission['source_records'], sort_keys=True).encode()).hexdigest() == admission['source_sha256']
    split_keys, all_ids, move_ids, result = {}, set(), set(), {}
    with Path(pgn_path).open('rb') as stream:
        for split in ('train', 'validation', 'test'):
            keys, movers, roots, children = set(), Counter(), Counter(), 0
            families = admission['source_records'][split]
            assert len(families) == admission['families'][split]
            for family in families:
                assert family['game_id'] not in all_ids
                assert family['moves_sha256'] not in move_ids
                all_ids.add(family['game_id']); move_ids.add(family['moves_sha256'])
                stream.seek(family['source_offset'])
                chunk = stream.read(family['source_end_offset'] - family['source_offset'])
                assert hashlib.sha256(chunk).hexdigest() == family['game_sha256']
                game = chess.pgn.read_game(io.StringIO(chunk.decode('utf-8')))
                assert game is not None and not game.errors
                assert game.headers.get('Variant', 'Standard').casefold() in ('standard', 'chess')
                assert game.headers.get('SetUp') != '1' and 'FEN' not in game.headers
                event = game.headers.get('Event', '').casefold()
                assert 'rated' in event and 'casual' not in event
                assert re.fullmatch(r'https://lichess\.org/[A-Za-z0-9]{8}', game.headers['Site'])
                assert hashlib.sha256(game.headers['Site'].encode()).hexdigest() == family['game_id']
                assert game.board().fen() == chess.STARTING_FEN
                moves = [move.uci() for move in game.mainline_moves()]
                assert moves == family['all_moves'] and 73 <= len(moves) <= 512
                assert hashlib.sha256(' '.join(moves).encode()).hexdigest() == family['moves_sha256']
                assert [int(game.headers[k]) for k in ('WhiteElo', 'BlackElo')] == family['ratings']
                assert min(family['ratings']) >= admission['filter']['both_min_rating']
                base, increment = map(int, game.headers['TimeControl'].split('+'))
                assert game.headers['TimeControl'] == family['time_control']
                assert base + 40 * increment >= admission['filter']['estimated_base_plus40increments_min_seconds']
                parity = int(family['game_id'][-1], 16) % 2
                plies = [ply + ((i + parity) % 2) for i, ply in enumerate(admission['plies'])]
                assert [row['ply'] for row in family['positions']] == plies
                observed = {row['ply']: row for row in family['positions']}
                board, history, family_keys = chess.Board(), [], set()
                for ply in range(max(plies) + 1):
                    if ply in observed:
                        row = observed[ply]
                        assert row['fen'] == board.fen(en_passant='fen')
                        assert row['moves'] == moves[:ply] and row['previous_hashes'] == history
                        assert row['game_id'] == family['game_id']
                        assert not board.is_game_over(claim_draw=False) and not board.is_repetition(3)
                        assert board.halfmove_clock < 100
                        key = feature_key(board)
                        assert key not in family_keys
                        family_keys.add(key); movers['white' if board.turn else 'black'] += 1
                    if ply < max(plies):
                        history.append(repetition_hash(board))
                        move = chess.Move.from_uci(moves[ply]); assert move in board.legal_moves
                        board.push(move)
                root_row = family['positions'][family['action_root_index']]
                assert root_row['ply'] == admission['action_root_ply'] + parity
                root = chess.Board(root_row['fen'])
                roots['white' if root.turn else 'black'] += 1
                children += root.legal_moves.count()
                family_keys.update(keys_at(root, 1))
                assert not family_keys & keys
                keys.update(family_keys)
            assert len(keys) == admission['canonical_counts'][split]
            assert dict(movers) == admission['mover_counts'][split]['observations']
            assert dict(roots) == admission['mover_counts'][split]['action_roots']
            assert children == admission['mover_counts'][split]['all_legal_child_rows']
            assert not any(keys & other for other in split_keys.values())
            split_keys[split] = keys
            result[split] = dict(families=len(families), observations=sum(movers.values()),
                                 canonical_inputs=len(keys), movers=dict(movers), roots=dict(roots),
                                 legal_children=children)
    spent_keys = set()
    for spent in admission['spent_reports']:
        path = Path(spent_repo) / spent['path']; assert sha(path) == spent['sha256']
        data = json.loads(path.read_text())
        if 'split' in data and 'source_records' in data['split']:
            for families in data['split']['source_records'].values():
                for family in families:
                    for row in family['roots']:
                        spent_keys.update(keys_at(chess.Board(row['fen']), 2))
        elif 'canonical_position_fens' in data:
            spent_keys.update(feature_key(chess.Board(fen)) for fen in data['canonical_position_fens'])
        elif data.get('kind') == 'cpu_only_frozen_plan_actionblind_batch_geometry_diagnostic':
            spent_keys.update(feature_key(chess.Board(row['root_fen']))
                              for run in data['runs'] for row in run['roots'])
        else:
            raise ValueError('unknown spent-input schema')
    assert len(spent_keys) == admission['spent_position_count']
    assert not any(keys & spent_keys for keys in split_keys.values())
    return dict(kind='independent_semantic_admission_replay', strength_evidence=False,
                admission_sha256=sha(admission_path), pgn_sha256=sha(pgn_path),
                auditor_sha256=sha(__file__), families=result,
                previous_hashes_and_raw_ep_fens_replayed=True, source_chunks_reparsed=True,
                spent_position_count=len(spent_keys), spent_reports=admission['spent_reports'],
                cross_split_overlap=0, spent_input_overlap=0,
                teacher_searches=0, model_training=False, heldout_score_metrics_read=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--admission', required=True)
    parser.add_argument('--pgn', required=True)
    parser.add_argument('--spent-repo', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    report = audit(args.admission, args.pgn, args.spent_repo)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
