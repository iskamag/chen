#!/usr/bin/env python3
"""Differential check of codec legal/check relations against python-chess.

Uses seeded reachable trajectories and explicit rare-rule roots. This is rules
coverage, not a playing-strength sample or a proof over every chess position.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import subprocess
import time

import chess

FIXTURES = [
    'r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1',
    '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1',
    '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
    'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
    'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1',
    '7k/8/8/r4pPK/8/8/8/8 w - f6 0 1',
    '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
    '4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1',
    '4k3/8/8/8/1b6/8/4r3/4K3 w - - 0 1',
    '7k/6Q1/6K1/8/8/8/8/8 b - - 0 1',
    '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
    '1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
    '4k3/8/8/8/8/8/p7/1R2K3 b - - 0 1',
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--games', type=int, default=64)
    parser.add_argument('--plies', type=int, default=128)
    parser.add_argument('--seed', type=int, default=20261015)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    if args.games < 1 or args.plies < 1:
        parser.error('games and plies must be positive')
    rng = random.Random(args.seed)
    fens = list(FIXTURES)
    for game in range(args.games):
        board = chess.Board(FIXTURES[game % len(FIXTURES)]) if game % 4 == 0 else chess.Board()
        for _ in range(args.plies):
            fens.append(board.fen(en_passant='fen'))
            moves = list(board.legal_moves)
            if not moves:
                break
            board.push(rng.choice(moves))
    args.out.mkdir(parents=True, exist_ok=False)
    text = '\n'.join(fens) + '\n'
    (args.out / 'positions.fen').write_text(text)
    started = time.perf_counter()
    run = subprocess.run([str(args.oracle.resolve())], input=text, text=True,
                         capture_output=True, timeout=180, check=True)
    (args.out / 'oracle.stdout').write_text(run.stdout)
    (args.out / 'oracle.stderr').write_text(run.stderr)
    lines = run.stdout.splitlines()
    assert len(lines) == len(fens), (len(lines), len(fens))
    legal_count = 0
    checking_count = 0
    special = {'castling': 0, 'en_passant': 0, 'promotion': 0, 'in_check': 0}
    for index, (fen, line) in enumerate(zip(fens, lines)):
        board = chess.Board(fen)
        assert board.is_valid(), (index, fen, board.status())
        expected = {move.uci(): board.gives_check(move) for move in board.legal_moves}
        got = {}
        for item in line.split():
            move, check = item.split(':')
            assert move not in got, (index, 'duplicate', move)
            assert check in ('0', '1')
            got[move] = check == '1'
        if expected != got:
            failure = {'index': index, 'fen': fen, 'expected': expected, 'got': got}
            (args.out / 'failure.json').write_text(json.dumps(failure, indent=2) + '\n')
            raise AssertionError(f'codec oracle mismatch at row {index}: {fen}; see failure.json')
        legal_count += len(expected)
        checking_count += sum(expected.values())
        special['in_check'] += board.is_check()
        for move in board.legal_moves:
            special['castling'] += board.is_castling(move)
            special['en_passant'] += board.is_en_passant(move)
            special['promotion'] += move.promotion is not None
    summary = {
        'positions': len(fens), 'legal_actions': legal_count, 'checking_actions': checking_count,
        'special_actions': special, 'seed': args.seed, 'games': args.games,
        'max_plies': args.plies, 'seconds': time.perf_counter() - started,
        'python_chess': chess.__version__, 'oracle_sha256': sha(args.oracle),
        'script_sha256': sha(__file__),
        'input_sha256': hashlib.sha256(text.encode()).hexdigest(),
        'scope': 'Exact legal action sets and check effects on sampled roots; no strength claim',
    }
    (args.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
