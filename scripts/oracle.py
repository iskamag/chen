#!/usr/bin/env python3
"""Optional independent move/state oracle. Requires python-chess only for QA."""
import argparse
import random
import subprocess
import chess
import chess.engine


def perft(board, depth):
    if depth == 0:
        return 1
    count = 0
    for move in list(board.legal_moves):
        board.push(move)
        count += perft(board, depth - 1)
        board.pop()
    return count


def check(binary, board):
    fen = board.fen(en_passant='fen')
    rows = subprocess.check_output([binary, 'divide', '2', fen], text=True).splitlines()
    observed = dict((row.split(': ')[0], int(row.split(': ')[1])) for row in rows if ': ' in row)
    expected = {}
    for move in list(board.legal_moves):
        board.push(move)
        expected[move.uci()] = perft(board, 1)
        board.pop()
    assert observed == expected, (fen, observed, expected)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--engine', default='./target/release/chen')
    parser.add_argument('--seed', type=int, default=20260930)
    parser.add_argument('--games', type=int, default=8)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    checks = 0
    for fen in ['r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1', '8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1', '8/8/8/r4pPK/8/8/8/7k w - f6 0 1', '1r5k/P7/8/8/8/8/8/7K w - - 0 1']:
        check(args.engine, chess.Board(fen))
        checks += 1
    for _ in range(args.games):
        board = chess.Board()
        sequence = []
        for ply in range(96):
            moves = list(board.legal_moves)
            if not moves:
                break
            move = rng.choice(moves)
            sequence.append(move.uci())
            board.push(move)
            if ply % 8 == 0:
                check(args.engine, board)
                # UCI replay exercises the actual protocol's board transition path.
                session = 'position startpos moves ' + ' '.join(sequence) + '\nd\nquit\n'
                lines = subprocess.check_output([args.engine], input=session, text=True).splitlines()
                assert 'info string ' + board.fen(en_passant='fen') in lines, (sequence, lines)
                checks += 1
    # A short protocol/legality game; this does not estimate strength.
    plies = 0
    with chess.engine.SimpleEngine.popen_uci(args.engine, timeout=10) as engine:
        engine.configure({'Hash': 1})
        board = chess.Board()
        for _ in range(60):
            if board.is_game_over(claim_draw=True):
                break
            result = engine.play(board, chess.engine.Limit(depth=2))
            assert result.move in board.legal_moves
            board.push(result.move)
            plies += 1
    print(f'PASS: {checks} independent legal-move/depth-2 divide comparisons + UCI FEN replays; {plies} legal self-play plies; seed={args.seed}')


if __name__ == '__main__':
    main()
