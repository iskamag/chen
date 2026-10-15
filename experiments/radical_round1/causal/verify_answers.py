#!/usr/bin/env python3
"""Independent python-chess attack-union differential; never part of timing."""
import argparse
import json
import hashlib
import chess
p = argparse.ArgumentParser()
p.add_argument('answers')
p.add_argument('--out')
a = p.parse_args()
rows = 0
for line in open(a.answers):
    row = json.loads(line)
    board = chess.Board(row['fen'])
    expected = [0, 0]
    for square, piece in board.piece_map().items():
        expected[0 if piece.color else 1] |= int(board.attacks(square))
    assert row['white'] == expected[0], (row['id'], 'white', row)
    assert row['black'] == expected[1], (row['id'], 'black', row)
    rows += 1
result = {'oracle': 'python-chess', 'version': chess.__version__, 'rows': rows,
          'boolean_answers': rows * 128, 'all_exact': True,
          'answers_sha256': hashlib.sha256(open(a.answers, 'rb').read()).hexdigest()}
text = json.dumps(result, sort_keys=True) + '\n'
if a.out:
    open(a.out, 'w').write(text)
print(text, end='')
