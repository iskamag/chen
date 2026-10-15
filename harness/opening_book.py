"""A shared, controller-supplied book from pinned Lichess ECO TSV files.

Both engines receive the resulting move history before searching. This is not
a Stockfish native option; a common controller book gives both sides the same
data, selection rule and half-move cutoff.
"""
from collections import Counter, defaultdict
import hashlib
import io
from pathlib import Path

import chess
import chess.pgn


def position_key(board):
    return ' '.join(board.fen(en_passant='legal').split()[:4])


class SharedOpeningBook:
    def __init__(self, directory, plies=12, seed=0):
        if not 1 <= plies <= 64:
            raise ValueError('book plies must be 1..64')
        self.plies, self.seed = plies, seed
        self.edges = defaultdict(Counter)
        self.files = {}
        self.lines = 0
        for letter in 'abcde':
            path = Path(directory) / (letter + '.tsv')
            self.files[str(path.resolve())] = hashlib.sha256(path.read_bytes()).hexdigest()
            for line in path.read_text().splitlines():
                fields = line.split('\t', 2)
                if len(fields) != 3 or fields[0] == 'eco':
                    continue
                game = chess.pgn.read_game(io.StringIO(fields[2]))
                if game is None or game.errors:
                    raise ValueError(f'invalid book line: {line}')
                board = game.board()
                for index, move in enumerate(game.mainline_moves()):
                    if index >= plies:
                        break
                    self.edges[position_key(board)][move.uci()] += 1
                    board.push(move)
                self.lines += 1
        if not self.edges:
            raise ValueError('opening book has no moves')

    def choose(self, board):
        if len(board.move_stack) >= self.plies:
            return None
        choices = self.edges.get(position_key(board))
        if not choices:
            return None
        legal = {move.uci() for move in board.legal_moves}
        weighted = [(move, weight) for move, weight in sorted(choices.items()) if move in legal]
        if not weighted:
            return None
        if self.seed == 0:
            move = max(weighted, key=lambda item: item[1])[0]
        else:
            key = f'{self.seed}|{position_key(board)}'.encode()
            pick = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), 'little') % sum(w for _, w in weighted)
            for move, weight in weighted:
                if pick < weight:
                    break
                pick -= weight
        return chess.Move.from_uci(move)

    def metadata(self):
        return {'kind': 'shared_controller_opening_book', 'source': 'CC0 Lichess chess-openings ECO TSV',
                'plies': self.plies, 'seed': self.seed, 'lines': self.lines,
                'positions': len(self.edges), 'files_sha256': self.files,
                'selection': 'most common for seed 0, otherwise deterministic weighted position-based selection',
                'applies_to': ['base', 'candidate']}
