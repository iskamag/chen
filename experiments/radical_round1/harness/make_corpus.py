#!/usr/bin/env python3
"""Deterministic CPU architecture corpus with exact python-chess obligations.

DEV supports construction/debugging. CERT is reserved for root-run frozen
variants. Constructed positions are locally valid, not claimed reachable.
Positive/negative tactical stratification is a challenge design, not game sampling.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import time

import chess

FIXTURES = [
 ('start','rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'),
 ('kiwipete','r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1'),
 ('rook-end','8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1'),
 ('middlegame','r1bq1rk1/pp2bppp/2n1pn2/2pp4/3P4/2PBPN2/PP1N1PPP/R1BQ1RK1 w - - 2 8'),
 ('promotion','4k3/P7/8/8/8/8/7p/4K3 w - - 0 1'),
 ('mate-one','7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'),
 ('mate','7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'),
 ('stalemate','7k/5Q2/6K1/8/8/8/8/8 b - - 0 1'),
 ('castle','r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'),
 ('castle-transit','4kr2/8/8/8/8/8/8/R3K2R w KQ - 0 1'),
 ('castle-in-check','4k3/8/8/8/8/8/4r3/R3K2R w KQ - 0 1'),
 ('ep','4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),
 ('ep-pin','4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),
 ('ep-rook-discovery','7k/8/8/r4pPK/8/8/8/8 w - f6 0 1'),
 ('ep-check-evasion','4k3/8/8/3pP3/4K3/8/8/8 w - d6 0 1'),
 ('promotion-capture','1r2k3/P7/8/8/8/8/7p/4K3 w - - 0 1'),
 ('double-check','4k3/8/8/8/1b6/8/4r3/4K3 w - - 0 1'),
 ('phase-promotions','k7/8/8/8/1Q1Q1Q1Q/8/1Q1Q1Q1Q/7K w - - 0 1'),
 ('clock99','4k3/8/8/8/8/8/8/R3K3 w - - 99 1'),
 ('clock100','4k3/8/8/8/8/8/8/R3K3 w - - 100 1'),
]


def fen(board):
    return board.fen(en_passant='fen')


def identity(board):
    return ' '.join(fen(board).split()[:4])


def oracle(board):
    legal = sorted(board.legal_moves, key=lambda move: move.uci())
    checking, mating, stalemating = [], [], []
    for move in legal:
        board.push(move)
        check = board.is_check()
        if check:
            checking.append(move.uci())
        if next(iter(board.legal_moves), None) is None:
            (mating if check else stalemating).append(move.uci())
        board.pop()
    return {'legal': [move.uci() for move in legal], 'checking': checking,
            'mates': mating, 'stalemates': stalemating,
            'root_check': board.is_check(), 'root_mate': not legal and board.is_check(),
            'root_stalemate': not legal and not board.is_check(),
            'draw_clock': board.halfmove_clock >= 100,
            'insufficient_material': board.is_insufficient_material()}


def sparse(rng, kind):
    board = chess.Board(None)
    side = bool(rng.getrandbits(1))
    board.turn = side
    # Constrain defending king toward an edge in tactical strata, with randomized
    # geometry. This mines challenge motifs; it is not a natural-game distribution.
    defender = not side
    squares = list(range(64))
    enemy_square = rng.choice([0, 1, 6, 7, 8, 15, 48, 55, 56, 57, 62, 63])
    board.set_piece_at(enemy_square, chess.Piece(chess.KING, defender))
    own_square = rng.choice(squares)
    if own_square == enemy_square:
        return None
    board.set_piece_at(own_square, chess.Piece(chess.KING, side))
    if kind == 'promotion':
        specs = [(chess.PAWN, side), (rng.choice([chess.ROOK,chess.BISHOP,chess.KNIGHT]), defender)]
    elif kind == 'rook':
        specs = [(chess.ROOK, side), (rng.choice([chess.PAWN,chess.KNIGHT,chess.BISHOP]), defender)]
    elif kind == 'minor':
        specs = [(chess.KNIGHT, side),(chess.BISHOP, side),(chess.PAWN, defender)]
    else:
        specs = [(chess.QUEEN, side)]
        if rng.random() < 0.6:
            specs.append((rng.choice([chess.PAWN,chess.KNIGHT,chess.BISHOP,chess.ROOK]), defender))
    for piece, color in specs:
        choices = [sq for sq in squares if board.piece_at(sq) is None
                   and (piece != chess.PAWN or sq // 8 in (1,2,3,4,5,6))]
        if kind == 'promotion' and piece == chess.PAWN:
            choices = [sq for sq in choices if sq // 8 == (6 if color else 1)]
        if not choices:
            return None
        board.set_piece_at(rng.choice(choices), chess.Piece(piece, color))
    board.fullmove_number = 20
    return board if board.is_valid() else None


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=2026101501)
    args=parser.parse_args()
    args.out.mkdir(parents=True,exist_ok=True)
    started=time.perf_counter()
    all_seen=set()
    inventory={}
    for split, seed in [('dev',args.seed),('cert',args.seed+1_000_003)]:
        rng=random.Random(seed)
        rows=[]; events=[]; labels=[]
        def add(board,family,origin,known=None):
            key=identity(board)
            if family.startswith('fixture-clock'):
                key += '|clock=' + str(board.halfmove_clock)
            if key in all_seen:
                return None
            assert board.is_valid(), (family,fen(board),board.status())
            all_seen.add(key)
            rid=f'{split}-{len(rows):05d}'
            row={'id':rid,'split':split,'family':family,'fen':fen(board),'origin':origin}
            rows.append(row)
            labels.append({**row,**(oracle(board) if known is None else known)})
            return rid
        if split=='dev':
            for name,value in FIXTURES:
                board=chess.Board(value)
                add(board,'fixture-'+name,'authored-regression')
                add(board.mirror(),'fixture-'+name,'color-mirrored-regression')
        # Held-out trajectory roots come from disjoint seeds/games, not color
        # flips or different clocks of the DEV roots.
        for game in range(24):
            mode=['uniform','capture','quiet','check'][game%4]
            board=chess.Board()
            for ply in range(144):
                legal=sorted(board.legal_moves,key=lambda move:move.uci())
                if not legal:
                    break
                if ply>=8 and ply%6==2:
                    add(board,'reachable-'+mode,f'game-{seed}-{game}-ply-{ply}')
                candidates=legal
                if mode=='capture':
                    candidates=[m for m in legal if board.is_capture(m)] or legal
                elif mode=='quiet':
                    candidates=[m for m in legal if not board.is_capture(m) and not board.gives_check(m)] or legal
                elif mode=='check':
                    candidates=[m for m in legal if board.gives_check(m)] or legal
                move=rng.choice(candidates)
                before=fen(board)
                board.push(move)
                events.append({'id':f'{split}-g{game}-p{ply}','parent_id':f'{split}-g{game}-p{ply-1}',
                               'family':'trajectory-'+mode,'fen':before,'uci':move.uci(),
                               'after_fen':fen(board)})
                if board.is_game_over(claim_draw=True):
                    break
        for kind in ['queen','rook','minor','promotion']:
            counts={'mate':0,'nonmate':0}
            target={'mate':20 if kind!='minor' else 4,'nonmate':28}
            tries=0
            while any(counts[k]<target[k] for k in counts) and tries<40_000:
                tries+=1
                board=sparse(rng,kind)
                if board is None or identity(board) in all_seen:
                    continue
                answer=oracle(board)
                category='mate' if answer['mates'] else 'nonmate'
                if counts[category]>=target[category]:
                    continue
                if add(board,f'constructed-{kind}-{category}',f'sparse-rejection-{seed}-{tries}',answer):
                    counts[category]+=1
            inventory[f'{split}-sparse-{kind}']={'counts':counts,'target':target,'trials':tries}
        text='\n'.join('|'.join(row[k] for k in ['id','split','family','fen']) for row in rows)+'\n'
        (args.out/f'{split}.tsv').write_text(text)
        (args.out/f'{split}-oracle.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in labels))
        event_text='\n'.join('|'.join(row[k] for k in ['id','parent_id','family','fen','uci','after_fen']) for row in events)+'\n'
        (args.out/f'{split}-transitions.tsv').write_text(event_text)
        inventory[split]={'roots':len(rows),'events':len(events),
                         'mate_positive_roots':sum(bool(row['mates']) for row in labels),
                         'mate_actions':sum(len(row['mates']) for row in labels),
                         'legal_actions':sum(len(row['legal']) for row in labels),
                         'families':{family:sum(row['family']==family for row in rows) for family in sorted({row['family'] for row in rows})}}
    files={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in args.out.iterdir() if p.is_file() and p.name!='manifest.json'}
    report={'seed':args.seed,'python_chess':chess.__version__,'seconds':time.perf_counter()-started,
            'scope':'Exact board-local legal/check/mate1/stalemate obligations; no repetition-history or reachability claim for constructed roots; no strength inference',
            'generator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'inventory':inventory,'sha256':files}
    (args.out/'manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    main()
