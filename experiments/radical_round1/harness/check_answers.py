#!/usr/bin/env python3
"""Independent complete-mask audit of architecture answer JSONL against python-chess."""
import argparse
import hashlib
import json
from pathlib import Path
import chess

ALL=(1<<64)-1


def number(value):
    return int(value,0) if isinstance(value,str) else int(value)


def expected(board,color):
    union=0
    pinned=0
    for square,piece in board.piece_map().items():
        if piece.color==color:
            union |= board.attacks_mask(square)
            if piece.piece_type!=chess.KING and board.is_pinned(color,square):
                pinned |= 1<<square
    king=board.king(color)
    checkers=int(board.attackers(not color,king))
    evasion=ALL
    for checker in chess.scan_forward(checkers):
        evasion &= (1<<checker) | chess.between(king,checker)
    return union,checkers,pinned,evasion


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--answers',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--allow-partial',action='store_true')
    p.add_argument('--require-children',action='store_true')
    a=p.parse_args()
    corpus={}
    for line in a.corpus.read_text().splitlines():
        rid,split,family,fen=line.split('|',3)
        assert rid not in corpus
        corpus[rid]={'fen':fen,'family':family}
    seen={}; rows=0; predicates=0; extra=0; children={}
    for line_number,line in enumerate(a.answers.read_text().splitlines(),1):
        if not line.strip():continue
        row=json.loads(line)
        variant=row.get('variant','default')
        rid=row.get('id',row.get('source_id'))
        move=row.get('uci',row.get('move'))
        if rid not in corpus and isinstance(rid,str) and '#' in rid:
            rid,suffix=rid.rsplit('#',1)
            assert rid in corpus and suffix.isdigit(),(line_number,'invalid child id')
            if rid not in children:
                parent=chess.Board(corpus[rid]['fen'])
                children[rid]={}
                for candidate in parent.legal_moves:
                    child=parent.copy();child.push(candidate)
                    key=child.fen(en_passant='fen')
                    assert key not in children[rid],'ambiguous child state'
                    children[rid][key]=candidate.uci()
            given=chess.Board(row['fen']).fen(en_passant='fen')
            assert given in children[rid],(line_number,'child not legal successor',rid)
            move=children[rid][given]
        assert rid in corpus, (line_number,'unexpected id',rid)
        board=chess.Board(corpus[rid]['fen'])
        if move:
            mv=chess.Move.from_uci(move)
            assert mv in board.legal_moves,(line_number,'illegal transition',rid,move)
            board.push(mv)
        given=row.get('fen')
        if given is not None:
            assert chess.Board(given).fen(en_passant='fen')==board.fen(en_passant='fen'),(line_number,'wrong state',rid)
        key=(rid,move)
        assert key not in seen.setdefault(variant,set()),(line_number,'duplicate',variant,key)
        seen[variant].add(key)
        for color,prefix in [(chess.WHITE,'white'),(chess.BLACK,'black')]:
            union,checkers,pinned,evasion=expected(board,color)
            got=row.get(prefix)
            if got is None and 'attacks' in row:
                got=row['attacks'][0 if color else 1]
            assert got is not None,(line_number,'missing mask',prefix)
            assert number(got)==union,(line_number,rid,variant,prefix,hex(number(got)),hex(union))
            predicates+=64
            for suffix,value in [('checkers',checkers),('pinned',pinned),('evasion',evasion)]:
                field=f'{prefix}_{suffix}'
                if field in row:
                    assert number(row[field])==value,(line_number,rid,variant,field,hex(number(row[field])),hex(value))
                    extra+=1
        rows+=1
    assert rows,'empty output'
    if not a.allow_partial:
        for variant,keys in seen.items():
            roots={rid for rid,move in keys if move is None}
            assert roots==set(corpus),(variant,'missing root coverage',len(roots),len(corpus))
            if a.require_children:
                required={(rid,mv.uci()) for rid,item in corpus.items()
                          for mv in chess.Board(item['fen']).legal_moves}
                actual={(rid,move) for rid,move in keys if move is not None}
                assert actual==required,(variant,'incomplete child coverage',len(actual),len(required))
    report={'status':'PASS','rows':rows,'variants':{v:len(k) for v,k in seen.items()},
            'full_attack_predicates':predicates,'extra_obligation_masks':extra,
            'python_chess':chess.__version__,
            'sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [a.corpus,a.answers,Path(__file__)]}}
    a.out.parent.mkdir(parents=True,exist_ok=True)
    a.out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
