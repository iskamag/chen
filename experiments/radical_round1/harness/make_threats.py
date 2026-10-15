#!/usr/bin/env python3
"""Exact board-local existence of opposing mate-in-one after every root action.

No repetition history or draw-claim policy is inferred. A witness proves the
existence of a legal two-action path ending in mate; absent witnesses exhaust
all legal replies. Positive/negative results are not search value or Elo.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import chess


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--corpus',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();started=time.perf_counter()
    count=0; actions=0; pairs=0; checks=0; threats=0
    a.out.parent.mkdir(parents=True,exist_ok=True)
    with a.out.open('w') as out:
        for line in a.corpus.read_text().splitlines():
            rid,split,family,fen=line.split('|',3)
            board=chess.Board(fen)
            witnesses={};safe=[]
            legal=sorted(board.legal_moves,key=lambda move:move.uci())
            for move in legal:
                actions+=1;board.push(move)
                witness=None
                for reply in sorted(board.legal_moves,key=lambda move:move.uci()):
                    pairs+=1;board.push(reply)
                    if board.is_check():
                        checks+=1
                        if next(iter(board.legal_moves),None) is None:
                            witness=reply.uci()
                    board.pop()
                    if witness is not None:break
                board.pop()
                if witness is None:safe.append(move.uci())
                else:witnesses[move.uci()]=witness
            count+=1;threats+=len(witnesses)
            out.write(json.dumps({'id':rid,'split':split,'family':family,
                                  'legal':[m.uci() for m in legal],
                                  'allows_mate1':witnesses,'safe_from_mate1':safe})+'\n')
            out.flush()
    report={'roots':count,'root_actions':actions,'tested_reply_pairs':pairs,
            'checked_replies':checks,'threat_allowing_actions':threats,
            'seconds':time.perf_counter()-started,'python_chess':chess.__version__,
            'scope':'Complete board-local opposing mate1 existence after each legalrootmove; no history/draw-policy strength inference',
            'sha256':{str(x):hashlib.sha256(x.read_bytes()).hexdigest() for x in [a.corpus,a.out,Path(__file__)]}}
    a.out.with_suffix('.manifest.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
