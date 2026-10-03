import chess,json,hashlib,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parent))
from neural.ordinary_packets import select_packet
p=Path(__file__).resolve().parent;d=json.loads((p/'declaration-draft.json').read_text())
for root in d['roots']:
    b=chess.Board(root['fen']);assert b.is_valid();moves=sorted(b.legal_moves,key=lambda m:m.uci());root['legal_count']=len(moves)
    mates=[]
    for m in moves:
        child=b.copy();child.push(m)
        if child.is_checkmate():mates.append(m.uci())
    root['mate_in_one_moves']=mates
    def emission(depth,bound=''):
        return [f'info depth {depth} seldepth {depth} multipv {i+1} score cp {100-i}'+bound+f' nodes 262144 tbhits 0 pv {m.uci()}'for i,m in enumerate(moves)]
    rows,meta,error=select_packet(emission(6)+emission(7,' upperbound')+['bestmove '+moves[0].uci()],b,all_legal=True,min_depth=6)
    assert error is None and len(rows)==len(moves) and meta['selected_common_depth']==6 and meta['last_emitted_packet_classification']=='bounded'
    rows,_,error=select_packet(emission(6)+['info depth 7 seldepth 7 multipv 1 score mate 1 nodes 262144 tbhits 0 pv '+moves[0].uci(),'bestmove '+moves[0].uci()],b,all_legal=True,min_depth=6)
    assert rows is None and error=='decisive_mate_tablebase_or_extreme_evidence'
d['python_chess_version']=chess.__version__;d['chess_source_sha256']=hashlib.sha256(Path(chess.__file__).read_bytes()).hexdigest()
(p/'declaration.json').write_text(json.dumps(d,indent=2,sort_keys=True)+'\n')
print(json.dumps({'preparation':'PASS','roots':len(d['roots']),'legal_actions':sum(r['legal_count']for r in d['roots']),'ordinary_control_cases':12,'noGPU':True}))
