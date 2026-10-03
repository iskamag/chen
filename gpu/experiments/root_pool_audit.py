#!/usr/bin/env python3
"""Saved-result mechanical audit only; no native/engine/teacher execution."""
import argparse,hashlib,json
from pathlib import Path
import chess

def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--teacher',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.packet)
    d=json.loads((root/'declaration.json').read_text());n=json.loads((root/'probe-results.json').read_text());u=json.loads((root/'smoke-results.json').read_text());e=json.loads((root/'decisions-results.json').read_text());t=json.loads(Path(a.teacher).read_text())
    assert n['status']==u['status']==e['status']==t['status']=='PASS';assert len(n['rows'])==23 and len(n['controls'])==5 and len(u['rows'])==7 and len(e['rows'])==18
    assert n['library_sha256']==d['library_sha256'] and n['cases_sha256']==d['cases_sha256'];assert all(x['exit_code']==0 for x in e['retired']) and u['retired']['exit_code']==0
    roots={r['name']:r for r in d['roots']};teachers={r['root']:r for r in t['records']};results=[]
    assert {(r['root'],r['backend'])for r in e['rows']}=={(name,backend)for name in roots for backend in ('alphabeta','hip-bounded','hip-root-pool')}
    for row in e['rows']:
        r=roots[row['root']];board=chess.Board(r['fen']);move=chess.Move.from_uci(row['bestmove']);assert move in board.legal_moves
        assert row['trace'].count('> go movetime 1000 nodes 1000000 depth 64')==1
        for line in row['trace']:
            if line.startswith('< info depth ')and' pv 'in line:
                b=board.copy()
                for text in line.split(' pv ')[1].split():m=chess.Move.from_uci(text);assert m in b.legal_moves;b.push(m)
        child=board.copy();child.push(move);assert child.is_checkmate()==row['chosen_mates_in_one']
        item={k:row[k]for k in ('root','backend','bestmove','reported_depth','reported_nodes','wall_ms','chosen_mates_in_one')}
        bounded=row['bounded'];pool=row['root_pool']
        if bounded:
            nodes=sum(it['nodes']for it in bounded['iterations']);qnodes=sum(it['qnodes']for it in bounded['iterations']);assert qnodes<=nodes<=1000000
            item.update(attempted_nodes=nodes,attempted_qnodes=qnodes,unfinished_iteration_nodes=sum(it['nodes']for it in bounded['iterations']if it['kind']==3),stop_reason=bounded['stop_reason'])
        if pool:
            epochs=pool['epochs'];assert pool['abi']==3 and pool['completed_depth']==row['reported_depth'];assert all(it['qnodes']<=it['nodes']<=1000000 for it in epochs)
            assert all(x['nodes']<=y['nodes']for x,y in zip(epochs,epochs[1:]));last=epochs[-1]
            for depth in set(it['depth']for it in epochs):
                per=[it for it in epochs if it['depth']==depth];assert all(x['completed_children']<=y['completed_children']for x,y in zip(per,per[1:]));assert sum(it['iteration_complete']!=0 for it in per)<=1
            assert all(it['abort_reason']==0 for it in epochs[:-1]);assert last['abort_reason']in (0,2)
            assert max((it['depth']for it in epochs if it['iteration_complete']),default=0)==pool['completed_depth']
            item.update(attempted_nodes=last['nodes'],attempted_qnodes=last['qnodes'],unfinished_iteration_nodes=0 if last['iteration_complete']else last['iteration_nodes'],completed_children=last['completed_children'],exact_children=last['exact_children'],upper_children=last['upper_children'],paused_children=last['paused_children'],epoch_count=len(epochs),stop_reason=pool['stop_reason'])
        teacher=teachers[row['root']];assert teacher['fen']==r['fen'];labels=teacher['labels']
        if labels:
            assert {x['move']for x in labels}=={m.uci()for m in board.legal_moves};chosen=next(x for x in labels if x['move']==row['bestmove']);winner=max(labels,key=lambda x:x['reported_cp'])
            gap=winner['reported_cp']-chosen['reported_cp'];interval=[0,0]if chosen['move']==winner['move']else[max(0,winner['converted']['lower_cp']-chosen['converted']['upper_cp']),max(0,winner['converted']['upper_cp']-chosen['converted']['lower_cp'])]
            item.update(teacher_rank=chosen['rank'],teacher_reported_gap_cp=gap,teacher_reference_gap_interval=interval,teacher_common_depth=chosen['depth'])
        else:item['teacher_unscored_reason']=teacher['rejection']
        results.append(item)
    for name in ('native','smoke','decisions'):
        terminal=json.loads((root/(name+'-terminal.json')).read_text());assert terminal['exit_code']==0 and terminal['host_retired']
    summary={'status':'PASS','rows':results,'teacher_binding':{'relative_path':'../hip-bounded-decisions-2026-10-03/teacher-results.json','sha256':hashlib.sha256(Path(a.teacher).read_bytes()).hexdigest(),'source_commit':'8eb2681','scope':'Previously closed ordinary full-root diagnostic estimates, reused unchanged; no teacher search rerun.'},'scope':'Cold saved-artifact legality/counter/packet summary; no quality/rating or hardware causal proof'}
    Path(a.output).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','decisions':18,'exhaustive_native_roots':23,'uci_controls':7,'teacher_eligible_queries':5}))
if __name__=='__main__':main()
