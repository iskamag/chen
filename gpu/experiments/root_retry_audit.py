#!/usr/bin/env python3
"""Cold saved-ABI4 evidence audit; no native/engine/teacher execution."""
import argparse,hashlib,json
from pathlib import Path
import chess

def main():
    p=argparse.ArgumentParser();p.add_argument('--packet',required=True);p.add_argument('--teacher',required=True);p.add_argument('--output',required=True);a=p.parse_args();root=Path(a.packet)
    d=json.loads((root/'declaration.json').read_text());n=json.loads((root/'probe-results.json').read_text());u=json.loads((root/'smoke-results.json').read_text());e=json.loads((root/'decisions-results.json').read_text());t=json.loads(Path(a.teacher).read_text())
    assert n['status']==u['status']==e['status']==t['status']=='PASS';assert len(n['rows'])==23 and len(n['controls'])==6 and len(u['rows'])==7 and len(e['rows'])==3
    assert n['library_sha256']==d['library_sha256'] and n['cases_sha256']==d['cases_sha256'];assert all(x['exit_code']==0 for x in e['retired']) and u['retired']['exit_code']==0
    soft=next(r for r in n['controls'] if r['name']=='soft-resource-no-bound-other-workers-continue');assert soft['failed_events']>=2 and soft['retry_count']>=1 and soft['completed_children']>0 and soft['stalled']==1
    kiwi=d['roots'][0];assert kiwi['name']=='kiwipete';teacher=next(r for r in t['records']if r['root']=='kiwipete');assert teacher['fen']==kiwi['fen'];labels=teacher['labels'];board=chess.Board(kiwi['fen']);assert {x['move']for x in labels}=={m.uci()for m in board.legal_moves}
    results=[]
    for row in e['rows']:
        assert row['root']=='kiwipete';move=chess.Move.from_uci(row['bestmove']);assert move in board.legal_moves
        assert row['trace'].count('> go movetime 1000 nodes 1000000 depth 64')==1
        for line in row['trace']:
            if line.startswith('< info depth ')and' pv 'in line:
                b=board.copy()
                for text in line.split(' pv ')[1].split():m=chess.Move.from_uci(text);assert m in b.legal_moves;b.push(m)
        item={k:row[k]for k in ('root','backend','bestmove','reported_depth','reported_nodes','wall_ms')}
        pool=row['root_pool']
        if pool:
            epochs=pool['epochs'];assert pool['completed_depth']==row['reported_depth'];assert all(it['qnodes']<=it['nodes']<=1000000 for it in epochs);assert all(x['nodes']<=y['nodes']for x,y in zip(epochs,epochs[1:]))
            last=epochs[-1];item.update(attempted_nodes=last['nodes'],attempted_qnodes=last['qnodes'],completed_children=last['completed_children'],exact_children=last['exact_children'],upper_children=last['upper_children'],paused_children=last['paused_children'],stop_reason=pool['stop_reason'],epoch_count=len(epochs))
            if row['backend']=='hip-root-retry':
                assert pool['abi']==4;assert all(it['abort_reason']==0 for it in epochs);assert all(x['failed_events']<=y['failed_events']for x,y in zip(epochs,epochs[1:]));assert all(x['retry_count']<=y['retry_count']for x,y in zip(epochs,epochs[1:]));assert last['retry_count']==0 and last['exact_children']==0
                item.update(failed_children=last['failed_children'],failed_events=last['failed_events'],retry_count=last['retry_count'],stalled=last['stalled'])
        chosen=next(x for x in labels if x['move']==row['bestmove']);winner=max(labels,key=lambda x:x['reported_cp']);interval=[0,0]if chosen['move']==winner['move']else[max(0,winner['converted']['lower_cp']-chosen['converted']['upper_cp']),max(0,winner['converted']['upper_cp']-chosen['converted']['lower_cp'])]
        item.update(teacher_rank=chosen['rank'],teacher_reported_gap_cp=winner['reported_cp']-chosen['reported_cp'],teacher_reference_gap_interval=interval,teacher_common_depth=chosen['depth']);results.append(item)
    mechanism=n['kiwipete_mechanism'];assert len(mechanism['witnesses'])>0
    for witness in mechanism['witnesses']:
        b=board.copy()
        for i,text in enumerate(witness['legal_tactical_or_evasion_witness']):
            move=chess.Move.from_uci(text);assert move in b.legal_moves
            if i:assert b.is_check()or b.is_capture(move)or move.promotion
            b.push(move)
        assert b.fen(en_passant='fen')==witness['failure_fen'];assert witness['failure_reason']==2 and len(witness['legal_tactical_or_evasion_witness'])==33
        bound=witness['bound'];assert bound['kind']==3 and bound['abort_reason']==2 and bound['score']==bound['completed_depth']==0 and not bound['pv']
    for name in ('native','smoke','decisions'):
        terminal=json.loads((root/(name+'-terminal.json')).read_text());assert terminal['exit_code']==0 and terminal['host_retired']
    summary={'status':'PASS','rows':results,'soft_cap_control':soft,'legal_qcap32_witnesses':len(mechanism['witnesses']),'teacher_binding':{'relative_path':'../hip-bounded-decisions-2026-10-03/teacher-results.json','sha256':hashlib.sha256(Path(a.teacher).read_bytes()).hexdigest(),'source_commit':'8eb2681','scope':'Previously closed ordinary complete-root estimates reused unchanged; no teacher search rerun.'},'scope':'Saved-artifact legality/identity/counter/mechanism summary; no general strength/rating/hardware causal proof'}
    Path(a.output).write_text(json.dumps(summary,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','decisions':3,'native_roots':23,'native_controls':6,'uci_controls':7,'legal_qcap32_witnesses':len(mechanism['witnesses'])}))
if __name__=='__main__':main()
