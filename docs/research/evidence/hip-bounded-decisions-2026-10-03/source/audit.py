#!/usr/bin/env python3
"""Cold CPU replay of decision legality, ordinary packet selection and intervals.

Reads saved traces only; never starts an engine, training or native GPU call.
The packet grammar and cold score-frame primitive are pinned shared controls;
this is a mechanical replay, not independent proof of teacher quality.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import chess

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--packet',required=True);parser.add_argument('--output',required=True);args=parser.parse_args();p=Path(args.packet)
    sys.path.insert(0,str(p/'source'))
    from neural.ordinary_packets import select_packet
    from neural.score_frames import PHASE_WEIGHTS,sf19_reference_score
    d=json.loads((p/'declaration.json').read_text());e=json.loads((p/'engine-results.json').read_text());t=json.loads((p/'teacher-results.json').read_text());s=json.loads((p/'summary.json').read_text())
    assert e['status']==t['status']==s['status']=='PASS'
    for name,h in d['teacher_source_sha256'].items():assert hashlib.sha256((p/'source'/name).read_bytes()).hexdigest()==h
    roots={r['name']:r for r in d['roots']};assert len(roots)==6;assert len(e['rows'])==len(s['rows'])==18;assert len(t['records'])==6
    assert {(r['root'],r['backend'])for r in e['rows']}=={(root,b)for root in roots for b in ('alphabeta','hip-selective','hip-bounded')}
    for row in e['rows']:
        board=chess.Board(roots[row['root']]['fen']);assert chess.Move.from_uci(row['bestmove'])in board.legal_moves
        assert row['trace'].count('> go movetime 1000 depth 64')==1
        for line in row['trace']:
            if line.startswith('< info depth ')and' pv 'in line:
                position=board.copy()
                for text in line.split(' pv ')[1].split():move=chess.Move.from_uci(text);assert move in position.legal_moves;position.push(move)
        if row['bounded']:
            stats=row['bounded'];completed=[i for i in stats['iterations']if i['kind']==0]
            assert stats['completed_depth']==(completed[-1]['requested_depth']if completed else 0)
            if stats['iterations'][-1]['kind']==3:assert stats['iterations'][-1]['abort_reason']==6
        child=board.copy();child.push_uci(row['bestmove']);assert child.is_checkmate()==row['chosen_mates_in_one']
    teachers={}
    for record in t['records']:
        root=roots[record['root']];board=chess.Board(root['fen']);assert record['fen']==root['fen'];assert root['history']==[]
        trace=record['trace'];assert '> position fen '+root['fen']in trace;assert '> go nodes 262144'in trace
        query_start=trace.index('> go nodes 262144')+1
        lines=[line[2:]for line in trace[query_start:] if line.startswith('< ')];selected,provenance,reason=select_packet(lines,board,all_legal=True,min_depth=6)
        assert reason==record['rejection'];assert provenance==record['provenance'];labels=record['labels'];assert (selected is None)==(labels is None)
        if labels:
            assert len(labels)==root['legal_count'];assert {r['move']for r in labels}=={m.uci()for m in board.legal_moves}
            material=sum(len(board.pieces(kind,color))*PHASE_WEIGHTS[kind-1]for color in (True,False)for kind in range(1,7));assert material==record['root_material']
            for native,row in zip(selected,labels):
                assert native.bound is None and native.score_kind=='cp'and not native.tablebase_hits
                assert native.score==row['reported_cp']and native.pv[0]==row['move']and native.multipv==row['rank']
                assert row['converted']==sf19_reference_score(native.score,material,score_kind='ordinary_unbounded_cp_no_tablebase',negate=False)
            assert len({r['depth']for r in labels})==len({r['nodes']for r in labels})==1 and labels[0]['depth']>=6
        teachers[record['root']]=record
    indexed={(r['root'],r['backend']):r for r in s['rows']}
    for row in e['rows']:
        summary=indexed[(row['root'],row['backend'])];labels=teachers[row['root']]['labels'];assert summary['bestmove']==row['bestmove']
        if labels:
            chosen=next(r for r in labels if r['move']==row['bestmove']);winner=max(labels,key=lambda r:r['reported_cp'])
            assert summary['teacher_reported_gap_cp']==winner['reported_cp']-chosen['reported_cp'];assert summary['teacher_rank']==chosen['rank']
            expected=[0,0]if chosen['move']==winner['move']else[max(0,winner['converted']['lower_cp']-chosen['converted']['upper_cp']),max(0,winner['converted']['upper_cp']-chosen['converted']['lower_cp'])]
            assert summary['teacher_reference_gap_interval']==expected
        if row['bounded']:
            aborted=sum(i['nodes']for i in row['bounded']['iterations']if i['kind']==3);total=sum(i['nodes']for i in row['bounded']['iterations'])
            assert summary['attempted_nodes']==total and summary['aborted_nodes']==aborted and summary['aborted_fraction']==aborted/max(total,1)
    assert all(r['exit_code']==0 for r in e['retired'])and t['exit_code']==0
    for name in ('engine-terminal.json','teacher-terminal.json'):
        terminal=json.loads((p/name).read_text());assert terminal['exit_code']==0 and terminal['host_retired']and not terminal['candidate_present']
    report={'status':'PASS','decisions':18,'legal_teacher_support':sum(len(r['labels']or[])for r in t['records']),'eligible_queries':sum(r['labels']is not None for r in t['records']),'rejected_queries':1,'all_move_PV_packet_frame_gap_and_retirement_fields_replayed':True,'scope':'CPU mechanical replay, shared pinned packet/frame controls; no new engine/GPU execution or quality proof'}
    Path(args.output).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n');print(json.dumps(report))
if __name__=='__main__':main()
