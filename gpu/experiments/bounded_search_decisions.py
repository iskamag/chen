#!/usr/bin/env python3
"""Declared six-position1s decisions; ordinary complete-root SF labels separately.

No games/rating claim. Existing benchmark FENs are diagnostic roots with empty
history, not a legal-reachable synthetic training pack or closed holdout.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import chess

BACKENDS=('alphabeta','hip-selective','hip-bounded')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
def action_id(move,mover):
    flip=0 if mover else 56;promotion=0 if move.promotion is None else ' nbrq'.index(chess.piece_symbol(move.promotion))
    return (((move.from_square^flip)*64+(move.to_square^flip))*5+promotion)

class Engine:
    def __init__(self,binary,backend):
        self.backend=backend;self.process=subprocess.Popen(['taskset','-c','5',binary,'uci'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        self.queue=queue.Queue();self.trace=[]
        def read():
            for line in self.process.stdout:self.queue.put(line.rstrip())
        threading.Thread(target=read,daemon=True).start()
        self.send('uci');self.until('uciok')
        for key,value in [('Eval','tapered'),('MaxPositions','1000000'),('SelectiveBatch','32'),('SelectiveCandidates','1024'),('Backend',backend)]:self.send(f'setoption name {key} value {value}')
        self.send('isready');self.until('readyok')
        self.setup=self.trace;self.trace=[]
    def send(self,line):self.trace.append('> '+line);self.process.stdin.write(line+'\n');self.process.stdin.flush()
    def until(self,prefix):
        result=[];deadline=time.monotonic()+10
        while True:
            line=self.queue.get(timeout=max(.001,deadline-time.monotonic()));self.trace.append('< '+line);result.append(line)
            assert 'failed:'not in line,line
            if line.startswith(prefix):return result
    def search(self,root):
        board=chess.Board(root['fen']);self.send('ucinewgame');self.send('isready');self.until('readyok')
        self.send('position fen '+root['fen']);start=time.monotonic();self.send('go movetime 1000 depth 64');lines=self.until('bestmove ');wall=(time.monotonic()-start)*1000
        best=lines[-1].split()[1];assert chess.Move.from_uci(best)in board.legal_moves
        infos=[line for line in lines if line.startswith('info depth ')];last=infos[-1].split()if infos else []
        for line in infos:
            if ' pv 'in line:
                b=board.copy()
                for uci in line.split(' pv ')[1].split():
                    m=chess.Move.from_uci(uci);assert m in b.legal_moves,(root['name'],self.backend,line,b.fen());b.push(m)
        bounded=[json.loads(line.removeprefix('info string bounded '))for line in lines if line.startswith('info string bounded ')]
        if self.backend=='hip-bounded':assert len(bounded)==1
        child=board.copy();child.push_uci(best)
        result={'root':root['name'],'backend':self.backend,'bestmove':best,'canonical_action':action_id(chess.Move.from_uci(best),board.turn),'wall_ms':wall,'legal':True,
            'reported_depth':int(last[last.index('depth')+1])if last else 0,'reported_seldepth':int(last[last.index('seldepth')+1])if 'seldepth'in last else None,
            'reported_nodes':int(last[last.index('nodes')+1])if last else 0,'reported_score':{'kind':last[last.index('score')+1],'value':int(last[last.index('score')+2])}if last else None,
            'final_info':infos[-1]if infos else None,'bounded':bounded[0]if bounded else None,'chosen_mates_in_one':child.is_checkmate(),'trace':self.trace}
        self.trace=[];return result
    def close(self):
        self.send('quit');code=self.process.wait(timeout=10);stderr=self.process.stderr.read();assert code==0,(self.backend,code,stderr)
        return {'backend':self.backend,'pid':self.process.pid,'exit_code':code,'stderr':stderr}

def engine_phase(args,decl):
    assert sha(args.engine)==decl['rust_binary_sha256'];assert sha(os.environ['CHEN_BOUNDED_LIBRARY'])==decl['native_library_sha256'];assert os.environ['CHEN_GPU_LIBRARY']==os.environ['CHEN_BOUNDED_LIBRARY']
    engines={};rows=[];retired=[]
    try:
        for backend in BACKENDS:engines[backend]=Engine(args.engine,backend)
        for i,root in enumerate(decl['roots']):
            order=BACKENDS[i%3:]+BACKENDS[:i%3]
            for backend in order:
                row=engines[backend].search(root);rows.append(row);save(args.output,{'status':'INCOMPLETE','rows':rows})
                print(json.dumps({k:row[k]for k in ('root','backend','bestmove','wall_ms','reported_depth','reported_nodes')}),flush=True)
        setup={key:e.setup for key,e in engines.items()}
        for e in engines.values():retired.append(e.close())
        save(args.output,{'status':'PASS','policy_scope':decl['policies'],'rows':rows,'setup':setup,'retired':retired,'scope':decl['scope'],'overlap':decl['overlap']})
    finally:
        for e in engines.values():
            if e.process.poll()is None:e.process.kill();e.process.wait()

def teacher_phase(args,decl):
    from neural.ordinary_packets import POLICY,parse_pv,select_packet
    from neural.teacher_session import StockfishSession
    from neural.score_frames import PHASE_WEIGHTS,sf19_reference_score,score_frame_manifest
    assert sha(args.stockfish)==decl['stockfish']['binary_sha256'];assert POLICY==decl['packet_policy']
    records=[]
    with StockfishSession(args.stockfish,expected_sha256=decl['stockfish']['binary_sha256'],timeout=30)as session:
        os.sched_setaffinity(session.process.pid,{4});setup=session.setup_trace;identity=session.identity
        for root in decl['roots']:
            board=chess.Board(root['fen']);width=len(list(board.legal_moves));assert width==root['legal_count']
            session.send('setoption name MultiPV value '+str(width));session.send('ucinewgame');session.send('isready');session.receive(lambda l:l=='readyok')
            # Authored benchmark FEN/history-empty root, distinct from the
            # startpos-prefix synthetic teacher's reachable family contract.
            session.send('position fen '+root['fen']);session.send('go nodes '+str(decl['teacher_nodes']))
            start=time.monotonic();lines=session.receive(lambda l:l.startswith('bestmove '));wall=(time.monotonic()-start)*1000
            selected,provenance,reason=select_packet(lines,board,all_legal=True,min_depth=6)
            material=sum(len(board.pieces(kind,color))*PHASE_WEIGHTS[kind-1]for color in (True,False)for kind in range(1,7))
            targets=None
            if selected is not None:
                targets=[{'canonical_action':action_id(chess.Move.from_uci(r.pv[0]),board.turn),'move':r.pv[0],'rank':r.multipv,'reported_cp':r.score,'depth':r.depth,'nodes':r.nodes,'pv':list(r.pv),'converted':sf19_reference_score(r.score,material,score_kind='ordinary_unbounded_cp_no_tablebase',negate=False)}for r in selected]
                assert {r['move']for r in targets}=={m.uci()for m in board.legal_moves}
            evidence=[]
            for offset,line in enumerate(lines):
                try:parsed=parse_pv(line)
                except ValueError:evidence.append({'offset':offset,'kind':'malformed'});continue
                if parsed is not None:evidence.append({'offset':offset,**asdict(parsed)})
            record={'root':root['name'],'fen':root['fen'],'history':'empty authored benchmark root, not a training family','requested_nodes':decl['teacher_nodes'],'wall_ms':wall,'labels':targets,'rejection':reason,'provenance':provenance,'root_material':material,'score_frame':score_frame_manifest(),'typed_scores':evidence,'trace':session.take_trace()}
            records.append(record);save(args.output,{'status':'INCOMPLETE','records':records});print(json.dumps({'root':root['name'],'teacher_eligible':targets is not None,'rejection':reason,'wall_ms':wall}),flush=True)
        teacher_pid=session.process.pid
    assert session.process.poll()==0
    save(args.output,{'status':'PASS','records':records,'setup':setup,'identity':identity,'stockfish':decl['stockfish'],'retired_pid':teacher_pid,'exit_code':session.process.returncode,'scope':'Bounded ordinary full-support SF19 estimates, not exact minimax/outcome/playing-strength labels; rejected queries remain unscored'})

def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['engines','teacher'],required=True);p.add_argument('--declaration',required=True);p.add_argument('--engine');p.add_argument('--stockfish');p.add_argument('--output',required=True);a=p.parse_args();decl=json.loads(Path(a.declaration).read_text())
    if a.phase=='engines':engine_phase(a,decl)
    else:teacher_phase(a,decl)
if __name__=='__main__':main()
