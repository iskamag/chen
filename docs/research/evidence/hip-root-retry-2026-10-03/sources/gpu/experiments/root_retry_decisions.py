#!/usr/bin/env python3
"""Declared six-position1s root pool decisions; teacher labels reused only cold.

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

BACKENDS=('hip-root-pool','hip-root-retry','alphabeta')

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
        if backend=='alphabeta':
            self.send('setoption name PVS value true');self.send('setoption name AspirationWindow value 25')
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
        self.send('position fen '+root['fen']);start=time.monotonic();self.send('go movetime 1000 nodes 1000000 depth 64');lines=self.until('bestmove ');wall=(time.monotonic()-start)*1000
        best=lines[-1].split()[1];assert chess.Move.from_uci(best)in board.legal_moves
        infos=[line for line in lines if line.startswith('info depth ')];last=infos[-1].split()if infos else []
        for line in infos:
            if ' pv 'in line:
                b=board.copy()
                for uci in line.split(' pv ')[1].split():
                    m=chess.Move.from_uci(uci);assert m in b.legal_moves,(root['name'],self.backend,line,b.fen());b.push(m)
        bounded=[json.loads(line.removeprefix('info string bounded '))for line in lines if line.startswith('info string bounded ')]
        if self.backend=='hip-bounded':assert len(bounded)==1
        pool=[json.loads(line.removeprefix('info string root_pool '))for line in lines if line.startswith('info string root_pool ')]
        if self.backend in('hip-root-pool','hip-root-retry'):assert len(pool)==1
        child=board.copy();child.push_uci(best)
        result={'root':root['name'],'backend':self.backend,'bestmove':best,'canonical_action':action_id(chess.Move.from_uci(best),board.turn),'wall_ms':wall,'legal':True,
            'reported_depth':int(last[last.index('depth')+1])if last else 0,'reported_seldepth':int(last[last.index('seldepth')+1])if 'seldepth'in last else None,
            'reported_nodes':int(last[last.index('nodes')+1])if last else 0,'reported_score':{'kind':last[last.index('score')+1],'value':int(last[last.index('score')+2])}if last else None,
            'final_info':infos[-1]if infos else None,'bounded':bounded[0]if bounded else None,'root_pool':pool[0]if pool else None,'chosen_mates_in_one':child.is_checkmate(),'trace':self.trace}
        self.trace=[];return result
    def close(self):
        self.send('quit');code=self.process.wait(timeout=10);stderr=self.process.stderr.read();assert code==0,(self.backend,code,stderr)
        return {'backend':self.backend,'pid':self.process.pid,'exit_code':code,'stderr':stderr}

def engine_phase(args,decl):
    assert sha(args.engine)==decl['rust_binary_sha256'];assert sha(os.environ['CHEN_ROOT_POOL_LIBRARY'])==decl['control_library_sha256'];assert sha(os.environ['CHEN_ROOT_RETRY_LIBRARY'])==decl['library_sha256']
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

def smoke(args,decl):
    engine=Engine(args.engine,'hip-root-retry');rows=[]
    fixtures=[('zero-time','position startpos','go movetime 0 depth 64',0,None),('zero-nodes','position startpos','go nodes 0 depth 64',0,None),('one-node','position startpos','go nodes 1 depth 64',0,None),('exact-d2','position startpos','go depth 2 nodes 1000000',2,None),('mate','position fen 7k/6Q1/6K1/8/8/8/8/8 b - - 0 1','go movetime 1000 depth 64',0,None),('threefold','position startpos moves g1f3 g8f6 f3g1 f6g8 g1f3 g8f6 f3g1 f6g8','go movetime 1000 depth 64',0,None),('stop','position fen '+decl['roots'][0]['fen'],'go infinite depth 16',None,.020)]
    try:
        for name,position,go,depth,stop in fixtures:
            engine.send('ucinewgame');engine.send('isready');engine.until('readyok');engine.send(position);begin=time.monotonic();engine.send(go)
            if stop:time.sleep(stop);engine.send('stop');cancelled=time.monotonic()
            lines=engine.until('bestmove ');wall=(time.monotonic()-begin)*1000;best=lines[-1].split()[1]
            stats=[json.loads(line.removeprefix('info string root_pool '))for line in lines if line.startswith('info string root_pool ')];assert len(stats)==1
            if name=='mate':assert best=='0000'and stats[0]['stop_reason']=='terminal'
            else:
                board=chess.Board(decl['roots'][0]['fen'])if name=='stop'else chess.Board();assert chess.Move.from_uci(best)in board.legal_moves
            if depth is not None:assert stats[0].get('completed_depth',0)==depth,(name,stats)
            if name=='threefold':assert stats[0]['stop_reason']=='terminal'
            if name=='stop':assert (time.monotonic()-cancelled)<.100
            rows.append({'name':name,'bestmove':best,'wall_ms':wall,'stop_response_ms':(time.monotonic()-cancelled)*1000 if stop else None,'stats':stats[0],'trace':engine.trace});engine.trace=[]
            save(args.output,{'status':'INCOMPLETE','rows':rows})
        retired=engine.close();save(args.output,{'status':'PASS','rows':rows,'retired':retired,'scope':'actual Rust UCI ABI3 ownership/publication/stop controls; no strength claim'})
    finally:
        if engine.process.poll()is None:engine.process.kill();engine.process.wait()
    print(json.dumps({'status':'PASS','uci_controls':len(rows)}),flush=True)
def main():
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','decisions'],required=True);p.add_argument('--declaration',required=True);p.add_argument('--engine',required=True);p.add_argument('--output',required=True);a=p.parse_args();d=json.loads(Path(a.declaration).read_text())
    assert sha(a.engine)==d['rust_binary_sha256'];assert sha(os.environ['CHEN_ROOT_RETRY_LIBRARY'])==d['library_sha256']
    if a.phase=='smoke':smoke(a,d)
    else:engine_phase(a,d)
if __name__=='__main__':main()
