#!/usr/bin/env python3
"""Actual Rust/UCI GPU publication, history, legality and wall-budget smoke."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import threading
import time
import chess

class Engine:
    def __init__(self,path):
        self.process=subprocess.Popen([path,'uci'],stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE,text=True,bufsize=1)
        self.lines=queue.Queue();self.transcript=[]
        def read():
            for line in self.process.stdout:self.lines.put(line.rstrip())
        self.reader=threading.Thread(target=read,daemon=True);self.reader.start()
    def send(self,text):
        self.transcript.append('> '+text);self.process.stdin.write(text+'\n');self.process.stdin.flush()
    def until(self,prefix):
        result=[];deadline=time.monotonic()+10
        while True:
            line=self.lines.get(timeout=max(.001,deadline-time.monotonic()));self.transcript.append('< '+line);result.append(line)
            assert 'failed:' not in line,line
            if line.startswith(prefix):return result
    def search(self,position,go,board,stop_ms=None):
        self.send(position);began=time.monotonic();self.send(go)
        if stop_ms is not None:time.sleep(stop_ms/1000);requested=time.monotonic();self.send('stop')
        else:requested=None
        lines=self.until('bestmove ');wall=(time.monotonic()-began)*1000;best=lines[-1].split()[1]
        if list(board.legal_moves):assert chess.Move.from_uci(best) in board.legal_moves,(board.fen(),best)
        else:assert best=='0000'
        bounded=[json.loads(line.removeprefix('info string bounded ')) for line in lines if line.startswith('info string bounded ')];assert len(bounded)==1
        infos=[line for line in lines if line.startswith('info depth ')]
        for line in infos:
            if ' pv ' in line:
                b=board.copy()
                for word in line.split(' pv ')[1].split():
                    move=chess.Move.from_uci(word);assert move in b.legal_moves,(line,b.fen());b.push(move)
        if infos:
            last=infos[-1].split();assert bounded[0]['completed_depth']==int(last[2]);assert best==last[last.index('pv')+1]
        return {'position':position,'go':go,'bestmove':best,'wall_ms':wall,'stop_response_ms':None if requested is None else(time.monotonic()-requested)*1000,'infos':infos,'bounded':bounded[0]}
    def close(self):
        self.send('quit');code=self.process.wait(timeout=10);stderr=self.process.stderr.read();assert code==0,(code,stderr);return stderr

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--engine',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    engine=Engine(args.engine);rows=[]
    try:
        engine.send('uci');lines=engine.until('uciok');assert any('var hip-bounded'in x for x in lines)
        engine.send('setoption name Backend value hip-bounded');engine.send('isready');engine.until('readyok')
        board=chess.Board()
        for go,reason in [('go nodes 0','nodes'),('go movetime 0','deadline')]:
            r=engine.search('position startpos',go,board);assert r['bounded']['completed_depth']==0 and not r['infos'];assert r['bounded']['stop_reason']==reason;rows.append(r)
        r=engine.search('position startpos','go nodes 24 depth 16',board)
        assert r['bounded']['completed_depth']==1 and r['bounded']['stop_reason']=='nodes'
        assert sum(x['nodes']for x in r['bounded']['iterations'])<=24;assert r['bounded']['iterations'][-1]['kind']==3;rows.append(r)
        base=chess.Board();base.remove_piece_at(chess.D8);fen=base.fen();moves=['g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1']
        for move in moves:base.push_uci(move)
        r=engine.search('position fen '+fen+' moves '+' '.join(moves),'go depth 1',base)
        assert r['bestmove']=='f6g8' and 'score cp 0'in r['infos'][-1];rows.append(r)
        mate=chess.Board('7k/6Q1/6K1/8/8/8/8/8 b - - 100 1')
        r=engine.search('position fen '+mate.fen(),'go movetime 1000',mate);assert r['bounded']['stop_reason']=='terminal';rows.append(r)
        r=engine.search('position startpos','go movetime 1000 depth 16',board,stop_ms=25)
        assert r['bounded']['stop_reason']=='cancelled' and r['stop_response_ms']<100;rows.append(r)
        r=engine.search('position startpos','go depth 1',board);assert r['bounded']['completed_depth']==1 and r['bounded']['stop_reason']=='depth';rows.append(r)
        r=engine.search('position startpos','go movetime 1000 depth 16',board)
        assert r['bounded']['stop_reason']=='deadline' and r['bounded']['completed_depth']>=1
        assert r['wall_ms']<1300;rows.append(r)
        stderr=engine.close();packet={'status':'PASS','rust_binary_sha256':hashlib.sha256(Path(args.engine).read_bytes()).hexdigest(),'library_path':os.environ['CHEN_BOUNDED_LIBRARY'],'rows':rows,'transcript':engine.transcript,'stderr':stderr,'scope':'Runnable scalar fullGPU reference; no strength/parallel-efficiency claim; root CPU match may share system power'}
        Path(args.output).write_text(json.dumps(packet,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','cases':len(rows),'one_second':rows[-1],'cancel_response_ms':rows[-3]['stop_response_ms']}),flush=True)
    finally:
        if engine.process.poll()is None:engine.process.kill();engine.process.wait()

if __name__=='__main__':main()
