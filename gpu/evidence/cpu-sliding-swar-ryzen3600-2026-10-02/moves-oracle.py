#!/usr/bin/env python3
"""Independent python-chess legal and child-legal counts for both pinned arms."""
import chess,hashlib,json,random,subprocess
from pathlib import Path
p=Path(__file__).resolve().parent
paths={a:Path('/tmp/chen-cpu-sliding-2026-10-02')/a/'target/release/chen' for a in ('baseline','candidate')}
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
pins={a:sha(f) for a,f in paths.items()}
fixtures=['rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1','r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 99 1','r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1','8/8/8/r4pPK/8/8/8/7k w - f6 0 1','4r2k/8/8/3pP3/8/8/8/4K3 w - d6 0 1','4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12','4k3/P7/8/8/8/8/7p/4K3 w - - 0 1','4k3/P7/8/8/8/8/7p/4K3 b - - 0 1','7k/6Q1/6K1/8/8/8/8/8 b - - 100 1','7k/5Q2/6K1/8/8/8/8/8 b - - 0 1']
boards=[chess.Board(f) for f in fixtures];rng=random.Random(0xc4e1030);board=chess.Board()
while len(boards)<256:
 moves=list(board.legal_moves)
 if not moves or board.halfmove_clock>=100:board=chess.Board()
 else:board.push(rng.choice(moves))
 boards.append(board.copy(stack=False))
results=[];children=0
for index,b in enumerate(boards):
 wanted={}
 for move in b.legal_moves:
  b.push(move);wanted[move.uci()]=b.legal_moves.count();b.pop()
 children+=len(wanted)
 actual={}
 for arm,file in paths.items():
  assert sha(file)==pins[arm]
  argv=['taskset','-c','2',str(file),'divide','2',b.fen(en_passant='fen')]
  run=subprocess.run(argv,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=3,check=True,text=True)
  got={}
  for line in run.stdout.splitlines():
   if ':' in line:
    move,count=line.split(': ');assert move not in got;got[move]=int(count)
  total=int(run.stdout.splitlines()[-1].split()[1]);assert total==sum(got.values())
  assert got==wanted,(index,b.fen(),arm,got,wanted)
  actual[arm]={'child_legal_counts':got,'depth2_nodes':total,'exit_code':run.returncode}
 results.append({'index':index,'fen':b.fen(en_passant='fen'),'mover':int(b.turn),'expected_child_legal_counts':wanted,'actual':actual})
(p/'moves-oracle-results.json').write_text(json.dumps({'status':'passed','source_sha256':sha(Path(__file__)),'python_chess':chess.__version__,'binary_sha256':pins,'roots_per_arm':len(boards),'child_positions_per_arm':children,'movers':{str(c):sum(b.turn==c for b in boards) for c in (False,True)},'rows':results},indent=2)+'\n')
print('PASS',len(boards),'roots and',children,'child legal move sets per arm')
