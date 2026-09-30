import os,sys
from pathlib import Path
Path(sys.argv[2]).write_text(str(os.getpid()))
for line in sys.stdin:
    if line.strip() == 'uci':
        print('id name Fault fixture',flush=True); print('uciok',flush=True)
    elif line.strip() == 'isready': print('readyok',flush=True)
    elif line.startswith('go') and sys.argv[1] == 'illegal': print('bestmove h1h8',flush=True)
    elif line.strip() == 'quit': break
