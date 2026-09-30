#!/home/iskam/src/vibe/chen/results/qa-venv/bin/python
import os
os.chdir('/home/iskam/src/vibe/chen')
env = dict(os.environ)
overrides = {}
for key, value in overrides.items():
    if value is None: env.pop(key, None)
    else: env[key] = value
command = ['/usr/bin/python3.14', '/home/iskam/src/vibe/chen/results/fastchess-qa-inputs/fault_engine.py', 'stall', '/home/iskam/src/vibe/chen/results/fastchess-qa-inputs/stall-pid.txt']
os.execvpe(command[0], command, env)
