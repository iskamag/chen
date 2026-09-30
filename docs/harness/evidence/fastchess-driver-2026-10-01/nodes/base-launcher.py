#!/home/iskam/src/vibe/chen/results/qa-venv/bin/python
import os
os.chdir('/home/iskam/src/vibe/chen')
env = dict(os.environ)
overrides = {}
for key, value in overrides.items():
    if value is None: env.pop(key, None)
    else: env[key] = value
command = ['/home/iskam/src/vibe/chen/target/release/chen']
os.execvpe(command[0], command, env)
