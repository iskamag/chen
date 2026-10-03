import json,pathlib,subprocess,time
r=pathlib.Path(__file__).resolve().parent
cmd=json.loads((r/'declaration.json').read_text())['command']
start=time.monotonic()
with (r/'match.stdout').open('xb') as out,(r/'match.stderr').open('xb') as err:
 p=subprocess.Popen(cmd,stdout=out,stderr=err)
 (r/'live.json').write_text(json.dumps(dict(host_pid=p.pid,start_monotonic=start,command=cmd))+'\n')
 rc=p.wait()
(r/'terminal.json').write_text(json.dumps(dict(exitcode=rc,elapsed_seconds=time.monotonic()-start))+'\n')
print(rc,flush=True)
raise SystemExit(rc)
