import json,os,subprocess,sys
from pathlib import Path
trial=int(sys.argv[1]);root=Path(__file__).resolve().parent
records=[json.loads(line) for line in (root/f'trial-{trial}.stdout.jsonl').read_text().splitlines()]
ready={r['pid'] for r in records if r['record']=='parent_context_UCI_ready'}
ready.update(r['diagnostic']['engine_pid'] for r in records if r['record']=='parent_context_UCI_timeout')
retired={r['pid'] for r in records if r['record'] in ('parent_context_owned_child_closed','parent_context_owned_child_retired')}
parents=[r['parent_pid'] for r in records if r['record']=='parent_context_trial_start']
pids=sorted(ready|set(parents))
program='''import json,os
from pathlib import Path
pids=json.loads(os.environ["CHECK_PIDS"])
observed={}
for pid in pids:
 try:observed[str(pid)]=Path(f"/proc/{pid}/status").read_text()
 except FileNotFoundError:observed[str(pid)]=None
engines=[]
for p in Path("/proc").iterdir():
 if not p.name.isdigit():continue
 try:
  target=os.readlink(p/"exe")
  if target=="/tmp/chen-graph-plugin/chen":engines.append({"pid":int(p.name),"exe":target,"status":(p/"status").read_text()})
 except (FileNotFoundError,PermissionError,ProcessLookupError):pass
print(json.dumps({"observed_owned_processes":observed,"live_engine_executables":engines}))
'''
completed=subprocess.run(['docker','exec','-e','CHECK_PIDS='+json.dumps(pids),'cpp-rocm','python','-c',program],capture_output=True,text=True,timeout=10)
assert completed.returncode==0,completed.stderr
inspection=json.loads(completed.stdout)
assert ready<=retired, ('missing owned retirement',ready-retired)
assert not inspection['live_engine_executables'],inspection
assert all(value is None or '\nState:\tZ' in value for value in inspection['observed_owned_processes'].values()),inspection
result={'trial':trial,'owned_ready_or_timedout_pids':sorted(ready),'recorded_retirement_pids':sorted(retired),'consumer_parent_pids':parents,
        'complete_record':any(r['record']=='parent_context_trial_complete' for r in records),
        'comparisons':sum(r['record']=='parent_context_exact_comparison' for r in records),
        'games':[{'opening':r['opening'],'plies':r['plies_after_opening']} for r in records if r['record']=='neural_UCI_legal_game'],
        'timeouts':[{k:r.get(k) for k in ('waiting_for','diagnostic')} for r in records if r['record']=='parent_context_UCI_timeout'],
        'actual_inspection':inspection,'scope':'CPU/proc-only inspection after completed transport; no GPU call'}
with (root/f'trial-{trial}.retirement.json').open('x') as stream:
 json.dump(result,stream,indent=2);stream.write('\n');stream.flush();os.fsync(stream.fileno())
print(json.dumps({k:result[k] for k in ('trial','complete_record','comparisons','games','timeouts')}))
