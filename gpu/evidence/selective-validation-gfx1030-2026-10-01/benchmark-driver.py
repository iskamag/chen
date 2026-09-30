import json, subprocess, pathlib, time, hashlib, datetime
root=pathlib.Path('/home/iskam/src/vibe/chen')
out=root/'results/selective-validation'
configs=[]
for backend in ['cpu-resident','hip-resident','cpu-selective','hip-selective']:
 for trial in range(3):
  cmd=['docker','exec','-w','/tmp/chen-selective-build','-e','CHEN_GPU_LIBRARY=/tmp/chen-selective-build/libchen_gpu.so','cpp-rocm','timeout','--signal=TERM','--kill-after=5s','30s','/tmp/chen-selective-build/chen','bench','--backend',backend,'--eval','tapered','--depth','4','--nodes','2000','--selective-batch','8','--max-positions','250000','--suite','benches/positions.txt']
  start=time.monotonic(); run=subprocess.run(cmd,text=True,capture_output=True)
  stem=f'bench-{backend}-{trial}'
  (out/(stem+'.stdout')).write_text(run.stdout); (out/(stem+'.stderr')).write_text(run.stderr)
  if run.returncode: raise RuntimeError(f'{stem} exit{run.returncode}: {run.stderr}')
  rows=[json.loads(line) for line in run.stdout.splitlines()]
  assert len(rows)==6 and all(r['nodes']<=2000 for r in rows)
  configs.append({'backend':backend,'trial':trial,'argv':cmd,'exit_code':run.returncode,'wall_seconds':time.monotonic()-start,'rows':rows})
record={'kind':'bounded_classical_selective_vs_fullwidth_diagnostic','requested_node_cap':2000,'requested_max_ply':4,'selective_batch_leaves':8,'candidate_limit':1024,'trials_per_backend':3,'position_suite_sha256':hashlib.sha256((root/'benches/positions.txt').read_bytes()).hexdigest(),'ended_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'runs':configs,'scope':'Six authored positions, three cold-process repeats; CPU/GPU scheduler parity and finite budget/depth, not strength or equal-work speedup. One retained selective arena versus fullwidth iterative re-expansion has different work.'}
(out/'benchmark.json').write_text(json.dumps(record,indent=2)+'\n')
for cpu,gpu in [('cpu-resident','hip-resident'),('cpu-selective','hip-selective')]:
 for trial in range(3):
  a=next(r['rows'] for r in configs if r['backend']==cpu and r['trial']==trial)
  b=next(r['rows'] for r in configs if r['backend']==gpu and r['trial']==trial)
  for x,y in zip(a,b):
   for field in ('position','depth','score_cp','bestmove','nodes','qnodes','stopped','pv','evaluated_positions'):
    assert x[field]==y[field],(cpu,gpu,trial,field,x,y)
   if cpu=='cpu-selective':
    for field in x['selective']:
     if field!='kernel_ms': assert x['selective'][field]==y['selective'][field],(field,x,y)
print('12 processes /72 rows complete; all CPU/GPU same-scheduler fields identical; no node-cap overshoot')
for backend in ['cpu-resident','hip-resident','cpu-selective','hip-selective']:
 rows=next(r['rows'] for r in configs if r['backend']==backend)
 print(backend,[(r['position'],r['nodes'],r['depth'],r['selective']['maximum_frontier_depth'] if r['selective'] else None) for r in rows])
