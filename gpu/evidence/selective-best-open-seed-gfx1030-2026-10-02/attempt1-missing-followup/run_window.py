import subprocess,pathlib,json,datetime,hashlib,time,os
out=pathlib.Path(__file__).resolve().parent
base='/tmp/chen-selective-cache-2026-10-02/libchen_gpu_selection_cache.so'
cand='/tmp/chen-best-open-seed-2026-10-02/libchen_gpu_best_open_seed.so'
root='/tmp/chen-best-open-seed-2026-10-02'
engine='/tmp/chen-fusion-hardware-2026-10-02/chen'
tests='/tmp/chen-fusion-hardware-2026-10-02/device-selective-tests'
pins=['--baseline',base,'--candidate',cand,'--baseline-sha256','a47b8c4e5704703280834b4de864d1daf059b63ce79eb1f1fb22462994d3edaf','--candidate-sha256','f71eabeeaae9db538c60b1f348d7cc571b174be519fc66e1f1d108200b51040f']
commands=[('seed-trace',['timeout','30s','taskset','-c','5',root+'/seed-trace']),('seed-bounds',['timeout','30s','taskset','-c','5',root+'/seed-bounds']),('deep',['timeout','45s','taskset','-c','5','python3','/tmp/chen-selective-cache-2026-10-02/gpu/selective_sparse_followup.py','deep',*pins,'--output',root+'/deep']),('exact-work',['timeout','60s','python3',root+'/selective_seed_probe.py','--engine',engine,'--tests',tests,*pins,'--engine-sha256','a31be5d14e6d62a6861b2aa838fd6fa8ac97de22373c625d5ba2903153018392','--tests-sha256','32930f0682f136d4ad464805efbdbc82f8172031e532446095ed387c894def15','--suite','/tmp/chen-fusion-hardware-2026-10-02/positions.txt','--output',root+'/exact-work'])]
commands=[(name,['docker','exec','cpp-rocm','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/tmp/chen-gpu','LD_LIBRARY_PATH=/tmp/chen-gpu/profiler-deps/usr/lib/x86_64-linux-gnu:/opt/rocm/lib',*cmd]) for name,cmd in commands]
rec={'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'variant':'backup_best_open_seed_v1','scope':'Exclusive bounded GPU admission/parity/alternating fixed-work; no profile or neural run overlap','cpu':5,'baseline_sha256':pins[5],'candidate_sha256':pins[7],'commands':[{'name':name,'argv':argv,'outer_timeout_seconds':70} for name,argv in commands]}
(out/'window-declaration.json').write_text(json.dumps(rec,indent=2)+'\n')
records=[]
try:
 for name,argv in commands:
  start=time.monotonic();result=subprocess.run(argv,capture_output=True,timeout=70)
  (out/(name+'.stdout')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
  records.append({'name':name,'exit_code':result.returncode,'wall_seconds':time.monotonic()-start})
  print(json.dumps(records[-1]),flush=True)
  if result.returncode:raise RuntimeError(name+' failed')
finally:
 (out/'window-result.json').write_text(json.dumps({'status':'passed' if len(records)==len(commands) and all(x['exit_code']==0 for x in records) else 'failed_or_partial','records':records},indent=2)+'\n')
