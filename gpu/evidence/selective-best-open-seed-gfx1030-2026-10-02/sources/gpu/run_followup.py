import pathlib,json,subprocess,time,datetime
out=pathlib.Path(__file__).resolve().parent
root='/tmp/chen-best-open-seed-2026-10-02'
common=['--baseline','/tmp/chen-selective-cache-2026-10-02/libchen_gpu_selection_cache.so','--candidate',root+'/libchen_gpu_best_open_seed.so','--baseline-sha256','a47b8c4e5704703280834b4de864d1daf059b63ce79eb1f1fb22462994d3edaf','--candidate-sha256','f71eabeeaae9db538c60b1f348d7cc571b174be519fc66e1f1d108200b51040f','--engine','/tmp/chen-fusion-hardware-2026-10-02/chen','--engine-sha256','a31be5d14e6d62a6861b2aa838fd6fa8ac97de22373c625d5ba2903153018392','--suite','/tmp/chen-selective-cache-2026-10-02/suite.json']
commands=[(mode,['docker','exec','cpp-rocm','env','PYTHONDONTWRITEBYTECODE=1','PYTHONPATH=/tmp/chen-gpu','LD_LIBRARY_PATH=/tmp/chen-gpu/profiler-deps/usr/lib/x86_64-linux-gnu:/opt/rocm/lib','timeout','45s','taskset','-c','5','python3','/tmp/chen-selective-cache-2026-10-02/gpu/selective_seed_followup.py',mode,*common,'--output',root+'/'+mode]) for mode in ['uci','trace']]
(out/'followup-declaration.json').write_text(json.dumps({'timestamp_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'exclusive_GPU':True,'profile_separate_from_latency':True,'commands':[{'name':name,'argv':argv,'timeout_seconds':55} for name,argv in commands]},indent=2)+'\n')
records=[]
try:
 for name,argv in commands:
  start=time.monotonic();result=subprocess.run(argv,capture_output=True,timeout=55)
  (out/(name+'.stdout')).write_bytes(result.stdout);(out/(name+'.stderr')).write_bytes(result.stderr)
  records.append({'name':name,'exit_code':result.returncode,'wall_seconds':time.monotonic()-start})
  print(json.dumps(records[-1]),flush=True)
  if result.returncode:raise RuntimeError(name+' failed')
finally:
 (out/'followup-result.json').write_text(json.dumps({'status':'passed' if len(records)==2 and all(x['exit_code']==0 for x in records) else 'failed_or_partial','records':records},indent=2)+'\n')
