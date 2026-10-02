import pathlib,subprocess,json,time,datetime
out=pathlib.Path(__file__).resolve().parent
root='/tmp/chen-best-open-seed-2026-10-02'
argv=['docker','exec','cpp-rocm','env','PYTHONDONTWRITEBYTECODE=1','timeout','50s','python3',root+'/selective_seed_pairs.py',
 '--engine','/tmp/chen-fusion-hardware-2026-10-02/chen','--engine-sha256','a31be5d14e6d62a6861b2aa838fd6fa8ac97de22373c625d5ba2903153018392',
 '--baseline','/tmp/chen-selective-cache-2026-10-02/libchen_gpu_selection_cache.so','--baseline-sha256','a47b8c4e5704703280834b4de864d1daf059b63ce79eb1f1fb22462994d3edaf',
 '--candidate',root+'/libchen_gpu_best_open_seed.so','--candidate-sha256','f71eabeeaae9db538c60b1f348d7cc571b174be519fc66e1f1d108200b51040f',
 '--suite','/tmp/chen-fusion-hardware-2026-10-02/positions.txt','--output',root+'/paired-confirmation','--cpu','5']
(out/'pairs-transport-declaration.json').write_text(json.dumps(dict(argv=argv,timeout_seconds=60,
 timestamp_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),kind='separately_declared_timing_only_balanced_12_pairs',
 correctness_reruns=False,exclusive_gpu_after_coherence_release=True),indent=2)+'\n')
start=time.monotonic();result=subprocess.run(argv,capture_output=True,timeout=60)
(out/'pairs-transport.stdout').write_bytes(result.stdout);(out/'pairs-transport.stderr').write_bytes(result.stderr)
(out/'pairs-transport-result.json').write_text(json.dumps(dict(exit_code=result.returncode,wall_seconds=time.monotonic()-start),indent=2)+'\n')
print(result.stdout.decode(),flush=True)
if result.returncode:raise RuntimeError(result.stderr.decode())
