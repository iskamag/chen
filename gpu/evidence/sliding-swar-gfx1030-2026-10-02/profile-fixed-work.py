#!/usr/bin/env python3
"""Separate repaired-profiler phase; original clean runs are not repeated."""
import hashlib,json,os,sys
from pathlib import Path
from selective_fusion_probe import run,normalized,sha,write_json
from sliding_swar_probe import parse_rows,WORKLOADS
base=Path('/tmp/chen-sliding-swar-2026-10-02')
out=base/'profiles-repaired';out.mkdir(exist_ok=False)
engine=Path('/tmp/chen-fusion-hardware-2026-10-02/chen')
libs={'baseline':Path('/tmp/chen-sparse-ply-2026-10-02/libchen_gpu_sparse_ply.so'),'candidate':base/'libchen_gpu_swar.so'}
pins={'baseline':'c33fc8f97e3fd546af72a3e3f22a45ffb57082a6071aa7b07066ada6d50c14cb','candidate':'979d6ad1dbd7a439db7eb6a0dffbd4c6c82d447a28a4f02c83e1163dd2700424'}
assert sha(engine)=='a31be5d14e6d62a6861b2aa838fd6fa8ac97de22373c625d5ba2903153018392'
for arm,path in libs.items():assert sha(path)==pins[arm]
reference_path=base/'study/large-arena-six-1-baseline.stdout'
reference=[normalized(x) for x in parse_rows(reference_path.read_text(),6,WORKLOADS[-1])]
env=dict(os.environ,LD_LIBRARY_PATH='/tmp/chen-gpu/profiler-deps/usr/lib/x86_64-linux-gnu:/opt/rocm/lib')
for key in ('HSA_OVERRIDE_GFX_VERSION','LD_PRELOAD','CHEN_NEURAL_MODEL','CHEN_NEURAL_LIBRARY'):assert not env.get(key)
declaration={'kind':'separate_repaired_profiler_fixed_work_phase','source_sha256':sha(__file__),'engine_sha256':sha(engine),'libraries':pins,'reference_stdout_sha256':sha(reference_path),'libdw_sha256':sha('/tmp/chen-gpu/profiler-deps/usr/lib/x86_64-linux-gnu/libdw.so.1'),'profiler_launcher_sha256':sha('/usr/bin/rocprofv3'),'LD_LIBRARY_PATH':env['LD_LIBRARY_PATH'],'scope':'Two instrumented fixed-work traces; compare counters and PV against already completed clean run, not clean latency inference'}
write_json(out/'declaration.json',declaration)
records=[]
try:
 for arm,path in libs.items():
  assert sha(path)==pins[arm]
  local_env=dict(env,CHEN_GPU_LIBRARY=str(path))
  argv=['taskset','-c','5','rocprofv3','--kernel-trace','--stats','--output-format','csv','--output-directory',str(out/('trace-'+arm)),'--output-file','fixed-work','--',str(engine),'bench','--backend','hip-selective','--eval','tapered','--depth','64','--nodes','50000','--selective-batch','32','--max-positions','250000','--suite','/tmp/chen-fusion-hardware-2026-10-02/positions.txt']
  text,record=run(argv,local_env,out/('profile-'+arm),30)
  assert [normalized(x) for x in parse_rows(text,6,WORKLOADS[-1])]==reference
  records.append({'arm':arm,'record':record,'exact_work_parity':True})
 for arm,path in libs.items():assert sha(path)==pins[arm]
 write_json(out/'result.json',{'status':'passed','profiles':records,'scope':declaration['scope']})
except Exception as e:
 write_json(out/'result.json',{'status':'failed_or_partial','error':str(e),'profiles':records});raise
