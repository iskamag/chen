#!/usr/bin/env python3
"""Pinned, controlled CPU SWAR comparison; explicit binaries and fresh output."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import statistics
import subprocess
import time

ORDER=('baseline','candidate','candidate','baseline','baseline','candidate','candidate','baseline')

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def write(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')

def run(argv,env,stem,timeout=30):
    started=time.monotonic();p=subprocess.Popen(argv,env=env,stdout=subprocess.PIPE,stderr=subprocess.PIPE,start_new_session=True)
    record={'argv':argv,'pid':p.pid,'timeout_seconds':timeout}
    try:stdout,stderr=p.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid,signal.SIGKILL);stdout,stderr=p.communicate(timeout=5)
        record['timed_out']=True
    try:os.killpg(p.pid,0);absent=False
    except ProcessLookupError:absent=True
    Path(str(stem)+'.stdout').write_bytes(stdout);Path(str(stem)+'.stderr').write_bytes(stderr)
    record.update(exit_code=p.returncode,group_absent=absent,wall_seconds=time.monotonic()-started)
    write(str(stem)+'.json',record)
    if p.returncode or not absent or record.get('timed_out'):raise RuntimeError('child failed or not retired: '+str(stem))
    return [json.loads(line) for line in stdout.decode().splitlines() if line.startswith('{')]

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    for field in ('baseline','candidate','suite','perf','output'):ap.add_argument('--'+field,type=Path,required=True)
    for field in ('baseline','candidate','perf'):ap.add_argument('--'+field+'-sha256',required=True)
    ap.add_argument('--cpu',type=int,default=2)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    paths={k:getattr(args,k).resolve(strict=True) for k in ('baseline','candidate','suite','perf')}
    for name in ('baseline','candidate','perf'):
        if sha(paths[name])!=getattr(args,name+'_sha256'):ap.error(name+' pin mismatch')
    if args.cpu not in os.sched_getaffinity(0):ap.error('CPU outside permitted mask')
    expected=sum(bool(line.strip()) and not line.startswith('#') for line in paths['suite'].read_text().splitlines())
    declaration={'kind':'cpu_swar_sliding_controlled_candidate','source_sha256':sha(__file__),
                 'artifacts':{k:{'path':str(p),'sha256':sha(p)} for k,p in paths.items()},'cpu_affinity':[args.cpu],
                 'order':ORDER,'positions':expected,'fixed_nodes':200000,'seconds_per_timed_search':1,
                 'perf_event':'cpu-clock','perf_frequency':997,'scope':'Same compiler flags in both candidate-copy builds; fixed-work timings, separate one-second search throughput and instrumented sampling; no strength inference'}
    write(args.output/'declaration.json',declaration);results=[]
    env=dict(os.environ)
    env['LD_LIBRARY_PATH']='/tmp/chen-perf-2026-10-02/usr/lib'+(':'+env['LD_LIBRARY_PATH'] if env.get('LD_LIBRARY_PATH') else '')
    def verify():
        for k,p in paths.items():
            if sha(p)!=declaration['artifacts'][k]['sha256']:raise ValueError('artifact changed: '+k)
    try:
        for arm in ('baseline','candidate'):
            verify();rows=run(['taskset','-c',str(args.cpu),str(paths[arm]),'verify'],env,args.output/('exhaustive-'+arm))
            if len(rows)!=1 or rows[0].get('status')!='passed':raise ValueError('exhaustive not passed')
        for mode in ('perft','nodes'):
            for arm in ('baseline','candidate'):
                verify();run(['taskset','-c',str(args.cpu),str(paths[arm]),mode,str(paths['suite'])],env,args.output/('warm-'+mode+'-'+arm))
            reference=None;timings={'baseline':[],'candidate':[]};raw=[]
            for index,arm in enumerate(ORDER):
                verify();rows=run(['taskset','-c',str(args.cpu),str(paths[arm]),mode,str(paths['suite'])],env,args.output/(mode+'-'+str(index)+'-'+arm))
                if len(rows)!=expected:raise ValueError('row count changed')
                comparable=[{k:v for k,v in r.items() if k!='elapsed_ns'} for r in rows]
                if reference is None:reference=comparable
                if comparable!=reference:raise ValueError('fixed-work semantic mismatch: '+mode)
                timings[arm].append(sum(r['elapsed_ns'] for r in rows)/1e6);raw.append({'arm':arm,'rows':rows})
            medians={a:statistics.median(v) for a,v in timings.items()}
            results.append({'mode':mode,'median_sum_ms':medians,'baseline_over_candidate':medians['baseline']/medians['candidate'],'runs':raw,'exact_work_parity':True})
        # A one-second search can finish a different iteration: counters/scores
        # are telemetry and explicitly not treated as fixed-work equivalents.
        timed=[]
        for index,arm in enumerate(('baseline','candidate','candidate','baseline')):
            verify();rows=run(['taskset','-c',str(args.cpu),str(paths[arm]),'timed',str(paths['suite'])],env,args.output/('timed-'+str(index)+'-'+arm))
            if len(rows)!=expected:raise ValueError('timed row count changed')
            timed.append({'arm':arm,'rows':rows})
        for arm in ('baseline','candidate'):
            verify();stem=args.output/('perf-'+arm)
            run(['taskset','-c',str(args.cpu),str(paths['perf']),'record','-e','cpu-clock','-F','997','--call-graph','fp','-o',str(stem)+'.data','--',str(paths[arm]),'timed',str(paths['suite'])],env,stem)
            report=subprocess.run([str(paths['perf']),'report','--stdio','--no-children','--percent-limit','0.5','--sort','comm,dso,symbol','-i',str(stem)+'.data'],env=env,capture_output=True,timeout=15,check=True)
            Path(str(stem)+'.report.txt').write_bytes(report.stdout);Path(str(stem)+'.report.stderr').write_bytes(report.stderr)
        verify();write(args.output/'result.json',{'status':'passed','fixed_work':results,'timed':timed,'scope':declaration['scope']})
    except Exception as e:
        write(args.output/'result.json',{'status':'failed_or_partial','error':str(e),'completed_fixed_work':results});raise

if __name__=='__main__':main()
