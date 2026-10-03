#!/usr/bin/env python3
"""ABI2 semantics/timing/cancel admission, not a GPU throughput benchmark."""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import threading
import time
import bounded_search_probe as reference
from rules_probe import pack
import chess

class TimedJob(c.Structure):
    _fields_=[('search',reference.Job),('elapsed_ns',c.c_uint64),('abi_version',c.c_uint32),('struct_size',c.c_uint32)]
class TimedResult(c.Structure):
    _fields_=[('search',reference.Result),('elapsed_ticks',c.c_uint64),('legal_root_count',c.c_uint32),('fallback',reference.Move),('abi_version',c.c_uint32),('struct_size',c.c_uint32)]
class Telemetry(c.Structure):
    _fields_=[('h2d_ms',c.c_double),('kernel_ms',c.c_double),('d2h_ms',c.c_double),('wall_clock_khz',c.c_uint64),('scratch_bytes',c.c_uint64),('history_capacity',c.c_uint64)]
assert c.sizeof(TimedJob)==208 and c.sizeof(TimedResult)==328 and c.sizeof(Telemetry)==48

def job(case,history_offset=0,elapsed_ns=5_000_000_000):
    p=pack(chess.Board(case['fen']))
    if case.get('corrupt_hash'):p.hash^=1
    old=reference.Job(p,case['node_budget'],history_offset,len(case.get('history',[])),1,192,case['depth'],case['max_qply'],case['evaluator'],case['quiescence'],case['root_ply'],0,case['alpha'],case['beta'])
    return TimedJob(old,elapsed_ns,2,208)

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--cases',required=True);parser.add_argument('--library');parser.add_argument('--output');args=parser.parse_args()
    if args.prepare:
        cases=reference.fixtures();Path(args.cases).write_text(json.dumps(cases,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'preparation':'PASS','cases':len(cases),'exhaustive':sum('exhaustive_reference'in x for x in cases)}));return
    cases=json.loads(Path(args.cases).read_text())
    for case in cases:
        if not case.get('corrupt_hash'):assert reference.oracle(case)==case['expected']
        if 'exhaustive_reference'in case:assert reference.oracle(dict(case,alpha=-32000,beta=32000),exhaustive=True)==case['exhaustive_reference']
    lib=c.CDLL(args.library);error=c.create_string_buffer(1024)
    runtime_version=c.c_int();lib.hipRuntimeGetVersion.argtypes=[c.POINTER(c.c_int)]
    assert lib.hipRuntimeGetVersion(c.byref(runtime_version))==0
    providers=sorted({line.split()[-1] for line in Path('/proc/self/maps').read_text().splitlines() if 'libamdhip64.so'in line})
    assert len(providers)==1,providers
    runtime_binding={'hipRuntimeGetVersion':runtime_version.value,'path':providers[0],'sha256':hashlib.sha256(Path(providers[0]).read_bytes()).hexdigest()}
    lib.chen_bounded_timed_abi_version.restype=c.c_uint32;assert lib.chen_bounded_timed_abi_version()==2
    lib.chen_bounded_timed_create.argtypes=[c.c_int32,c.c_uint32,c.c_char_p,c.c_uint64];lib.chen_bounded_timed_create.restype=c.c_void_p
    lib.chen_bounded_timed_run.argtypes=[c.c_void_p,c.POINTER(TimedJob),c.c_uint32,c.POINTER(TimedResult),c.POINTER(Telemetry),c.c_char_p,c.c_uint64]
    lib.chen_bounded_timed_history.argtypes=[c.c_void_p,c.POINTER(c.c_uint64),c.c_uint64,c.POINTER(c.c_double),c.c_char_p,c.c_uint64]
    for name in ('cancel','clear_cancel','destroy'):getattr(lib,'chen_bounded_timed_'+name).argtypes=[c.c_void_p]
    lib.chen_bounded_timed_destroy.restype=c.c_int32
    ctx=lib.chen_bounded_timed_create(0,64,error,len(error));assert ctx,error.value
    packet={'status':'INCOMPLETE','runtime':runtime_binding,'library_sha256':hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),'cases_sha256':hashlib.sha256(Path(args.cases).read_bytes()).hexdigest(),'policy':'unchanged finite ABI1 semantics; independent Python-chess traversal; reused Tapered arithmetic','rows':[],'deadlines':[],'rejections':[]}
    def run(j):
        out=TimedResult();stats=Telemetry();start=time.monotonic()
        assert lib.chen_bounded_timed_run(ctx,c.byref(j),1,c.byref(out),c.byref(stats),error,len(error))==0,error.value
        assert out.abi_version==2 and out.struct_size==328
        return out,stats,(time.monotonic()-start)*1000
    try:
        flattened=[];jobs=[]
        for case in cases:jobs.append(job(case,len(flattened)));flattened.extend(case.get('history',[]))
        history=(c.c_uint64*len(flattened))(*flattened);history_ms=c.c_double()
        assert lib.chen_bounded_timed_history(ctx,history,len(flattened),c.byref(history_ms),error,len(error))==0,error.value
        native_jobs=(TimedJob*len(jobs))(*jobs);outputs=(TimedResult*len(jobs))();stats=Telemetry()
        assert lib.chen_bounded_timed_run(ctx,native_jobs,len(jobs),outputs,c.byref(stats),error,len(error))==0,error.value
        for case,out in zip(cases,outputs):
            assert out.abi_version==2 and out.struct_size==328
            observed=reference.packet(out.search);reference.validate(case,observed)
            if out.legal_root_count:
                legal=reference.canonical(chess.Board(case['fen']));assert out.legal_root_count==len(legal);assert out.fallback.uci()==legal[0].uci()
            packet['rows'].append({'case':case['name'],'result':observed,'device_elapsed_ticks':out.elapsed_ticks,'legal_root_count':out.legal_root_count})
        packet['admission']={name:getattr(stats,name) for name,_ in stats._fields_};packet['history_h2d_ms']=history_ms.value
        # Empty history after batched semantic jobs: no accidental history reuse.
        assert lib.chen_bounded_timed_history(ctx,None,0,c.byref(history_ms),error,len(error))==0
        deep=dict(cases[0],depth=16,max_qply=32,node_budget=1_000_000,history=[])
        for ns in (1,1_000_000,20_000_000,50_000_000):
            out,timing,wall=run(job(deep,elapsed_ns=ns));observed=reference.packet(out.search)
            assert observed['kind']==3 and observed['abort_reason']==6 and observed['score']==0 and not observed['pv'] and observed['completed_depth']==0,observed
            assert bytes(out.search.pv)==bytes(c.sizeof(out.search.pv))
            device_ms=out.elapsed_ticks/timing.wall_clock_khz
            assert device_ms>=ns/1e6 and device_ms<=ns/1e6+20, (ns,device_ms)
            assert device_ms<=wall+2 and device_ms<=timing.kernel_ms+2,(device_ms,wall,timing.kernel_ms)
            packet['deadlines'].append({'allowance_ns':ns,'result':observed,'device_ms':device_ms,'wall_ms':wall,'kernel_event_ms':timing.kernel_ms,'overshoot_ms':device_ms-ns/1e6})
        lib.chen_bounded_timed_cancel(ctx)
        out,_,_=run(job(deep));assert out.search.kind==3 and out.search.abort_reason==7 and out.search.nodes==0
        lib.chen_bounded_timed_clear_cancel(ctx)
        requested=[]
        def cancel():
            time.sleep(.02);requested.append(time.monotonic());lib.chen_bounded_timed_cancel(ctx)
        thread=threading.Thread(target=cancel);thread.start();out,timing,wall=run(job(deep));completed=time.monotonic();thread.join()
        assert out.search.kind==3 and out.search.abort_reason==7 and out.search.nodes>0,(reference.packet(out.search),wall)
        assert (completed-requested[0])*1000<100
        packet['cancellation']={'result':reference.packet(out.search),'wall_ms':wall,'response_ms':(completed-requested[0])*1000,'device_ms':out.elapsed_ticks/timing.wall_clock_khz,'pre_cancelled_nodes':0}
        lib.chen_bounded_timed_clear_cancel(ctx)
        out,_,_=run(job(dict(cases[0],history=[])));reference.validate(cases[0],reference.packet(out.search))
        base=job(dict(cases[0],history=[]));out=TimedResult();c.memset(c.byref(out),0x5a,c.sizeof(out));timing=Telemetry();c.memset(c.byref(timing),0x5a,c.sizeof(timing));oldout=bytes(out);oldstats=bytes(timing)
        for field,value in [('abi_version',1),('struct_size',200),('elapsed_ns',0),('elapsed_ns',5_000_000_001)]:
            broken=TimedJob.from_buffer_copy(base);setattr(broken,field,value)
            assert lib.chen_bounded_timed_run(ctx,c.byref(broken),1,c.byref(out),c.byref(timing),error,len(error))!=0
            assert bytes(out)==oldout and bytes(timing)==oldstats
            packet['rejections'].append({'field':field,'value':value,'error':error.value.decode()})
        packet['status']='PASS';Path(args.output).write_text(json.dumps(packet,indent=2,sort_keys=True)+'\n')
        print(json.dumps({'status':'PASS','cases':len(cases),'deadlines':len(packet['deadlines']),'cancellation':packet['cancellation'],'admission':packet['admission'],'host_rejections':len(packet['rejections'])}),flush=True)
    finally:
        assert lib.chen_bounded_timed_destroy(ctx)==0,'retire process: uncertain HIP cleanup'

if __name__=='__main__':main()
