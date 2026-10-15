#!/usr/bin/env python3
"""Small frozen-protocol finite-board alignment experiment; NumPy only."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time
import numpy as np

HERE = Path(__file__).resolve().parent
OFF = range(-7, 8)
KNIGHT = [(y,x) for y in [-2,-1,1,2] for x in [-2,-1,1,2] if abs(x)+abs(y)==3]
VARIANTS = ['spatial_direct','complex_phase_padded16','magnitude_only_padded16','power_only_padded16','complex_phase_circular8']


def direct(s,t):
    out=np.empty((15,15))
    for iy,dy in enumerate(OFF):
        for ix,dx in enumerate(OFF):
            sy=s[max(0,-dy):min(8,8-dy),max(0,-dx):min(8,8-dx)]
            ty=t[max(0,dy):min(8,8+dy),max(0,dx):min(8,8+dx)]
            out[iy,ix]=np.sum(sy*ty)
    return out


def encode(s,t,name):
    n=8 if name=='complex_phase_circular8' else 16
    a=np.fft.fft2(s,s=(n,n)); b=np.fft.fft2(t,s=(n,n))
    if name=='magnitude_only_padded16':a,b=np.abs(a),np.abs(b)
    if name=='power_only_padded16':a,b=np.abs(a)**2,np.abs(b)**2
    return a,b


def compose(a,b,name):
    # Power-only has identical information to magnitude-only for nonnegative
    # powers; square root gives the fair corresponding phase-erased decoder.
    if name=='power_only_padded16':a,b=np.sqrt(a),np.sqrt(b)
    c=np.fft.ifft2(np.conj(a)*b).real
    idx=np.arange(-7,8)%a.shape[0]
    return c[np.ix_(idx,idx)]


def query(s,t,name):
    return direct(s,t) if name=='spatial_direct' else compose(*encode(s,t,name),name)


def knight_oracle(s,t):
    total=0.
    for y in range(8):
        for x in range(8):
            for dy,dx in KNIGHT:
                if 0<=y+dy<8 and 0<=x+dx<8:total+=s[y,x]*t[y+dy,x+dx]
    return total


def knight(c):return float(sum(c[y+7,x+7] for y,x in KNIGHT))


def generate(seed):
    eye=np.eye(64).reshape(64,8,8)
    yield 'exhaustive_impulse_pairs',[(s,t) for s in eye for t in eye]
    rng=np.random.default_rng(seed)
    yield 'sparse_binary',[(rng.binomial(1,.125,(8,8)).astype(float),rng.binomial(1,.125,(8,8)).astype(float)) for _ in range(64)]
    yield 'dense_signed',[(rng.integers(-3,4,(8,8)).astype(float),rng.integers(-3,4,(8,8)).astype(float)) for _ in range(64)]
    edge=np.ones((8,8));edge[1:7,1:7]=0
    yield 'edge_binary',[(rng.binomial(1,.5,(8,8))*edge,rng.binomial(1,.5,(8,8))*edge) for _ in range(64)]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',default=str(HERE/'results.json'));ap.add_argument('--timing-out',default=str(HERE/'timing.json'));args=ap.parse_args()
    p=json.loads((HERE/'protocol.json').read_text()); tol=p['tolerance_absolute']
    result={'protocol_sha256':hashlib.sha256((HERE/'protocol.json').read_bytes()).hexdigest(),'families':{}}
    timing_cases=None
    for family,cases in generate(p['random_seed']):
        stats={n:{'cases':len(cases),'full_mismatches':0,'knight_mismatches':0,'max_full_error':0.,'max_knight_error':0.} for n in VARIANTS}
        for s,t in cases:
            truth=direct(s,t); kt=knight_oracle(s,t)
            assert knight(truth)==kt
            for name in VARIANTS:
                actual=query(s,t,name);err=float(np.max(np.abs(actual-truth)));ke=abs(knight(actual)-kt)
                z=stats[name];z['full_mismatches']+=int(err>tol);z['knight_mismatches']+=int(ke>tol);z['max_full_error']=max(z['max_full_error'],err);z['max_knight_error']=max(z['max_knight_error'],ke)
        result['families'][family]=stats
        if family=='dense_signed':timing_cases=cases
    a=np.zeros((8,8));a[0,0]=1;b=np.zeros((8,8));b[0,1]=1;t=np.zeros((8,8));t[1,2]=1
    result['alignment_counterexample']={'sources':['a1','b1'],'target':'c2','source_magnitude_max_difference':float(np.max(np.abs(np.abs(np.fft.fft2(a,s=(16,16)))-np.abs(np.fft.fft2(b,s=(16,16)))))),'true_knight_counts':[knight_oracle(a,t),knight_oracle(b,t)],'phase_erased_knight_counts':[knight(query(a,t,'magnitude_only_padded16')),knight(query(b,t,'magnitude_only_padded16'))]}
    # All4096 impulse pairs have the same two per-channel magnitude spectra.
    deltas={(ty-sy,tx-sx) for sy in range(8) for sx in range(8) for ty in range(8) for tx in range(8)}
    result['impulse_collision']={'pairs_in_one_magnitude_power_class':4096,'distinct_full_correlations':len(deltas),'true_knight_pairs':sum(1 for sy in range(8) for sx in range(8) for ty in range(8) for tx in range(8) if (ty-sy,tx-sx) in KNIGHT)}
    wrap=np.zeros((8,8));wrap[1,6]=1
    result['boundary_counterexample']={'source':'a1','target':'g2','finite_knight':knight_oracle(a,wrap),'circular8_knight':knight(query(a,wrap,'complex_phase_circular8')),'padded16_knight':knight(query(a,wrap,'complex_phase_padded16'))}
    failures=0
    for blockers in range(128):
        expected=np.zeros(7)
        for x in range(7):
            expected[x]=1
            if blockers&(1<<x):break
        failures+=int(not np.array_equal(expected,np.ones(7)))
    result['blocked_slider_control']={'cases':128,'fixed_linear_unblocked_kernel_failures':failures,'query':'rook a1 ray b1..h1, first blocker included; all7 occupancy bits exhaustive','explanation':'A fixed unblocked kernel has no occupancy-dependent stopping operation. Signed phase alone does not supply one.'}
    result['explicit_array_footprints_bytes']={'input_two_8x8_float64':1024,'encoded_two_16x16_complex128':8192,'encoded_two_16x16_float64_magnitudes_or_powers':4096,'encoded_two_8x8_complex128':2048,'full225_float64_output':1800,'scope':'Explicit persistent arrays only; FFT temporaries/Python object overhead excluded. No compression claimed.'}
    assert all(result['families'][f]['complex_phase_padded16']['full_mismatches']==0 for f in result['families'])
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    times={'environment':{'numpy':np.__version__,'python':platform.python_version(),'platform':platform.platform()},'scope':p['timing'],'variants':{}}
    for name in VARIANTS:
        cached=[encode(s,t,name) for s,t in timing_cases] if name!='spatial_direct' else None
        for mode in (['end_to_end','cached_composition_decode'] if cached is not None else ['end_to_end']):
            samples=[]
            for rep in range(p['timing']['warmup']+p['timing']['repeats']):
                start=time.perf_counter_ns()
                for i,(s,t) in enumerate(timing_cases):
                    value=query(s,t,name) if mode=='end_to_end' else compose(*cached[i],name)
                    assert value.shape==(15,15)
                elapsed=(time.perf_counter_ns()-start)/len(timing_cases)/1000
                if rep>=p['timing']['warmup']:samples.append(elapsed)
            times['variants'][name+'/'+mode]={'median_us':float(np.median(samples)),'min_us':min(samples),'max_us':max(samples)}
    Path(args.timing_out).write_text(json.dumps(times,indent=2)+'\n')
    print(json.dumps({'families':{f:{n:v['full_mismatches'] for n,v in st.items()} for f,st in result['families'].items()},'counterexample':result['alignment_counterexample']}))


if __name__=='__main__':main()
