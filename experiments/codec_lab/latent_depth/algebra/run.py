#!/usr/bin/env python3
"""Fixed Boolean polynomial state: interaction depth, not path expansion."""
import argparse
import hashlib
import json
from pathlib import Path
import time
import numpy as np

HERE=Path(__file__).resolve().parent
VARIANTS=['nilpotent_closure','frozen_state','additive_reinject_same_edges','omit_resource_exclusion']

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def independent_maximum(columns,resources):
    # Independent sequential assignment DP, used only as external oracle.
    masks={0}
    for neighbors in columns:
        following=set(masks)
        for used in masks:
            for d in range(resources):
                if neighbors&(1<<d) and not used&(1<<d):following.add(used|(1<<d))
        masks=following
    return max(m.bit_count() for m in masks)

def pair_indices(t,d,exclude_resources):
    target_pairs=[(a,b) for a in range(1<<t) for b in range(1<<t) if not a&b]
    resource_pairs=[(a,b) for a in range(1<<d) for b in range(1<<d) if not exclude_resources or not a&b]
    tl,tr=np.array(target_pairs,dtype=np.uint16).T
    rl,rr=np.array(resource_pairs,dtype=np.uint16).T
    left=(np.repeat(tl,len(rl))<<d)|np.tile(rl,len(tl))
    right=(np.repeat(tr,len(rr))<<d)|np.tile(rr,len(tr))
    return left,right,left|right

def initial(columns,t,d):
    state=np.zeros(1<<(t+d),dtype=bool);state[0]=True
    for target,neighbors in enumerate(columns):
        for resource in range(d):
            if neighbors&(1<<resource):state[((1<<target)<<d)|(1<<resource)]=True
    return state

def compose(state,indices):
    left,right,out=indices
    updated=state.copy()
    np.logical_or.at(updated,out,state[left]&state[right])
    return updated

def readout(state,t,d,saturated,variant):
    full=bool(np.any(state[((1<<t)-1)<<d:]))
    if full:answer='feasible'
    elif saturated and variant in ('nilpotent_closure','omit_resource_exclusion'):answer='impossible'
    else:answer='unknown'
    cardinalities=np.array([m.bit_count() for m in range(1<<t)],dtype=np.uint8)
    target_active=state.reshape(1<<t,1<<d).any(axis=1)
    return {'answer':answer,'saturated_under_update':bool(saturated),'max_covered_targets':int(np.max(cardinalities[target_active])),'active_coordinates':int(np.count_nonzero(state))}

def choose(corpus):
    allowed=[r for r in corpus['rows'] if r['split'] in ('eval','structure') and r['targets']<=6 and r['resources']<=6]
    result=[r for r in allowed if r['split']=='structure']
    groups={}
    for row in allowed:
        if row['split']=='eval':groups.setdefault((row['resources'],row['targets']),[]).append(row)
    for key in sorted(groups):
        result+=sorted(groups[key],key=lambda r:hashlib.sha256(('latent-algebra-v1|'+r['id']).encode()).digest())[:16]
    for name,cols in [('gram_star',[3,5,9,3]),('gram_triangle',[3,5,6,3])]:
        maximum=independent_maximum(cols,4)
        result.append({'id':name,'split':'worked','family':'gram_equal','targets':4,'resources':4,'columns':cols,'maximum_matching':maximum,'feasible':maximum==4})
    return sorted(result,key=lambda r:r['id'])

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--corpus',required=True);ap.add_argument('--out',default=str(HERE/'results.json'));ap.add_argument('--timing-out',default=str(HERE/'timing.json'));args=ap.parse_args()
    protocol=json.loads((HERE/'protocol.json').read_text());rows=choose(json.loads(Path(args.corpus).read_text()))
    if not rows:raise ValueError('empty corpus selection')
    tables={};footprints={};timings={v:[[] for _ in range(5)] for v in VARIANTS};records=[]
    for row in rows:
        t,d=row['targets'],row['resources'];cols=row['columns']
        assert len(cols)==t and all(0<=c<(1<<d) for c in cols)
        maximum=independent_maximum(cols,d)
        assert maximum==row['maximum_matching'] and (maximum==t)==row['feasible'],row['id']
        for excluded in [True,False]:
            key=(t,d,excluded)
            if key not in tables:tables[key]=pair_indices(*key)
            footprints[str(key)]={'state_bytes':1<<(t+d),'composition_pairs':len(tables[key][0]),'composition_indices_bytes':sum(a.nbytes for a in tables[key])}
        start=initial(cols,t,d)
        record={k:row[k] for k in ['id','split','family','targets','resources','columns','maximum_matching','feasible']};record['variants']={}
        for variant in VARIANTS:
            state=start.copy();stages=[readout(state,t,d,False,variant)]
            for depth in range(1,5):
                tick=time.perf_counter_ns()
                if variant=='frozen_state':new=state.copy()
                elif variant=='additive_reinject_same_edges':new=state|start
                else:new=compose(state,tables[(t,d,variant=='nilpotent_closure')])
                elapsed=time.perf_counter_ns()-tick
                saturated=np.array_equal(new,state);state=new
                stages.append(readout(state,t,d,saturated,variant));timings[variant][depth].append(elapsed)
            record['variants'][variant]=stages
        records.append(record)
    summaries={}
    for split in ['all','eval','structure','worked']:
        subset=[r for r in records if split=='all' or r['split']==split]
        summaries[split]={}
        for variant in VARIANTS:
            curve=[]
            for depth in range(5):
                counts=dict(correct_feasible=0,correct_impossible=0,unknown=0,false_positive=0,false_negative=0)
                for row in subset:
                    answer=row['variants'][variant][depth]['answer'];truth=row['feasible']
                    key='unknown' if answer=='unknown' else ('correct_feasible' if truth else 'false_positive') if answer=='feasible' else ('false_negative' if truth else 'correct_impossible')
                    counts[key]+=1
                curve.append({'depth':depth,'cases':len(subset),**counts})
            summaries[split][variant]=curve
    assert not any(r['false_positive'] or r['false_negative'] for r in summaries['all']['nilpotent_closure'])
    assert summaries['all']['nilpotent_closure'][4]['unknown']==0
    result={'protocol_sha256':sha(HERE/'protocol.json'),'source_sha256':sha(__file__),'corpus_sha256':sha(args.corpus),'selection_ids':[r['id'] for r in rows],'oracle_label_fields':['maximum_matching','feasible'],'oracle_mismatches':0,'footprints':footprints,'summaries':summaries,'records':records,'limits':protocol['resource_limits']}
    Path(args.out).write_text(json.dumps(result,indent=2)+'\n')
    timing={'numpy':np.__version__,'scope':'Single exploratory update per selected instance per depth. No optimized-kernel or throughput comparison. Readout, oracle, index precomputation excluded; full fixed state and fixed pair tables traversed.','variants':{v:[{'depth':i,'samples':len(xs),'median_us':float(np.median(xs))/1000 if xs else None,'total_ms':sum(xs)/1e6} for i,xs in enumerate(rounds)] for v,rounds in timings.items()}}
    Path(args.timing_out).write_text(json.dumps(timing,indent=2)+'\n')
    print(json.dumps(summaries['all']))

if __name__=='__main__':main()
