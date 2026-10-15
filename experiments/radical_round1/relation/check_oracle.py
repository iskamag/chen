#!/usr/bin/env python3
"""Compare complete experiment classification records with frozen oracle JSONL."""
import json, sys
oracle = {r['id']:r for r in map(json.loads,open(sys.argv[1]))}
counts={}; covered={}; mismatches=[]
for line in open(sys.argv[2]):
    if not line.startswith('{'): continue
    r=json.loads(line)
    if r.get('kind')!='classification': continue
    expected=oracle[r['id']]; variant=r['variant']; counts[variant]=counts.get(variant,0)+1
    covered.setdefault(variant,set()).add(r['id'])
    for field in ['legal','checking','mates','stalemates']:
        if sorted(r[field])!=sorted(expected[field]): mismatches.append([r['id'],variant,field,r[field],expected[field]])
    if r['unknown']: mismatches.append([r['id'],variant,'unexpected-unknown'])
for variant,ids in covered.items():
    if ids!=set(oracle):mismatches.append([variant,'missing-roots',sorted(set(oracle)-ids)])
assert covered,'no classification records'
print(json.dumps({'roots':len(oracle),'records':counts,'mismatches':mismatches}))
assert not mismatches
