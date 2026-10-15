#!/usr/bin/env python3
"""CPU-only paired method-order/frontend/ISA experiment driver."""
import argparse, hashlib, json, pathlib, subprocess
p=argparse.ArgumentParser();p.add_argument('--corpus',required=True);p.add_argument('--out',required=True);p.add_argument('--generic',required=True);p.add_argument('--native',required=True);p.add_argument('--repeats',type=int,default=100);p.add_argument('--runs',type=int,default=4);a=p.parse_args()
out=pathlib.Path(a.out);out.mkdir(parents=True,exist_ok=True)
manifest={'corpus_sha256':hashlib.sha256(pathlib.Path(a.corpus).read_bytes()).hexdigest(),'cpu':3,'runs':[]}
for run in range(a.runs):
 for isa in (['generic','native'] if run%2==0 else ['native','generic']):
  binary=getattr(a,isa)
  for frontend in ['own','codec','orthodox']:
   cmd=['taskset','-c','3',binary,a.corpus,str(a.repeats),'--frontend='+frontend]
   if run%2:cmd+=['--reverse']
   name=f'{isa}-{frontend}-{run}.jsonl'
   with (out/name).open('w') as f:r=subprocess.run(cmd,stdout=f,stderr=subprocess.PIPE,text=True,check=True)
   manifest['runs'].append({'file':name,'command':cmd,'binary_sha256':hashlib.sha256(pathlib.Path(binary).read_bytes()).hexdigest(),'stderr':r.stderr})
   (out/'matrix-manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
   print(name,flush=True)
