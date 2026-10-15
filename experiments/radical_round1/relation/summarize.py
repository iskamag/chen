#!/usr/bin/env python3
import collections,json,pathlib,statistics,sys
root=pathlib.Path(sys.argv[1]);groups=collections.defaultdict(list);families=collections.defaultdict(list)
for file in root.glob('*.jsonl'):
 parts=file.stem.split('-')
 if len(parts)!=3 or parts[0] not in ['generic','native']:continue
 isa,frontend,_=parts
 for line in file.open():
  if not line.startswith('{'):continue
  r=json.loads(line)
  if r.get('kind') in ['aggregate','kernel_aggregate']:groups[(isa,frontend,r['kind'],r['subset'],r['variant'])].append(r['us_per_root'])
  if r.get('kind') in ['case_timing','kernel_case']:families[(isa,frontend,r['kind'],r['subset'],r['variant'],r['family'])].append(r['us_per_root'])
summary=[dict(zip(['isa','frontend','kind','subset','variant'],k),median_us=statistics.median(v),minimum_us=min(v),maximum_us=max(v),runs=len(v))for k,v in sorted(groups.items())]
(root/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
fams=[dict(zip(['isa','frontend','kind','subset','variant','family'],k),mean_us=statistics.mean(v),observations=len(v))for k,v in sorted(families.items())]
(root/'families.json').write_text(json.dumps(fams,indent=2)+'\n')
for row in summary:print(json.dumps(row))
