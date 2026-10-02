#!/usr/bin/env python3
"""Cold stdlib verification of frozen source, exact workloads and process records."""
import hashlib,json,re,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
source=json.loads((p/'source-declaration.json').read_text())
for name,digest in source['source_sha256'].items():
 arm,rest=name.split('/',1);assert sha(p/(arm+'-source')/rest)==digest
run=p/'runs';d=json.loads((run/'declaration.json').read_text());result=json.loads((run/'result.json').read_text());assert result['status']=='passed'
assert d['source_sha256']==sha(p/'cpu_sliding_probe.py')
assert d['cpu_affinity']==[2]
for arm in ('baseline','candidate'):
 assert json.loads((run/('exhaustive-'+arm+'.stdout')).read_text())=={'phase':'exhaustive','checks':4878976,'status':'passed'}
 text=(p/(arm+'-tests.stdout')).read_text()
 assert '0 failed' in text and not re.search(r'[1-9]\d* failed',text)
for file in run.glob('*.json'):
 if file.name in ('declaration.json','result.json'):continue
 r=json.loads(file.read_text());assert r['exit_code']==0 and r['group_absent'] and not r.get('timed_out')
fixed=[];checked=0
for mode in ('perft','nodes'):
 reference=None;times={a:[] for a in ('baseline','candidate')};perposition={}
 for index,arm in enumerate(d['order']):
  data=[json.loads(s) for s in (run/(mode+'-'+str(index)+'-'+arm+'.stdout')).read_text().splitlines()]
  assert len(data)==d['positions'];stable=[{k:v for k,v in r.items() if k!='elapsed_ns'} for r in data]
  if reference is None:reference=stable
  assert stable==reference;checked+=len(data)
  times[arm].append(sum(r['elapsed_ns'] for r in data)/1e6)
  for row in data:perposition.setdefault(row['position'],{'baseline':[],'candidate':[]})[arm].append(row['elapsed_ns']/1e6)
 med={a:statistics.median(v) for a,v in times.items()}
 fixed.append({'mode':mode,'median_sum_ms':med,'baseline_over_candidate':med['baseline']/med['candidate'],'positions':{name:{'median_ms':{a:statistics.median(v) for a,v in arms.items()},'baseline_over_candidate':statistics.median(arms['baseline'])/statistics.median(arms['candidate'])} for name,arms in perposition.items()}})
moves=json.loads((p/'moves-oracle-results.json').read_text());assert moves['status']=='passed' and moves['roots_per_arm']==256 and moves['child_positions_per_arm']==6975
for row in moves['rows']:
 for a in ('baseline','candidate'):assert row['actual'][a]['child_legal_counts']==row['expected_child_legal_counts']
profiles={}
for a in ('baseline','candidate'):
 text=(run/('perf-'+a+'.report.txt')).read_text();assert '# Total Lost Samples: 0' in text
 profiles[a]=[line.strip() for line in text.splitlines() if re.match(r'^\s+\d+\.\d+%',line)]
report={'status':'passed','audit_source_sha256':sha(Path(__file__)),'exact_work_rows':checked,'exhaustive_checks_per_arm':4878976,'independent_move_roots_per_arm':256,'independent_child_legal_count_maps_per_arm':6975,'fixed_work':fixed,'profiles_self_sample_rows':profiles,'scope':'Six FENs, affinity2, identical release flags, no rating inference; timed searches may differ in completed iteration, scores and PV. Shared cache/power are uncontrolled.'}
(p/'audit-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n');print('PASS',checked,'fixed-work rows,source/provenance/process checks')
