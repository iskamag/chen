#!/usr/bin/env python3
"""Cold stdlib audit of both completed phases; no producer imports."""
import csv,hashlib,json,re,statistics
from pathlib import Path
p=Path(__file__).resolve().parent
sha=lambda f:hashlib.sha256(f.read_bytes()).hexdigest()
def rows(file):return [json.loads(s) for s in file.read_text().splitlines() if s.startswith('{')]
def stable(row):
 row=json.loads(json.dumps(row))
 for k in ('elapsed_ns','nps','kernel_ms'):row.pop(k,None)
 row['selective'].pop('kernel_ms',None)
 return row
study=p/'study-attempt1'; declaration=json.loads((study/'declaration.json').read_text()); repaired=p/'profiles-repaired'
assert declaration['artifacts']['baseline']['sha256']==sha(p/'libchen_gpu_baseline.so')
assert declaration['artifacts']['candidate']['sha256']==sha(p/'libchen_gpu_swar.so')
assert declaration['source_sha256']==sha(p/'sliding_swar_probe.py')
assert declaration['helper_sha256']==sha(p/'selective_fusion_probe.py')
assert declaration['rules_oracle_sha256']==sha(p/'rules_probe.py')
source=json.loads((p/'source-declaration.json').read_text())
for path,digest in source['source_sha256'].items():assert sha(p/path)==digest
assert (p/'baseline-source/chen_selective.inc').read_bytes()==(p/'candidate-source/chen_selective.inc').read_bytes()
for arm in ('baseline','candidate'):
 oracle=rows(study/('rules-'+arm+'.stdout'));assert oracle[0]['positions']==1024 and oracle[0]['legal_transitions']==19574 and oracle[0]['exact_packet_and_hash']
 assert any(r.get('record')=='resident_perft' and r['depth']==4 and r['nodes']==197281 for r in oracle)
 assert any(r.get('record')=='resident_minimax' and r['score']==36 and r['depth']==3 for r in oracle)
 for phase in ('rules','tests'):
  rec=json.loads((study/(phase+'-'+arm+'.json')).read_text());assert rec['exit_code']==0 and rec['process_group_absent'] and rec['owned_leader_retired'] and not rec['timed_out']
 test=(study/('tests-'+arm+'.stdout')).read_text()
 for name in declaration['tests']:assert 'test '+name+' ... ok' in test
workloads=[];checked=0
for config in declaration['workloads']:
 reference=None;times={'baseline':[],'candidate':[]};kernel={'baseline':[],'candidate':[]}
 for index,arm in enumerate(declaration['order'],1):
  stem=study/(config['name']+'-'+str(index)+'-'+arm)
  rec=json.loads(Path(str(stem)+'.json').read_text());assert rec['exit_code']==0 and rec['process_group_absent'] and not rec['timed_out']
  data=rows(Path(str(stem)+'.stdout'));assert len(data)==6
  comparable=[stable(r) for r in data]
  if reference is None:reference=comparable
  assert comparable==reference
  assert all(r['requested_nodes']==config['nodes'] and r['requested_depth']==config['depth'] and r['selective']['batch_leaves']==config['batch'] for r in data)
  checked+=len(data);times[arm].append(sum(r['elapsed_ns'] for r in data)/1e6);kernel[arm].append(sum(r['kernel_ms'] for r in data))
 med={a:statistics.median(v) for a,v in times.items()}
 workloads.append({'config':config,'median_search_sum_ms':med,'baseline_over_candidate':med['baseline']/med['candidate'],'median_aggregate_kernel_ms':{a:statistics.median(v) for a,v in kernel.items()}})
profiles={}
for arm in ('baseline','candidate'):
 rec=json.loads((repaired/('profile-'+arm+'.json')).read_text());assert rec['exit_code']==0 and rec['process_group_absent'] and not rec['timed_out']
 data=rows(repaired/('profile-'+arm+'.stdout'));assert len(data)==6 and [stable(r) for r in data]==reference
 profiles[arm]=[{**r,'TotalDurationNs':int(r['TotalDurationNs']),'Calls':int(r['Calls']),'Percentage':float(r['Percentage'])} for r in csv.DictReader(open(repaired/('trace-'+arm)/'fixed-work_kernel_stats.csv'))]
assert json.loads((repaired/'result.json').read_text())['status']=='passed'
assert json.loads((study/'profile-baseline.json').read_text())['exit_code']==127
assert 'libdw.so.1' in (study/'profile-baseline.stderr').read_text()
report={'status':'passed_completed_clean_work_and_separate_repaired_profile','audit_source_sha256':sha(Path(__file__)),'cpu_exhaustive':json.loads((p/'cpu-exhaustive.json').read_text()),'clean_rows_checked':checked,'profile_rows_checked':12,'rules_per_arm':{'roots':1024,'transitions':19574,'perft_and_boundaries':True},'workloads':workloads,'profiles':profiles,'adoption':'Keep as controlled candidate; clean integrated latency difference below 0.2 percent, no demonstrated whole-engine gain.','initial_profile_failure':'Original phase preserved; exit127 missing libdw. Only two instrumented traces repeated with recorded profiler dependency path; no clean benchmark repeated.','scope':'Correctness and fixed-work kernel study, no strength or rating inference'}
(p/'audit-report.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print('PASS',checked,'clean rows,12 profile rows, both full rules/selective arms')
