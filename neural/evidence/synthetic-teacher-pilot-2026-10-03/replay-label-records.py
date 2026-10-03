"""Mechanical saved-transcript replay, not an independent packet implementation.

Imports the frozen collector closure, never opens an engine or reads a model.
"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys

base=Path(sys.argv[1]).resolve()
sys.path.insert(0,str(base/'source'))
from neural.synthetic_teacher import ordinary_labels,compare_budgets

sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
declaration=json.loads((base/'declaration.json').read_text())
plan=json.loads((base/'plan.json').read_text())
report=json.loads((base/'attempt/report.json').read_text())
assert report['status']=='complete' and report['stockfish_retired']
assert sha(base/'declaration.json')==report['declaration_sha256']
assert sha(base/'plan.json')==declaration['plan_sha256']
for name,pin in declaration['source_sha256'].items():assert sha(base/'source'/name)==pin
roots={r['root_id']:r for r in plan['roots']}
records=[]
for job in plan['jobs']:
 record=json.loads((base/'attempt'/(job['query_id']+'.json')).read_text())
 assert record['job']==job
 marker='> go nodes '+str(job['nodes'])
 assert record['transcript'].count(marker)==1
 start=record['transcript'].index(marker)
 lines=[s[2:] for s in record['transcript'][start+1:] if s.startswith('< ')]
 labels,meta,reason,evidence=ordinary_labels(lines,roots[job['root_id']])
 assert json.dumps((labels,meta,reason,evidence),sort_keys=True)==json.dumps(
     (record['labels'],record['packet_provenance'],record['missing_reason'],record['typed_score_evidence']),sort_keys=True)
 records.append(record)
by={(r['root_id'],r['job']['budget']):r for r in records}
cal=[dict(root_id=key,**compare_budgets(by[key,'base']['labels'],by[key,'calibration_4x']['labels'])) for key in plan['calibration_root_ids']]
assert cal==report['calibration']
counts={budget:dict(queries=sum(r['job']['budget']==budget for r in records),
 accepted=sum(r['job']['budget']==budget and r['labels'] is not None for r in records),
 ordinary_action_rows=sum(len(r['labels']['rows']) for r in records if r['job']['budget']==budget and r['labels'])) for budget in ('base','calibration_4x')}
details=[]
for root in plan['roots']:
 r=by[root['root_id'],'base']
 details.append(dict(root_id=root['root_id'],family_id=root['family_id'],mover=root['mover'],ply=root['ply'],motifs=root['motifs'],
 ordinary_accepted=r['labels'] is not None,missing_reason=r['missing_reason'],
 depth=None if not r['labels'] else r['labels']['common_reported_depth'],
 final_choice=r['packet_provenance']['final_bestmove']))
final_choice_changes=[]
for key in plan['calibration_root_ids']:
 a,b=by[key,'base'],by[key,'calibration_4x']
 final_choice_changes.append(dict(root_id=key,base_final_bestmove=a['packet_provenance']['final_bestmove'],
 high_final_bestmove=b['packet_provenance']['final_bestmove'],
 final_choice_changed=a['packet_provenance']['final_bestmove']!=b['packet_provenance']['final_bestmove'],
 ordinary_pair_admitted=a['labels'] is not None and b['labels'] is not None))
result=dict(schema='chen_synthetic_teacher_mechanical_replay_v1',passed_queries=len(records),
 declaration_sha256=report['declaration_sha256'],counts=counts,
 base_mover_acceptance=dict(Counter(roots[r['root_id']]['mover'] for r in records if r['job']['budget']=='base' and r['labels'])),
 accepted_base_depths=dict(Counter(r['labels']['common_reported_depth'] for r in records if r['job']['budget']=='base' and r['labels'])),
 ordinary_calibration=cal,final_choice_calibration=final_choice_changes,base_roots=details,
 teacher_execution_repeated=False,model_or_heldout_read=False,
 scope='saved transcript/packet/frame/support replay with SAME frozen parser; parent independent review remains separate')
print(json.dumps(result,indent=2))
