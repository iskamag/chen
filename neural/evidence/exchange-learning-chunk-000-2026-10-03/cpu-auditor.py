import gzip,hashlib,io,json,tarfile
from pathlib import Path
import torch
from neural.exchange_study import Plateau,next_fit
p=Path('neural/evidence/exchange-learning-chunk-000-2026-10-03');sha=lambda raw:hashlib.sha256(raw).hexdigest()
manifest=json.loads((p/'manifest.json').read_text())
for name,row in manifest['payloads'].items():
 raw=(p/name).read_bytes();assert len(raw)==row['bytes'] and sha(raw)==row['sha256']
envelope=json.loads((p/'archive-members.json').read_text());raw=(p/'runtime-and-run.tar.gz').read_bytes();assert sha(raw)==envelope['archive_sha256']
with tarfile.open(fileobj=io.BytesIO(raw),mode='r:gz') as tar:
 members=tar.getmembers();assert len({m.name for m in members})==len(members)==len(envelope['members'])
 payloads={}
 for member in members:
  assert member.isfile() and not member.name.startswith('/') and '..' not in Path(member.name).parts
  content=tar.extractfile(member).read();row=envelope['members'][member.name]
  assert len(content)==member.size==row['bytes'] and sha(content)==row['sha256'];payloads[member.name]=content
execution=json.loads(payloads['control/execution-declaration.json']);host=json.loads(payloads['control/host-reviewed-declaration.json'])
assert execution['directory']!=host['directory'];normalized=dict(execution);del normalized['parent_declaration_sha256'];normalized['directory']=host['directory'];assert normalized==host
assert execution['parent_declaration_sha256']==sha(payloads['control/host-reviewed-declaration.json'])
for name,pin in execution['source_sha256'].items():assert sha(payloads['closure/neural/'+name])==pin
state=json.loads(payloads['run/state.json']);assert state['active_job'] is None and state['test_gate']=='CLOSED' and len(state['completed'])==2
reports=[]
for name,binding in state['completed'].items():
 report=json.loads(payloads['run/'+binding['report_path']]);assert report==binding['report'];assert sha(payloads['run/'+binding['report_path']])==binding['report_sha256']
 checkpoint_bytes=payloads['run/'+binding['checkpoint_path']];assert sha(checkpoint_bytes)==binding['checkpoint_sha256']
 checkpoint=torch.load(io.BytesIO(checkpoint_bytes),map_location='cpu',weights_only=True)
 assert checkpoint['strategic_value'] is False and checkpoint['test_gate']=='CLOSED' and checkpoint['seed']==6101 and checkpoint['steps']==1 and checkpoint['typed'] is True
 assert checkpoint['learning_rate']==.001 and checkpoint['weight_decay']==report['weight_decay']
 assert sum(t.numel() for t in checkpoint['state_dict'].values())==1067521 and all(torch.isfinite(t).all() for t in checkpoint['state_dict'].values())
 history=[json.loads(line) for line in payloads['run/'+name+'/history.jsonl'].splitlines()];controller=Plateau(execution['protocol'])
 for epoch,row in enumerate(history,1):assert row['epoch']==epoch and controller.update(row['metrics']['macro_family_mae_cp'],epoch)==row['decision']
 assert controller.converged and history[-1]['decision']=='plateau' and report['converged']
 best=min(history,key=lambda r:r['metrics']['macro_family_mae_cp'])
 assert best['epoch']==checkpoint['epoch'] and best['metrics']==report['best_dev_metrics']
 assert best['metrics']['macro_family_mae_cp']==report['best_dev_macro_mae_cp']
 reports.append(dict(name=name,epochs=len(history),best_epoch=checkpoint['epoch'],best_macro_family_mae=report['best_dev_macro_mae_cp'],final_macro_family_mae=history[-1]['metrics']['macro_family_mae_cp']))
job,selected=next_fit(execution['protocol'],state['completed']);assert job['name']=='r1-typed-hpo-lr0.0003-wd0.0' and not selected
assert not any(name.startswith('data/') or name.endswith('pack-test.pt') or 'test-evaluated' in name for name in payloads)
audit=dict(scope='independent_CPU_byte_envelope_checkpoint_schema_and_plateau_replay_no_prediction_quality_reexecution',passed=True,
 payloads_verified=len(manifest['payloads']),archive_members_verified=len(payloads),source_modules_verified=len(execution['source_sha256']),fits=reports,next_fit=job,test_gate='CLOSED',architecture_winner=None)
(p/'cpu-audit.json').write_text(json.dumps(audit,indent=2)+'\n')
(p/'manifest.json').write_text(json.dumps({'payloads':{q.name:{'sha256':sha(q.read_bytes()),'bytes':q.stat().st_size} for q in sorted(p.iterdir()) if q.is_file() and q.name!='manifest.json'},'training_performed':True,'test_gate':'CLOSED','no_architecture_winner':True},indent=2)+'\n')
print(json.dumps(audit,indent=2));print('updated_manifest_sha256',sha((p/'manifest.json').read_bytes()))
