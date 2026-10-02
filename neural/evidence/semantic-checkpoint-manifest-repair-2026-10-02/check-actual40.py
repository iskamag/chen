"""Read-only mechanics over archived attempt-1 states; no admission gate."""
from pathlib import Path
import json,torch
from neural import run_semantic_corpus_study as r
from neural import semantic_corpus_study as s
base=Path(__file__).resolve().parent;attempt=base.parent/'runtime-2026-10-02/attempt-1'
torch.set_num_threads(1);anchors={};transitions={};phases=[]
for path in sorted(attempt.glob('anchor-*.pt'))+sorted(attempt.glob('transition-*.pt')):
    value=torch.load(path,map_location='cpu',weights_only=True)
    anchor=path.name.startswith('anchor-');key=f"{value['encoder']}/{value['seed']}"if anchor else f"{value['encoder']}/{value['control']}/{value['seed']}"
    (anchors if anchor else transitions)[key]=r.checkpoint_entry(path,r.sha(path))
    phases.append(dict(key=key,**value['report']));scale=value['scale'];del value
record=dict(kind=r.FREEZE_KIND,version=1,declaration_sha256='5567f57a43f8738b3f5170de58ad8301cc0867ec7b760e5362ef61c4a0e4cd42',scale=scale,anchors=anchors,transitions=transitions,phase_reports=phases)
path=base/'mechanical-only-failed-attempt-file-manifest.json';pin=r.publish_json(path,record)
r.read_checkpoint_manifest(attempt,path,pin,record['declaration_sha256'],s.Configuration())
result=dict(kind='semantic_manifest_repair_actual40_mechanical_check_v1',passed=True,files=40,logical_individual_bytes=sum(x['bytes']for table in (anchors,transitions)for x in table.values()),freeze_manifest_bytes=path.stat().st_size,manifest_sha256=pin,all_original_phase_fingerprints_matched=True,heldout_gates_created=False,quality_metrics_computed=False,training=False,attempt1_resumed=False)
print(json.dumps(result,sort_keys=True));r.publish_json(base/'actual40-mechanical-check.json',result)
