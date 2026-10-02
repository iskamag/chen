"""CPU orchestration probes only; no real labels evaluated or model updates."""
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest.mock import patch
import torch
from neural import run_semantic_corpus_study as r

torch.set_num_threads(1)
proposal_path=Path('results/semantic-corpus-study/outer-runner/host-unsealed-proposal.json')
p=r.read_json(proposal_path)
with TemporaryDirectory() as td:
    kernel=r.read_json(p['kernel_declaration']['path'])
    kernel['validation_manifest_sha256']='0'*64
    kernel_path=Path(td)/'kernel.json'
    kernel_path.write_text(json.dumps(kernel))
    accepted=r.suite_proposal(p['main_root'],Path(p['splits']['train']['path']).parent,kernel_path,
       p['baseline_witness']['path'],p['baseline_witness']['sha256'],Path(td)/'attempt')
    print(json.dumps({'probe':'mismatched_kernel_validation_corpus','accepted':True,
        'kernel_validation':kernel['validation_manifest_sha256'],
        'suite_validation':accepted['splits']['validation']['manifest_sha256']}))
clock=[0.]
observation=SimpleNamespace(midpoint=torch.tensor([0.]),halfwidth=torch.zeros(1),ordinary=torch.tensor([True]))
root=SimpleNamespace(legal_actions=torch.tensor([[1,2]]),targets=SimpleNamespace(midpoint=torch.tensor([[0.,-1.]]),
      halfwidth=torch.zeros(1,2),reported_cp=torch.tensor([[0,1]]),terminal=torch.zeros(1,2,dtype=torch.bool)))
corpus=SimpleNamespace(prepare_structural_schedule=lambda:([0],[0]),observations=lambda plan:observation,roots=lambda plan:root)
def root_predict(batch):
    clock[0]=11.
    return torch.tensor([[1.,0.]])
with patch.object(r.time,'monotonic',side_effect=lambda:clock[0]):
    result=r.evaluate(corpus,observer=lambda batch,plan:torch.zeros_like(batch.midpoint),root_predictor=root_predict,deadline=10.)
print(json.dumps({'probe':'final_batch_exceeds_wall','returned_metrics':result['metrics'],'deadline':10.,'completed_at':clock[0]}))
