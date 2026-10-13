"""Cold CPU checkpoint mechanics only. No heldout access or quality metrics."""
from pathlib import Path
import json,hashlib,torch
from neural import run_semantic_corpus_study as r
from neural import semantic_corpus_study as s
from neural.semantic_strategy import SemanticAnchor,ENCODERS
base=Path(__file__).resolve().parent;attempt=base/'attempt-1';control=base/'closed-control'
config=s.Configuration();torch.set_num_threads(1)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
terminal=r.read_json(control/'watchdog-terminal.json');assert terminal['exitcode']==1 and terminal['child_retired']
assert not any((attempt/name).exists()for name in ['training-complete.json','validation-checkpoint-gate.json','validation-rocm-admission.json','validation-results.json','selection.json','test-pass-consumed.json','test-results.json','complete.json'])
results=[]
for seed in config.seeds:
    for encoder in ENCODERS:
        for control_name in [None,*s.CONTROLS]:
            filename=f'anchor-{encoder}-{seed}.pt'if control_name is None else f'transition-{encoder}-{control_name}-{seed}.pt'
            path=attempt/filename;assert path.is_file()and path.stat().st_size<=128<<20
            record=torch.load(path,map_location='cpu',weights_only=True)
            assert record['seed']==seed and record['encoder']==encoder
            report=record['report'];assert report['status']=='complete'
            assert report['updates']==(config.anchor_updates if control_name is None else config.transition_updates)
            if control_name is None:
                model=SemanticAnchor(encoder,config.width,config.depth,record['scale']);model.load_state_dict(record['state'])
            else:
                assert record['control']==control_name;model=r.restore_transition(record,config,'cpu')
            assert s.fingerprint(model)==report['checkpoint_state_sha256'],filename
            for tensor in record['state'].values():
                assert tensor.device.type=='cpu'
                if tensor.is_floating_point():assert bool(torch.isfinite(tensor).all())
            results.append(dict(file=filename,sha256=h(path),bytes=path.stat().st_size,phase=report['phase'],updates=report['updates'],elapsed_seconds=report['elapsed_seconds'],stored_active_parameters=report['parameter_counts'],checkpoint_state_sha256=report['checkpoint_state_sha256']))
            del model,record
bundle=attempt/'all-frozen-checkpoints.pt';assert bundle.stat().st_size>128<<20
summary=dict(kind='semantic_failed_training_cold_mechanical_audit_v1',status='failed_at_frozen_bundle_size_gate',GPU_training_observed=True,CPU_audit_only=True,heldout_opened=False,quality_metrics_computed=False,checkpoint_files=len(results),anchor_phases=8,transition_phases=32,anchor_updates=3072,transition_updates=6144,individual_checkpoint_bytes=sum(x['bytes']for x in results),rejected_bundle_bytes=bundle.stat().st_size,rejected_bundle_sha256=h(bundle),gate_limit_bytes=128<<20,all_individual_state_fingerprints_match_phase_reports=True,all_individual_weights_finite=True,external_terminal=terminal,phase_records=results,phase_elapsed_sum=sum(x['elapsed_seconds']for x in results),baseline_witness_sha256=h(control/'runtime-native-baselines.json'),suite_sha256=h(control/'suite-sealed.json'),kernel_sha256=h(control/'kernel-sealed.json'))
with (base/'failed-training-cold-audit.json').open('x')as f:json.dump(summary,f,indent=2,sort_keys=True);f.write('\n')
print(json.dumps({k:v for k,v in summary.items()if k not in ['phase_records','external_terminal']},sort_keys=True))
