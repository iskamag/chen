"""Declared TRAIN/DEV-only restricted-exchange convergence runner.

No strategic CP, quality gate, online backprop or TEST read. Deployment selection
requires a separate frozen evaluation; a resource stop is not convergence.
"""
import argparse
import ast
from dataclasses import asdict,fields
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .exchange_model import ExchangeModel,task_loss
from .exchange_study_data import SCHEMA,digest,read_json,sha
from .tactical_inputs import tactical_config
from .tactical_operator import TacticalBatch,admit_contents

KIND='chen_restricted_exchange_convergence_v1'


def default_protocol(width=256):
    return dict(kind=KIND,width=width,steps=[1,2,4,8],typed=[True,False],
        seeds=[6101,6102,6103],hpo_seed=6101,learning_rates=[.001,.0003],weight_decays=[0.,.0001],
        batch_size=8,max_epochs=200,min_epochs=20,patience=8,absolute_improvement_cp=.5,
        relative_improvement=.002,lr_reductions=2,lr_decay=.3,gradient_clip=1.,
        job_wall_seconds=1800,total_wall_seconds=21600,max_fits_per_chunk=2,chunk_wall_seconds=3900,
        training_loss='active_closed_action_mean_smooth_L1_correction_scaled_800',
        development_primary='macro_family_root_action_mean_absolute_correction_error_cp',
        hpo='all_four_LR_WD_pairs_per_arm_one_paired_seed_then_best_converged_pair_three_paired_seeds_reuse_screen_seed',
        convergence='minimum_epochs_then_three_plateaus_with_two_LR_decays_no_TEST',
        exact_child_control='same_head_destination_seam_only_not_in_this_training_declaration',
        precision='FP32_weights_activations_autograd_no_AMP_no_quantization',
        test_gate='CLOSED',teacher_inputs=False,strategic_value=False)


def validate_protocol(p):
    if p.get('kind')!=KIND or p.get('test_gate')!='CLOSED' or p.get('teacher_inputs') is not False or p.get('strategic_value') is not False:
        raise ValueError('restricted task and closed TEST contract required')
    if p['steps']!=[1,2,4,8] or p['typed']!=[True,False] or p['seeds']!=[6101,6102,6103] or p['hpo_seed']!=6101:
        raise ValueError('paired predeclared recurrence/type controls required')
    if p['learning_rates']!=[.001,.0003] or p['weight_decays']!=[0.,.0001]:raise ValueError('bounded development-only HPO required')
    for name in ('width','batch_size','max_epochs','min_epochs','patience','lr_reductions','job_wall_seconds','total_wall_seconds','max_fits_per_chunk','chunk_wall_seconds'):
        if type(p[name]) is not int or p[name]<1:raise ValueError('positive resource controls required')
    if p['width']<8 or p['max_epochs']<p['min_epochs'] or p['max_fits_per_chunk']>2:raise ValueError('invalid model/convergence controls')
    for name in ('absolute_improvement_cp','relative_improvement','lr_decay','gradient_clip'):
        if type(p[name]) not in (int,float) or not math.isfinite(p[name]) or p[name]<=0:raise ValueError('finite convergence controls required')
    if p['lr_decay']>=1:raise ValueError('learning rate decay must reduce')


class Plateau:
    """Predeclared LR continuation; exhausting safety bounds never sets plateau."""
    def __init__(self,protocol):
        self.protocol=protocol;self.best=math.inf;self.bad=0;self.reductions=0;self.converged=False

    def update(self,value,epoch):
        if not math.isfinite(value):raise ValueError('nonfinite DEV metric')
        threshold=max(self.protocol['absolute_improvement_cp'],self.protocol['relative_improvement']*self.best) if math.isfinite(self.best) else 0.
        improved=value<self.best-threshold
        if improved:self.best=value;self.bad=0
        else:self.bad+=1
        if epoch<self.protocol['min_epochs'] or self.bad<self.protocol['patience']:return 'continue'
        self.bad=0
        if self.reductions<self.protocol['lr_reductions']:
            self.reductions+=1;return 'reduce_lr'
        self.converged=True;return 'plateau'


def load_pack(directory,split):
    if split not in ('train','dev'):raise ValueError('TEST gate CLOSED before pack open')
    directory=Path(directory);path=directory/f'pack-{split}.pt'
    pack=torch.load(path,map_location='cpu',weights_only=True)
    admission=read_json(directory/'admission.json');labels=read_json(directory/f'labels-{split}.json')
    if (pack.get('schema')!=SCHEMA or pack.get('split')!=split or admission.get('test_gate')!='CLOSED' or
        pack['admission_sha256']!=sha(directory/'admission.json') or
        pack['input_sha256']!=labels['input_sha256'] or pack['labels_sha256']!=labels['target_sha256']):
        raise ValueError('pack admission/source binding changed')
    allowed=set(admission['family_ids'][split]);records=pack['records']
    if not records:raise ValueError('no closed task targets')
    for record in records:
        batch=TacticalBatch(**record['observation']);admit_contents(batch,tactical_config())
        mask=record['mask'];target=record['correction'];rounds=record['rounds']
        k=batch.actions.shape[1]
        if (record['family_id'] not in allowed or mask.dtype!=torch.bool or target.dtype!=torch.float32 or
            mask.shape!=(k,) or target.shape!=(k,) or rounds.shape!=(k,) or rounds.dtype!=torch.int64 or
            not bool(mask.any()) or not bool(torch.isfinite(target[mask]).all()) or
            not bool((rounds[mask]>=1).all()) or not bool(batch.action_mask[0,mask].all())):
            raise ValueError('cold task label admission failed')
        # Exact observed action schema: capture channel4 / promotion suffix8.
        if not bool(((batch.actions[0,:,4]>0)|(batch.actions[0,:,8]>0))[mask].all()):
            raise ValueError('task mask includes quiet nonpromotion action')
    return records


def collate(records,device='cpu'):
    """Offline ragged reference collation; active capacities, no global E6144."""
    observations=[r['observation'] for r in records]
    b=len(records);k=max(o['actions'].shape[1] for o in observations);e=max(o['edge_index'].shape[1] for o in observations)
    action_fields={'actions','action_mask','action_ids','action_prior_cp'}
    edge_fields={'edge_index','edge_kind','edge_mask','edge_features'}
    result={}
    for field in fields(TacticalBatch):
        name=field.name;values=[o[name] for o in observations]
        if not isinstance(values[0],torch.Tensor):
            if any(v!=values[0] for v in values):raise ValueError('mixed observation schemas')
            result[name]=values[0];continue
        shape=list(values[0].shape);shape[0]=b
        if name in action_fields:shape[1]=k
        if name in edge_fields:shape[1]=e
        fill=-1 if name in ('action_ids','piece_squares','edge_kind') else 0
        value=torch.full(shape,fill,dtype=values[0].dtype)
        for row,source in enumerate(values):
            if name in action_fields or name in edge_fields:value[row,:source.shape[1]]=source[0]
            else:value[row]=source[0]
        result[name]=value.to(device)
    pad=lambda name,fill=0:torch.stack([F.pad(r[name],(0,k-len(r[name])),value=fill) for r in records]).to(device)
    return TacticalBatch(**result),pad('correction'),pad('mask'),pad('rounds'),dict(
        families=[r['family_id'] for r in records],motifs=[r['motifs'] for r in records],
        capacity_edges=b*e,active_edges=sum(int(o['edge_mask'].sum()) for o in observations),
        capacity_nodes=b*(32+64+k),active_nodes=sum(int(o['piece_mask'].sum())+64+int(o['action_mask'].sum()) for o in observations))


def ordered_batches(records,batch_size,seed=None):
    # Bucket by edge work, then shuffle bucket/batch order reproducibly per epoch.
    ordered=sorted(records,key=lambda r:r['observation']['edge_index'].shape[1])
    buckets=[ordered[i:i+batch_size*8] for i in range(0,len(ordered),batch_size*8)]
    rng=random.Random(seed)
    if seed is not None:
        for bucket in buckets:rng.shuffle(bucket)
        rng.shuffle(buckets)
    return [bucket[i:i+batch_size] for bucket in buckets for i in range(0,len(bucket),batch_size)]


def summarize_rows(rows):
    """DEV diagnostics, not strategic ranking or whole-chess model selection."""
    groups={};nontrivial={};strata={}
    for row in rows:
        fid=row['family_id'];groups.setdefault(fid,[]).append(sum(row['errors'])/len(row['errors']))
        values=[error for error,target in zip(row['errors'],row['targets']) if target!=0]
        if values:nontrivial.setdefault(fid,[]).append(sum(values)/len(values))
        for error,rounds,target in zip(row['errors'],row['rounds'],row['targets']):
            depth='1' if rounds==1 else '2' if rounds==2 else '3-4' if rounds<=4 else '5+'
            for name in ('dependency_'+depth,'correction_zero' if target==0 else 'correction_nonzero',*('motif_'+m for m in row['motifs'])):
                strata.setdefault(name,[]).append(error)
    mean=lambda values:sum(values)/len(values) if values else None
    return dict(macro_family_mae_cp=mean([mean(v) for v in groups.values()]),
        macro_nontrivial_family_mae_cp=mean([mean(v) for v in nontrivial.values()]),
        micro_action_mae_cp=mean([x for row in rows for x in row['errors']]),
        strata={k:dict(count=len(v),mae_cp=mean(v)) for k,v in sorted(strata.items())},
        roots=len(rows),families=len(groups),closed_actions=sum(len(r['errors']) for r in rows))


@torch.no_grad()
def evaluate(model,records,protocol,device,steps,typed,deadline):
    if model is not None:model.eval()
    rows=[]
    for group in ordered_batches(records,protocol['batch_size']):
        if time.monotonic()>=deadline:raise TimeoutError('evaluation resource deadline')
        batch,target,mask,rounds,meta=collate(group,device)
        prediction=torch.zeros_like(target) if model is None else model(batch,steps=steps,typed=typed)['correction_cp']
        if not bool(torch.isfinite(prediction[mask]).all()):raise ValueError('nonfinite task prediction')
        errors=(prediction-target).abs().cpu();target=target.cpu();mask=mask.cpu();rounds=rounds.cpu()
        for i,record in enumerate(group):rows.append(dict(family_id=record['family_id'],motifs=record['motifs'],
            errors=errors[i,mask[i]].tolist(),targets=target[i,mask[i]].tolist(),rounds=rounds[i,mask[i]].tolist()))
    return summarize_rows(rows)


def fit(train,dev,protocol,device,steps,typed,seed,lr,wd,output,deadline):
    output=Path(output);output.mkdir(exist_ok=False);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    model=ExchangeModel(tactical_config(width=protocol['width'],steps=steps)).to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=lr,weight_decay=wd)
    plateau=Plateau(protocol);started=time.monotonic();best=math.inf;best_metrics=None;status='unconverged_epoch_cap';history=[];updates=0
    cold_collate_upload_wall=0.;training_event_ms=0.;processed_edges=0;processed_nodes=0
    deadline=min(deadline,started+protocol['job_wall_seconds'])
    try:
        for epoch in range(1,protocol['max_epochs']+1):
            model.train()
            losses=[]
            for group in ordered_batches(train,protocol['batch_size'],seed+epoch):
                if time.monotonic()>=deadline:raise TimeoutError('training resource deadline')
                tick=time.monotonic();batch,target,mask,rounds,meta=collate(group,device)
                torch.cuda.synchronize(device);cold_collate_upload_wall+=time.monotonic()-tick
                begin,end=torch.cuda.Event(enable_timing=True),torch.cuda.Event(enable_timing=True)
                begin.record();optimizer.zero_grad(set_to_none=True)
                loss=task_loss(model(batch,steps=steps,typed=typed),target,mask)
                loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),protocol['gradient_clip'],error_if_nonfinite=True)
                optimizer.step();end.record();end.synchronize()
                training_event_ms+=begin.elapsed_time(end);updates+=1
                losses.append(float(loss.detach().cpu()))
                processed_edges+=steps*meta['capacity_edges'];processed_nodes+=steps*meta['capacity_nodes']
            metrics=evaluate(model,dev,protocol,device,steps,typed,deadline)
            metric=metrics['macro_family_mae_cp'];decision=plateau.update(metric,epoch)
            if metric<best:
                best=metric;best_metrics=metrics;torch.save(dict(kind=KIND,model_config=model.config(),state_dict=model.state_dict(),
                    seed=seed,steps=steps,typed=typed,learning_rate=lr,weight_decay=wd,epoch=epoch,
                    test_gate='CLOSED',strategic_value=False),output/'best.pt')
            row=dict(epoch=epoch,updates=updates,metrics=metrics,decision=decision,
                train_batch_mean_smooth_L1_loss=sum(losses)/len(losses),
                current_lr=optimizer.param_groups[0]['lr'],wall_seconds=time.monotonic()-started)
            history.append(row)
            with (output/'history.jsonl').open('a') as stream:stream.write(json.dumps(row)+'\n')
            if decision=='reduce_lr':
                for group in optimizer.param_groups:group['lr']*=protocol['lr_decay']
            if decision=='plateau':status='declared_dev_plateau';break
    except TimeoutError:status='unconverged_wall_cap'
    torch.cuda.synchronize(device)
    if time.monotonic()>=deadline and status=='declared_dev_plateau':status='unconverged_final_fence_deadline'
    report=dict(status=status,converged=status=='declared_dev_plateau',best_dev_macro_mae_cp=None if best==math.inf else best,
        best_dev_metrics=best_metrics,
        epochs=len(history),updates=updates,steps=steps,typed=typed,seed=seed,learning_rate=lr,weight_decay=wd,
        wall_seconds=time.monotonic()-started,training_forward_backward_optimizer_event_ms=training_event_ms,
        cold_collation_upload_synchronized_wall_seconds=cold_collate_upload_wall,
        recurrence_capacity_edges=processed_edges,recurrence_capacity_nodes=processed_nodes,
        peak_device_allocated_bytes=torch.cuda.max_memory_allocated(device),test_gate='CLOSED',quality_claim=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n');return report


def resource_estimate(protocol,train,dev):
    groups=ordered_batches(train,protocol['batch_size'],0);d=protocol['width'];max_e=max(max(r['observation']['edge_index'].shape[1] for r in g) for g in groups)
    max_k=max(r['observation']['actions'].shape[1] for r in train);b=protocol['batch_size']
    # Structural work, not FLOPs/sec, cache-fit or actual autograd allocation.
    work=sum(len(g)*(16*(32+64+max(r['observation']['actions'].shape[1] for r in g))*d*d+
              max(r['observation']['edge_index'].shape[1] for r in g)*4*d) for g in groups)
    return dict(model=ExchangeModel.estimate(tactical_config(width=d)),train_roots=len(train),dev_roots=len(dev),
        train_closed_actions=sum(int(r['mask'].sum()) for r in train),batch_size=b,max_active_capacity_edges=max_e,
        max_legal_actions=max_k,one_step_forward_epoch_structural_MACs=work,
        max_edge_expanded_fp32_tensor_bytes=4*b*max_e*d,
        edge_state_8_steps_8_saved_tensors_structural_bytes=4*b*max_e*d*8*8,
        optimizer_parameter_gradient_adam_arrays_bytes=16*ExchangeModel.estimate(tactical_config(width=d))['parameters'],
        planned_hpo_fits=32,planned_additional_selected_seed_fits=16,reused_selected_hpo_seed_fits=8,
        total_wall_safety_seconds=protocol['total_wall_seconds'],job_wall_safety_seconds=protocol['job_wall_seconds'],
        measured_time=None,convergence_guaranteed=False,resource_stop_means_unconverged=True)


def declare(directory,output,width=256):
    directory=Path(directory).resolve();output=Path(output)
    if output.exists():raise ValueError('fresh declaration path required')
    protocol=default_protocol(width);validate_protocol(protocol)
    train=load_pack(directory,'train');dev=load_pack(directory,'dev')
    # Recheck new split identities without opening TEST, whose family/key admission
    # was fixed before labels by the separate data program.
    if {r['family_id'] for r in train}&{r['family_id'] for r in dev}:raise ValueError('family split collision')
    sources=source_closure(('exchange_study.py','int8_inference.py'))
    declaration=dict(kind=KIND,protocol=protocol,directory=str(directory),source_sha256=sources,
        data_sha256={name:sha(directory/name) for name in
            ('admission.json','pack-train.pt','pack-dev.pt','labels-train.json','labels-dev.json')},
        resource_estimate=resource_estimate(protocol,train,dev),
        runtime_requirements=dict(torch='2.10.0+rocm7.0',gfx='gfx1030',hip_version=70051831,
            hip_provider_sha256='c13a873d6444aa9c32d846feaa888de2108a5ed3c15f8a1b7708c5742a894e81'),
        execution_status='not_started',test_gate='CLOSED',quality_claim=False)
    output.write_text(json.dumps(declaration,indent=2)+'\n');return declaration


def next_fit(protocol,completed):
    """Static arm schedule with declared conditional DEV selection, no TEST."""
    selected={}
    for steps in protocol['steps']:
        for typed in protocol['typed']:
            arm=f'r{steps}-'+('typed' if typed else 'untyped');screen=[]
            for lr in protocol['learning_rates']:
                for wd in protocol['weight_decays']:
                    name=f'{arm}-hpo-lr{lr}-wd{wd}'
                    if name not in completed:
                        return dict(name=name,steps=steps,typed=typed,seed=protocol['hpo_seed'],lr=lr,wd=wd),selected
                    screen.append(completed[name]['report'])
            converged=[r for r in screen if r['converged']]
            if len(converged)!=4:
                selected[arm]=dict(status='no_selection_all_four_HPO_controls_must_plateau');continue
            chosen=min(converged,key=lambda r:(r['best_dev_macro_mae_cp'],r['learning_rate'],r['weight_decay']))
            selected[arm]=dict(learning_rate=chosen['learning_rate'],weight_decay=chosen['weight_decay'],
                reused_seed=protocol['hpo_seed'],reused_hpo_report=chosen,status='DEV_HPO_only')
            for seed in protocol['seeds']:
                if seed==protocol['hpo_seed']:continue
                name=f'{arm}-selected-seed{seed}'
                if name not in completed:
                    return dict(name=name,steps=steps,typed=typed,seed=seed,
                        lr=chosen['learning_rate'],wd=chosen['weight_decay']),selected
    return None,selected


def save_state(path,state):
    temporary=path.with_suffix('.pending')
    temporary.write_text(json.dumps(state,indent=2)+'\n');os.replace(temporary,path)


def run(declaration_path,expected_sha,output,max_fits=2):
    """Continue BETWEEN retired fits only; never restart an interrupted fit."""
    if sha(declaration_path)!=expected_sha:raise ValueError('declaration hash mismatch')
    declaration=read_json(declaration_path);protocol=declaration['protocol'];validate_protocol(protocol)
    if type(max_fits)is not int or not 1<=max_fits<=protocol['max_fits_per_chunk']:
        raise ValueError('predeclared bounded fit group required')
    for name,pin in declaration['source_sha256'].items():
        if sha(Path(__file__).parent/name)!=pin:raise ValueError('source freeze changed')
    directory=Path(declaration['directory'])
    for name,pin in declaration['data_sha256'].items():
        if sha(directory/name)!=pin:raise ValueError('data freeze changed')
    if torch.__version__!=declaration['runtime_requirements']['torch']:raise ValueError('actual declared ROCm Torch required')
    from .int8_inference import admit_gfx1030,loaded_hip_runtime
    device,identity=admit_gfx1030();runtime=loaded_hip_runtime();requirements=declaration['runtime_requirements']
    if runtime['sha256']!=requirements['hip_provider_sha256'] or runtime['hipRuntimeGetVersion']!=requirements['hip_version']:
        raise ValueError('actual loaded HIP runtime differs')
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    # Single writer, in addition to external exclusive GPU scheduling.
    import fcntl
    lock=(output/'run.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    state_path=output/'state.json'
    state=read_json(state_path) if state_path.exists() else dict(declaration_sha256=expected_sha,
        completed={},active_job=None,groups=[],active_wall_seconds=0.,test_gate='CLOSED')
    if state['declaration_sha256']!=expected_sha or state['active_job'] is not None:
        raise ValueError('changed declaration or interrupted fit: no automatic restart')
    for record in state['completed'].values():
        if sha(output/record['report_path'])!=record['report_sha256']:
            raise ValueError('retired fit report changed')
        if record['checkpoint_path'] is not None and sha(output/record['checkpoint_path'])!=record['checkpoint_sha256']:
            raise ValueError('retired best DEV checkpoint changed')
    started=time.monotonic();remaining=protocol['total_wall_seconds']-state['active_wall_seconds']
    if remaining<=0:raise ValueError('declared aggregate active wall cap already consumed')
    deadline=started+min(remaining,protocol['chunk_wall_seconds'])
    group_id=len(state['groups']);group_path=output/f'group-{group_id:03d}';group_path.mkdir(exist_ok=False)
    start=dict(declaration_sha256=expected_sha,pid=os.getpid(),device=identity,runtime=runtime,
        started_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),max_fits=max_fits,
        actual_inner_wall_seconds=min(remaining,protocol['chunk_wall_seconds']))
    (group_path/'start.json').write_text(json.dumps(start,indent=2)+'\n')
    train=load_pack(directory,'train');dev=load_pack(directory,'dev');status='between_fit_boundary';jobs=[]
    if not (output/'immediate-material-baseline.json').exists():
        baseline=evaluate(None,dev,protocol,device,1,True,deadline)
        (output/'immediate-material-baseline.json').write_text(json.dumps(baseline,indent=2)+'\n')
        state['baseline_sha256']=sha(output/'immediate-material-baseline.json')
    elif sha(output/'immediate-material-baseline.json')!=state['baseline_sha256']:
        raise ValueError('baseline changed between groups')
    save_state(state_path,state)
    for _ in range(max_fits):
        job,selected=next_fit(protocol,state['completed'])
        if job is None:status='all_declared_fits_retired';break
        if time.monotonic()>=deadline:status='group_resource_cap';break
        state['active_job']=dict(**job,group=group_id,pid=os.getpid());save_state(state_path,state)
        torch.cuda.reset_peak_memory_stats(device)
        report=fit(train,dev,protocol,device,job['steps'],job['typed'],job['seed'],job['lr'],job['wd'],
            output/job['name'],deadline)
        path=output/job['name']/'report.json';checkpoint=output/job['name']/'best.pt'
        state['completed'][job['name']]=dict(report=report,report_path=str(path.relative_to(output)),report_sha256=sha(path),
            checkpoint_path=str(checkpoint.relative_to(output)) if checkpoint.exists() else None,
            checkpoint_sha256=sha(checkpoint) if checkpoint.exists() else None)
        state['active_job']=None;save_state(state_path,state);jobs.append(job['name'])
    torch.cuda.synchronize(device)
    elapsed=time.monotonic()-started;state['active_wall_seconds']+=elapsed
    job,selected=next_fit(protocol,state['completed'])
    if job is None:status='all_declared_fits_retired'
    elif state['active_wall_seconds']>=protocol['total_wall_seconds']:status='incomplete_aggregate_wall_cap'
    elif time.monotonic()>=deadline:status='group_resource_cap'
    terminal=dict(status=status,wall_seconds=elapsed,retired_jobs=jobs,next_job=job,
        hpo_selected=selected,test_gate='CLOSED',test_evaluated=False,architecture_winner=None,
        strategic_value=False,quality_claim=False,all_device_fences_completed=True)
    (group_path/'terminal.json').write_text(json.dumps(terminal,indent=2)+'\n')
    state['groups'].append(dict(path=group_path.name,terminal_sha256=sha(group_path/'terminal.json'),wall_seconds=elapsed))
    state['hpo_selected']=selected;save_state(state_path,state)
    fcntl.flock(lock,fcntl.LOCK_UN);lock.close();return terminal


def source_closure(roots):
    """Actual local relative-import closure including runtime admission helper."""
    directory=Path(__file__).parent;todo=list(roots);found={}
    if (directory/'__init__.py').is_file():todo.append('__init__.py')
    while todo:
        name=todo.pop()
        if name in found:continue
        path=directory/name
        if not path.is_file():raise ValueError('source dependency missing: '+name)
        found[name]=sha(path)
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node,ast.ImportFrom) and node.level==1 and node.module:
                dependency=node.module.replace('.','/')+'.py'
                if dependency not in found:todo.append(dependency)
    return found


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    dec=sub.add_parser('declare');dec.add_argument('--directory',required=True);dec.add_argument('--output',required=True);dec.add_argument('--width',type=int,default=256)
    launch=sub.add_parser('run');launch.add_argument('--declaration',required=True);launch.add_argument('--sha256',required=True);launch.add_argument('--output',required=True);launch.add_argument('--max-fits',type=int,default=2)
    args=parser.parse_args()
    print(json.dumps(declare(args.directory,args.output,args.width) if args.command=='declare' else run(args.declaration,args.sha256,args.output,args.max_fits),indent=2))

if __name__=='__main__':main()
