"""Pinned TRAIN-only whole-model numerical fidelity and latency, never learning."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

import torch

from .int8_inference import (NativeInt8Backend, NativeDynamicInt8Backend,
                            PackedInt8Linear, mixed_int8_copy, value_only_copy)
from .semantic_strategy import SemanticAnchor, SemanticTransition
from . import semantic_corpus_study as study

VARIANTS = ('fp32_full', 'fp32_value', 'int8_multi', 'int8_fused', 'int8_hybrid')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, record):
    with Path(path).open('x') as stream:
        json.dump(record, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')


def errors(actual, reference):
    if actual.device.type != 'cuda' or actual.shape != reference.shape:
        raise ValueError('expected same-shape resident model outputs')
    if not bool(torch.isfinite(actual).all()):
        raise ValueError('nonfinite numerical variant')
    difference = (actual - reference).abs()
    return dict(count=difference.numel(), sum=float(difference.sum()),
                maximum=float(difference.max()), squared_sum=float(difference.square().sum()))


def accumulate(total, row):
    for key in ('count', 'sum', 'squared_sum'):
        total[key] = total.get(key, 0) + row[key]
    total['maximum'] = max(total.get('maximum', 0), row['maximum'])


def finish_errors(row):
    return dict(count=row['count'], maximum_absolute=row['maximum'],
                mean_absolute=row['sum'] / row['count'], mean_squared=row['squared_sum'] / row['count'])


def fidelity(models, batches, phase, control, deadline):
    stats = {name:dict(cp={}, latent={}, roots=0, agreements=0, changed_gaps=[])
             for name in VARIANTS[1:]}
    for features, actions in batches():
        if time.monotonic() > deadline:
            raise TimeoutError('whole-model fidelity deadline')
        predictions = {name:(model(features) if phase == 'anchor' else
                             study.branch(model, control, features, actions))
                       for name, model in models.items()}
        reference = predictions['fp32_full']
        for name, actual in predictions.items():
            if name == 'fp32_full':continue
            if name == 'fp32_value':
                if not (torch.equal(actual['value_cp'], reference['value_cp']) and
                        torch.equal(actual['latent'], reference['latent'])):
                    raise ValueError('same-checkpoint FP32 value-only parity failed')
            accumulate(stats[name]['cp'], errors(actual['value_cp'], reference['value_cp']))
            accumulate(stats[name]['latent'], errors(actual['latent'], reference['latent']))
            if phase == 'transition':
                # Next-mover CP: parent prefers minimum; argmin keeps first
                # canonical action. No teacher target or terminal label enters.
                expected_choice = reference['value_cp'].argmin(-1)
                choice = actual['value_cp'].argmin(-1)
                same = choice == expected_choice
                stats[name]['roots'] += same.numel()
                stats[name]['agreements'] += int(same.sum())
                if actions.shape[1] > 1:
                    sorted_cp = reference['value_cp'].sort(-1).values
                    gap = sorted_cp[:, 1] - sorted_cp[:, 0]
                    stats[name]['changed_gaps'].extend(gap[~same].cpu().tolist())
        del predictions
    result = {}
    for name, row in stats.items():
        gaps = row['changed_gaps']
        result[name] = dict(cp=finish_errors(row['cp']), latent=finish_errors(row['latent']),
            roots=row['roots'], first_canonical_action_agreement=None if not row['roots'] else row['agreements']/row['roots'],
            changed_roots=len(gaps), changed_fp32_best_second_gap_cp=dict(
                minimum=None if not gaps else min(gaps), mean=None if not gaps else statistics.mean(gaps),
                maximum=None if not gaps else max(gaps), exact_ties=sum(x==0 for x in gaps),
                at_most_1=sum(x<=1 for x in gaps), at_most_5=sum(x<=5 for x in gaps),
                at_most_10=sum(x<=10 for x in gaps)))
    return result


def timed(call, count, warmups, deadline):
    for _ in range(warmups):call()
    torch.cuda.synchronize()
    started, ended = (torch.cuda.Event(enable_timing=True) for _ in range(2))
    wall = time.monotonic(); started.record()
    for _ in range(count):call()
    ended.record();torch.cuda.synchronize()
    result = dict(event_us=started.elapsed_time(ended)*1000/count,
                  synchronized_wall_us=(time.monotonic()-wall)*1e6/count, iterations=count)
    if time.monotonic()>deadline:raise TimeoutError('whole-model timing deadline')
    return result


def timings(models, features, actions, phase, control, config, deadline):
    records = {name:[] for name in VARIANTS}
    for round_id in range(config['timing_rounds']):
        order = VARIANTS if round_id % 2 == 0 else tuple(reversed(VARIANTS))
        for name in order:
            model=models[name]
            call=(lambda:model(features)) if phase=='anchor' else (lambda:study.branch(model,control,features,actions))
            records[name].append(timed(call,config['timing_iterations'],config['warmups'],deadline))
    return {name:dict(samples=rows,event_us_median=statistics.median(r['event_us']for r in rows),
                      synchronized_wall_us_median=statistics.median(r['synchronized_wall_us']for r in rows))
            for name,rows in records.items()}


def linear_profile(model, features, actions, phase, control):
    """Separate instrumented pass; hook/event costs do not enter clean timings."""
    active, records, handles = {}, [], []
    def before(name, module, inputs):
        event=torch.cuda.Event(enable_timing=True);event.record()
        active[name]=(event, tuple(inputs[0].shape))
    def after(name, module, inputs, output):
        event=torch.cuda.Event(enable_timing=True);event.record()
        start,shape=active.pop(name)
        records.append((name,start,event,shape,tuple(output.shape),isinstance(module,PackedInt8Linear)))
    for name,module in model.named_modules():
        if isinstance(module,(torch.nn.Linear,PackedInt8Linear)):
            handles.append(module.register_forward_pre_hook(lambda m,i,name=name:before(name,m,i)))
            handles.append(module.register_forward_hook(lambda m,i,o,name=name:after(name,m,i,o)))
    try:
        model(features) if phase=='anchor' else study.branch(model,control,features,actions)
        torch.cuda.synchronize()
        return [dict(name=name,input_shape=shape,output_shape=out,int8_module=integer,
                     event_us=start.elapsed_time(end)*1000)for name,start,end,shape,out,integer in records]
    finally:
        for handle in handles:handle.remove()


def run(path, pin, output):
    if sha(path)!=pin:raise ValueError('diagnostic declaration pin mismatch')
    declaration=json.loads(Path(path).read_text())
    if (declaration['kind']!='chen_frozen_train_only_int8_diagnostic_v1' or
            declaration['training'] or declaration['calibration'] or
            declaration['splits']!=['train'] or declaration['promotion']):
        raise ValueError('frozen TRAIN-only mechanical diagnostic required')
    for source,digest in declaration['dependencies'].items():
        if sha(source)!=digest:raise ValueError('diagnostic source closure changed')
    frozen_path=Path(declaration['frozen_manifest']['path'])
    if sha(frozen_path)!=declaration['frozen_manifest']['sha256']:raise ValueError('checkpoint manifest changed')
    frozen=json.loads(frozen_path.read_text())
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();deadline=started+declaration['config']['wall_seconds']
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    torch.backends.cudnn.benchmark=False
    native=declaration['native']
    common=dict(device='cuda:0',runtime_path=native['runtime_path'],runtime_sha256=native['runtime_sha256'])
    backend=NativeInt8Backend(native['library_path'],native['library_sha256'],**common)
    fused=NativeDynamicInt8Backend(native['library_path'],native['library_sha256'],**common)
    hybrid=NativeDynamicInt8Backend(native['library_path'],native['library_sha256'],
                                  wider_rows=declaration['config']['fp32_wider_rows'],**common)
    if backend.runtime_identity['hipRuntimeGetVersion']!=native['runtime_version']:
        raise ValueError('HIP runtime API version changed')
    write(output/'device-admission.json',dict(device=backend.device_identity,runtime=backend.runtime_identity,
          native=native,variant='mixed INT8 linears; FP32 attention/norm/nonlinearity/residuals'))
    from experiments.semantic_corpus import resident
    train=declaration['train']
    admission=dict(kind='semantic_resident_rocm_admission_v1',corpus_manifest_sha256=train['manifest_sha256'],
                   packing_source_sha256=train['packing_source_sha256'],resident_source_sha256=sha(resident.__file__),
                   stage='train',device_arch='gfx1030')
    write(output/'train-rocm-admission.json',admission)
    corpus=resident.load_resident(train['path'],expected_manifest_sha256=train['manifest_sha256'],
          expected_source_sha256=train['packing_source_sha256'],stage='train',device='cuda:0',
          rocm_admission_path=output/'train-rocm-admission.json',rocm_admission_sha256=sha(output/'train-rocm-admission.json'))
    roots, observations=corpus.prepare_structural_schedule()
    # Structural selection only, no score/label-dependent plan choice.
    root_sample=sorted(roots,key=lambda p:(abs(p.legal_count-32),-len(p.root_ids),p.key))[0]
    records={}
    with torch.inference_mode():
        for phase,table in [('anchor',frozen['anchors']),('transition',frozen['transitions'])]:
            for key,entry in table.items():
                checkpoint=frozen_path.parent/entry['file']
                if checkpoint.name!=entry['file'] or checkpoint.is_symlink() or checkpoint.stat().st_size!=entry['bytes']:
                    raise ValueError('checkpoint bounded regular identity mismatch')
                raw=checkpoint.read_bytes()
                if hashlib.sha256(raw).hexdigest()!=entry['sha256']:raise ValueError('checkpoint pin mismatch before load')
                state=torch.load(io.BytesIO(raw),map_location='cpu',weights_only=True)
                anchor=SemanticAnchor(state['encoder'],declaration['config']['width'],declaration['config']['depth'],state['scale'])
                control=state.get('control')
                source=anchor if phase=='anchor' else SemanticTransition(anchor,'conditioned'if control=='shuffled_conditioned'else control)
                source.load_state_dict(state['state'],strict=True)
                source.eval().requires_grad_(False).to('cuda:0')
                models={'fp32_full':source,'fp32_value':value_only_copy(source)}
                setup={}
                for name,implementation in [('int8_multi',backend),('int8_fused',fused),('int8_hybrid',hybrid)]:
                    start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
                    torch.cuda.synchronize();wall=time.monotonic();start.record()
                    models[name],linears=mixed_int8_copy(source,implementation)
                    end.record();torch.cuda.synchronize()
                    setup[name]=dict(event_ms=start.elapsed_time(end),synchronized_wall_ms=(time.monotonic()-wall)*1000,
                                     linears=list(linears),scope='one_time_owned_copy_finite_admission_weight_quantization_and_static_packing')
                def batches():
                    for plan in observations if phase=='anchor'else roots:
                        batch=corpus.observations(plan)if phase=='anchor'else corpus.roots(plan)
                        yield batch.features,None if phase=='anchor'else batch.legal_actions
                row=dict(checkpoint_sha256=entry['sha256'],phase=phase,control=control,setup=setup,
                         fidelity=fidelity(models,batches,phase,control,deadline),timings={})
                if phase=='anchor':
                    first=corpus.observations(observations[0]).features
                    for count in declaration['config']['anchor_batches']:
                        features=first.repeat((count+first.shape[0]-1)//first.shape[0],1,1,1)[:count].contiguous()
                        row['timings'][str(count)]=timings(models,features,None,phase,control,declaration['config'],deadline)
                    profile_features=first;profile_actions=None
                else:
                    batch=corpus.roots(root_sample);profile_features=batch.features;profile_actions=batch.legal_actions
                    row['timing_root_plan']=dict(key=root_sample.key,batch=len(root_sample.root_ids),K=root_sample.legal_count,
                                                selection='closest_K32_then_largest_batch_then_key_no_labels')
                    row['timings']['complete_root']=timings(models,profile_features,profile_actions,phase,control,declaration['config'],deadline)
                row['separate_instrumented_linear_profile']={name:linear_profile(models[name],profile_features,profile_actions,phase,control)
                                                            for name in ('fp32_value','int8_multi','int8_fused','int8_hybrid')}
                records[key]=row
                write(output/f"{phase}-{key.replace('/','-')}.json",row)
                print(json.dumps(dict(completed_key=key,phase=phase,elapsed_seconds=time.monotonic()-started)),flush=True)
                del models,source,anchor,state,raw
                torch.cuda.synchronize()
    write(output/'complete.json',dict(status='complete',declaration_sha256=pin,records=records,
          elapsed_seconds=time.monotonic()-started,split='train',teacher_quality_metrics=False,
          calibration=False,training=False,promotion=False,rating_claim=None,
          timing_scope='eager_device_event_intervals_include_host_dispatch_gaps; synchronized_wall_includes_completion',
          policy_scope='raw_neural_complete_root_choices_against_same_FP32;simulator_terminal_adjudication_excluded',
          corpus_ingress_scope='CPU_validated_packed_TRAIN_upload_then_GPU_feature_decode;not_native_search_node_producer',
          graph_capture=False,engine_latency_claim=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--declaration',required=True)
    parser.add_argument('--sha256',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();run(args.declaration,args.sha256,args.output)
