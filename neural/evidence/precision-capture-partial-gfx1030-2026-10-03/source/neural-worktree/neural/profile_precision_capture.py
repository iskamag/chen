"""Separately declared same-checkpoint TRAIN precision/dispatch controls.

Reuses frozen original checkpoints and data, never repeats the 40-model study.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
from pathlib import Path
import statistics
import time

import torch

from .int8_inference import NativeDynamicInt8Backend, mixed_int8_copy, value_only_copy
from .precision_inference import NativeHalfBackend, StaticCapturedInference, mixed_fp16_copy
from .profile_int8_inference import accumulate, errors, finish_errors, sha, timed, write
from .semantic_strategy import SemanticAnchor, SemanticTransition
from . import semantic_corpus_study as study

VARIANTS = ('fp32_value', 'int8_fused', 'torch_fp16_affine', 'native_fp16_affine', 'torch_autocast_fp16')


def forward(model, features, actions, control, *, autocast=False):
    # Cache disabled: conversion is explicit model work in both eager/capture.
    with torch.autocast('cuda', dtype=torch.float16, enabled=autocast, cache_enabled=False):
        return model(features) if actions is None else study.branch(model, control, features, actions)


def check_deadline(deadline):
    if time.monotonic() > deadline:
        raise TimeoutError('precision/capture diagnostic deadline')


def fidelity(calls, batches, deadline):
    totals = {name:dict(cp={}, latent={}, roots=0, agreements=0, changed_gaps=[])
              for name in VARIANTS[1:]}
    for features, actions in batches():
        check_deadline(deadline)
        reference = calls['fp32_value'](features, actions)
        for name in VARIANTS[1:]:
            actual = calls[name](features, actions)
            for key in ('value_cp', 'latent'):
                accumulate(totals[name]['cp' if key == 'value_cp' else 'latent'], errors(actual[key].float(), reference[key]))
            if actions is not None:
                same = actual['value_cp'].argmin(-1) == reference['value_cp'].argmin(-1)
                totals[name]['roots'] += same.numel()
                totals[name]['agreements'] += int(same.sum())
                if actions.shape[1] > 1:
                    sorted_cp = reference['value_cp'].sort(-1).values
                    totals[name]['changed_gaps'].extend((sorted_cp[:, 1] - sorted_cp[:, 0])[~same].cpu().tolist())
    return {name:dict(cp=finish_errors(row['cp']), latent=finish_errors(row['latent']),
                     roots=row['roots'], changed_roots=len(row['changed_gaps']),
                     first_canonical_action_agreement=None if not row['roots'] else row['agreements'] / row['roots'],
                     changed_fp32_best_second_gaps_cp=row['changed_gaps']) for name, row in totals.items()}


def graph_numerical_check(graph, call, features, actions, alternate_features=None):
    """Both original and changed resident inputs; graph output is borrowed."""
    result = []
    # Rotate whole rows of exact admitted positions and their complete actions.
    for shift in (0, 1):
        candidate = features if shift == 0 else (features.roll(1, 0).contiguous()
                    if alternate_features is None else alternate_features)
        moves = None if actions is None else actions.roll(shift, 0).contiguous()
        expected = call(candidate, moves)
        actual = graph.copy_and_replay(candidate, moves)
        torch.cuda.synchronize()
        row = dict(input_features_changed=not torch.equal(candidate, features),
                   input_actions_changed=False if actions is None else not torch.equal(moves, actions))
        for key in ('value_cp', 'latent'):
            row[key] = finish_errors(errors(actual[key].float(), expected[key].float()))
            if not torch.equal(actual[key], expected[key]):
                raise ValueError('capture differs from its own eager numerical variant')
        result.append(row)
    graph.copy_and_replay(features, actions)
    torch.cuda.synchronize()
    return result


def routes(calls, features, actions, config, deadline, alternate_features=None):
    graphs, setup = {}, {}
    for name in VARIANTS:
        check_deadline(deadline)
        torch.cuda.synchronize()
        start = time.monotonic()
        graphs[name] = StaticCapturedInference(calls[name], features, actions, warmups=config['warmups'])
        setup[name] = dict(synchronized_wall_ms=(time.monotonic() - start) * 1000,
                           numerical_checks=graph_numerical_check(graphs[name], calls[name], features, actions, alternate_features))
    samples = {name:{route:[] for route in ('eager', 'graph_replay', 'resident_copy_graph')} for name in VARIANTS}
    pairs = [(name, route) for name in VARIANTS for route in samples[name]]
    for round_id in range(config['timing_rounds']):
        for name, route in pairs if round_id % 2 == 0 else reversed(pairs):
            check_deadline(deadline)
            if route == 'eager':
                call = lambda name=name: calls[name](features, actions)
            elif route == 'graph_replay':
                call = graphs[name].replay
            else:
                call = lambda name=name: graphs[name].copy_and_replay(features, actions)
            samples[name][route].append(timed(call, config['timing_iterations'], config['warmups'], deadline))
    torch.cuda.synchronize()
    check_deadline(deadline)
    return dict(setup=setup, samples=samples,
                medians={name:{route:dict(event_us=statistics.median(r['event_us'] for r in rows),
                                         synchronized_wall_us=statistics.median(r['synchronized_wall_us'] for r in rows))
                               for route, rows in table.items()} for name, table in samples.items()})


def run(path, pin, output):
    if sha(path) != pin:
        raise ValueError('precision declaration pin mismatch')
    declaration = json.loads(Path(path).read_text())
    if (declaration['kind'] != 'chen_train_precision_capture_diagnostic_v1' or
            declaration['splits'] != ['train'] or declaration['training'] or
            declaration['calibration'] or declaration['promotion']):
        raise ValueError('TRAIN-only no-learning declaration required')
    expected_keys = [f'{encoder}/{20261021}' for encoder in ('square', 'relation', 'plan', 'threat')]
    expected_keys += [f'{encoder}/conditioned/{20261021}' for encoder in ('square', 'relation', 'plan', 'threat')]
    if declaration['checkpoint_keys'] != expected_keys:
        raise ValueError('eight fixed structural representative checkpoints required')
    config = declaration['config']
    if config != dict(anchor_batches=[1, 32, 256], timing_rounds=4, timing_iterations=50,
                      warmups=3, wall_seconds=600):
        raise ValueError('fixed bounded precision/capture configuration required')
    for source, digest in declaration['dependencies'].items():
        if sha(source) != digest:
            raise ValueError('precision source closure changed')
    base_path = Path(declaration['baseline_declaration']['path'])
    if sha(base_path) != declaration['baseline_declaration']['sha256']:
        raise ValueError('original diagnostic declaration changed')
    base = json.loads(base_path.read_text())
    frozen_path = Path(base['frozen_manifest']['path'])
    if sha(frozen_path) != base['frozen_manifest']['sha256']:
        raise ValueError('original 40-checkpoint manifest changed')
    frozen = json.loads(frozen_path.read_text())
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    deadline = started + config['wall_seconds']
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False
    common = dict(device='cuda:0', runtime_path=base['native']['runtime_path'], runtime_sha256=base['native']['runtime_sha256'])
    integer = NativeDynamicInt8Backend(base['native']['library_path'], base['native']['library_sha256'], **common)
    half = NativeHalfBackend(declaration['native_half']['library_path'], declaration['native_half']['library_sha256'], **common)
    if integer.runtime_identity['hipRuntimeGetVersion'] != base['native']['runtime_version'] or half.runtime != integer.runtime_identity:
        raise ValueError('producing HIP runtime version/identity changed')
    write(output / 'device-admission.json', dict(device=integer.device_identity, runtime=integer.runtime_identity,
          native_half=declaration['native_half'], accumulation='native FDOT2 FP32; Torch output FP16 and reduced_precision_reduction disabled'))
    from experiments.semantic_corpus import resident
    train = base['train']
    admission = dict(kind='semantic_resident_rocm_admission_v1', corpus_manifest_sha256=train['manifest_sha256'],
                     packing_source_sha256=train['packing_source_sha256'], resident_source_sha256=sha(resident.__file__),
                     stage='train', device_arch='gfx1030')
    write(output / 'train-rocm-admission.json', admission)
    corpus = resident.load_resident(train['path'], expected_manifest_sha256=train['manifest_sha256'],
        expected_source_sha256=train['packing_source_sha256'], stage='train', device='cuda:0',
        rocm_admission_path=output / 'train-rocm-admission.json', rocm_admission_sha256=sha(output / 'train-rocm-admission.json'))
    roots, observations = corpus.prepare_structural_schedule()
    root_sample = min(roots, key=lambda p:(abs(p.legal_count - 32), -len(p.root_ids), p.key))
    records = {}
    with torch.inference_mode():
        for key in declaration['checkpoint_keys']:
            check_deadline(deadline)
            phase = 'anchor' if key in frozen['anchors'] else 'transition'
            entry = frozen['anchors' if phase == 'anchor' else 'transitions'][key]
            checkpoint = frozen_path.parent / entry['file']
            if checkpoint.name != entry['file'] or checkpoint.is_symlink() or not checkpoint.is_file() or checkpoint.stat().st_size != entry['bytes']:
                raise ValueError('checkpoint regular identity mismatch')
            raw = checkpoint.read_bytes()
            if hashlib.sha256(raw).hexdigest() != entry['sha256']:
                raise ValueError('checkpoint pin changed before loading')
            state = torch.load(io.BytesIO(raw), map_location='cpu', weights_only=True)
            anchor = SemanticAnchor(state['encoder'], base['config']['width'], base['config']['depth'], state['scale'])
            control = state.get('control')
            source = anchor if phase == 'anchor' else SemanticTransition(anchor, control)
            source.load_state_dict(state['state'], strict=True)
            source.eval().requires_grad_(False).to('cuda:0')
            models = dict(fp32_value=value_only_copy(source))
            setup = {}
            for name, builder in [('int8_fused', lambda:mixed_int8_copy(source, integer)),
                                  ('torch_fp16_affine', lambda:mixed_fp16_copy(source)),
                                  ('native_fp16_affine', lambda:mixed_fp16_copy(source, half))]:
                torch.cuda.synchronize(); start = time.monotonic()
                models[name], linears = builder()
                torch.cuda.synchronize()
                setup[name] = dict(synchronized_wall_ms=(time.monotonic() - start) * 1000, linears=list(linears))
            models['torch_autocast_fp16'] = value_only_copy(source)
            calls = {name:(lambda features, actions, name=name:forward(models[name], features, actions, control,
                           autocast=name == 'torch_autocast_fp16')) for name in VARIANTS}
            def batches():
                for plan in observations if phase == 'anchor' else roots:
                    batch = corpus.observations(plan) if phase == 'anchor' else corpus.roots(plan)
                    yield batch.features, None if phase == 'anchor' else batch.legal_actions
            row = dict(checkpoint_sha256=entry['sha256'], phase=phase, control=control, setup=setup,
                       fidelity=fidelity(calls, batches, deadline), timings={})
            if phase == 'anchor':
                first = corpus.observations(observations[0]).features
                for count in config['anchor_batches']:
                    features = first.repeat((count + len(first) - 1) // len(first), 1, 1, 1)[:count].contiguous()
                    alternate = first.roll(1, 0).repeat((count + len(first) - 1) // len(first), 1, 1, 1)[:count].contiguous()
                    row['timings'][str(count)] = routes(calls, features, None, config, deadline, alternate)
            else:
                batch = corpus.roots(root_sample)
                row['timing_root_plan'] = dict(key=root_sample.key, batch=len(root_sample.root_ids), K=root_sample.legal_count)
                row['timings']['complete_root'] = routes(calls, batch.features, batch.legal_actions, config, deadline)
            torch.cuda.synchronize(); check_deadline(deadline)
            records[key] = row
            write(output / f"{phase}-{key.replace('/', '-')}.json", row)
            print(json.dumps(dict(completed_key=key, elapsed_seconds=time.monotonic() - started)), flush=True)
            del models, calls, source, anchor, state, raw
            torch.cuda.synchronize(); check_deadline(deadline)
    torch.cuda.synchronize(); check_deadline(deadline)
    write(output / 'complete.json', dict(status='complete', declaration_sha256=pin, records=records,
          elapsed_seconds=time.monotonic() - started, training=False, calibration=False, promotion=False,
          split='train', teacher_quality_metrics=False, engine_latency_claim=False,
          timing_scope='events include host gaps; synchronized wall; graph_replay excludes input copies; resident_copy_graph includes device copies',
          graph_ownership='one frozen model/input/output set per fixed shape; no CPU feature reconstruction'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--declaration', required=True)
    parser.add_argument('--sha256', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    run(args.declaration, args.sha256, args.output)
