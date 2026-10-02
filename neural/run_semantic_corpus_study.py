"""Outer, fail-closed runner for the frozen semantic corpus experiment.

`prepare` emits a review proposal. `run` requires its externally pinned seal,
actual ROCm gfx1030, and the separately sealed phase-kernel declaration. There
is no CPU training, resume, retry, partial promotion, or heldout optimization.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time

import torch
from . import semantic_corpus_study as study
from .semantic_strategy import SemanticAnchor, SemanticTransition, ENCODERS
from .value_heads import material_cp

KIND = 'semantic_corpus_suite_v1'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read_json(path, pin=None, cap=128 << 20):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size > cap:
        raise ValueError('bounded regular source file required')
    raw = path.read_bytes()
    if len(raw) > cap or (pin is not None and hashlib.sha256(raw).hexdigest() != pin):
        raise ValueError('external file pin/cap mismatch')
    def unique(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError('duplicate JSON key')
            result[key] = value
        return result
    def finite(value):
        number = float(value)
        if not math.isfinite(number):
            raise ValueError('nonfinite JSON numeric')
        return number
    return json.loads(raw, object_pairs_hook=unique, parse_float=finite,
                      parse_constant=lambda value: (_ for _ in ()).throw(ValueError('nonfinite JSON')))


def publish_json(path, value):
    raw = (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + '\n').encode()
    with Path(path).open('xb') as stream:
        stream.write(raw); stream.flush(); os.fsync(stream.fileno())
    return hashlib.sha256(raw).hexdigest()


def checkpoint(path, record):
    """Persist only owned CPU tensors; no optimizer state is promoted."""
    with Path(path).open('xb') as stream:
        torch.save(record, stream); stream.flush(); os.fsync(stream.fileno())
    if Path(path).stat().st_size > 128 << 20:
        raise ValueError('checkpoint exceeds resident stage-gate bound')
    return sha(path)


def cpu_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def suite_proposal(main_root, packs_root, kernel_declaration_path, baseline_path, baseline_pin, attempt_output):
    """Read metadata and label-free simulator witnesses, never heldout metrics."""
    main_root, packs_root = Path(main_root).resolve(), Path(packs_root).resolve()
    kernel_path = Path(kernel_declaration_path).resolve()
    kernel = read_json(kernel_path, cap=1 << 20)
    baseline_path = Path(baseline_path).resolve()
    baseline = read_json(baseline_path, baseline_pin)
    if baseline.get('kind') != 'semantic_exact_baseline_witness_v1':
        raise ValueError('native baseline/draw witness required')
    splits = {}
    for split in ('train', 'validation', 'test'):
        directory = packs_root / split
        manifest = read_json(directory / 'manifest.json', cap=1 << 20)
        splits[split] = dict(path=str(directory), manifest_sha256=sha(directory / 'manifest.json'),
                             counts=manifest['counts'], packing_source_sha256=manifest['source_sha256'])
    for split in ('train', 'validation'):
        if kernel.get(split + '_manifest_sha256') != splits[split]['manifest_sha256']:
            raise ValueError('kernel and suite ' + split + ' corpus identities differ')
    dependencies = study.dependency_source_pins(main_root)
    dependencies[str(Path(__file__).resolve())] = sha(__file__)
    if kernel.get('dependencies') != {key: value for key, value in dependencies.items() if key != str(Path(__file__).resolve())}:
        raise ValueError('kernel proposal source closure differs from current sources')
    return dict(kind=KIND, status='unsealed', runner_source_sha256=sha(__file__),
                dependencies=dependencies, main_root=str(main_root), device='cuda:0', attempt_output=str(Path(attempt_output).resolve()),
                kernel_declaration=dict(path=str(kernel_path), sha256=sha(kernel_path)),
                baseline_witness=dict(path=str(baseline_path), sha256=baseline_pin), splits=splits,
                checkpoint_policy='all_8_anchors_and_32_transitions_frozen_before_validation;only_selected_pair_before_test',
                failure_policy='no_resume_no_retry_no_partial_promotion',
                test_policy='NONE_keeps_test_sealed;otherwise_one_selected_pair_pass',
                wall_scope=f"kernel_{kernel['config']['total_wall_seconds']}_seconds_includes_ingress_calibration_all_phases_validation_and_test",
                baseline_score_frame='native_current_mover_raw_TaperedCP_visible_uncalibrated_control',
                equal_compute_claim=False, cpu_training_substitution=False)


def verify_suite(path, pin):
    declaration = read_json(path, pin, 1 << 20)
    if declaration.get('kind') != KIND or declaration.get('status') != 'sealed_for_gpu_training':
        raise ValueError('root-sealed suite declaration required')
    if declaration.get('runner_source_sha256') != sha(__file__):
        raise ValueError('runner source mismatch')
    for source, digest in declaration.get('dependencies', {}).items():
        if not Path(source).is_absolute() or sha(source) != digest:
            raise ValueError('suite source closure changed')
    kernel = declaration['kernel_declaration']
    read_json(kernel['path'], kernel['sha256'], 1 << 20)
    expected = suite_proposal(declaration['main_root'], Path(declaration['splits']['train']['path']).parent,
                              kernel['path'], declaration['baseline_witness']['path'],
                              declaration['baseline_witness']['sha256'], declaration['attempt_output'])
    expected['status'] = 'sealed_for_gpu_training'
    if json.dumps(expected, sort_keys=True) != json.dumps(declaration, sort_keys=True):
        raise ValueError('suite differs from exact reviewed proposal')
    if (torch.version.hip is None or not torch.cuda.is_available() or os.environ.get('HSA_OVERRIDE_GFX_VERSION') or
            torch.cuda.get_device_properties(declaration['device']).gcnArchName.split(':')[0] != 'gfx1030'):
        raise ValueError('actual non-overridden gfx1030 ROCm required')
    return declaration


def validate_witness(witness, splits):
    """Require exact state-index coverage and draw-only terminal classification.

    Root review pins native source/binary provenance. This verifies row identity
    and simulator evidence; it cannot turn fabricated JSON into native proof.
    """
    if set(witness) != {'kind', 'splits', 'provenance'} or witness['kind'] != 'semantic_exact_baseline_witness_v1':
        raise ValueError('exact baseline witness schema required')
    provenance = witness['provenance']
    required = {'sources_sha256', 'cpu_binary_path', 'cpu_binary_sha256', 'terminal_oracle_path', 'terminal_oracle_sha256',
                'terminal_oracle_build_path', 'terminal_oracle_build_sha256'}
    if set(provenance) != required or not provenance['sources_sha256']:
        raise ValueError('exact CPU evaluator and terminal oracle provenance required')
    for path, pin in provenance['sources_sha256'].items():
        if not Path(path).is_absolute() or sha(path) != pin:
            raise ValueError('baseline native source pin mismatch')
    for prefix in ('cpu_binary', 'terminal_oracle', 'terminal_oracle_build'):
        if sha(provenance[prefix + '_path']) != provenance[prefix + '_sha256']:
            raise ValueError('baseline native artifact pin mismatch')
    if set(witness['splits']) != set(splits):
        raise ValueError('all three structural split witnesses required before training')
    for split, info in splits.items():
        row, counts = witness['splits'][split], info['counts']
        if set(row) != {'manifest_sha256', 'raw_tapered_state_cp', 'native_material_state_cp', 'native_hashes', 'observation_state_ids', 'terminal_states'}:
            raise ValueError('unknown baseline witness field')
        if row['manifest_sha256'] != info['manifest_sha256']:
            raise ValueError('witness corpus identity mismatch')
        values = row['raw_tapered_state_cp']
        if len(values) != counts['states'] or any(type(value) is not int or abs(value) > 100000 for value in values):
            raise ValueError('bounded exact native CP table required')
        material = row['native_material_state_cp']
        hashes = row['native_hashes']
        if len(material) != counts['states'] or any(type(x) is not int or abs(x) > 100000 for x in material):
            raise ValueError('native material support mismatch')
        if len(hashes) != counts['states'] or any(type(x) is not int or not 0 <= x < 1 << 64 for x in hashes):
            raise ValueError('native state hash support mismatch')
        ids = row['observation_state_ids']
        if len(ids) != counts['observations'] or any(type(value) is not int or not 0 <= value < counts['states'] for value in ids):
            raise ValueError('standalone state support mismatch')
        seen = set()
        for terminal in row['terminal_states']:
            if (set(terminal) != {'state_id', 'kind', 'in_check', 'legal_count'} or terminal['kind'] != 'draw' or
                    type(terminal['state_id']) is not int or not 0 <= terminal['state_id'] < counts['states'] or
                    type(terminal['in_check']) is not bool or type(terminal['legal_count']) is not int or
                    terminal['legal_count'] < 0 or (terminal['in_check'] and terminal['legal_count'] == 0) or
                    terminal['state_id'] in seen):
                raise ValueError('unique exact draw-only simulator witness required; mate rejected')
            seen.add(terminal['state_id'])
    return witness


def load_split(declaration, resident, split, output, gate=None):
    row = declaration['splits'][split]
    stage = 'train' if split == 'train' else split + '_eval'
    admission = dict(kind='semantic_resident_rocm_admission_v1', corpus_manifest_sha256=row['manifest_sha256'],
                     packing_source_sha256=row['packing_source_sha256'],
                     resident_source_sha256=sha(resident.__file__), stage=stage, device_arch='gfx1030')
    path = output / (split + '-rocm-admission.json')
    pin = publish_json(path, admission)
    return resident.load_resident(row['path'], expected_manifest_sha256=row['manifest_sha256'],
             expected_source_sha256=row['packing_source_sha256'], stage=stage, gate=gate,
             device=declaration['device'], rocm_admission_path=path, rocm_admission_sha256=pin)


def admit_terminal_support(corpus, witness):
    """Check all serialized terminal bits against the frozen draw witness."""
    expected = {row['state_id'] for row in witness['terminal_states']}
    actual = set(corpus._views['states.terminal'].nonzero().flatten().cpu().tolist())
    if actual != expected:
        raise ValueError('terminal witness does not cover exact resident terminal support')
    hashes = [value & ((1 << 64) - 1) for value in corpus._views['states.hash'].cpu().tolist()]
    if hashes != witness['native_hashes']:
        raise ValueError('native state hash identity mismatch')
    # Mechanical binary-board parity; no teacher label comparison here.
    for begin in range(0, len(hashes), 64):
        ids = torch.arange(begin, min(begin + 64, len(hashes)), device=corpus.device)
        actual_material = material_cp(corpus._features(ids)).cpu().tolist()
        expected_material = [max(-29000, min(29000, value)) for value in witness['native_material_state_cp'][begin:begin + 64]]
        if actual_material != expected_material:
            raise ValueError('native/feature material score disagreement')
    observation_states = corpus._views['observations.state'].cpu().tolist()
    if observation_states != witness['observation_state_ids']:
        raise ValueError('baseline standalone state index mismatch')


def metric_row(scalar_sum, scalar_count, regret_sum, agree_sum, root_count):
    if scalar_count < 1 or root_count < 1:
        raise ValueError('full scalar and root support required')
    values = dict(scalar_error_cp=scalar_sum / scalar_count,
                  interval_regret_cp=regret_sum / root_count, optimal_agreement=agree_sum / root_count)
    if any(not math.isfinite(value) for value in values.values()):
        raise ValueError('nonfinite evaluation')
    return values


def check_deadline(deadline, scope):
    if time.monotonic() > deadline:
        raise TimeoutError('study wall cap during ' + scope)


def evaluate(corpus, *, observer, root_predictor, deadline, model=None):
    """Sweep every standalone and whole legal root once; equal root weights."""
    roots, observations = corpus.prepare_structural_schedule()
    scalar_sum = regret_sum = agree_sum = drift_sum = refresh_sum = 0.
    scalar_count = root_count = edge_count = 0
    with torch.no_grad():
        for plan in observations:
            check_deadline(deadline, 'standalone evaluation')
            batch = corpus.observations(plan)
            prediction = observer(batch, plan)
            if prediction.shape != batch.midpoint.shape or not bool(torch.isfinite(prediction).all()):
                raise ValueError('finite prediction for each standalone observation required')
            distance = ((prediction - batch.midpoint).abs() - batch.halfwidth).clamp_min(0)
            scalar_sum += float(distance[batch.ordinary].double().sum().cpu())
            scalar_count += int(batch.ordinary.sum().cpu())
        for plan in roots:
            check_deadline(deadline, 'root evaluation')
            batch = corpus.roots(plan)
            result = root_predictor(batch)
            prediction = result['value_cp'] if isinstance(result, dict) else result
            if not bool(torch.isfinite(prediction).all()):
                raise ValueError('finite prediction for every root candidate required')
            metrics = study.complete_root_metrics(prediction, batch, terminal_cp=torch.zeros_like(prediction))
            n = metrics['complete_roots']
            regret_sum += float(metrics['interval_regret_cp'].double().cpu()) * n
            agree_sum += float(metrics['optimal_agreement'].double().cpu()) * n
            root_count += n; edge_count += metrics['legal_candidates']
            if model is not None:
                b, k = prediction.shape
                true = model.anchor(batch.targets.features.reshape(-1, 18, 8, 8))
                roles = true['latent'].reshape(b, k, 8, model.anchor.width)
                drift_sum += float((result['latent'] - roles).square().double().mean().cpu()) * n
                gap = ((true['value_cp'].reshape(b, k) - batch.targets.midpoint).abs() - batch.targets.halfwidth).clamp_min(0)
                ordinary = batch.targets.ordinary
                refresh_sum += float(((gap * ordinary).sum(1) / ordinary.sum(1).clamp_min(1)).double().sum().cpu())
    # CPU reductions drain the relevant GPU work. The last batch has no next
    # loop guard, so it needs an explicit completion check before promotion.
    check_deadline(deadline, 'completed evaluation')
    return dict(metrics=metric_row(scalar_sum, scalar_count, regret_sum, agree_sum, root_count),
                support=dict(ordinary_observations=scalar_count, roots=root_count, legal_candidates=edge_count),
                diagnostics=None if model is None else dict(mean_child_role_latent_mse=drift_sum / root_count,
                     mean_parentPV_interval_vs_frozen_child_value_gap_cp=refresh_sum / root_count))


def baseline_metrics(corpus, witness, scale, deadline):
    scores = torch.tensor(witness['raw_tapered_state_cp'], dtype=torch.float32, device=corpus.device)
    _, plans = corpus.prepare_structural_schedule()
    observed = torch.tensor(witness['observation_state_ids'], dtype=torch.int64, device=corpus.device)
    indices = {plan.observation_ids: torch.tensor(plan.observation_ids, dtype=torch.int64, device=corpus.device) for plan in plans}
    rows = {}
    for name, multiplier in (('raw_material100', 1.), ('scaled_material', scale)):
        rows[name] = evaluate(corpus, observer=lambda batch, plan: material_cp(batch.features) * multiplier,
                 root_predictor=lambda batch: material_cp(batch.targets.features.flatten(0, 1)).reshape(batch.legal_actions.shape) * multiplier,
                 deadline=deadline)
    rows['raw_tapered'] = evaluate(corpus,
         observer=lambda batch, plan: scores.index_select(0, observed.index_select(0, indices[plan.observation_ids])),
         root_predictor=lambda batch: scores[batch.targets.child_state_ids], deadline=deadline)
    return rows


def restore_transition(record, config, device):
    anchor = SemanticAnchor(record['encoder'], config.width, config.depth, record['scale']).to(device)
    anchor.requires_grad_(False).eval()
    mode = 'conditioned' if record['control'] == 'shuffled_conditioned' else record['control']
    model = SemanticTransition(anchor, mode).to(device)
    model.load_state_dict(record['state']); model.eval(); model.anchor.eval()
    return model


def frozen_gate(resident, output, split, checkpoint_path, checkpoint_pin, row, selection_source=None):
    record = dict(kind='semantic_frozen_checkpoint_v1' if split == 'validation' else 'semantic_selected_checkpoint_v1',
                  corpus_manifest_sha256=row['manifest_sha256'], packing_source_sha256=row['packing_source_sha256'],
                  checkpoint_sha256=checkpoint_pin)
    if selection_source is not None:
        record['selection_source_sha256'] = selection_source[1]
    path = output / (split + '-checkpoint-gate.json')
    pin = publish_json(path, record)
    return resident.EvaluationGate(path, pin, checkpoint_path, checkpoint_pin,
              None if selection_source is None else selection_source[0],
              None if selection_source is None else selection_source[1])


def run(declaration_path, declaration_pin, output):
    declaration = verify_suite(declaration_path, declaration_pin)
    output = Path(output).resolve()
    if str(output) != declaration['attempt_output']:
        raise ValueError('attempt output differs from sealed unique run identity')
    output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    # The lock is never removed: failure consumes this output/attempt identity.
    publish_json(output / 'attempt.json', dict(declaration_sha256=declaration_pin, status='started',
                 no_retry=True, runtime=dict(torch=str(torch.__version__), hip=torch.version.hip,
                 threads=torch.get_num_threads(), interop_threads=torch.get_num_interop_threads(),
                 deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                 cudnn_enabled=torch.backends.cudnn.enabled, cudnn_deterministic=torch.backends.cudnn.deterministic,
                 cudnn_benchmark=torch.backends.cudnn.benchmark,
                 cuda_matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
                 cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)))
    witness = validate_witness(read_json(**dict(path=declaration['baseline_witness']['path'], pin=declaration['baseline_witness']['sha256'])), declaration['splits'])
    main_root = Path(declaration['main_root'])
    relative_pins = {name: declaration['dependencies'][str(main_root / name)] for name in (
       'experiments/semantic_corpus/__init__.py', 'experiments/semantic_corpus/packing.py', 'experiments/semantic_corpus/resident.py')}
    resident = study.admit_resident_dependency(main_root, relative_pins)
    kernel = declaration['kernel_declaration']
    train = load_split(declaration, resident, 'train', output)
    admit_terminal_support(train, witness['splits']['train'])
    config = study.require_gpu_learning_authority(kernel['path'], kernel['sha256'], train)
    deadline = started + config.total_wall_seconds
    check_deadline(deadline, 'TRAIN admission')
    scale = study.fit_train_material(train, declaration_path=kernel['path'], declaration_sha256=kernel['sha256'])
    check_deadline(deadline, 'TRAIN calibration')
    publish_json(output / 'train-calibration.json', dict(scale=scale, split='train', objective='squared_reporting_interval_distance', kernel_declaration_sha256=kernel['sha256']))
    states, anchors_frozen, phases = {}, {}, []
    for seed in config.seeds:
        anchors = study.shared_anchors(seed, scale, config, train.device)
        for encoder in ENCODERS:
            anchor = anchors.pop(encoder)
            report = study.train_phase(anchor, train, phase='anchor', declaration_path=kernel['path'],
                                       declaration_sha256=kernel['sha256'], study_started=started)
            anchor.requires_grad_(False).eval()
            anchor_record = dict(encoder=encoder, seed=seed, scale=scale, state=cpu_state(anchor), report=report)
            key = f'{encoder}/{seed}'
            anchors_frozen[key] = anchor_record
            checkpoint(output / f'anchor-{encoder}-{seed}.pt', anchor_record)
            phases.append(dict(key=key, **report))
            # Exactly the same seeded full transition state is shared by controls.
            controls = study.transition_controls(anchor, seed)
            for control in study.CONTROLS:
                model = controls.pop(control).to(train.device)
                report = study.train_phase(model, train, phase='transition', control=control,
                         declaration_path=kernel['path'], declaration_sha256=kernel['sha256'], study_started=started)
                arm = f'{encoder}/{control}'
                record = dict(encoder=encoder, control=control, seed=seed, scale=scale,
                              state=cpu_state(model), report=report)
                states[f'{arm}/{seed}'] = record
                checkpoint(output / f'transition-{encoder}-{control}-{seed}.pt', record)
                phases.append(dict(key=f'{arm}/{seed}', **report)); del model
            del anchor
    if len(states) != 32 or len(phases) != 40:
        raise RuntimeError('incomplete study cannot enter validation')
    bundle_path = output / 'all-frozen-checkpoints.pt'
    bundle_pin = checkpoint(bundle_path, dict(kind='semantic_all_frozen_checkpoints_v1', declaration_sha256=declaration_pin,
                                           scale=scale, anchors=anchors_frozen, transitions=states, phase_reports=phases))
    publish_json(output / 'training-complete.json', dict(status='complete', anchors=8, transitions=32,
                   bundle_sha256=bundle_pin, phase_reports=phases))
    if sha(bundle_path) != bundle_pin:
        raise ValueError('frozen bundle changed before validation')
    states = torch.load(bundle_path, map_location='cpu', weights_only=True)['transitions']
    del train, anchors_frozen
    check_deadline(deadline, 'checkpoint freeze before validation admission')
    gate = frozen_gate(resident, output, 'validation', bundle_path, bundle_pin, declaration['splits']['validation'])
    validation = load_split(declaration, resident, 'validation', output, gate)
    admit_terminal_support(validation, witness['splits']['validation'])
    baseline_rows = baseline_metrics(validation, witness['splits']['validation'], scale, deadline)
    candidates, diagnostics = {}, {}
    for key, record in states.items():
        model = restore_transition(record, config, validation.device)
        result = evaluate(validation, observer=lambda batch, plan: model.anchor(batch.features)['value_cp'],
                  root_predictor=lambda batch: study.branch(model, record['control'], batch.features, batch.legal_actions),
                  deadline=deadline, model=model)
        arm = f"{record['encoder']}/{record['control']}"
        candidates.setdefault(arm, {})[record['seed']] = result['metrics']
        diagnostics[key] = result; del model
    baselines = {name: row['metrics'] for name, row in baseline_rows.items()}
    check_deadline(deadline, 'validation completion before selection')
    selected = study.select_complete_study_or_none(candidates, baselines, paired_seeds=config.seeds)
    validation_pin = publish_json(output / 'validation-results.json', dict(candidates=candidates, baselines=baseline_rows,
                      diagnostics=diagnostics, selected=selected, checkpoint_bundle_sha256=bundle_pin))
    selection = dict(kind='semantic_suite_selection_v1', selected=selected, declaration_sha256=declaration_pin,
                     validation_results_sha256=validation_pin, test_opened=False)
    if selected == 'NONE':
        publish_json(output / 'selection.json', selection)
        check_deadline(deadline, 'NONE completion')
        publish_json(output / 'complete.json', dict(status='complete', selection='NONE', test_opened=False,
                     elapsed_seconds=time.monotonic() - started, rating_claim=None))
        return selection
    selected_states = {str(seed): states[f'{selected}/{seed}'] for seed in config.seeds}
    selected_path = output / 'selected-paired-checkpoints.pt'
    selected_pin = checkpoint(selected_path, dict(kind='semantic_selected_paired_checkpoints_v1',
                         selected=selected, transitions=selected_states, declaration_sha256=declaration_pin))
    selection.update(selected_checkpoint_sha256=selected_pin)
    publish_json(output / 'selection.json', selection)
    source_path = output / 'test-selection-source.json'
    source_pin = publish_json(source_path, dict(kind='semantic_selection_source_v1', selected_checkpoint_sha256=selected_pin,
                              selection_splits=['train', 'validation'], test_policy='sealed_out'))
    del validation
    gate = frozen_gate(resident, output, 'test', selected_path, selected_pin, declaration['splits']['test'], (source_path, source_pin))
    check_deadline(deadline, 'selected checkpoint freeze before TEST admission')
    # Durable consumption marker BEFORE opening TEST: no failed-pass retry.
    publish_json(output / 'test-pass-consumed.json', dict(selected=selected, checkpoint_sha256=selected_pin, single_pass=True))
    test = load_split(declaration, resident, 'test', output, gate)
    admit_terminal_support(test, witness['splits']['test'])
    if sha(selected_path) != selected_pin:
        raise ValueError('selected bundle changed before test')
    selected_states = torch.load(selected_path, map_location='cpu', weights_only=True)['transitions']
    test_rows = {}
    for seed, record in selected_states.items():
        model = restore_transition(record, config, test.device)
        test_rows[seed] = evaluate(test, observer=lambda batch, plan: model.anchor(batch.features)['value_cp'],
                    root_predictor=lambda batch: study.branch(model, record['control'], batch.features, batch.legal_actions),
                    deadline=deadline, model=model)
        del model
    publish_json(output / 'test-results.json', dict(selected=selected, checkpoint_sha256=selected_pin,
                 paired_seed_results=test_rows, baselines=baseline_metrics(test, witness['splits']['test'], scale, deadline)))
    check_deadline(deadline, 'TEST completion')
    publish_json(output / 'complete.json', dict(status='complete', selection=selected, test_opened=True,
                 elapsed_seconds=time.monotonic() - started, rating_claim=None))
    return selection


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    for name in ('main-root', 'packs-root', 'kernel-declaration', 'baseline-witness', 'baseline-sha256', 'attempt-output', 'output'):
        prepare.add_argument('--' + name, required=True)
    execute = commands.add_parser('run')
    for name in ('declaration', 'declaration-sha256', 'output'):
        execute.add_argument('--' + name, required=True)
    args = parser.parse_args()
    if args.command == 'prepare':
        record = suite_proposal(args.main_root, args.packs_root, args.kernel_declaration, args.baseline_witness, args.baseline_sha256, args.attempt_output)
        print(publish_json(args.output, record))
    else:
        print(json.dumps(run(args.declaration, args.declaration_sha256, args.output), sort_keys=True))


if __name__ == '__main__':
    main()
