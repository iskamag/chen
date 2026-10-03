"""Bounded, paired semantic experiments over externally pinned resident data.

This module supplies mechanics and an UNSEALED proposal. It never admits
heldout data and has no command which trains from a proposal. Actual TRAIN
updates require a separately hashed root-reviewed GPU learning declaration.
One-step transitions refresh exact pieces only; no imaginary trajectories are
constructed from sparsely sampled observations.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import importlib
import io
import json
import math
from pathlib import Path
import sys
import time

import torch
from torch.nn import functional as F

from .semantic_strategy import (ENCODERS, SemanticAnchor, SemanticTransition,
                               anchor_losses, parameter_counts)
from .strategic_transport import derive_effects, successor_pieces
from .value_heads import material_cp, SCALE_CP

CONTROLS = ('transport', 'conditioned', 'action_blind', 'shuffled_conditioned')
COEFFICIENTS = dict(value=1., board=.05, maps=.05, rights=.02, ep=.02,
                    clock=.02, latent=.1)


@dataclass(frozen=True)
class Configuration:
    seeds: tuple[int, ...] = (20261021, 20261022)
    width: int = 32
    depth: int = 1
    anchor_updates: int = 384
    transition_updates: int = 192
    learning_rate: float = 0.0003
    weight_decay: float = 0.
    clip_grad_norm: float = 1.
    material_lower: float = .05
    material_upper: float = 5.
    material_bisections: int = 64
    phase_wall_seconds: int = 900
    total_wall_seconds: int = 7200

    def __post_init__(self):
        if (type(self.seeds) is not tuple or len(self.seeds) != 2 or
                len(set(self.seeds)) != 2 or any(type(s) is not int or s < 0 for s in self.seeds)):
            raise ValueError('exactly two distinct paired integer seeds required')
        for name, low, high in (('width', 8, 128), ('depth', 1, 8),
                               ('anchor_updates', 1, 4096), ('transition_updates', 1, 4096),
                               ('material_bisections', 1, 128), ('phase_wall_seconds', 1, 3600),
                               ('total_wall_seconds', 1, 28800)):
            value = getattr(self, name)
            if type(value) is not int or not low <= value <= high:
                raise ValueError('invalid bounded ' + name)
        for name in ('learning_rate', 'weight_decay', 'clip_grad_norm', 'material_lower', 'material_upper'):
            value = getattr(self, name)
            if type(value) not in (int, float) or not math.isfinite(value):
                raise ValueError('finite numeric configuration required')
        if not (0 < self.learning_rate <= .01 and 0 <= self.weight_decay <= .1 and
                0 < self.clip_grad_norm <= 10 and 0 < self.material_lower < 1 < self.material_upper <= 10):
            raise ValueError('optimizer/positive-scale bounds')


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dependency_source_pins(main_root):
    """Explicit absolute workspace bindings; call after source review/freeze."""
    neural_root = Path(__file__).resolve().parent
    paths = [neural_root / name for name in (
        'semantic_corpus_study.py', 'semantic_strategy.py', 'strategic_transport.py',
        'value_heads.py', 'score_frames.py', 'models.py', 'features.py', 'dynamics.py', 'diamonds.py')]
    shared_root = Path(main_root).resolve() / 'experiments/semantic_corpus'
    paths.extend(shared_root / name for name in ('__init__.py', 'packing.py', 'resident.py'))
    return {str(path): _sha(path) for path in paths}


def proposal(*, train_manifest_sha256, validation_manifest_sha256, dependencies,
             config=Configuration()):
    """Return a concrete review artifact, never a training authorization."""
    return dict(kind='semantic_corpus_learning_v1', status='unsealed',
                config=asdict(config), encoders=list(ENCODERS), controls=list(CONTROLS),
                train_manifest_sha256=train_manifest_sha256,
                validation_manifest_sha256=validation_manifest_sha256,
                dependencies=dict(dependencies), target_gpu='gfx1030',
                optimizer=dict(name='AdamW', betas=[.9, .999], eps=1e-8,
                               learning_rate=config.learning_rate, weight_decay=config.weight_decay),
                loss_coefficients=COEFFICIENTS.copy(),
                calibration=dict(split='train', query_kind='standalone_only',
                                 objective='mean_squared_distance_outside_reporting_interval',
                                 shared_across_all_models=True, tie_policy='prefer_one_if_one_is_optimal_else_leftmost_minimizer',
                                 bounds=[config.material_lower, config.material_upper],
                                 bisections=config.material_bisections),
                ordering=dict(schedule='resident_validated_seed20261008_exactK_roots4_observations32_keep_tails',
                              update_path='cyclic_repetition_of_complete_one_epoch_plans_from_plan0',
                              anchor='standalone_observations_only_semantics_plus_value',
                              transition='complete_legal_roots_only_equal_root_weight',
                              tail_weight='batch_mean_times_actual_rows_over_declared_batch_size'),
                initialization=dict(common='copy_matching_decoder_and_role_pool_parameters_from_square_seed_template',
                                    transitions='identical_full_transition_state_for_each_frozen_anchor',
                                    anchor='freeze_and_hash_before_all_transition_updates'),
                shuffle=dict(mode='cyclic_left_rotation_one_within_each_complete_root',
                             inputs='action_event_and_action_derived_child_pieces_maps',
                             fixed='true_root_anchor_and_true_action_material_prior',
                             singleton='unchanged'),
                rollout=dict(plies=1, simulator='exact_parent_actions_and_child_piece_targets',
                             latent_target='detached_frozen_anchor_on_exact_child',
                             longer_rollout='requires_new_contiguous_sequence_admission'),
                metrics=dict(scalar='mean_distance_outside_interval_cp_standalone',
                             root='mean_interval_regret_cp_and_raw_parentCP_optimal_agreement',
                             tie='first_canonical_action', drift='mean_squared_frozen_child_role_latent',
                             parentPV_context='absolute_parent_query_CP_supervision_not_standalone_child_CP',
                             refresh_disagreement='record_parentPV_interval_vs_frozen_anchor_exact_child_value_gap_not_new_Stockfish_query',
                             terminal='pinned_simulator_draw_witness_overrides_prediction_and_decision_target_to_zero_never_drop_actions',
                             mate='first_study_draw_only_admission_rejects_mates_before_training_requires_separate_lexicographic_policy'),
                selection=dict(splits=['train', 'validation'], test='sealed_out',
                               baselines=['raw_material100', 'scaled_material', 'raw_tapered'],
                               paired_seed_gate='each_seed_scalar_error_strictly_below_all_baselines_and_root_interval_regret_strictly_below_all_baselines_and_agreement_at_least_all_baselines',
                               decision='seedmean_regret_then_seedmean_error_then_encoder_control_names',
                               failed_gate='NONE'),
                equal_compute_claim=False, cpu_training_substitution=False)


def transformer_followup_proposal():
    """Separate experiment, requiring new classes and a separate declaration.

    Existing relation and plan are single-head attention controls, not proof of
    a modern multihead transformer comparison. Do not silently add these arms
    to an already sealed experiment.
    """
    return dict(status='unsealed_followup_not_implemented',
                data='same_pinned_complete_root_corpus_no_new_heldout_peeks',
                paired_seeds=list(Configuration().seeds),
                shared_protocol='same_eight_role_decoder_value_intervals_anchor_freeze_and_transition_controls',
                arms=['dense_square_multihead_preLN_transformer',
                      'typed_attack_support_bias_multihead_transformer',
                      'occupied_piece_object_transformer_with_presence_mask',
                      'eight_latent_query_bottleneck_with_square_cross_attention'],
                primary_ablation='same_attention_block_change_token_ownership_and_exact_rule_bias',
                initialization='match_shared_parameters_where_tensor_shapes_and_meanings_match',
                compute='record_per_phase_wall_GPU_event_times_peak_memory_stored_active_parameters_and_batchK',
                equal_compute_claim=False,
                architecture_hypothesis='typed_piece_relations_or_latent_query_routing_reduce_empty_square_attention_cost_without_losing_root_choice_quality',
                mechanical_gate='causal_inputs_complete_root_support_exact_blind_ties_simulator_terminal_parity',
                quality_gate='separately_sealed_joint_scalar_interval_error_and_complete_root_regret_against_all_visible_baselines',
                test='selected_configuration_source_checkpoint_frozen_then_one_test_pass_or_NONE')


def admit_resident_dependency(root, pins):
    """Normal-import the shared implementation only after exact file pin checks.

    The caller's reviewed declaration owns both pins. This is a source binding,
    not an adversarial Python isolation boundary.
    """
    root = Path(root).resolve()
    names = ('experiments/semantic_corpus/__init__.py',
             'experiments/semantic_corpus/packing.py', 'experiments/semantic_corpus/resident.py')
    if set(pins) != set(names) or any(_sha(root / name) != pins[name] for name in names):
        raise ValueError('shared resident dependency source mismatch')
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    module = importlib.import_module('experiments.semantic_corpus.resident')
    if Path(module.__file__).resolve() != root / names[-1]:
        raise ValueError('normal import resolved another resident implementation')
    return module


def fit_positive_material(material, midpoint, halfwidth, mask, config=Configuration()):
    """Convex interval objective on already admitted TRAIN standalone vectors.

    All 64 search iterations stay on the device. This low-level mechanical
    kernel does not itself establish which split its tensors came from.
    """
    if (any(not isinstance(x, torch.Tensor) for x in (material, midpoint, halfwidth, mask)) or
            any(x.shape != material.shape or x.device != material.device for x in (midpoint, halfwidth, mask)) or
            material.ndim != 1 or material.numel() == 0 or mask.dtype != torch.bool):
        raise ValueError('same-device nonempty score vectors and boolean mask required')
    x, y, w = (v.double() for v in (material, midpoint, halfwidth))
    weight = mask.double()
    def derivative(scale):
        delta = scale * x - y
        residual = (delta.abs() - w).clamp_min(0) * delta.sign()
        return (2 * x * residual * weight).sum()
    low, high = x.new_tensor(config.material_lower), x.new_tensor(config.material_upper)
    for _ in range(config.material_bisections):
        middle = (low + high) * .5
        nonnegative = derivative(middle) >= 0
        low, high = torch.where(nonnegative, low, middle), torch.where(nonnegative, middle, high)
    one = x.new_tensor(1.)
    return torch.where(derivative(one) == 0, one, (low + high) * .5)


def shared_anchors(seed, scale, config=Configuration(), device='cpu'):
    """Pair common head/role parameters; encoder-specific structure is explicit."""
    result = {}
    devices = [torch.device(device).index or 0] if torch.device(device).type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        for encoder in ENCODERS:
            torch.manual_seed(seed)
            result[encoder] = SemanticAnchor(encoder, config.width, config.depth, scale).to(device)
    shared = result['square'].state_dict()
    prefixes = ('queries', 'key.', 'value.', 'norm.', 'board.', 'maps.', 'rights.', 'ep.', 'clock.', 'cp.')
    for encoder, anchor in result.items():
        if encoder == 'square':
            continue
        state = anchor.state_dict()
        for name in state:
            if name.startswith(prefixes):
                state[name] = shared[name].clone()
        anchor.load_state_dict(state)
    return result


def fingerprint(model):
    """Checkpoint boundary only: copy state to CPU and hash deterministic bytes."""
    stream = io.BytesIO()
    torch.save({name: value.detach().cpu().clone() for name, value in model.state_dict().items()}, stream)
    return hashlib.sha256(stream.getvalue()).hexdigest()


def transition_controls(anchor, seed):
    anchor.requires_grad_(False).eval()
    device = anchor.queries.device
    devices = [device.index or 0] if device.type == 'cuda' else []
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(seed)
        template = SemanticTransition(anchor, 'transport')
        result = {}
        for name in CONTROLS:
            mode = 'conditioned' if name == 'shuffled_conditioned' else name
            model = SemanticTransition(anchor, mode)
            model.load_state_dict(template.state_dict())
            result[name] = model
    return result


def branch(model, control, features, actions):
    """No target features/CP/history enter a root forward pass."""
    if control not in CONTROLS or model.mode != ('conditioned' if control == 'shuffled_conditioned' else control):
        raise ValueError('control/model routing mismatch')
    if control != 'shuffled_conditioned':
        return model.branch(features, actions)
    latent = model.encode(features)
    outputs = model.branch(features, actions.roll(-1, dims=1), latent)
    b, k = actions.shape
    roots = features[:, None].expand(-1, k, -1, -1, -1).reshape(-1, 18, 8, 8)
    pieces = successor_pieces(roots, derive_effects(roots, actions.flatten()))
    prior = (model.anchor.diagram.material(pieces) * model.anchor.material_scale).reshape(b, k)
    # Only action-dependent learned inputs are shuffled. Readout scalar prior
    # belongs to the actual candidate being scored, with the correct mover sign.
    outputs.update(pieces=pieces.reshape(b, k, 12, 8, 8), material_cp=prior,
                   value_cp=outputs['residual_cp'] + prior,
                   utility=((outputs['residual_cp'] + prior) / SCALE_CP).tanh())
    return outputs


def transition_loss(model, outputs, batch):
    """Equal root weight, masked scalar intervals, detached true-child targets."""
    b, k = batch.legal_actions.shape
    target_features = batch.targets.features.reshape(-1, 18, 8, 8).detach()
    flattened = {name: value.reshape(b * k, *value.shape[2:]) for name, value in outputs.items()}
    # Per-root reductions keep a 40-action position from outweighing a 4-action
    # position. K is constant within the reader's complete, unpadded batch.
    with torch.no_grad():
        true_latent = model.anchor.encode(target_features).detach().reshape(b, k, 8, model.anchor.width)
        target = model.anchor.semantic_targets(target_features)
    ordinary = batch.targets.ordinary.detach().to(outputs['value_cp'].dtype)
    distance = ((outputs['value_cp'] - batch.targets.midpoint.detach()).abs() -
                batch.targets.halfwidth.detach()).clamp_min(0) / SCALE_CP
    per_edge = F.smooth_l1_loss(distance, torch.zeros_like(distance), reduction='none')
    value = ((per_edge * ordinary).sum(1) / ordinary.sum(1).clamp_min(1)).mean()
    losses = dict(value=value, latent=F.mse_loss(outputs['latent'], true_latent),
                  board=F.cross_entropy(flattened['board_logits'], target['board']),
                  maps=F.binary_cross_entropy_with_logits(flattened['map_logits'], target['maps']),
                  rights=F.binary_cross_entropy_with_logits(flattened['castling_logits'], target['castling']),
                  ep=F.cross_entropy(flattened['ep_logits'], target['ep']),
                  clock=F.mse_loss(flattened['halfmove'], target['halfmove']))
    return sum(COEFFICIENTS[name] * value for name, value in losses.items()), losses


def scalar_interval_error(prediction, midpoint, halfwidth):
    return ((prediction - midpoint).abs() - halfwidth).clamp_min(0).mean()


def complete_root_metrics(prediction, batch, *, terminal_cp=None):
    """Lower child-mover CP wins; exact canonical ordering breaks exact ties.

    Every candidate stays present. This first study admits certified DRAW
    terminals only. Caller must pin a simulator classification witness; zero
    tensors alone do not prove a terminal is a draw. Mate terminals require a
    separate lexicographic policy, never conversion into SF ordinary CP.
    Labels remain finite-search parent-query intervals, not minimax truth.
    """
    if prediction.shape != batch.legal_actions.shape:
        raise ValueError('one prediction for every legal candidate required')
    terminal = batch.targets.terminal
    if terminal_cp is None:
        # This check is evaluation-boundary logic, not part of the hot forward.
        if bool(terminal.any()):
            raise ValueError('exact simulator terminal scores required for complete root metrics')
    else:
        if terminal_cp.shape != prediction.shape or terminal_cp.device != prediction.device:
            raise ValueError('terminal scores must preserve full candidate support')
        if bool((terminal_cp[terminal] != 0).any()):
            raise ValueError('first study admits certified draw terminals only')
        prediction = torch.where(terminal, terminal_cp, prediction)
    index = prediction.argmin(1)
    center, width = batch.targets.midpoint, batch.targets.halfwidth
    # A simulator draw is exact utility0, separate from parent-PV ordinary CP
    # intervals. Ordinary scalar supervision still uses its false mask.
    center = torch.where(terminal, torch.zeros_like(center), center)
    width = torch.where(terminal, torch.zeros_like(width), width)
    lower, upper = center - width, center + width
    selected_lower = lower.gather(1, index[:, None]).squeeze(1)
    regret = (selected_lower - upper.amin(1)).clamp_min(0)
    raw = batch.targets.reported_cp
    raw = torch.where(terminal, torch.zeros_like(raw), raw)
    agreement = raw.gather(1, index[:, None]).squeeze(1) == raw.amax(1)
    return dict(interval_regret_cp=regret.mean(), optimal_agreement=agreement.float().mean(),
                selected_canonical_action=batch.legal_actions.gather(1, index[:, None]).squeeze(1),
                complete_roots=prediction.shape[0], legal_candidates=prediction.numel())


def select_or_none(candidates, baselines, *, paired_seeds=Configuration().seeds):
    """Pure selection boundary with strict paired-seed joint gates and NONE."""
    required = {'raw_material100', 'scaled_material', 'raw_tapered'}
    if set(baselines) != required:
        raise ValueError('all predeclared visible baselines required')
    def valid(row):
        return (set(row) == {'scalar_error_cp', 'interval_regret_cp', 'optimal_agreement'} and
                all(type(x) in (int, float) and math.isfinite(x) for x in row.values()) and
                row['scalar_error_cp'] >= 0 and row['interval_regret_cp'] >= 0 and 0 <= row['optimal_agreement'] <= 1)
    if not all(valid(row) for row in baselines.values()):
        raise ValueError('finite baseline metric schema required')
    eligible = []
    for name, rows in candidates.items():
        if set(rows) != set(paired_seeds) or not all(valid(row) for row in rows.values()):
            raise ValueError('complete finite paired seed metrics required')
        if all(row['scalar_error_cp'] < base['scalar_error_cp'] and
               row['interval_regret_cp'] < base['interval_regret_cp'] and
               row['optimal_agreement'] >= base['optimal_agreement']
               for row in rows.values() for base in baselines.values()):
            eligible.append((sum(r['interval_regret_cp'] for r in rows.values()) / len(rows),
                             sum(r['scalar_error_cp'] for r in rows.values()) / len(rows), name))
    return min(eligible)[2] if eligible else 'NONE'


def select_complete_study_or_none(candidates, baselines, *, paired_seeds=Configuration().seeds):
    """Actual-study boundary: all sixteen representation/control arms required."""
    expected = {f'{encoder}/{control}' for encoder in ENCODERS for control in CONTROLS}
    if set(candidates) != expected:
        raise ValueError('all four encoders times four declared controls required before selection')
    return select_or_none(candidates, baselines, paired_seeds=paired_seeds)


def require_gpu_learning_authority(declaration_path, declaration_sha256, corpus):
    """Fail closed before constructing an optimizer or fitting real labels."""
    raw = Path(declaration_path).read_bytes()
    if len(raw) > 1 << 20 or hashlib.sha256(raw).hexdigest() != declaration_sha256:
        raise ValueError('externally pinned learning declaration required')
    def unique(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError('duplicate declaration key')
            result[key] = value
        return result
    declaration = json.loads(raw, object_pairs_hook=unique)
    dependencies = declaration.get('dependencies')
    if (type(dependencies) is not dict or not dependencies or
            any(not Path(name).is_absolute() or not Path(name).is_file() or Path(name).is_symlink() or
                Path(name).stat().st_size > 64 << 20 or _sha(name) != pin
                for name, pin in dependencies.items())):
        raise ValueError('all explicit absolute source dependencies must match frozen pins')
    resident_module = sys.modules.get(type(corpus).__module__)
    if resident_module is None or not hasattr(resident_module, 'ResidentCorpus') or not isinstance(corpus, resident_module.ResidentCorpus):
        raise ValueError('shared admitted ResidentCorpus instance required')
    main_root = Path(resident_module.__file__).resolve().parents[2]
    required_dependencies = dependency_source_pins(main_root)
    if any(dependencies.get(path) != pin for path, pin in required_dependencies.items()):
        raise ValueError('complete neural/shared-resident source closure required')
    if (declaration.get('kind') != 'semantic_corpus_learning_v1' or
            declaration.get('status') != 'sealed_for_gpu_training' or
            declaration.get('study_source_sha256') != _sha(__file__) or
            declaration.get('target_gpu') != 'gfx1030' or corpus.stage != 'train' or
            corpus.manifest_sha256 != declaration.get('train_manifest_sha256') or
            corpus.device.type != 'cuda' or torch.version.hip is None):
        raise ValueError('sealed matching TRAIN gfx1030 learning authority required')
    if torch.cuda.get_device_properties(corpus.device).gcnArchName.split(':')[0] != 'gfx1030':
        raise ValueError('actual gfx1030 required')
    config_record = declaration['config'].copy()
    config_record['seeds'] = tuple(config_record['seeds'])
    config = Configuration(**config_record)
    expected = proposal(train_manifest_sha256=declaration['train_manifest_sha256'],
                        validation_manifest_sha256=declaration['validation_manifest_sha256'],
                        dependencies=declaration['dependencies'], config=config)
    expected.update(status='sealed_for_gpu_training', study_source_sha256=_sha(__file__))
    if json.dumps(declaration, sort_keys=True) != json.dumps(expected, sort_keys=True):
        raise ValueError('sealed declaration differs from exact reviewed protocol')
    return config


def fit_train_material(corpus, *, declaration_path, declaration_sha256):
    """One scale, fitted only on the authorized resident TRAIN observations."""
    config = require_gpu_learning_authority(declaration_path, declaration_sha256, corpus)
    _, plans = corpus.prepare_structural_schedule()
    if not plans:
        raise ValueError('nonempty TRAIN standalone schedule required')
    vectors = [[], [], [], []]
    for plan in plans:
        batch = corpus.observations(plan)
        for target, value in zip(vectors, (material_cp(batch.features), batch.midpoint,
                                          batch.halfwidth, batch.ordinary)):
            target.append(value.detach())
    material, midpoint, halfwidth, mask = (torch.cat(v) for v in vectors)
    if not bool(mask.any()):
        raise ValueError('at least one admitted TRAIN ordinary scalar label required')
    scale = fit_positive_material(material, midpoint, halfwidth, mask, config)
    # Explicit phase boundary host transfer; every model gets this same number.
    return float(scale.cpu())


def profile_branch_on_train(model, control, corpus, plan, *, expected_train_manifest_sha256,
                            warmups=4, repeats=8):
    """Untrained mechanical forward timings, without losses or target metrics.

    A separately ROCm-admitted resident TRAIN slab is required. Event timing
    excludes corpus ingress and the one batch gather/decode; end-to-end search
    and learning time remain different measurements. CPU fallback is rejected.
    """
    if (type(warmups) is not int or not 0 <= warmups <= 32 or
            type(repeats) is not int or not 1 <= repeats <= 128 or corpus.stage != 'train' or
            corpus.manifest_sha256 != expected_train_manifest_sha256 or corpus.device.type != 'cuda' or
            torch.version.hip is None):
        raise ValueError('bounded profile requires admitted pinned TRAIN on ROCm')
    if torch.cuda.get_device_properties(corpus.device).gcnArchName.split(':')[0] != 'gfx1030':
        raise ValueError('actual gfx1030 profiling required')
    batch = corpus.roots(plan)
    old_training = model.training
    old_anchor_training = model.anchor.training
    model.eval()
    try:
        with torch.cuda.device(corpus.device), torch.no_grad():
            for _ in range(warmups):
                branch(model, control, batch.features, batch.legal_actions)
            torch.cuda.synchronize(corpus.device)
            torch.cuda.reset_peak_memory_stats(corpus.device)
            events = [(torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True))
                      for _ in range(repeats)]
            started = time.monotonic()
            for first, last in events:
                first.record()
                branch(model, control, batch.features, batch.legal_actions)
                last.record()
            torch.cuda.synchronize(corpus.device)
            wall = time.monotonic() - started
            milliseconds = [first.elapsed_time(last) for first, last in events]
    finally:
        model.train(old_training)
        model.anchor.train(old_anchor_training)
    parameters = list(model.active_parameters())
    return dict(kind='semantic_untrained_branch_GPU_profile', control=control,
                encoder=model.anchor.encoder, train_manifest_sha256=corpus.manifest_sha256,
                study_source_sha256=_sha(__file__), root_ids=list(batch.root_ids),
                batch_roots=batch.features.shape[0], legal_count=batch.legal_actions.shape[1],
                warmups=warmups, repeats=repeats, GPU_event_ms=milliseconds,
                measured_wall_seconds=wall,
                stored_parameters=sum(p.numel() for p in model.parameters()),
                active_parameters=sum(p.numel() for p in parameters),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(corpus.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(corpus.device),
                memory_scope='whole_device_allocator_including_resident_corpus_and_other_models',
                time_scope='branch_forward_only_excludes_ingress_gather_decode',
                quality_metrics_computed=False, training=False, equal_compute_claim=False)


def train_phase(model, corpus, *, phase, control=None, declaration_path,
                declaration_sha256, study_started):
    """One bounded TRAIN phase; caller persists/fingerprints checkpoints.

    No label-driven early stopping or replacement. Exhaustion/fault is a failed
    run; partial checkpoints cannot be promoted. Per-update losses stay resident.
    """
    config = require_gpu_learning_authority(declaration_path, declaration_sha256, corpus)
    if phase not in ('anchor', 'transition') or (phase == 'transition' and control not in CONTROLS):
        raise ValueError('declared phase/control required')
    anchor = model if phase == 'anchor' else model.anchor
    if (not isinstance(anchor, SemanticAnchor) or anchor.encoder not in ENCODERS or
            anchor.width != config.width or anchor.depth != config.depth or
            anchor.queries.device != corpus.device or
            anchor.anchor_kind != 'material' or
            not config.material_lower <= anchor.material_scale <= config.material_upper):
        raise ValueError('model differs from declared representation/positive shared scale')
    if phase == 'transition' and any(p.requires_grad for p in model.anchor.parameters()):
        raise ValueError('transition anchor/readout must already be frozen')
    root_plans, observation_plans = corpus.prepare_structural_schedule()
    plans = observation_plans if phase == 'anchor' else root_plans
    if not plans:
        raise ValueError('complete nonempty resident schedule required')
    parameters = list(model.active_parameters())
    optimizer = torch.optim.AdamW(parameters, lr=config.learning_rate,
                                  weight_decay=config.weight_decay, betas=(.9, .999), eps=1e-8)
    updates = config.anchor_updates if phase == 'anchor' else config.transition_updates
    torch.cuda.reset_peak_memory_stats(corpus.device)
    started = time.monotonic()
    model.train()
    if phase == 'transition':
        model.anchor.eval()
        anchor_before = fingerprint(model.anchor)
    accumulated = torch.zeros((), device=corpus.device)
    for update in range(updates):
        now = time.monotonic()
        if now - started > config.phase_wall_seconds or now - study_started > config.total_wall_seconds:
            raise TimeoutError('declared phase/study wall ceiling; run inadmissible')
        optimizer.zero_grad(set_to_none=True)
        plan = plans[update % len(plans)]
        if phase == 'anchor':
            batch = corpus.observations(plan)
            losses = anchor_losses(model, model(batch.features), batch.features,
                                   batch.midpoint, batch.ordinary, teacher_halfwidth=batch.halfwidth)
            loss = sum(COEFFICIENTS[name] * value for name, value in losses.items())
            loss = loss * (batch.features.shape[0] / 32)
        else:
            batch = corpus.roots(plan)
            output = branch(model, control, batch.features, batch.legal_actions)
            loss, _ = transition_loss(model, output, batch)
            loss = loss * (batch.features.shape[0] / 4)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(parameters, config.clip_grad_norm, foreach=False)
        optimizer.step()
        accumulated += loss.detach()
    torch.cuda.synchronize(corpus.device)
    elapsed = time.monotonic() - started
    if elapsed > config.phase_wall_seconds or time.monotonic() - study_started > config.total_wall_seconds:
        raise TimeoutError('declared wall ceiling exceeded on final GPU drain; run inadmissible')
    if phase == 'transition' and fingerprint(model.anchor) != anchor_before:
        raise RuntimeError('frozen anchor/readout changed')
    mean_loss = float((accumulated / updates).cpu())
    if not math.isfinite(mean_loss):
        raise RuntimeError('nonfinite completed TRAIN objective; run inadmissible')
    return dict(status='complete', phase=phase, control=control, updates=updates,
                elapsed_seconds=elapsed, parameter_counts=parameter_counts(model) if phase == 'anchor' else
                dict(total=sum(p.numel() for p in model.parameters()), active=sum(p.numel() for p in parameters)),
                peak_allocated_bytes=torch.cuda.max_memory_allocated(corpus.device),
                peak_reserved_bytes=torch.cuda.max_memory_reserved(corpus.device),
                memory_scope='device_allocator_including_preexisting_resident_data_and_other_models',
                mean_train_loss=mean_loss, checkpoint_state_sha256=fingerprint(model))
