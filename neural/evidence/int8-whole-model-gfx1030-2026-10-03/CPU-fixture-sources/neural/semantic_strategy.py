"""Train-only typed chess semantics and a causal, frozen-readout transition.

Schema1 observations never include repetition context. Diagram pseudoattacks
are declared exact rules priors, not predictions of legality or winning threats.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .models import ChessLatent, build_model
from .score_frames import FRAME_KIND, interval_smooth_l1, score_frame_manifest
from .strategic_transport import _inputs, derive_effects, successor_pieces
from .value_heads import MATERIAL_VALUES, SCALE_CP, STATIC_LIMIT, ZeroLinear

ENCODERS = ('square', 'relation', 'plan', 'threat')
MODES = ('transport', 'conditioned', 'action_blind')
ROLES = ('own_king_pressure', 'enemy_king_pressure', 'own_loose', 'enemy_loose',
         'own_pseudoattacks', 'enemy_pseudoattacks', 'own_pawn_advance', 'enemy_pawn_advance')
KIND = 'chen_semantic_strategy_checkpoint'
LIMIT = 128 * 1024 * 1024


class Diagram(nn.Module):
    """Resident blocked pseudoattack graph and eight bounded diagram targets."""
    def __init__(self):
        super().__init__()
        self.register_buffer('patterns', ChessLatent._attack_patterns())
        self.register_buffer('between', ChessLatent._between())
        self.register_buffer('sliding', torch.tensor([0, 0, 1, 1, 1, 0] * 2).float())
        self.register_buffer('orientation', torch.arange(64) ^ 56)
        square = torch.arange(64)
        df = (square[:, None] % 8 - square[None] % 8).abs()
        dr = (square[:, None] // 8 - square[None] // 8).abs()
        self.register_buffer('king_zone', (torch.maximum(df, dr) <= 1).float())
        self.register_buffer('coordinates', torch.stack((square % 8 / 7, square // 8 / 7), -1))
        self.register_buffer('material_values', torch.tensor(MATERIAL_VALUES).float())
        self.register_buffer('role_flip', torch.tensor([1, 0, 3, 2, 5, 4, 7, 6]))

    def edges(self, pieces):
        flat = pieces.flatten(2)
        clear = (torch.einsum('bs,ijs->bij', flat.sum(1), self.between) == 0).to(flat.dtype)
        patterns = self.patterns[None] * ((1 - self.sliding[None, :, None, None]) +
                                        self.sliding[None, :, None, None] * clear[:, None])
        return torch.stack((torch.einsum('bps,bpst->bst', flat[:, :6], patterns[:, :6]),
                            torch.einsum('bps,bpst->bst', flat[:, 6:], patterns[:, 6:])), 1)

    def maps(self, pieces):
        flat = pieces.flatten(2)
        counts = self.edges(pieces).sum(2)
        own, enemy = flat[:, :6].sum(1), flat[:, 6:].sum(1)
        king_zones = torch.stack((flat[:, 5] @ self.king_zone, flat[:, 11] @ self.king_zone), 1)
        own_loose = (own - flat[:, 5]) * (counts[:, 1] > 0) * (counts[:, 0] == 0)
        enemy_loose = (enemy - flat[:, 11]) * (counts[:, 0] > 0) * (counts[:, 1] == 0)
        occupied = own + enemy
        square = torch.arange(64, device=pieces.device)
        own_empty = (1 - occupied[:, (square + 8).clamp_max(63)]) * (square < 56)
        enemy_empty = (1 - occupied[:, (square - 8).clamp_min(0)]) * (square >= 8)
        return torch.stack((king_zones[:, 0] * counts[:, 1].clamp_max(8) / 8,
                            king_zones[:, 1] * counts[:, 0].clamp_max(8) / 8,
                            own_loose, enemy_loose, counts[:, 0].clamp_max(8) / 8,
                            counts[:, 1].clamp_max(8) / 8,
                            (flat[:, 0] * own_empty).roll(8, -1), (flat[:, 6] * enemy_empty).roll(-8, -1)), 1).reshape(-1, 8, 8, 8)

    def material(self, pieces):
        counts = pieces.sum((-1, -2))
        return ((counts[:, :6] - counts[:, 6:]) * self.material_values).sum(-1).clamp(-STATIC_LIMIT, STATIC_LIMIT)

    def typed_edges(self, pieces):
        flat = pieces.flatten(2)
        own, enemy = flat[:, :6].sum(1), flat[:, 6:].sum(1)
        edges = self.edges(pieces)
        # [B,type,receiver,sender]: attack/support messages only to pieces.
        return torch.stack((edges[:, 0] * enemy[:, None], edges[:, 0] * own[:, None],
                            edges[:, 1] * own[:, None], edges[:, 1] * enemy[:, None]), 1).transpose(-1, -2)


class TypedMessage(nn.Module):
    """Four explicit attack/support message channels, not dense attention."""
    def __init__(self, width):
        super().__init__()
        self.messages = nn.ModuleList(nn.Linear(width, width, bias=False) for _ in range(4))
        self.update = nn.Sequential(nn.LayerNorm(width * 5), nn.Linear(width * 5, width * 2),
                                    nn.GELU(), nn.Linear(width * 2, width))

    def forward(self, objects, edges, occupied):
        normalized = edges / edges.sum(-1, keepdim=True).clamp_min(1)
        messages = [normalized[:, i] @ layer(objects) for i, layer in enumerate(self.messages)]
        return (objects + self.update(torch.cat([objects, *messages], -1))) * occupied[:, :, None]


class SemanticAnchor(nn.Module):
    """Eight typed plan tokens, shared semantic decoder and CP residual head.

    For independent 'threat' encoding, occupied piece objects exchange typed
    attack/support messages. Old encoders are preserved as controlled backends.
    material_scale converts the declared physical pawn100 prior into teacher CP;
    fit it on TRAIN only and freeze before validation/test. It is not WDL.
    """
    def __init__(self, encoder='threat', width=32, depth=1, material_scale=1.0, anchor_kind='material'):
        super().__init__()
        if (encoder not in ENCODERS or type(width) is not int or not 8 <= width <= 128 or
                type(depth) is not int or not 1 <= depth <= 8 or
                type(material_scale) not in (int, float) or not math.isfinite(material_scale) or
                not 0 < material_scale <= 10 or anchor_kind not in ('material', 'zero')):
            raise ValueError('invalid semantic anchor configuration')
        self.encoder, self.width, self.depth = encoder, width, depth
        self.material_scale, self.anchor_kind = float(material_scale), anchor_kind
        self.diagram = Diagram()
        self.base = build_model(encoder, width=width, depth=depth, plan_tokens=8) if encoder != 'threat' else None
        self.objects = nn.Linear(20, width) if encoder == 'threat' else None
        self.messages = nn.ModuleList(TypedMessage(width) for _ in range(depth)) if encoder == 'threat' else nn.ModuleList()
        self.queries = nn.Parameter(torch.randn(8, width) * .02)
        self.key, self.value = nn.Linear(width, width), nn.Linear(width, width)
        self.norm = nn.LayerNorm(width)
        # Board and rule decoders have no bypass around the eight-token bottleneck.
        self.board = nn.Linear(8 * width, 64 * 13)
        self.maps = nn.ModuleList(nn.Linear(width, 64) for _ in range(8))
        self.rights = nn.Linear(8 * width, 4)
        self.ep = nn.Linear(8 * width, 65)
        self.clock = nn.Linear(8 * width, 1)
        self.cp = ZeroLinear(8 * width, 1)

    def encode(self, features):
        _inputs(features)
        pieces = features[:, :12]
        occupied = pieces.flatten(2).sum(1)
        if self.encoder == 'threat':
            inputs = torch.cat((features.flatten(2).transpose(1, 2),
                                self.diagram.coordinates[None].expand(features.shape[0], -1, -1)), -1)
            objects = self.objects(inputs) * occupied[:, :, None]
            edges = self.diagram.typed_edges(pieces)
            for layer in self.messages:
                objects = layer(objects, edges, occupied)
        else:
            objects = self.base.encode(features)
        logits = self.queries[None] @ self.key(objects).transpose(-1, -2) / math.sqrt(self.width)
        # Object presence is explicit; finite mask also keeps malformed empty
        # diagrams numerically defined (ingress must certify real chess states).
        logits = logits.masked_fill(occupied[:, None] == 0, -1e4)
        return self.norm(self.queries[None] + logits.softmax(-1) @ self.value(objects))

    def decode(self, latent, pieces):
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (8, self.width):
            raise ValueError('expected semantic latent [B,8,D]')
        flat = latent.flatten(1)
        material = self.diagram.material(pieces) * self.material_scale
        residual = self.cp(flat).squeeze(-1) * SCALE_CP
        cp = residual + material if self.anchor_kind == 'material' else residual
        return dict(latent=latent, pieces=pieces, value_cp=cp, residual_cp=residual, material_cp=material,
                    utility=(cp / SCALE_CP).tanh(), board_logits=self.board(flat).reshape(-1, 13, 8, 8),
                    map_logits=torch.stack([head(latent[:, i]) for i, head in enumerate(self.maps)], 1).reshape(-1, 8, 8, 8),
                    castling_logits=self.rights(flat), ep_logits=self.ep(flat), halfmove=self.clock(flat).squeeze(-1).sigmoid())

    def forward(self, features):
        return self.decode(self.encode(features), features[:, :12])

    def semantic_targets(self, features):
        _inputs(features)
        pieces = features[:, :12]
        classes = torch.cat((1 - pieces.sum(1, keepdim=True), pieces), 1).argmax(1)
        ep = features[:, 16].flatten(1)
        return dict(board=classes, maps=self.diagram.maps(pieces),
                    castling=features[:, 12:16, 0, 0],
                    ep=torch.where(ep.sum(-1) == 0, 64, ep.argmax(-1)), halfmove=features[:, 17, 0, 0])

    def active_parameters(self):
        # Unused policy/dynamics/old scalar heads never receive optimizer decay.
        if self.base is not None:
            for name, parameter in self.base.named_parameters():
                if name in ('position', 'plans') or name.startswith(('input.', 'square_layers.', 'relation_layers.', 'plan_', 'latent_norm.')):
                    yield parameter
        for name, parameter in self.named_parameters():
            if not name.startswith('base.'):
                yield parameter

    def config(self):
        return dict(semantic_version=1, feature_schema=1, encoder=self.encoder, width=self.width,
                    depth=self.depth, roles=list(ROLES), material_scale=self.material_scale,
                    anchor_kind=self.anchor_kind, material_values=list(MATERIAL_VALUES),
                    scale_cp=SCALE_CP, target_kind=FRAME_KIND, score_frame=score_frame_manifest(),
                    diagram='blocked_pseudoattacks_no_pins_v1', terminal_responsibility='simulator')


def anchor_losses(anchor, outputs, features, teacher_cp, ordinary_mask, *, teacher_halfwidth=None):
    """Unweighted components; study MUST predeclare coefficients/sampling.

    ordinary_mask is bool[B]; terminal/mate/bounded teacher labels never silently
    train scalar calibration. No host scalar extraction in this resident path.
    """
    _inputs(features)
    if (not isinstance(teacher_cp, torch.Tensor) or teacher_cp.dtype != torch.float32 or
            teacher_cp.shape != (features.shape[0],) or teacher_cp.device != features.device or
            not isinstance(ordinary_mask, torch.Tensor) or ordinary_mask.dtype != torch.bool or
            ordinary_mask.shape != teacher_cp.shape or ordinary_mask.device != features.device):
        raise ValueError('expected resident float32 teacher CP and bool ordinary mask [B]')
    targets = anchor.semantic_targets(features)
    weight = ordinary_mask.to(outputs['value_cp'].dtype)
    if teacher_halfwidth is None:
        teacher_halfwidth = torch.zeros_like(teacher_cp)  # synthetic/exact mechanical targets only
    if (not isinstance(teacher_halfwidth, torch.Tensor) or teacher_halfwidth.dtype != torch.float32 or
            teacher_halfwidth.shape != teacher_cp.shape or teacher_halfwidth.device != features.device):
        raise ValueError('expected admitted resident float32 teacher interval halfwidth[B]')
    value = interval_smooth_l1(outputs['value_cp'], teacher_cp, teacher_halfwidth, SCALE_CP)
    return dict(value=(value * weight).sum() / weight.sum().clamp_min(1),
                board=F.cross_entropy(outputs['board_logits'], targets['board']),
                maps=F.binary_cross_entropy_with_logits(outputs['map_logits'], targets['maps']),
                rights=F.binary_cross_entropy_with_logits(outputs['castling_logits'], targets['castling']),
                ep=F.cross_entropy(outputs['ep_logits'], targets['ep']),
                clock=F.mse_loss(outputs['halfmove'], targets['halfmove']))


def parameter_counts(model):
    """Report stored and actually optimized parameters, never assume equality."""
    return dict(total=sum(p.numel() for p in model.parameters()),
                active=sum(p.numel() for p in model.active_parameters()))


class SemanticTransition(nn.Module):
    """Frozen trained semantics + learned action correction with one CP decoder.

    No successor observation ingress. 'transport' uses typed role-biased routing of derived child
    pieces; 'conditioned' uses the SAME child diagram with generic routing; 'blind'
    uses neither action nor derived child diagram in learned latent prediction.
    The material prior remains an explicit action-sensitive algebraic baseline.
    """
    def __init__(self, anchor, mode='transport'):
        super().__init__()
        if not isinstance(anchor, SemanticAnchor) or mode not in MODES:
            raise ValueError('invalid semantic transition')
        self.anchor, self.mode = anchor, mode
        anchor.requires_grad_(False)
        width = anchor.width
        self.local = nn.Sequential(nn.Linear(12 + 8 + 2, width), nn.GELU(), nn.Linear(width, width))
        self.event = nn.Sequential(nn.Linear(12 * 3 + 5 + 4, width), nn.GELU(), nn.Linear(width, width))
        self.query = nn.Linear(width, width)
        self.update = nn.Sequential(nn.Linear(width * 3, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.norm = nn.LayerNorm(width)
        self.to(device=anchor.queries.device, dtype=torch.float32)

    def encode(self, features):
        with torch.no_grad():
            return self.anchor.encode(features)

    def predict(self, root_features, actions, root_latent=None):
        _inputs(root_features, actions)
        effects = derive_effects(root_features, actions)
        pieces = successor_pieces(root_features, effects)
        latent = self.encode(root_features) if root_latent is None else root_latent
        if (not isinstance(latent, torch.Tensor) or latent.dtype != torch.float32 or
                latent.shape != (root_features.shape[0], 8, self.anchor.width) or latent.device != root_features.device):
            raise ValueError('expected resident cached root plans [B,8,D]')
        # Do NOT detach caller predicted latents: multistep gradients are useful.
        oriented = latent[:, self.anchor.diagram.role_flip]
        if self.mode == 'action_blind':
            context = torch.zeros_like(oriented)
            event = torch.zeros_like(oriented[:, 0])
        else:
            local_pieces = pieces
            maps = self.anchor.diagram.maps(local_pieces).flatten(2).transpose(1, 2)
            local = self.local(torch.cat((local_pieces.flatten(2).transpose(1, 2), maps,
                self.anchor.diagram.coordinates[None].expand(latent.shape[0], -1, -1)), -1))
            logits = self.query(oriented) @ local.transpose(-1, -2) / math.sqrt(self.anchor.width)
            if self.mode == 'transport':
                # Typed prior: each role routes toward its own diagram signal.
                # Both controls consume identical maps; this is not extra data.
                logits = logits + maps.transpose(1, 2)
            context = logits.softmax(-1) @ local
            batch = torch.arange(latent.shape[0], device=latent.device)
            root = root_features[:, :12].flatten(2).transpose(1, 2)
            source, target, captured = (effects[key] for key in ('source', 'target', 'captured'))
            promotion = (effects['promotion'][:, None] == torch.arange(5, device=latent.device)[None]).float()
            squares = torch.stack((source % 8 / 7, source // 8 / 7, target % 8 / 7, target // 8 / 7), -1)
            event = self.event(torch.cat((root[batch, source], root[batch, target], root[batch, captured], promotion, squares), -1))
        correction = self.update(torch.cat((oriented, context, event[:, None].expand_as(oriented)), -1))
        return self.anchor.decode(self.norm(oriented + correction), pieces)

    def forward(self, root_features, actions, root_latent=None):
        return self.predict(root_features, actions, root_latent)

    def refresh(self, features):
        return self.anchor(features)

    def rollout(self, observations, actions, initial_latent=None):
        """Hybrid: observed current piece/rule sidecars + imagined strategic plans.

        observations [T,B,18,8,8] are simulator current states, never encoded after
        ply0. Each action is canonical to THAT mover. This is not autonomous rules.
        """
        if observations.ndim != 5 or actions.shape != observations.shape[:2]:
            raise ValueError('expected observations [T,B,18,8,8], actions [T,B]')
        if observations.shape[0] == 0:
            raise ValueError('rollout requires at least one legal ply')
        latent = initial_latent
        results = []
        for observation, action in zip(observations.unbind(0), actions.unbind(0)):
            output = self.predict(observation, action, latent)
            latent = output['latent']
            results.append(output)
        return results

    def branch(self, root_features, actions, root_latent=None):
        """Evaluate B roots x K legal candidates, caching each root once.

        Blind strategic residual is computed once per root then expanded, which
        guarantees exact same-material sibling ties rather than duplicate-root
        GEMM roundoff artifacts. No padding is supported: ingress groups counts.
        """
        _inputs(root_features)
        if (actions.ndim != 2 or actions.shape[0] != root_features.shape[0] or
                actions.dtype != torch.int64 or actions.device != root_features.device or actions.shape[1] == 0):
            raise ValueError('expected resident legal actions [B,K], K>0')
        batch, count = actions.shape
        latent = self.encode(root_features) if root_latent is None else root_latent
        if latent.dtype != torch.float32 or latent.shape != (batch, 8, self.anchor.width) or latent.device != root_features.device:
            raise ValueError('invalid root plans')
        roots = root_features[:, None].expand(-1, count, -1, -1, -1).reshape(-1, 18, 8, 8)
        if self.mode != 'action_blind':
            plans = latent[:, None].expand(-1, count, -1, -1).reshape(-1, 8, self.anchor.width)
            outputs = self.predict(roots, actions.flatten(), plans)
            return {k: v.reshape(batch, count, *v.shape[1:]) for k, v in outputs.items()}
        oriented = latent[:, self.anchor.diagram.role_flip]
        correction = self.update(torch.cat((oriented, torch.zeros_like(oriented), torch.zeros_like(oriented)), -1))
        singleton = self.anchor.decode(self.norm(oriented + correction), root_features[:, :12])
        outputs = {k: v[:, None].expand(-1, count, *v.shape[1:]) for k, v in singleton.items()}
        pieces = successor_pieces(roots, derive_effects(roots, actions.flatten()))
        material = (self.anchor.diagram.material(pieces) * self.anchor.material_scale).reshape(batch, count)
        cp = outputs['residual_cp'] + material if self.anchor.anchor_kind == 'material' else outputs['residual_cp']
        outputs.update(pieces=pieces.reshape(batch, count, 12, 8, 8), material_cp=material, value_cp=cp, utility=(cp / SCALE_CP).tanh())
        return outputs

    def active_parameters(self):
        for module in (self.update, self.norm):
            yield from module.parameters()
        if self.mode != 'action_blind':
            for module in (self.local, self.event, self.query):
                yield from module.parameters()

    def config(self):
        return dict(transition_version=1, anchor=self.anchor.config(), mode=self.mode,
                    action_schema=1, action_admission='legal_standard_chess',
                    perspective='successor_side_to_move', latent_target='frozen_train_only_semantic_anchor',
                    readout='shared_frozen_anchor', child_prior='same_exact_pieces_maps_in_both_conditioned_modes',
                    role_routing='diagram_bias_transport_only', rollout='observed_piece_rules_predicted_plans')


def save_semantic_checkpoint(model, path, provenance):
    if not isinstance(model, (SemanticAnchor, SemanticTransition)) or not isinstance(provenance, dict):
        raise ValueError('semantic model and primitive provenance required')
    try:
        json.dumps(provenance, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('nonfinite/nonprimitive provenance') from error
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint_version=1, kind=KIND, strength_evidence=False,
                   phase='anchor' if isinstance(model, SemanticAnchor) else 'transition',
                   config=model.config(), provenance=provenance,
                   state_dict={k: v.detach().cpu().clone() for k, v in model.state_dict().items()})
    torch.save(payload, path)
    if path.stat().st_size > LIMIT:
        path.unlink()
        raise ValueError('semantic checkpoint exceeds128MiB')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest())


def load_semantic_checkpoint(path, device='cpu'):
    path = Path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError('semantic checkpoint exceeds128MiB')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError('restricted semantic load failed') from error
    if (not isinstance(payload, dict) or set(payload) != {'checkpoint_version', 'kind', 'strength_evidence',
            'phase', 'config', 'provenance', 'state_dict'} or type(payload['checkpoint_version']) is not int or
            payload['checkpoint_version'] != 1 or payload['kind'] != KIND or payload['strength_evidence'] is not False or
            payload['phase'] not in ('anchor', 'transition')):
        raise ValueError('invalid semantic envelope')
    config = payload['config']
    if not isinstance(config, dict):
        raise ValueError('invalid semantic config')
    ac = config.get('anchor') if payload['phase'] == 'transition' else config
    if not isinstance(ac, dict):
        raise ValueError('invalid semantic config')
    try:
        model = SemanticAnchor(ac['encoder'], ac['width'], ac['depth'], ac['material_scale'], ac['anchor_kind'])
        if payload['phase'] == 'transition':
            model = SemanticTransition(model, config['mode'])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError('invalid semantic constructor config') from error
    # JSON equality is type-sensitive, unlike Python's True == 1.
    if json.dumps(config, sort_keys=True, allow_nan=False) != json.dumps(model.config(), sort_keys=True):
        raise ValueError('unsupported semantic config')
    if not isinstance(payload['provenance'], dict):
        raise ValueError('invalid semantic provenance')
    json.dumps(payload['provenance'], allow_nan=False)
    state, expected = payload['state_dict'], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('semantic tensor keys mismatch')
    for key, target in expected.items():
        value = state[key]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype or
                (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
            raise ValueError('semantic tensor schema/nonfinite mismatch: ' + key)
    for key, value in model.named_buffers():
        if not torch.equal(state[key], value):
            raise ValueError('semantic deterministic buffer mismatch: ' + key)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval(), {k: v for k, v in payload.items() if k != 'state_dict'}
