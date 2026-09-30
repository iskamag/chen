"""Action-sensitive chess residual prediction over explicitly known move algebra.

predict() accepts roots and legal canonical actions, never observed successors.
The exact piece/material/pseudoattack branch is a declared handcrafted prior.
The learned branch predicts finite-search correction in the NEXT mover's frame.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .dynamics import transport_slots
from .features import SCHEMA_VERSION
from .models import ChessLatent, MODELS, build_model
from .value_heads import MATERIAL_VALUES, SCALE_CP, STATIC_LIMIT, ZeroLinear

MODES = ('transport', 'conditioned', 'action_blind')
ANCHORS = ('material', 'zero')
KIND = 'chen_strategic_transport_checkpoint'
LIMIT = 128 * 1024 * 1024


def _inputs(features, actions=None):
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or
            features.ndim != 4 or tuple(features.shape[1:]) != (18, 8, 8)):
        raise ValueError('expected schema1 float32 [B,18,8,8] roots')
    if actions is not None and (not isinstance(actions, torch.Tensor) or
            actions.dtype != torch.int64 or actions.shape != (features.shape[0],) or
            actions.device != features.device):
        raise ValueError('expected resident canonical int64 [B] legal actions')


def derive_effects(features, actions):
    """Resident standard-chess effects for ALREADY LEGAL schema1 actions.

    This is algebra, not a legality check. Ingress must certify action range,
    legality, binary observations and standard (not Chess960) chess. No scalar
    host extraction is performed here. Black roots already flip ranks/colors.
    """
    _inputs(features, actions)
    pair, promotion = actions.div(5, rounding_mode='floor'), actions.remainder(5)
    source, target = pair.div(64, rounding_mode='floor'), pair.remainder(64)
    batch = torch.arange(features.shape[0], device=features.device)
    pieces = features[:, :12].flatten(2)
    pawn, king = pieces[batch, 0, source] > .5, pieces[batch, 5, source] > .5
    empty_target = pieces.sum(1)[batch, target] == 0
    ep = pawn & empty_target & (source.remainder(8) != target.remainder(8)) & (
        features[:, 16].flatten(1)[batch, target] > .5)
    castle = king & (source == 4) & ((target == 2) | (target == 6))
    rook_source = torch.where(castle, torch.where(target == 6, 7, 0), -1)
    rook_target = torch.where(castle, torch.where(target == 6, 5, 3), -1)
    return dict(source=source, target=target, promotion=promotion,
                captured=torch.where(ep, target - 8, target),
                rook_source=rook_source, rook_target=rook_target)


def successor_pieces(features, effects):
    """Same exact legal piece algebra, without one_hot content-to-host checks."""
    slots = transport_slots(features[:, :12].flatten(2).transpose(1, 2), effects)
    square = torch.arange(64, device=features.device)[None]
    promotion_mask = (effects['promotion'] > 0)[:, None] & (
        square == (effects['target'] ^ 56)[:, None])
    promoted = (effects['promotion'][:, None] ==
                torch.arange(12, device=features.device)[None]).to(features.dtype)
    slots = torch.where(promotion_mask[:, :, None], promoted[:, None], slots)
    slots = torch.cat((slots[:, :, 6:], slots[:, :, :6]), -1)
    return slots.transpose(1, 2).reshape(-1, 12, 8, 8)


def _model_config(base):
    return dict(name=base.kind, width=base.width, depth=base.depth,
                plan_tokens=base.plans.shape[0] if base.kind == 'plan' else 8)


def _validate_model_config(config):
    if (not isinstance(config, dict) or set(config) != {'name', 'width', 'depth', 'plan_tokens'} or
            config['name'] not in MODELS):
        raise ValueError('invalid strategic encoder config')
    for key, low, high in [('width', 8, 512), ('depth', 1, 32), ('plan_tokens', 1, 128)]:
        if type(config[key]) is not int or not low <= config[key] <= high:
            raise ValueError('invalid strategic encoder bound: ' + key)


class StrategicPredictor(nn.Module):
    """Owned root encoder, transported square latents and learned CP correction.

    Every mode constructs identical parameter/buffer keys and consumes identical
    initialization RNG; use one common state snapshot for controlled variants.
    Empty slot and scalar head start at zero. The movement-conditioned residual
    starts small rather than zero, making action identity measurable before the
    CP readout learns. No learned kernel replaces the simulator's legal rules.
    """
    def __init__(self, base, mode='transport', anchor_kind='material', freeze_encoder=False):
        super().__init__()
        if (not isinstance(base, ChessLatent) or base.position.dtype != torch.float32 or
                mode not in MODES or anchor_kind not in ANCHORS or type(freeze_encoder) is not bool):
            raise ValueError('unsupported strategic predictor configuration')
        _validate_model_config(_model_config(base))
        self.base, self.mode = base, mode
        self.anchor_kind, self.freeze_encoder = anchor_kind, freeze_encoder
        width = base.width
        self.empty = nn.Parameter(torch.zeros(width))
        self.perspective = nn.Linear(width, width)
        with torch.no_grad():
            self.perspective.weight.copy_(torch.eye(width))
            self.perspective.bias.zero_()
        self.condition = nn.Sequential(nn.Linear(width * 3 + 17, width), nn.GELU())
        self.correction = nn.Sequential(nn.Linear(width * 2 + 13, width * 2), nn.GELU(),
                                        nn.Linear(width * 2, width))
        with torch.no_grad():
            self.correction[-1].weight.mul_(.1)
            self.correction[-1].bias.zero_()
        self.norm = nn.LayerNorm(width)
        self.head = ZeroLinear(width, 1)
        self.register_buffer('material_values', torch.tensor(MATERIAL_VALUES, dtype=torch.float32))
        self.register_buffer('sliding', torch.tensor([0, 0, 1, 1, 1, 0] * 2, dtype=torch.float32))
        # These new modules follow an already-owned base device, without probing CUDA.
        self.to(device=base.position.device, dtype=torch.float32)
        if freeze_encoder:
            self.base.requires_grad_(False)

    def encode(self, features):
        _inputs(features)
        if self.freeze_encoder:
            with torch.no_grad():
                return self.base.encode(features)
        return self.base.encode(features)

    def material(self, pieces):
        counts = pieces.sum((-1, -2))
        return ((counts[:, :6] - counts[:, 6:12]) * self.material_values).sum(-1).clamp(-STATIC_LIMIT, STATIC_LIMIT)

    def attack_counts(self, pieces):
        """Exact blocker-aware PSEUDOattacks per side/target, not legal attacks.

        Endpoints are not blockers; own-occupied targets count as defended.
        Pins/check legality, castling, EP attacks and history are excluded.
        """
        flat = pieces.flatten(2)
        rays = torch.einsum('bs,ijs->bij', flat.sum(1), self.base.between)
        open_rays = (rays == 0).to(pieces.dtype)
        # Keep the side/color type dimension explicit while sharing occupancy.
        patterns = self.base.attack_patterns[None] * (
            (1 - self.sliding[None, :, None, None]) +
            self.sliding[None, :, None, None] * open_rays[:, None])
        per_piece = torch.einsum('bps,bpst->bpt', flat, patterns)
        return torch.stack((per_piece[:, :6].sum(1), per_piece[:, 6:].sum(1)), 1)

    def attack_maps(self, roots, pieces):
        """Eight declared exact-local strategic covariates in successor frame."""
        current = self.attack_counts(roots[:, :12])
        current = current.flip(1)[:, :, self.base.next_orientation]
        successor = self.attack_counts(pieces)
        delta = successor - current
        flat = pieces.flatten(2)
        return torch.stack((successor[:, 0] / 8, successor[:, 1] / 8,
                            delta[:, 0] / 8, delta[:, 1] / 8,
                            flat[:, :6].sum(1), flat[:, 6:].sum(1),
                            successor[:, 1] * flat[:, 5] / 8,
                            successor[:, 0] * flat[:, 11] / 8), -1)

    def _values(self, latent, pieces):
        material = self.material(pieces)
        residual = self.head(latent.mean(1)).squeeze(-1) * SCALE_CP
        cp = residual + material if self.anchor_kind == 'material' else residual
        return dict(latent=latent, pieces=pieces, material_cp=material,
                    residual_cp=residual, value_cp=cp, utility=(cp / SCALE_CP).tanh())

    def predict(self, root_features, actions, root_latent=None):
        _inputs(root_features, actions)
        effects = derive_effects(root_features, actions)
        pieces = successor_pieces(root_features, effects)
        latent = self.encode(root_features) if root_latent is None else root_latent
        if (not isinstance(latent, torch.Tensor) or latent.dtype != torch.float32 or
                latent.shape != (root_features.shape[0], 64, self.base.width) or
                latent.device != root_features.device):
            raise ValueError('root latent must be resident float32 [B,64,D]')
        if self.freeze_encoder:
            latent = latent.detach()
        if self.mode == 'transport':
            oriented = transport_slots(latent, effects, self.empty)
        else:
            oriented = latent[:, self.base.next_orientation]
        oriented = self.perspective(oriented)
        if self.mode == 'action_blind':
            # Exact material remains action-dependent; learned correction is blind.
            return self._values(self.norm(oriented), pieces)
        batch = torch.arange(latent.shape[0], device=latent.device)
        source, target, captured = (effects[key] for key in ('source', 'target', 'captured'))
        source_piece = root_features[:, :12].flatten(2).transpose(1, 2)[batch, source]
        condition = self.condition(torch.cat((latent[batch, source], latent[batch, target],
            latent[batch, captured], source_piece,
            (effects['promotion'][:, None] == torch.arange(5, device=latent.device)[None]).to(latent.dtype)), -1))
        square = torch.arange(64, device=latent.device)[None]
        flags = torch.stack([((square == (effects[key] ^ 56)[:, None]) &
                              (effects[key] >= 0)[:, None]).to(latent.dtype)
                             for key in ('source', 'target', 'captured', 'rook_source', 'rook_target')], -1)
        maps = self.attack_maps(root_features, pieces)
        correction = self.correction(torch.cat((oriented,
            condition[:, None].expand(-1, 64, -1), flags, maps), -1))
        return self._values(self.norm(oriented + correction), pieces)

    def forward(self, root_features, actions, root_latent=None):
        return self.predict(root_features, actions, root_latent)

    def static_scores(self, root_features, actions, root_latent=None):
        """Optional bounded integer heuristic; simulator supplies terminal/mate."""
        return self.predict(root_features, actions, root_latent)['value_cp'].round().clamp(
            -STATIC_LIMIT, STATIC_LIMIT).to(torch.int32)

    def refresh(self, successor_features):
        """Diagnostic full observation branch; never called inside predict()."""
        return self._values(self.encode(successor_features), successor_features[:, :12])

    def active_parameters(self):
        prefixes = ('input.', 'square_layers.', 'relation_layers.', 'plan_', 'latent_norm.')
        if not self.freeze_encoder:
            for name, parameter in self.base.named_parameters():
                if name in ('position', 'plans') or name.startswith(prefixes):
                    yield parameter
        for module in (self.perspective, self.norm, self.head):
            yield from module.parameters()
        if self.mode != 'action_blind':
            yield from self.condition.parameters()
            yield from self.correction.parameters()
        if self.mode == 'transport':
            yield self.empty

    def config(self):
        return dict(strategic_version=1, feature_schema=SCHEMA_VERSION,
            model_config=_model_config(self.base), mode=self.mode, anchor_kind=self.anchor_kind,
            freeze_encoder=self.freeze_encoder, material_values=list(MATERIAL_VALUES),
            scale_cp=SCALE_CP, action_schema=1, action_admission='legal_standard_chess',
            effect_algebra='root_pieces_raw_ep_legal_action_v1',
            exact_state='successor_piece_planes_only', pseudoattacks='blocker_aware_no_pins',
            attack_map_scale=8.0, perspective='successor_side_to_move',
            target_kind='finite_depth_successor_minimax_cp_regression',
            terminal_responsibility='simulator', head_init='zero_without_rng',
            latent_target='externally_fixed_encoder', float_cp_clamped=False,
            static_limit=STATIC_LIMIT, static_rounding='nearest_even')


def build_strategic_predictor(model_config, mode='transport', anchor_kind='material', freeze_encoder=False):
    _validate_model_config(model_config)
    return StrategicPredictor(build_model(**model_config), mode, anchor_kind, freeze_encoder)


def _metadata(value):
    if not isinstance(value, dict):
        raise ValueError('primitive provenance dictionary required')
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError('nonprimitive/nonfinite strategic provenance') from error


def save_strategic_checkpoint(model, path, provenance):
    if not isinstance(model, StrategicPredictor):
        raise ValueError('expected StrategicPredictor')
    _metadata(provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(checkpoint_version=1, kind=KIND, strength_evidence=False,
        predictor_config=model.config(), provenance=provenance,
        state_dict={key: value.detach().cpu().clone() for key, value in model.state_dict().items()})
    torch.save(payload, path)
    if path.stat().st_size > LIMIT:
        path.unlink()
        raise ValueError('strategic checkpoint exceeds128MiB')
    return dict(path=str(path), sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                kind=KIND, strength_evidence=False)


def load_strategic_checkpoint(path, device='cpu'):
    path = Path(path)
    if path.stat().st_size > LIMIT:
        raise ValueError('strategic checkpoint exceeds128MiB')
    try:
        payload = torch.load(path, map_location='cpu', weights_only=True)
    except Exception as error:
        raise ValueError('restricted strategic loading failed') from error
    if (not isinstance(payload, dict) or set(payload) !=
            {'checkpoint_version', 'kind', 'strength_evidence', 'predictor_config', 'provenance', 'state_dict'} or
            type(payload['checkpoint_version']) is not int or payload['checkpoint_version'] != 1 or
            payload['kind'] != KIND or payload['strength_evidence'] is not False):
        raise ValueError('invalid strategic checkpoint envelope')
    config = payload['predictor_config']
    if not isinstance(config, dict):
        raise ValueError('invalid strategic config')
    _validate_model_config(config.get('model_config'))
    model = build_strategic_predictor(config['model_config'], config.get('mode'),
        config.get('anchor_kind'), config.get('freeze_encoder'))
    if (type(config.get('strategic_version')) is not int or
            type(config.get('feature_schema')) is not int or type(config.get('action_schema')) is not int or
            not isinstance(config.get('material_values'), list) or
            any(type(value) is not int for value in config['material_values']) or
            config.get('float_cp_clamped') is not False or config != model.config()):
        raise ValueError('unsupported strategic semantics/config')
    _metadata(payload['provenance'])
    state, expected = payload['state_dict'], model.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected):
        raise ValueError('strategic tensor keys mismatch')
    for key, value in state.items():
        target = expected[key]
        if (not isinstance(value, torch.Tensor) or value.shape != target.shape or value.dtype != target.dtype or
                (value.is_floating_point() and not bool(torch.isfinite(value).all()))):
            raise ValueError('strategic tensor schema/nonfinite mismatch: ' + key)
    for key, value in model.named_buffers():
        if not torch.equal(state[key], value):
            raise ValueError('strategic deterministic buffer mismatch: ' + key)
    model.load_state_dict(state, strict=True)
    model.to(device).eval()
    return model, {key: value for key, value in payload.items() if key != 'state_dict'}
