"""Sparse tactical computation prototype, without training/deployment claims.

The caller supplies exact directed evidence and the COMPLETE legal action set.
This module learns messages over that evidence; it does not calculate attacks,
certify legality, execute SEE/minimax, or accept teacher targets in forward.
CPU admission is separate from tensor-only arithmetic so no forward host reads
are needed. A future resident producer must certify state/history/epoch itself.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Sequence

import torch
from torch import nn

from .features import ACTION_COUNT

KIND = 'chen_tactical_operator_mechanics_v1'
SCORE_FRAME = 'current_node_mover_cp'
PRIOR_KIND = 'causal_classical_current_state_cp'
STEPS = (1, 2, 4, 8)


class Relation(IntEnum):
    OCCUPIES = 0                  # piece -> square
    OCCUPIED_BY = 1               # square -> piece
    PSEUDOATTACK = 2              # piece -> square, including empty targets
    OWN_ATTACK_ENEMY = 3          # mover piece -> opponent piece
    OWN_SUPPORT_OWN = 4           # mover piece -> mover piece
    ENEMY_ATTACK_OWN = 5          # opponent piece -> mover piece
    ENEMY_SUPPORT_ENEMY = 6       # opponent piece -> opponent piece
    PINNED_TO_KING = 7            # pinned piece -> its king
    PINNER_TO_PINNED = 8          # pinner -> pinned piece
    CHECKER_TO_KING = 9           # checker -> current mover king
    ACTION_SOURCE = 10           # moving piece -> action
    ACTION_DESTINATION = 11      # target square -> action
    ACTION_VICTIM = 12           # captured piece -> action (EP victim != target)
    ACTION_EFFECT = 13           # action -> changed square, including rook/EP


@dataclass(frozen=True)
class TacticalConfig:
    width: int = 32
    steps: int = 4
    piece_channels: int = 16
    square_channels: int = 12
    action_channels: int = 12
    edge_channels: int = 4
    feature_schema: str = 'caller_declared_tactical_v1'
    cp_scale: float = 800.

    def __post_init__(self):
        if (type(self.width) is not int or not 8 <= self.width <= 256 or
                type(self.steps) is not int or self.steps not in STEPS):
            raise ValueError('width must be 8..256 and steps one of 1/2/4/8')
        for name in ('piece_channels', 'square_channels', 'action_channels', 'edge_channels'):
            if type(getattr(self, name)) is not int or not 1 <= getattr(self, name) <= 128:
                raise ValueError('feature widths must be integer 1..128')
        if (not isinstance(self.feature_schema, str) or not self.feature_schema or
                type(self.cp_scale) not in (int, float) or not 0 < self.cp_scale <= 10000):
            raise ValueError('a feature schema and finite positive CP scale are required')


@dataclass(frozen=True)
class TacticalBatch:
    # Node indices concatenate P pieces, 64 squares, K canonical action slots.
    pieces: torch.Tensor          # float32 [B,P,Cp], P<=32; caller feature schema
    squares: torch.Tensor         # float32 [B,64,Cs]
    actions: torch.Tensor         # float32 [B,K,Ca]
    piece_mask: torch.Tensor      # bool [B,P]
    piece_squares: torch.Tensor   # int64 [B,P], canonical square for each object
    action_mask: torch.Tensor     # bool [B,K], canonical sorted valid slots
    action_ids: torch.Tensor      # int64 [B,K], from neural.features schema
    edge_index: torch.Tensor      # int64 [B,E,2], source then receiver
    edge_kind: torch.Tensor       # int64 [B,E], Relation
    edge_mask: torch.Tensor       # bool [B,E]; no implicit reverse edges
    edge_features: torch.Tensor   # float32 [B,E,Ce], exact causal evidence
    edge_overflow: torch.Tensor   # bool [B]; emitter capacity failure, not truncation
    value_prior_cp: torch.Tensor  # float32 [B], same node mover frame
    action_prior_cp: torch.Tensor # float32 [B,K], PARENT mover frame, not child mover
    feature_schema: str = 'caller_declared_tactical_v1'
    score_frame: str = SCORE_FRAME
    prior_kind: str = PRIOR_KIND


def _metadata(batch: TacticalBatch, config: TacticalConfig):
    if not isinstance(batch, TacticalBatch):
        raise ValueError('expected TacticalBatch')
    if (batch.feature_schema != config.feature_schema or batch.score_frame != SCORE_FRAME or
            batch.prior_kind != PRIOR_KIND):
        raise ValueError('feature schema, mover score frame or causal prior kind mismatch')
    x = batch.pieces
    if (not isinstance(x, torch.Tensor) or x.ndim != 3 or x.shape[0] == 0 or
            not 1 <= x.shape[1] <= 32 or x.shape[2] != config.piece_channels or x.dtype != torch.float32):
        raise ValueError('pieces must be float32 nonempty [B,P<=32,Cp]')
    b, p = x.shape[:2]
    if not isinstance(batch.actions, torch.Tensor) or batch.actions.ndim != 3:
        raise ValueError('actions must be [B,K,Ca]')
    k = batch.actions.shape[1]
    if not isinstance(batch.edge_index, torch.Tensor) or batch.edge_index.ndim != 3:
        raise ValueError('edge_index must be [B,E,2]')
    e = batch.edge_index.shape[1]
    expected = dict(squares=((b, 64, config.square_channels), torch.float32),
                    actions=((b, k, config.action_channels), torch.float32),
                    piece_mask=((b, p), torch.bool), piece_squares=((b, p), torch.int64),
                    action_mask=((b, k), torch.bool), action_ids=((b, k), torch.int64),
                    edge_index=((b, e, 2), torch.int64), edge_kind=((b, e), torch.int64),
                    edge_mask=((b, e), torch.bool), edge_features=((b, e, config.edge_channels), torch.float32),
                    edge_overflow=((b,), torch.bool), value_prior_cp=((b,), torch.float32),
                    action_prior_cp=((b, k), torch.float32))
    for name, (shape, dtype) in expected.items():
        tensor = getattr(batch, name)
        if (not isinstance(tensor, torch.Tensor) or tuple(tensor.shape) != shape or
                tensor.dtype != dtype or tensor.device != x.device):
            raise ValueError(f'{name} shape/dtype/device mismatch')
    return b, p, k, e


def admit_contents(batch: TacticalBatch, config: TacticalConfig,
                   *, expected_legal_actions: Sequence[Sequence[int]] | None = None):
    """Cold CPU structural admission. External oracle optionally checks coverage.

    This does NOT establish that relations/features/priors are truthful chess
    evidence. The producer still owns that proof plus state/hash/history/epoch.
    Padded payload may be arbitrary, including NaNs and invalid indices.
    """
    b, p, k, _ = _metadata(batch, config)
    if batch.pieces.device.type != 'cpu':
        raise ValueError('content admission is CPU-only; forward never calls it')
    if bool(batch.edge_overflow.any()):
        raise ValueError('topology overflow: reject packet, never truncate')
    node_mask = torch.cat((batch.piece_mask, torch.ones(b, 64, dtype=torch.bool), batch.action_mask), 1)
    for name, mask in (('pieces', batch.piece_mask), ('squares', torch.ones(b, 64, dtype=torch.bool)),
                       ('actions', batch.action_mask), ('edge_features', batch.edge_mask),
                       ('action_prior_cp', batch.action_mask)):
        if not bool(torch.isfinite(getattr(batch, name)[mask]).all()):
            raise ValueError(f'nonfinite active {name}')
    if not bool(torch.isfinite(batch.value_prior_cp).all()):
        raise ValueError('nonfinite value prior')
    for row in range(b):
        locations = batch.piece_squares[row, batch.piece_mask[row]].tolist()
        if any(not 0 <= square < 64 for square in locations) or len(set(locations)) != len(locations):
            raise ValueError('piece objects require distinct canonical squares')
        ids = batch.action_ids[row, batch.action_mask[row]].tolist()
        if any(not 0 <= index < ACTION_COUNT for index in ids) or ids != sorted(set(ids)):
            raise ValueError('legal actions require unique ascending canonical IDs')
        if expected_legal_actions is not None:
            if len(expected_legal_actions) != b or ids != sorted(expected_legal_actions[row]):
                raise ValueError('complete legal action coverage mismatch')
        indices = batch.edge_index[row, batch.edge_mask[row]]
        kinds = batch.edge_kind[row, batch.edge_mask[row]]
        if (bool(((indices < 0) | (indices >= p + 64 + k)).any()) or
                bool(((kinds < 0) | (kinds >= len(Relation))).any())):
            raise ValueError('active edge index or relation outside declared range')
        if not bool(node_mask[row, indices].all()):
            raise ValueError('active edge touches a padded node')
        # Check node categories; pin/attack semantics themselves are producer-owned.
        for endpoints, kind in zip(indices.tolist(), kinds.tolist()):
            source, receiver = endpoints
            piece = lambda index: index < p
            square = lambda index: p <= index < p + 64
            action = lambda index: index >= p + 64
            if kind in (Relation.OCCUPIES, Relation.PSEUDOATTACK):
                good = piece(source) and square(receiver)
            elif kind == Relation.OCCUPIED_BY:
                good = square(source) and piece(receiver)
            elif Relation.OWN_ATTACK_ENEMY <= kind <= Relation.CHECKER_TO_KING:
                good = piece(source) and piece(receiver)
            elif kind in (Relation.ACTION_SOURCE, Relation.ACTION_VICTIM):
                good = piece(source) and action(receiver)
            elif kind == Relation.ACTION_DESTINATION:
                good = square(source) and action(receiver)
            else:
                good = action(source) and square(receiver)
            if not good:
                raise ValueError('relation endpoints have wrong node categories')
            if kind == Relation.OCCUPIES and receiver - p != batch.piece_squares[row, source]:
                raise ValueError('occupancy edge disagrees with piece object square')
            if kind == Relation.OCCUPIED_BY and source - p != batch.piece_squares[row, receiver]:
                raise ValueError('reverse occupancy edge disagrees with piece object square')
            if kind == Relation.ACTION_SOURCE:
                move_source = int(batch.action_ids[row, receiver - p - 64]) // (64 * 5)
                if int(batch.piece_squares[row, source]) != move_source:
                    raise ValueError('action source edge disagrees with canonical action ID')
            if kind == Relation.ACTION_DESTINATION:
                move_target = (int(batch.action_ids[row, receiver - p - 64]) // 5) % 64
                if source - p != move_target:
                    raise ValueError('action destination edge disagrees with canonical action ID')


def incoming_reductions(messages, receivers, edge_mask, node_count):
    """Sparse incoming mean/max/min, exact zero for no incoming evidence.

    Caller validates index bounds. Padded receivers are harmless after clamping;
    padded messages are ignored even when NaN. No dense N-by-N topology exists.
    """
    b, _, d = messages.shape
    index = receivers.clamp(0, node_count - 1)[:, :, None].expand(-1, -1, d)
    valid = edge_mask[:, :, None]
    zero = messages.new_zeros(b, node_count, d)
    total = zero.scatter_add(1, index, torch.where(valid, messages, torch.zeros_like(messages)))
    count = messages.new_zeros(b, node_count, 1).scatter_add(
        1, index[:, :, :1], edge_mask[:, :, None].to(messages.dtype))
    maximum = torch.full_like(zero, -torch.inf).scatter_reduce(
        1, index, torch.where(valid, messages, torch.full_like(messages, -torch.inf)),
        reduce='amax', include_self=True)
    minimum = torch.full_like(zero, torch.inf).scatter_reduce(
        1, index, torch.where(valid, messages, torch.full_like(messages, torch.inf)),
        reduce='amin', include_self=True)
    return (total / count.clamp_min(1), torch.where(count > 0, maximum, zero),
            torch.where(count > 0, minimum, zero))


class TacticalCell(nn.Module):
    """One learned processor reused at every iteration; NOT a chess algorithm."""
    def __init__(self, width, edge_channels):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.source = nn.Linear(width, width)
        self.receiver = nn.Linear(width, width, bias=False)
        self.evidence = nn.Linear(edge_channels, width, bias=False)
        self.relation = nn.Embedding(len(Relation), width)
        self.update = nn.Sequential(nn.Linear(4 * width, 2 * width), nn.GELU(), nn.Linear(2 * width, width))
        self.gate = nn.Linear(4 * width, width)
        self.output_norm = nn.LayerNorm(width)

    def forward(self, latent, node_mask, batch, *, typed=True):
        b, n, d = latent.shape
        normalized = self.norm(latent)
        endpoints = batch.edge_index.clamp(0, n - 1)
        # Project nodes ONCE, then gather along edges. Repeated per-edge node
        # projections would turn node work N*D^2 into avoidable E*D^2 work.
        sources, receivers = self.source(normalized), self.receiver(normalized)
        gather = lambda values, index: values.gather(1, index[:, :, None].expand(-1, -1, d))
        evidence = torch.where(batch.edge_mask[:, :, None], batch.edge_features,
                               torch.zeros_like(batch.edge_features))
        relation = self.relation(batch.edge_kind.clamp(0, len(Relation) - 1))
        if not typed:
            relation = torch.zeros_like(relation)
        messages = torch.nn.functional.gelu(gather(sources, endpoints[:, :, 0]) +
                                           gather(receivers, endpoints[:, :, 1]) +
                                           self.evidence(evidence) + relation)
        mean, maximum, minimum = incoming_reductions(messages, endpoints[:, :, 1], batch.edge_mask, n)
        joined = torch.cat((normalized, mean, maximum, minimum), -1)
        output = self.output_norm(latent + self.gate(joined).sigmoid() * self.update(joined))
        return torch.where(node_mask[:, :, None], output, torch.zeros_like(output))


def _action_outputs(scores, ids, mask):
    scores = scores.masked_fill(~mask, -torch.inf)
    if scores.shape[1] == 0:
        return dict(action_scores_cp=scores, action_probabilities=scores.clone(),
                    canonical_action=ids.new_full((ids.shape[0],), -1), has_legal_action=mask.any(1))
    present = mask.any(1)
    # All-masked rows are terminal: no fabricated softmax action, no NaNs.
    safe = torch.where(present[:, None], scores, torch.zeros_like(scores))
    probabilities = safe.softmax(1).masked_fill(~mask, 0.)
    slot = scores.argmax(1)
    chosen = ids.gather(1, slot[:, None]).squeeze(1)
    return dict(action_scores_cp=scores, action_probabilities=probabilities,
                canonical_action=torch.where(present, chosen, ids.new_full(chosen.shape, -1)),
                has_legal_action=present)


class TacticalOperator(nn.Module):
    def __init__(self, config=TacticalConfig()):
        super().__init__()
        if not isinstance(config, TacticalConfig):
            raise ValueError('expected TacticalConfig')
        self.settings = config
        d = config.width
        self.piece_input = nn.Linear(config.piece_channels, d)
        self.square_input = nn.Linear(config.square_channels, d)
        self.action_input = nn.Linear(config.action_channels, d)
        self.cell = TacticalCell(d, config.edge_channels)
        self.value = nn.Linear(3 * d, 1)
        self.action = nn.Linear(4 * d, 1)
        for head in (self.value, self.action):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def config(self):
        return dict(kind=KIND, **asdict(self.settings), relations=[kind.name for kind in Relation],
                    topology='caller_supplied_directed_sparse_edges', score_frame=SCORE_FRAME,
                    exact_rules='caller_owned', recurrent_weights='one_shared_cell',
                    trained=False, deployment_admitted=False)

    def parameter_report(self):
        total = sum(parameter.numel() for parameter in self.parameters())
        return dict(parameters=total, processor_parameters=sum(p.numel() for p in self.cell.parameters()),
                    steps=self.settings.steps, trained=False, measured_runtime=None)

    def forward(self, batch, *, steps=None, typed=True, return_trace=False):
        b, p, k, _ = _metadata(batch, self.settings)
        steps = self.settings.steps if steps is None else steps
        if type(steps) is not int or steps not in STEPS or type(typed) is not bool or type(return_trace) is not bool:
            raise ValueError('steps=1/2/4/8 and boolean control flags required')
        if batch.pieces.device != self.piece_input.weight.device:
            raise ValueError('batch and model must share device')
        if self.piece_input.weight.dtype != torch.float32:
            raise ValueError('prototype declares FP32 arithmetic')
        clean = lambda values, mask: torch.where(mask[:, :, None], values, torch.zeros_like(values))
        pieces = self.piece_input(clean(batch.pieces, batch.piece_mask))
        squares = self.square_input(batch.squares)
        actions = self.action_input(clean(batch.actions, batch.action_mask))
        mask = torch.cat((batch.piece_mask, torch.ones(b, 64, dtype=torch.bool, device=pieces.device),
                          batch.action_mask), 1)
        latent = torch.where(mask[:, :, None], torch.cat((pieces, squares, actions), 1), 0.)
        trace = []
        for _ in range(steps):
            latent = self.cell(latent, mask, batch, typed=typed)
            if return_trace:
                trace.append(latent)
        pool = lambda values, valid: values.sum(1) / valid.sum(1, keepdim=True).clamp_min(1)
        pooled = torch.cat((pool(latent[:, :p], batch.piece_mask), latent[:, p:p + 64].mean(1),
                            pool(latent[:, p + 64:], batch.action_mask)), 1)
        residual = self.value(pooled).squeeze(1) * self.settings.cp_scale
        value = batch.value_prior_cp + residual
        action_latent = latent[:, p + 64:]
        action_residual = self.action(torch.cat((action_latent, pooled[:, None].expand(-1, k, -1)), -1)).squeeze(-1)
        action_residual = action_residual * self.settings.cp_scale
        prior = torch.where(batch.action_mask, batch.action_prior_cp, 0.)
        result = dict(latent=latent, value_cp=value, residual_cp=residual,
                      action_residual_cp=action_residual.masked_fill(~batch.action_mask, 0.),
                      score_frame=SCORE_FRAME)
        result.update(_action_outputs(prior + action_residual, batch.action_ids, batch.action_mask))
        if return_trace:
            result['trace'] = torch.stack(trace)
        return result

    def exact_child_control(self, parent, children, child_index, *, steps=None, typed=True):
        """Same model re-encodes exact children; negate child-mover CP for parent.

        children contains independently admitted exact successor graphs with
        their COMPLETE child legal sets. child_index[B,K] maps parent slots to
        child rows. This cannot certify that caller supplied the correct states.
        No parent latent, teacher score or future search state is consumed.
        """
        b, _, k, _ = _metadata(parent, self.settings)
        count, _, _, _ = _metadata(children, self.settings)
        if (not isinstance(child_index, torch.Tensor) or child_index.dtype != torch.int64 or
                child_index.shape != (b, k) or child_index.device != parent.pieces.device or
                children.pieces.device != parent.pieces.device):
            raise ValueError('child row mapping must be same-device int64 [B,K]')
        # Mapping content must be admitted separately; no device scalar reads.
        output = self(children, steps=steps, typed=typed)
        values = output['value_cp'][child_index.clamp(0, count - 1)]
        return dict(**_action_outputs(-values, parent.action_ids, parent.action_mask),
                    child_value_cp=values, score_frame=SCORE_FRAME)


def admit_child_mapping(parent, children, child_index, config):
    """CPU structural one-to-one mapping; producer certifies actual successor IDs."""
    b, _, k, _ = _metadata(parent, config)
    count, _, _, _ = _metadata(children, config)
    if (parent.pieces.device.type != 'cpu' or children.pieces.device.type != 'cpu' or
            not isinstance(child_index, torch.Tensor) or child_index.device.type != 'cpu' or
            child_index.dtype != torch.int64 or child_index.shape != (b, k)):
        raise ValueError('CPU int64 [B,K] child mapping required')
    active = child_index[parent.action_mask].tolist()
    if sorted(active) != list(range(count)):
        raise ValueError('each child row must map to exactly one parent legal action')
