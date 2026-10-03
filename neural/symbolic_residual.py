"""Causal current-state symbolic channels injected into eight learned roles.

The exact rules producer owns attack/pin/check/history semantics. This seam
checks tensor metadata without reconstructing state or extracting values to
CPU during forward. A separately reviewed ingress must establish the producer
certificate, finite/binary values and matching state context. A certificate
string here is an identity binding, never a proof of its contents.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import re

import torch
from torch import nn

from .int8_inference import value_only_copy
from .semantic_strategy import SemanticAnchor
from .value_heads import ZeroLinear

KIND = 'chen_current_state_symbolic_residual_v1'
HISTORY_FIELDS = ('halfmove_clock', 'current_repetition_count',
                  'reversible_history_length', 'ply_from_search_root')
CHANNELS = ('own_incoming_pseudoattacks', 'enemy_incoming_pseudoattacks',
            'own_outgoing_pseudoattacks', 'enemy_outgoing_pseudoattacks',
            'own_pinned', 'enemy_pinned', 'own_king_checkers', 'phase', *HISTORY_FIELDS)


@dataclass(frozen=True)
class SymbolicState:
    """Only information available at the authoritative evaluated state.

    All squares/colors already share schema1's current-mover orientation.
    pseudoattacks[B,color,source,target] are occupancy-blocked, not legality.
    pinned[B,color,square] and checkers[B,square] come from the exact simulator;
    checkers identify enemy pieces checking the mover king. History includes
    this current state and is tied to its search epoch/context, not its labels.
    Phase is the simulator's normalized material phase in [0,1].

    This evaluated state may be an exact child node already produced by legal
    search. Hypothetical transition studies retain their distinct parent/action
    contract. Future search-control residuals require a new typed version with
    explicit dependencies; this V1 never accepts score/PV/teacher dictionaries.
    """
    features: torch.Tensor
    pseudoattacks: torch.Tensor
    pinned: torch.Tensor
    checkers: torch.Tensor
    phase: torch.Tensor
    history: torch.Tensor
    context_sha256: str
    producer_sha256: str
    epoch: int

    def validate_metadata(self):
        if (not isinstance(self.features, torch.Tensor) or self.features.ndim != 4 or
                tuple(self.features.shape[1:]) != (18, 8, 8)):
            raise ValueError('symbolic state requires schema1 [B,18,8,8]')
        b, device = self.features.shape[0], self.features.device
        if device.type != 'cuda' or b == 0:
            raise ValueError('symbolic state must remain on GPU')
        for value, dtype, shape in (
                (self.features, torch.float32, (b, 18, 8, 8)),
                (self.pseudoattacks, torch.bool, (b, 2, 64, 64)),
                (self.pinned, torch.bool, (b, 2, 64)),
                (self.checkers, torch.bool, (b, 64)),
                (self.phase, torch.float32, (b,)),
                (self.history, torch.int64, (b, 4))):
            if (not isinstance(value, torch.Tensor) or value.dtype != dtype or
                    tuple(value.shape) != shape or value.device != device or
                    not value.is_contiguous()):
                raise ValueError('symbolic field dtype/shape/device/ownership mismatch')
        if (type(self.epoch) is not int or self.epoch < 0 or
                any(not isinstance(pin, str) or not re.fullmatch('[0-9a-f]{64}', pin)
                    for pin in (self.context_sha256, self.producer_sha256))):
            raise ValueError('symbolic state requires exact context/producer/epoch identities')

    def channels(self):
        self.validate_metadata()
        # Clipping counts is a declared normalization, never a legality change.
        incoming = self.pseudoattacks.sum(-2).clamp_max(8).float() / 8
        outgoing = self.pseudoattacks.sum(-1).clamp_max(8).float() / 8
        history_scale = self.features.new_tensor((100., 3., 256., 256.))
        history = (self.history.float() / history_scale).clamp(0, 1)
        return torch.cat((incoming.transpose(1, 2), outgoing.transpose(1, 2),
                          self.pinned.transpose(1, 2).float(),
                          self.checkers[:, :, None].float(),
                          self.phase[:, None, None].expand(-1, 64, 1),
                          history[:, None].expand(-1, 64, -1)), -1)


@dataclass(frozen=True)
class CompactSymbolicState:
    """Same twelve channels, computed directly by the admitted native producer.

    Avoids materializing/reducing a dense pseudoattack graph during inference.
    History retains raw counters; normalized_channels includes their declared
    clipped broadcast. The producer owns value/identity certification; metadata
    checks do not independently establish contents or current arena selection.
    Snapshots remain owned and valid after the producing arena advances/closes.
    """
    features: torch.Tensor
    normalized_channels: torch.Tensor
    history: torch.Tensor
    position_hash: torch.Tensor
    node_id: torch.Tensor
    root_epoch: torch.Tensor
    context_sha256: str
    producer_sha256: str
    epoch: int

    def validate_metadata(self):
        if (not isinstance(self.features, torch.Tensor) or self.features.ndim != 4 or
                tuple(self.features.shape[1:]) != (18, 8, 8) or not self.features.shape[0]):
            raise ValueError('compact symbolic state requires nonempty schema1 features')
        b, device = self.features.shape[0], self.features.device
        if device.type != 'cuda':
            raise ValueError('compact symbolic state must remain on GPU')
        for value, dtype, shape in (
                (self.features, torch.float32, (b, 18, 8, 8)),
                (self.normalized_channels, torch.float32, (b, 64, len(CHANNELS))),
                (self.history, torch.int64, (b, 4)),
                (self.position_hash, torch.int64, (b,)),
                (self.node_id, torch.int64, (b,)),
                (self.root_epoch, torch.int64, (b,))):
            if (not isinstance(value, torch.Tensor) or value.dtype != dtype or
                    tuple(value.shape) != shape or value.device != device or
                    not value.is_contiguous()):
                raise ValueError('compact symbolic field dtype/shape/device mismatch')
        if (type(self.epoch) is not int or not 0 <= self.epoch <= 2**64 - 1 or
                any(not isinstance(pin, str) or not re.fullmatch('[0-9a-f]{64}', pin)
                    for pin in (self.context_sha256, self.producer_sha256))):
            raise ValueError('compact symbolic context/producer/epoch identity required')

    def channels(self):
        self.validate_metadata()
        return self.normalized_channels


class SymbolicResidualAnchor(nn.Module):
    """Separate trainable architecture; actual symbols affect learned roles.

    The value projection starts at zero for exact frozen-checkpoint parity.
    TRAIN updates can learn role-specific pin/check/attack/history corrections;
    attention keys and value projection are real forward dependencies. Training
    this new seam requires its own declaration and paired no-injection control.
    No fitting or training occurs in construction or forward.
    """
    def __init__(self, anchor):
        super().__init__()
        if not isinstance(anchor, SemanticAnchor):
            raise ValueError('symbolic residual requires an admitted semantic anchor')
        self.anchor = value_only_copy(anchor)
        self.key = nn.Linear(len(CHANNELS), anchor.width)
        self.value = ZeroLinear(len(CHANNELS), anchor.width)
        self.to(device=anchor.queries.device, dtype=torch.float32)

    def forward(self, state):
        if not isinstance(state, (SymbolicState, CompactSymbolicState)):
            raise ValueError('symbolic residual accepts the typed current-state seam only')
        symbols = state.channels()
        if state.features.device != self.anchor.queries.device:
            raise ValueError('symbolic state/model device mismatch')
        latent = self.anchor.encode(state.features)
        weights = (self.anchor.queries[None] @ self.key(symbols).transpose(-1, -2) /
                   math.sqrt(self.anchor.width)).softmax(-1)
        residual = weights @ self.value(symbols)
        output = self.anchor.decode(latent + residual, state.features[:, :12])
        output['symbolic_role_residual'] = residual
        return output

    def config(self):
        return dict(kind=KIND, anchor=self.anchor.config(), channels=list(CHANNELS),
                    history_fields=list(HISTORY_FIELDS),
                    producer='exact_current_state_gpu_simulator_under_separate_admission',
                    identity='producer_sha256_context_sha256_epoch',
                    injection='eight_role_query_attention_then_add_symbolic_value_projection',
                    normalization='counts_clip8_div8_history_clip1_phase_normalized',
                    initialization='zero_symbolic_value_projection_preserves_frozen_anchor',
                    teacher_or_future_target_input=False)
