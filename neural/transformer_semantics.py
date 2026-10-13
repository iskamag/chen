"""Separate multihead semantic controls; not an admitted trained evaluator.

Tensor forward is an arithmetic/training seam. Exact simulator admission owns
legality, history, current-state causality, terminal overrides and GPU lifetime.
No source in the frozen four-encoder study is changed by this module.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import torch
from torch import nn

from .models import ChessLatent
from .semantic_strategy import Diagram, ROLES, SemanticAnchor
from .score_frames import FRAME_KIND, score_frame_manifest
from .value_heads import MATERIAL_VALUES, SCALE_CP, ZeroLinear

KIND = 'chen_multihead_semantic_transformer_v1'


@dataclass(frozen=True)
class TransformerConfig:
    width: int = 64
    heads: int = 4
    depth: int = 2
    expansion: int = 4
    geometry_bias: bool = True
    rule_bias: bool = False
    piece_keys_only: bool = False
    symbolic_channels: int = 0

    def __post_init__(self):
        for name in ('width', 'heads', 'depth', 'expansion', 'symbolic_channels'):
            if type(getattr(self, name)) is not int:
                raise ValueError('transformer dimensions must be integers')
        if (not 8 <= self.width or not 1 <= self.heads or
                self.width % self.heads or not 1 <= self.depth or
                not 1 <= self.expansion <= 8 or not 0 <= self.symbolic_channels <= 64):
            raise ValueError('invalid transformer dimensions')
        if any(type(getattr(self, name)) is not bool for name in
               ('geometry_bias', 'rule_bias', 'piece_keys_only')):
            raise ValueError('transformer controls must be booleans')

    def estimate(self, *, batch_size=1, value_only=True):
        """Pure integer descriptor: never constructs a module or allocates tensors."""
        return estimate_transformer(self, batch_size=batch_size, value_only=value_only)


def estimate_transformer(config, *, batch_size=1, value_only=True):
    """Exact parameter counts and dense matmul MAC accounting, not runtime.

    Counts matrix multiply-accumulates as ONE MAC, not two FLOPs. Normalization,
    activations, softmax, material/evidence production, bias adds, allocator and
    synchronization are excluded. Attention elements are summed logit shapes,
    not a peak memory estimate. Stored FP32 weights exclude buffers/optimizers.
    """
    if (not isinstance(config, TransformerConfig) or type(batch_size) is not int or batch_size < 1 or
            type(value_only) is not bool):
        raise ValueError('configuration, positive integer batch and boolean value_only required')
    d, h, depth, r, c = config.width, config.heads, config.depth, config.expansion, config.symbolic_channels
    biases = (6 * h if config.geometry_bias else 0) + (4 * h if config.rule_bias else 0)
    groups = dict(input_position_ep=d * d + 83 * d,
                  square_blocks=depth * ((4 + 2 * r) * d * d + (r + 9) * d + biases),
                  role_pool=4 * d * d + 18 * d,
                  symbolic_injection=3 * c * d + 3 * d if c else 0,
                  value_head=8 * d + 1, auxiliary_heads=7728 * d + 1414)
    total = sum(groups.values())
    active = total - groups['auxiliary_heads']
    macs = dict(input_position_ep=64 * 18 * d + 64 * d + d * d,
                square_blocks=depth * (64 * (4 + 2 * r) * d * d + 2 * 64 * 64 * d),
                geometry_bias=depth * 64 * 64 * 6 * h if config.geometry_bias else 0,
                rule_bias=depth * 64 * 64 * 4 * h if config.rule_bias else 0,
                role_pool=(2 * 8 + 2 * 64) * d * d + 2 * 8 * 64 * d,
                symbolic_injection=3 * 64 * c * d + 2 * 8 * 64 * d if c else 0,
                value_head=8 * d,
                auxiliary_heads=0 if value_only else 7728 * d)
    logits = depth * h * 64 * 64 + h * 8 * 64 + (8 * 64 if c else 0)
    # Static geometry is computed once and broadcast, unlike batched rule bias.
    batched_macs = {name: count * (1 if name == 'geometry_bias' else batch_size)
                    for name, count in macs.items()}
    return dict(kind=KIND, architecture=asdict(config), stored_parameters=total,
                parameter_groups=groups, value_path_parameters=active,
                forward_parameters=active if value_only else total,
                stored_fp32_parameter_bytes=4 * total, batch_size=batch_size,
                value_only=value_only, forward_matmul_macs=sum(batched_macs.values()),
                matmul_macs_by_group=batched_macs,
                summed_attention_logit_elements=batch_size * logits,
                largest_attention_logit_elements=batch_size * h * 64 * 64,
                token_compaction=False, allocated=False, measured_runtime=None,
                exclusions=['pointwise_norm_activation_softmax', 'material_and_rule_topology_production',
                            'bias_adds', 'buffer_optimizer_activation_memory', 'dispatch_and_synchronization'])


def scaling_profiles():
    """Allocation-free candidate bands by VALUE-path parameters, not trained models."""
    bands = ((1000000, 128, 2, 5), (10000000, 320, 5, 8), (50000000, 640, 10, 10),
             (100000000, 768, 12, 14), (270000000, 1280, 20, 13))
    return [dict(target_value_parameters=target,
                 **TransformerConfig(width=d, heads=h, depth=depth).estimate())
            for target, d, h, depth in bands]


class MultiheadAttention(nn.Module):
    """Explicit heads and linears, usable by separately reviewed affine backends."""
    def __init__(self, width, heads):
        super().__init__()
        if (type(width) is not int or type(heads) is not int or
                width < 1 or heads < 1 or width % heads):
            raise ValueError('attention width must divide into nonempty integer heads')
        self.width, self.heads, self.head_width = width, heads, width // heads
        self.query = nn.Linear(width, width)
        self.key = nn.Linear(width, width)
        self.value = nn.Linear(width, width)
        self.output = nn.Linear(width, width)

    def forward(self, query, context, *, bias=None, keys=None):
        if (not isinstance(query, torch.Tensor) or not isinstance(context, torch.Tensor) or
                query.ndim != 3 or context.ndim != 3 or query.shape[0] != context.shape[0] or
                query.shape[0] == 0 or query.shape[1] == 0 or context.shape[1] == 0 or
                query.shape[2] != self.width or context.shape[2] != self.width or
                query.dtype != context.dtype or query.device != context.device):
            raise ValueError('attention requires compatible nonempty [B,Q,D] and [B,K,D] tensors')
        b, q, _ = query.shape
        k = context.shape[1]
        if bias is not None and (not isinstance(bias, torch.Tensor) or bias.ndim != 4 or
                bias.shape[0] not in (1, b) or bias.shape[1] not in (1, self.heads) or
                tuple(bias.shape[2:]) != (q, k) or bias.dtype != query.dtype or
                bias.device != query.device):
            raise ValueError('attention bias must broadcast only across batch or heads')
        split = lambda values: values.reshape(b, -1, self.heads, self.head_width).transpose(1, 2)
        logits = split(self.query(query)) @ split(self.key(context)).transpose(-1, -2)
        logits = logits / math.sqrt(self.head_width)
        if bias is not None:
            logits = logits + bias
        if keys is not None:
            if (not isinstance(keys, torch.Tensor) or keys.dtype != torch.bool or tuple(keys.shape) != (b, k) or
                    keys.device != query.device):
                raise ValueError('attention keys must be same-device bool [B,K]')
            # Exact exclusion for arbitrary finite logits, without a host read.
            # Real ingress includes kings. Empty arithmetic fixtures yield a
            # zero attention contribution before the output projection bias.
            present = keys.any(-1)[:, None, None, None]
            logits = logits.masked_fill(~keys[:, None, None, :], -torch.inf)
            logits = torch.where(present, logits, torch.zeros_like(logits))
        weights = logits.softmax(-1)
        if keys is not None:
            weights = torch.where(present, weights, torch.zeros_like(weights))
        attended = weights @ split(self.value(context))
        return self.output(attended.transpose(1, 2).reshape(b, q, self.width))


class SquareBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm = nn.LayerNorm(config.width)
        self.attention = MultiheadAttention(config.width, config.heads)
        self.feedforward = nn.Sequential(nn.LayerNorm(config.width),
            nn.Linear(config.width, config.width * config.expansion), nn.GELU(),
            nn.Linear(config.width * config.expansion, config.width))
        self.geometry_weight = (nn.Parameter(torch.zeros(config.heads, 6))
                                if config.geometry_bias else None)
        self.rule_weight = (nn.Parameter(torch.zeros(config.heads, 4))
                            if config.rule_bias else None)

    def forward(self, latent, geometry, rules, keys):
        normalized = self.norm(latent)
        bias = None
        if self.geometry_weight is not None:
            bias = torch.einsum('hc,cij->hij', self.geometry_weight, geometry)[None]
        if self.rule_weight is not None:
            typed = torch.einsum('hc,bcij->bhij', self.rule_weight, rules)
            bias = typed if bias is None else bias + typed
        latent = latent + self.attention(normalized, normalized, bias=bias, keys=keys)
        return latent + self.feedforward(latent)


class TransformerSemantics(nn.Module):
    """Dense square reasoning and an explicit eight-role multihead bottleneck.

    piece_keys_only changes ownership, not tensor compaction: it still computes
    64-square attention. An EP context preserves information on an empty EP square
    when keys are restricted to pieces. Learned symbolic projections enter both
    square and role residual streams and start at zero for shared-weight controls.
    """
    def __init__(self, config=TransformerConfig(), *, material_scale=1., anchor_kind='material',
                 max_parameters=None):
        super().__init__()
        if (not isinstance(config, TransformerConfig) or
                type(material_scale) not in (int, float) or not math.isfinite(material_scale) or
                not 0 < material_scale <= 10 or anchor_kind not in ('material', 'zero')):
            raise ValueError('invalid semantic transformer configuration')
        # Caller resource policy, checked BEFORE Diagram/buffers/weights allocate.
        # None is uncapped; there is no implied small-model architectural ceiling.
        if (max_parameters is not None and
                (type(max_parameters) is not int or max_parameters < 1)):
            raise ValueError('max_parameters must be a positive integer or None')
        if max_parameters is not None and config.estimate()['stored_parameters'] > max_parameters:
            raise ValueError('declared parameter allocation budget exceeded')
        self.max_parameters = max_parameters
        self.settings = config
        self.width, self.depth = config.width, config.depth
        self.material_scale, self.anchor_kind = float(material_scale), anchor_kind
        self.diagram = Diagram()
        self.register_buffer('geometry', ChessLatent._geometry())
        self.input = nn.Linear(18, self.width)
        self.position = nn.Parameter(torch.randn(64, self.width) * .02)
        self.ep_context = nn.Linear(self.width, self.width, bias=False)
        self.layers = nn.ModuleList(SquareBlock(config) for _ in range(config.depth))
        self.square_norm = nn.LayerNorm(self.width)
        self.queries = nn.Parameter(torch.randn(8, self.width) * .02)
        self.role_norm = nn.LayerNorm(self.width)
        self.role_attention = MultiheadAttention(self.width, config.heads)
        self.role_output_norm = nn.LayerNorm(self.width)
        if config.symbolic_channels:
            self.symbolic_square = ZeroLinear(config.symbolic_channels, self.width)
            self.symbolic_key = nn.Linear(config.symbolic_channels, self.width)
            self.symbolic_value = ZeroLinear(config.symbolic_channels, self.width)
        self.board = nn.Linear(8 * self.width, 64 * 13)
        self.maps = nn.ModuleList(nn.Linear(self.width, 64) for _ in ROLES)
        self.rights = nn.Linear(8 * self.width, 4)
        self.ep = nn.Linear(8 * self.width, 65)
        self.clock = nn.Linear(8 * self.width, 1)
        self.cp = ZeroLinear(8 * self.width, 1)

    def _inputs(self, features, symbols):
        if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or
                features.ndim != 4 or tuple(features.shape[1:]) != (18, 8, 8) or
                features.shape[0] == 0 or features.device != self.position.device or
                self.position.dtype != torch.float32):
            raise ValueError('expected same-device FP32 schema1 features [B,18,8,8]')
        channels = self.settings.symbolic_channels
        if channels:
            if (not isinstance(symbols, torch.Tensor) or symbols.dtype != torch.float32 or
                    tuple(symbols.shape) != (features.shape[0], 64, channels) or
                    symbols.device != features.device):
                raise ValueError('configured symbolic transformer requires current-state [B,64,C] channels')
        elif symbols is not None:
            raise ValueError('symbolic channels supplied to a no-injection control')

    def encode(self, features, symbolic_channels=None):
        self._inputs(features, symbolic_channels)
        occupied = features[:, :12].flatten(2).sum(1) > 0
        latent = self.input(features.flatten(2).transpose(1, 2)) + self.position
        # Empty-square EP information is available in both ownership controls.
        ep = features[:, 16].flatten(1) @ self.position
        latent = latent + self.ep_context(ep)[:, None]
        rules = self.diagram.typed_edges(features[:, :12]) if self.settings.rule_bias else None
        keys = occupied if self.settings.piece_keys_only else None
        if symbolic_channels is not None:
            latent = latent + self.symbolic_square(symbolic_channels)
        for layer in self.layers:
            latent = layer(latent, self.geometry, rules, keys)
        squares = self.square_norm(latent)
        queries = self.queries[None].expand(features.shape[0], -1, -1)
        roles = queries + self.role_attention(self.role_norm(queries), squares, keys=keys)
        if symbolic_channels is not None:
            weights = (queries @ self.symbolic_key(symbolic_channels).transpose(-1, -2) /
                       math.sqrt(self.width)).softmax(-1)
            roles = roles + weights @ self.symbolic_value(symbolic_channels)
        return self.role_output_norm(roles)

    def decode(self, latent, pieces, *, value_only=False):
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (8, self.width):
            raise ValueError('expected semantic roles [B,8,D]')
        flat = latent.flatten(1)
        material = self.diagram.material(pieces) * self.material_scale
        residual = self.cp(flat).squeeze(-1) * SCALE_CP
        cp = residual + material if self.anchor_kind == 'material' else residual
        output = dict(latent=latent, pieces=pieces, value_cp=cp,
                      residual_cp=residual, material_cp=material)
        if not value_only:
            output.update(utility=(cp / SCALE_CP).tanh(),
                board_logits=self.board(flat).reshape(-1, 13, 8, 8),
                map_logits=torch.stack([head(latent[:, i]) for i, head in enumerate(self.maps)], 1).reshape(-1, 8, 8, 8),
                castling_logits=self.rights(flat), ep_logits=self.ep(flat),
                halfmove=self.clock(flat).squeeze(-1).sigmoid())
        return output

    def forward(self, features, symbolic_channels=None, *, value_only=False):
        return self.decode(self.encode(features, symbolic_channels), features[:, :12], value_only=value_only)

    def forward_symbolic(self, state, *, value_only=False):
        # Lazy import keeps this candidate outside the frozen old source closure.
        from .symbolic_residual import SymbolicState, CompactSymbolicState
        if not isinstance(state, (SymbolicState, CompactSymbolicState)):
            raise ValueError('requires the typed admitted current-state symbolic seam')
        state.validate_metadata()
        return self(state.features, state.channels(), value_only=value_only)

    def semantic_targets(self, features):
        return SemanticAnchor.semantic_targets(self, features)

    def active_parameters(self):
        return self.parameters()

    def parameter_report(self, *, batch_size=1, value_only=True):
        report = self.settings.estimate(batch_size=batch_size, value_only=value_only)
        return dict(report, allocated=True, actual_stored_parameters=sum(p.numel() for p in self.parameters()),
                    max_parameters=self.max_parameters)

    def copy_shared_state(self, source):
        """Copy same-meaning controls before fitting; never a training shortcut."""
        if not isinstance(source, TransformerSemantics):
            raise ValueError('shared control must be a semantic transformer')
        left, right = asdict(self.settings), asdict(source.settings)
        for key in ('symbolic_channels', 'rule_bias', 'piece_keys_only'):
            left.pop(key)
            right.pop(key)
        if (left != right or self.material_scale != source.material_scale or
                self.anchor_kind != source.anchor_kind):
            raise ValueError('shared controls require matching tensor meaning and configuration')
        owned = self.state_dict()
        copies = []
        for name, tensor in source.state_dict().items():
            if name in owned:
                if tensor.shape != owned[name].shape or tensor.dtype != owned[name].dtype:
                    raise ValueError('shared-state tensor shape or dtype mismatch')
                copies.append((owned[name], tensor))
        with torch.no_grad():
            for target, tensor in copies:
                target.copy_(tensor)

    def config(self):
        return dict(kind=KIND, feature_schema=1, **asdict(self.settings),
            roles=list(ROLES), material_scale=self.material_scale, anchor_kind=self.anchor_kind,
            material_values=list(MATERIAL_VALUES), scale_cp=SCALE_CP,
            target_kind=FRAME_KIND, score_frame=score_frame_manifest(),
            token_ownership='occupied_piece_keys' if self.settings.piece_keys_only else 'all_64_square_keys',
            symbolic_token_ownership='all_64_squares' if self.settings.symbolic_channels else 'absent',
            token_compaction=False, ep_context='position_weighted_exact_EP_plane',
            symbolic_injection='zero_initial_square_and_eight_role_residual_projections',
            allocation_policy=dict(max_parameters=self.max_parameters),
            terminal_responsibility='exact_simulator', deployment_admitted=False)
