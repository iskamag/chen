"""Explicit value adapters for the unchanged Chen latent encoders.

Float value_cp is a differentiable finite-search heuristic, not calibrated WDL.
The material anchor is a declared handcrafted prior. Legality and terminal
adjudication remain simulator responsibilities. These artifacts are not accepted
by the existing deployment_version1 three-logit inference loader.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch
from torch import nn

from .features import SCHEMA_VERSION
from .models import ChessLatent, MODELS, build_model, checkpoint_buffers

HEAD_VERSION = 1
HEADS = ("three_logit", "direct_cp", "material_residual")
SCALE_CP = 800.0
INVERSE_EPSILON = 1e-6
STATIC_LIMIT = 29000
MATERIAL_VALUES = (100, 320, 330, 500, 900, 0)
CHECKPOINT_KIND = "chen_value_adapter_checkpoint"
CHECKPOINT_LIMIT = 128 * 1024 * 1024


def _features(features):
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or
        features.ndim != 4 or tuple(features.shape[1:]) != (18, 8, 8)):
        raise ValueError("value input must be schema1 float32 [B,18,8,8]")


def material_cp(features):
    """Schema1 material, exactly CPU-search/HIP static material on binary boards.

    Friendly planes already use the current-mover frame. Rights/raw EP/clocks
    have no material contribution. The static clamp matches search and HIP;
    Rust Material's raw evaluator is unclamped before search applies that bound.
    Metadata guards never extract tensor contents to host.
    """
    _features(features)
    weights = features.new_tensor(MATERIAL_VALUES)
    counts = features[:, :12].sum((-1, -2))
    return ((counts[:, :6] - counts[:, 6:12]) * weights).sum(-1).clamp(-STATIC_LIMIT, STATIC_LIMIT)


class ZeroLinear(nn.Linear):
    """A scalar readout whose initialization consumes no random numbers."""
    def reset_parameters(self):
        nn.init.zeros_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)


def _base_config(base):
    return {"name": base.kind, "width": base.width, "depth": base.depth,
            "plan_tokens": base.plans.shape[0] if base.kind == "plan" else 8}


def _validate_base_config(config):
    if (not isinstance(config, dict) or set(config) != {"name", "width", "depth", "plan_tokens"} or
        config["name"] not in MODELS):
        raise ValueError("invalid value-adapter encoder configuration")
    for name, low, high in (("width", 8, 512), ("depth", 1, 32), ("plan_tokens", 1, 128)):
        if type(config[name]) is not int or not low <= config[name] <= high:
            raise ValueError(f"value-adapter {name} outside supported bounds")


class ValueAdapter(nn.Module):
    """Take ownership of a base model; callers clone it for independent variants.

    All heads read the same mean pooled encode(features). The original control
    head retains its existing weights. New CP readouts start at zero. This makes
    direct_cp start at 0cp and material_residual start at the material prior.
    The first update learns the new readout; encoder gradients begin after its
    weights become nonzero. No policy or latent transition loss is implicit.
    """
    def __init__(self, base: ChessLatent, head_kind: str):
        super().__init__()
        if not isinstance(base, ChessLatent) or head_kind not in HEADS:
            raise ValueError("adapter requires a current encoder and supported head")
        if base.position.dtype != torch.float32:
            raise ValueError("value adapter requires a float32 encoder")
        _validate_base_config(_base_config(base))
        self.base, self.head_kind = base, head_kind
        self.head = None if head_kind == "three_logit" else ZeroLinear(base.width, 1)
        # Follow a pre-existing base device without initializing any GPU itself.
        self.register_buffer("material_values", base.position.detach().new_tensor(MATERIAL_VALUES))
        if self.head is not None:
            self.head.to(device=base.position.device, dtype=base.position.dtype)

    def encode(self, features):
        _features(features)
        return self.base.encode(features)

    def forward(self, features):
        latent = self.encode(features)
        pooled = latent.mean(1)
        counts = features[:, :12].sum((-1, -2))
        anchor = ((counts[:, :6] - counts[:, 6:12]) * self.material_values).sum(-1).clamp(-STATIC_LIMIT, STATIC_LIMIT)
        if self.head_kind == "three_logit":
            probabilities = self.base.wdl(pooled).softmax(-1)
            utility = probabilities[:, 2] - probabilities[:, 0]
            cp = utility.clamp(-1 + INVERSE_EPSILON, 1 - INVERSE_EPSILON).atanh() * SCALE_CP
            residual = cp - anchor
        else:
            predicted = self.head(pooled).squeeze(-1) * SCALE_CP
            cp = anchor + predicted if self.head_kind == "material_residual" else predicted
            utility = (cp / SCALE_CP).tanh()
            residual = cp - anchor
        return {"value_cp": cp, "utility": utility, "material_cp": anchor,
                "residual_cp": residual}

    def static_scores(self, features):
        """Nearest-even integer heuristic; simulator supplies all terminal scores."""
        return self(features)["value_cp"].round().clamp(-STATIC_LIMIT, STATIC_LIMIT).to(torch.int32)

    def active_parameters(self):
        """Optimizer parameters for this value-only experiment, with no dead heads."""
        prefixes = ("input.", "square_layers.", "relation_layers.", "plan_", "latent_norm.")
        for name, parameter in self.base.named_parameters():
            if name in ("position", "plans") or name.startswith(prefixes):
                yield parameter
            elif self.head_kind == "three_logit" and name.startswith("wdl."):
                yield parameter
        if self.head is not None:
            yield from self.head.parameters()

    def config(self):
        return {"head_version": HEAD_VERSION, "feature_schema": SCHEMA_VERSION,
                "model_config": _base_config(self.base), "head_kind": self.head_kind,
                "scale_cp": SCALE_CP, "inverse_epsilon": INVERSE_EPSILON,
                "residual_scale_cp": SCALE_CP, "material_values": list(MATERIAL_VALUES),
                "new_head_initialization": "zero_without_rng", "perspective": "side_to_move",
                "pooling": "mean64", "float_cp_clamped": False,
                "static_limit": STATIC_LIMIT, "static_rounding": "nearest_even",
                "target_kind": "finite_depth_minimax_cp_regression",
                "terminal_responsibility": "simulator"}


def build_value_adapter(model_config, head_kind):
    _validate_base_config(model_config)
    return ValueAdapter(build_model(**model_config), head_kind)


def _primitive_metadata(metadata):
    if not isinstance(metadata, dict):
        raise ValueError("explicit primitive provenance dictionary is required")
    try:
        json.dumps(metadata, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("nonprimitive/nonfinite value-adapter provenance") from error


def save_value_checkpoint(adapter, path, provenance):
    """Separate tensor artifact; original encoders and deterministic buffers retained."""
    if not isinstance(adapter, ValueAdapter):
        raise ValueError("expected ValueAdapter")
    _primitive_metadata(provenance)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"checkpoint_version": 1, "kind": CHECKPOINT_KIND,
               "strength_evidence": False, "adapter_config": adapter.config(),
               "provenance": provenance,
               "state_dict": {name: value.detach().cpu().clone() for name, value in adapter.state_dict().items()}}
    torch.save(payload, path)
    if path.stat().st_size > CHECKPOINT_LIMIT:
        path.unlink()
        raise ValueError("value-adapter checkpoint exceeds128MiB")
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "kind": CHECKPOINT_KIND, "strength_evidence": False}


def load_value_checkpoint(path, device="cpu"):
    """Restricted CPU-first tensor loading; exact config/state/buffer validation.

    This is a trusted local checkpoint boundary. weights_only does not promise
    safety against every malicious resource-exhaustion archive.
    """
    path = Path(path)
    if path.stat().st_size > CHECKPOINT_LIMIT:
        raise ValueError("value-adapter checkpoint exceeds128MiB")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as error:
        raise ValueError("restricted value-adapter loading failed") from error
    required = {"checkpoint_version", "kind", "strength_evidence", "adapter_config", "provenance", "state_dict"}
    if (not isinstance(payload, dict) or set(payload) != required or
        type(payload["checkpoint_version"]) is not int or payload["checkpoint_version"] != 1 or
        payload["kind"] != CHECKPOINT_KIND or payload["strength_evidence"] is not False):
        raise ValueError("invalid value-adapter checkpoint envelope")
    config = payload["adapter_config"]
    if not isinstance(config, dict):
        raise ValueError("invalid adapter configuration")
    _validate_base_config(config.get("model_config"))
    adapter = build_value_adapter(config["model_config"], config.get("head_kind"))
    if (type(config.get("head_version")) is not int or
        type(config.get("feature_schema")) is not int or
        not isinstance(config.get("material_values"), list) or
        any(type(value) is not int for value in config["material_values"]) or
        config.get("float_cp_clamped") is not False or config != adapter.config()):
        raise ValueError("unsupported value-adapter head/schema/semantics")
    _primitive_metadata(payload["provenance"])
    state = payload["state_dict"]
    expected_state = adapter.state_dict()
    if not isinstance(state, dict) or set(state) != set(expected_state):
        raise ValueError("value-adapter state keys mismatch")
    for name, expected in expected_state.items():
        value = state[name]
        if not isinstance(value, torch.Tensor) or value.shape != expected.shape or value.dtype != expected.dtype:
            raise ValueError(f"value-adapter tensor schema mismatch: {name}")
        if value.is_floating_point() and not bool(torch.isfinite(value).all()):
            raise ValueError(f"value-adapter nonfinite tensor: {name}")
    for name, expected in checkpoint_buffers(adapter):
        if not torch.equal(state[name], expected):
            raise ValueError(f"value-adapter deterministic buffer mismatch: {name}")
    adapter.load_state_dict(state, strict=True)
    adapter.to(device).eval()
    return adapter, {key: value for key, value in payload.items() if key != "state_dict"}
