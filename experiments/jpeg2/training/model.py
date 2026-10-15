"""Direct typed-channel action policy and independently fitted mechanism controls."""
import torch
from torch import nn

FORMAT = "jpeg2-channel-policy-f32-v1"
PAIR_NAMES = [
    ("slider_lines", "king_blocks"), ("destination_balance", "pressure"),
    ("action", "knight_pawn"), ("action", "slider_lines"),
    ("action", "pressure"), ("material", "pressure"),
    ("knight_pawn", "king_blocks"),
]


class ChannelPolicy(nn.Module):
    def __init__(self, channels, dim, architecture="interaction", mean=None, inverse_std=None):
        super().__init__()
        self.architecture = architecture
        self.dim = dim
        self.channels = list(channels)
        if architecture == "flat":
            self.channels = [{"name": "flat", "start": 0, "end": dim}]
        widths = {"interaction": 32, "additive": 124 if dim == 320 else 138,
                  "flat": 129 if dim == 320 else 144}
        self.width = widths[architecture]
        self.register_buffer("mean", torch.zeros(dim) if mean is None else mean.clone())
        self.register_buffer("inverse_std", torch.ones(dim) if inverse_std is None else inverse_std.clone())
        self.projections = nn.ModuleList([
            nn.Linear(c["end"] - c["start"], self.width) for c in self.channels
        ])
        self.base = nn.Linear(len(self.channels) * self.width, 1)
        names = [c["name"] for c in self.channels]
        self.pairs = [(names.index(a), names.index(b)) for a, b in PAIR_NAMES] if architecture == "interaction" else []
        self.residual = None
        if architecture == "interaction":
            self.residual = nn.Sequential(
                nn.Linear((len(self.channels) + len(self.pairs)) * self.width, 64),
                nn.ReLU(), nn.Linear(64, 1),
            )
            nn.init.zeros_(self.residual[2].weight)
            nn.init.zeros_(self.residual[2].bias)

    def forward(self, x, parts=False):
        x = ((x - self.mean) * self.inverse_std).clamp(-8, 8)
        groups = [torch.relu(p(x[..., c["start"]:c["end"]])) for p, c in zip(self.projections, self.channels)]
        base = self.base(torch.cat(groups, dim=-1)).squeeze(-1)
        full = base
        if self.residual is not None:
            joint = torch.cat(groups + [groups[a] * groups[b] for a, b in self.pairs], dim=-1)
            full = base + self.residual(joint).squeeze(-1)
        return (full, base) if parts else full

    def export(self, metadata=None):
        def linear(layer):
            return {"weight": layer.weight.detach().tolist(), "bias": layer.bias.detach().tolist()}
        residual = {"enabled": False, "weight1": [], "bias1": [], "weight2": [], "bias2": 0.0}
        if self.residual is not None:
            residual = {
                "enabled": True,
                "weight1": self.residual[0].weight.detach().tolist(),
                "bias1": self.residual[0].bias.detach().tolist(),
                "weight2": self.residual[2].weight[0].detach().tolist(),
                "bias2": self.residual[2].bias[0].item(),
            }
        return {
            "format": FORMAT, "dim": self.dim, "channels": self.channels,
            "feature_schema_sha256": (metadata or {}).get("feature_schema_sha256"),
            "architecture": self.architecture,
            "normalization": {"mean": self.mean.tolist(), "inverse_std": self.inverse_std.tolist(), "clip": 8.0},
            "projection_width": self.width, "projections": [linear(p) for p in self.projections],
            "base": {"weight": self.base.weight[0].detach().tolist(), "bias": self.base.bias[0].item()},
            "interaction_pairs": self.pairs, "residual": residual,
            "activation": "relu", "output": "base_plus_residual",
            "metadata": metadata or {},
        }


def exported_forward(model, features):
    """Independent NumPy evaluator of the actual deployment JSON layout."""
    import numpy as np
    x = np.asarray(features, dtype=np.float32)
    norm = model["normalization"]
    x = np.clip((x - np.asarray(norm["mean"], dtype=np.float32)) * np.asarray(norm["inverse_std"], dtype=np.float32), -8, 8)
    groups = []
    for c, p in zip(model["channels"], model["projections"]):
        groups.append(np.maximum(0, x[..., c["start"]:c["end"]] @ np.asarray(p["weight"], dtype=np.float32).T + np.asarray(p["bias"], dtype=np.float32)))
    base = np.concatenate(groups, axis=-1) @ np.asarray(model["base"]["weight"], dtype=np.float32) + model["base"]["bias"]
    if not model["residual"]["enabled"]:
        return base, base
    r = model["residual"]
    joint = np.concatenate(groups + [groups[a] * groups[b] for a, b in model["interaction_pairs"]], axis=-1)
    hidden = np.maximum(0, joint @ np.asarray(r["weight1"], dtype=np.float32).T + np.asarray(r["bias1"], dtype=np.float32))
    return base + hidden @ np.asarray(r["weight2"], dtype=np.float32) + r["bias2"], base
