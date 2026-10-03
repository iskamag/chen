"""Controlled chess latent models with shared transition and output heads.

Every model accepts schema-v1 [B,18,8,8] features. Square latents have
[B,64,width] shape. The transition changes orientation to the NEXT mover.
It predicts learned state; callers must use real chess legality outside it.
"""
from __future__ import annotations

import math
import torch
from torch import nn
from .features import ACTION_COUNT, FEATURE_PLANES

MODELS = ("square", "relation", "plan")


class RelationBlock(nn.Module):
    """Dense square attention conditioned on geometric and pseudoattack edges."""
    def __init__(self, width: int):
        super().__init__()
        self.norm = nn.LayerNorm(width)
        self.qkv = nn.Linear(width, width * 3)
        self.out = nn.Linear(width, width)
        self.ff = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.geometry_weight = nn.Parameter(torch.zeros(6))
        self.attack_weight = nn.Parameter(torch.tensor(0.1))

    def forward(self, latent: torch.Tensor, geometry: torch.Tensor, attacks: torch.Tensor) -> torch.Tensor:
        query, key, value = self.qkv(self.norm(latent)).chunk(3, -1)
        logits = query @ key.transpose(-1, -2) / math.sqrt(query.shape[-1])
        logits = logits + torch.einsum("c,cij->ij", self.geometry_weight, geometry)
        logits = logits + self.attack_weight * attacks
        latent = latent + self.out(logits.softmax(-1) @ value)
        return latent + self.ff(latent)


class ChessLatent(nn.Module):
    def __init__(self, kind: str, width: int = 64, depth: int = 2, plan_tokens: int = 8):
        super().__init__()
        if kind not in MODELS or width < 8 or depth < 1 or plan_tokens < 1:
            raise ValueError("invalid model kind, width, depth, or plan-token count")
        self.kind, self.width, self.depth = kind, width, depth
        self.input = nn.Linear(FEATURE_PLANES, width)
        self.position = nn.Parameter(torch.randn(64, width) * 0.02)
        self.square_layers = nn.ModuleList([
            nn.Sequential(nn.LayerNorm(width), nn.Linear(width, width * 2), nn.GELU(), nn.Linear(width * 2, width))
            for _ in range(depth)
        ]) if kind == "square" else nn.ModuleList()
        self.relation_layers = nn.ModuleList([RelationBlock(width) for _ in range(depth)]) if kind != "square" else nn.ModuleList()
        if kind == "plan":
            self.plans = nn.Parameter(torch.randn(plan_tokens, width) * 0.02)
            self.plan_query = nn.Linear(width, width)
            self.plan_key = nn.Linear(width, width)
            self.plan_value = nn.Linear(width, width)
            self.plan_output = nn.Linear(width, width)
        self.latent_norm = nn.LayerNorm(width)
        self.promotion = nn.Embedding(5, width)
        self.action_condition = nn.Sequential(nn.Linear(width * 3, width), nn.GELU(), nn.Linear(width, width))
        self.dynamics = nn.Sequential(nn.LayerNorm(width * 2 + 2), nn.Linear(width * 2 + 2, width * 2), nn.GELU(), nn.Linear(width * 2, width))
        self.wdl = nn.Linear(width, 3)
        self.board = nn.Linear(width, 12)
        self.castling = nn.Linear(width, 4)
        self.en_passant = nn.Linear(width, 1)
        self.no_en_passant = nn.Linear(width, 1)
        self.halfmove = nn.Linear(width, 1)
        rank = max(8, width // 2)
        self.policy_source = nn.Linear(width, rank)
        self.policy_target = nn.Linear(width, rank)
        self.promo_source = nn.Linear(width, 5)
        self.promo_target = nn.Linear(width, 5)
        self.register_buffer("geometry", self._geometry())
        self.register_buffer("attack_patterns", self._attack_patterns())
        self.register_buffer("between", self._between())
        self.register_buffer("sliding_mask", torch.tensor([0, 0, 1, 1, 1, 0] * 2).float(), persistent=False)
        self.register_buffer("next_orientation", torch.arange(64) ^ 56)

    @staticmethod
    def _geometry() -> torch.Tensor:
        square = torch.arange(64)
        df = (square[:, None] % 8 - square[None, :] % 8).abs()
        dr = (square[:, None] // 8 - square[None, :] // 8).abs()
        return torch.stack((df == 0, dr == 0, df == dr, (df * dr == 2) & (df + dr == 3), torch.maximum(df, dr) == 1, square[:, None] == square[None, :])).float()

    @staticmethod
    def _attack_patterns() -> torch.Tensor:
        square = torch.arange(64)
        df = square[None, :] % 8 - square[:, None] % 8
        dr = square[None, :] // 8 - square[:, None] // 8
        diagonal = (df.abs() == dr.abs()) & (df != 0)
        straight = ((df == 0) | (dr == 0)) & ((df != 0) | (dr != 0))
        knight = (df.abs() * dr.abs() == 2) & (df.abs() + dr.abs() == 3)
        king = torch.maximum(df.abs(), dr.abs()) == 1
        return torch.stack([(df.abs() == 1) & (dr == 1), knight, diagonal, straight, diagonal | straight, king,
                            (df.abs() == 1) & (dr == -1), knight, diagonal, straight, diagonal | straight, king]).float()

    @staticmethod
    def _between() -> torch.Tensor:
        # Between[source,target,blocker] for rays; excludes both endpoints.
        result = torch.zeros(64, 64, 64)
        for source in range(64):
            for target in range(64):
                df, dr = target % 8 - source % 8, target // 8 - source // 8
                if source == target or not (df == 0 or dr == 0 or abs(df) == abs(dr)):
                    continue
                step_f, step_r = (df > 0) - (df < 0), (dr > 0) - (dr < 0)
                for distance in range(1, max(abs(df), abs(dr))):
                    blocker = source + distance * (step_f + 8 * step_r)
                    result[source, target, blocker] = 1
        return result

    def attack_edges(self, features: torch.Tensor) -> torch.Tensor:
        """Pseudoattacks with exact occupancy blockers; pins/check legality excluded."""
        pieces = features[:, :12].flatten(2)
        rays = torch.einsum("bs,ijs->bij", pieces.sum(1), self.between)
        open_rays = (rays == 0).to(features.dtype)
        patterns = self.attack_patterns.to(features.dtype).unsqueeze(0)
        sliding = self.sliding_mask.to(features.dtype).view(1, 12, 1, 1)
        patterns = patterns * ((1 - sliding) + sliding * open_rays[:, None])
        return torch.einsum("bps,bpst->bst", pieces, patterns)

    def encode(self, features: torch.Tensor) -> torch.Tensor:
        if features.ndim != 4 or tuple(features.shape[1:]) != (FEATURE_PLANES, 8, 8):
            raise ValueError("expected features [B,18,8,8]")
        latent = self.input(features.flatten(2).transpose(1, 2)) + self.position
        if self.kind == "square":
            for layer in self.square_layers:
                latent = latent + layer(latent + latent.mean(1, keepdim=True))
        else:
            attacks = self.attack_edges(features)
            for layer in self.relation_layers:
                latent = layer(latent, self.geometry, attacks)
        if self.kind == "plan":
            plans = self.plans.unsqueeze(0).expand(latent.shape[0], -1, -1)
            weights = (self.plan_query(plans) @ self.plan_key(latent).transpose(-1, -2) / math.sqrt(self.width)).softmax(-1)
            plans = plans + weights @ self.plan_value(latent)
            back_weights = (latent @ plans.transpose(-1, -2) / math.sqrt(self.width)).softmax(-1)
            latent = latent + self.plan_output(back_weights @ plans)
        return self.latent_norm(latent)

    def transition(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (64, self.width) or actions.shape != (latent.shape[0],) or actions.dtype != torch.long:
            raise ValueError("expected latent [B,64,D] and int64 actions [B]")
        pair, promotion = actions.div(5, rounding_mode="floor"), actions.remainder(5)
        source, target = pair.div(64, rounding_mode="floor"), pair.remainder(64)
        batch = torch.arange(latent.shape[0], device=latent.device)
        condition = self.action_condition(torch.cat((latent[batch, source], latent[batch, target], self.promotion(promotion)), -1))
        squares = torch.arange(64, device=latent.device)
        source_mask = (squares[None] == source[:, None]).to(latent.dtype).unsqueeze(-1)
        target_mask = (squares[None] == target[:, None]).to(latent.dtype).unsqueeze(-1)
        # A learned residual operator can update capture, rights, EP and castles
        # globally. Spatial source/target flags preserve move locality explicitly.
        inputs = torch.cat((latent, condition[:, None].expand(-1, 64, -1), source_mask, target_mask), -1)
        successor = latent + self.dynamics(inputs)
        return self.latent_norm(successor[:, self.next_orientation])

    def decode_board(self, latent: torch.Tensor) -> torch.Tensor:
        return self.board(latent).transpose(1, 2).reshape(-1, 12, 8, 8)

    def decode_rules(self, latent: torch.Tensor) -> dict[str, torch.Tensor]:
        pooled = latent.mean(1)
        return {"castling_logits": self.castling(pooled),
                "ep_logits": torch.cat((self.en_passant(latent).squeeze(-1), self.no_en_passant(pooled)), -1),
                "halfmove": self.halfmove(pooled).squeeze(-1).sigmoid()}

    def branch(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Expand B positions over K caller-provided legal moves, in one batch."""
        if actions.ndim != 2 or actions.shape[0] != latent.shape[0]:
            raise ValueError("expected branching actions [B,K]")
        batch, candidates = actions.shape
        expanded = latent[:, None].expand(-1, candidates, -1, -1).reshape(batch * candidates, 64, self.width)
        return self.transition(expanded, actions.reshape(-1)).reshape(batch, candidates, 64, self.width)

    def rollout(self, latent: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Return root plus H learned successors. Actions use each ply's frame."""
        if actions.ndim != 2 or actions.shape[0] != latent.shape[0]:
            raise ValueError("expected rollout actions [B,H]")
        states = [latent]
        for ply in range(actions.shape[1]):
            states.append(self.transition(states[-1], actions[:, ply]))
        return torch.stack(states, 1)

    def decode(self, latent: torch.Tensor, legal_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        source, target = self.policy_source(latent), self.policy_target(latent)
        pair = source @ target.transpose(-1, -2) / math.sqrt(source.shape[-1])
        promotions = self.promo_source(latent)[:, :, None, :] + self.promo_target(latent)[:, None, :, :]
        logits = (pair.unsqueeze(-1) + promotions).reshape(latent.shape[0], ACTION_COUNT)
        if legal_mask is not None:
            if legal_mask.shape != logits.shape or legal_mask.dtype != torch.bool:
                raise ValueError("expected boolean legal mask [B,20480]")
            logits = logits.masked_fill(~legal_mask, torch.finfo(logits.dtype).min)
        return {"latent": latent, "wdl_logits": self.wdl(latent.mean(1)), "policy_logits": logits,
                "board_logits": self.decode_board(latent), **self.decode_rules(latent)}

    def forward(self, features: torch.Tensor, legal_mask: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        return self.decode(self.encode(features), legal_mask)


def build_model(name: str, width: int = 64, depth: int = 2, **kwargs: int) -> ChessLatent:
    return ChessLatent(name, width=width, depth=depth, **kwargs)
