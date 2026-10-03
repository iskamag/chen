"""Pinned SF19 UCI score-frame conversion, not a strength/WDL conversion.

An ordinary reported score is rounded after root-material normalization. Recover
an interval in reference-material58 CP-like units, never an exact hidden value.
Mate/tablebase/bounded scores must be rejected by the producer before ingress.
"""
from __future__ import annotations

import math

import torch
from torch.nn import functional as F

FRAME_KIND = 'sf19_reference_material58_approx_cp'
AS = (-142.72052667, 372.35176398, -340.71073572, 415.23490212)
PHASE_WEIGHTS = (1, 3, 3, 5, 9, 0)
REFERENCE_MATERIAL = 58
UCI_SOURCE_SHA256 = 'a31fa10719b1d9ecaa23f774f14a5a995be65eeb52722798c2f2bb4255991b21'
SCORE_SOURCE_SHA256 = 'c5877628d70984fef345f2ace3f155ea883f316505485f2115701aac313dfb01'
UCI_SOURCE_URL = 'https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_19/src/uci.cpp'
SCORE_SOURCE_URL = 'https://raw.githubusercontent.com/official-stockfish/Stockfish/sf_19/src/score.cpp'
NUMERIC_GUARD = 1e-9


def _a(material):
    m = min(max(material, 17), 78) / 58.0
    return ((AS[0] * m + AS[1]) * m + AS[2]) * m + AS[3]


REFERENCE_A = _a(REFERENCE_MATERIAL)


def score_frame_manifest():
    """Immutable math/provenance contract; copied into producer/checkpoints."""
    return dict(frame_version=1, kind=FRAME_KIND, stockfish_tag='sf_19',
                uci_source_url=UCI_SOURCE_URL, uci_source_sha256=UCI_SOURCE_SHA256,
                score_source_url=SCORE_SOURCE_URL, score_source_sha256=SCORE_SOURCE_SHA256,
                material_weights=list(PHASE_WEIGHTS), material_clamp=[17, 78],
                material_divisor=58.0, normalization_coefficients=list(AS),
                reference_material=REFERENCE_MATERIAL, reference_a=REFERENCE_A,
                source_rounding='std_round_nearest_half_away_from_zero',
                recovery='closed_half_uci_unit_interval_not_exact_hidden_value',
                numeric_guard_relative=NUMERIC_GUARD, batch_midpoint_dtype='float32',
                batch_halfwidth='float32_outward_including_midpoint_cast_error',
                perspective='caller_declared_query_root_mover_or_negation',
                required_source_kind='ordinary_unbounded_cp_no_tablebase')


def sf19_reference_score(reported_cp, root_material, *, score_kind, negate=False):
    """Cold primitive record from ACTUAL QUERY ROOT total phase material.

    Parent MultiPV uses parent's root material even when the score describes a
    child; convert first, negate for child perspective second. Standalone child
    queries use child's material. The returned interval includes ±0.5 UCI unit.
    """
    if (type(reported_cp) is not int or not -(2**31) <= reported_cp < 2**31 or
            type(root_material) is not int or not 0 <= root_material <= 576 or
            score_kind != 'ordinary_unbounded_cp_no_tablebase' or type(negate) is not bool):
        raise ValueError('ordinary unbounded SF19 integer CP/root material required')
    factor = _a(root_material) / REFERENCE_A
    value = reported_cp * factor * (-1 if negate else 1)
    width = .5 * factor
    guard = NUMERIC_GUARD * (abs(value) + width + 1)
    return dict(frame_kind=FRAME_KIND, reported_uci_cp=reported_cp,
                normalization_root_material=root_material, normalization_a=_a(root_material),
                reference_material=58, factor=factor, negated=negate,
                value_cp=value, quantization_halfwidth=width,
                interval_halfwidth=width + guard,
                lower_cp=value - width - guard, upper_cp=value + width + guard,
                source_score_kind=score_kind)


def sf19_reference_batch(reported_cp, normalization_root_features, *, score_kind, negate=False):
    """Resident tensor conversion; content admission is producer responsibility.

    reported_cp float32[B] must contain admitted ordinary integer UCI scores.
    root features are EXACTLY the queried SF position's root, not a PV child.
    No host scalar extraction. Return float32 midpoint+outward interval width,
    float64 interval bounds, and integer total material for audit sidecars.
    """
    features = normalization_root_features
    if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or features.ndim != 4 or
            tuple(features.shape[1:]) != (18, 8, 8) or not isinstance(reported_cp, torch.Tensor) or
            reported_cp.dtype != torch.float32 or reported_cp.shape != (features.shape[0],) or
            reported_cp.device != features.device or type(negate) is not bool or
            score_kind != 'ordinary_unbounded_cp_no_tablebase'):
        raise ValueError('expected admitted float32 CP[B]/query roots[B,18,8,8]')
    weights = torch.tensor(PHASE_WEIGHTS * 2, device=features.device, dtype=torch.float64)
    material = (features[:, :12].sum((-1, -2)).double() * weights).sum(-1)
    m = material.clamp(17, 78) / 58.0
    a = ((AS[0] * m + AS[1]) * m + AS[2]) * m + AS[3]
    factor = a / REFERENCE_A
    exact_midpoint = reported_cp.double() * factor * (-1 if negate else 1)
    midpoint = exact_midpoint.float()
    halfwidth = .5 * factor
    guard = NUMERIC_GUARD * (exact_midpoint.abs() + halfwidth + 1)
    total_width = halfwidth + (midpoint.double() - exact_midpoint).abs() + guard
    width = torch.nextafter(total_width.float(), torch.full_like(midpoint, float('inf')))
    return dict(value_cp=midpoint, interval_halfwidth=width,
                lower_cp=midpoint.double() - width.double(), upper_cp=midpoint.double() + width.double(),
                quantization_halfwidth=halfwidth, reported_uci_cp=reported_cp,
                normalization_root_material=material.to(torch.int64), factor=factor)


def interval_smooth_l1(predicted_cp, target_cp, interval_halfwidth, scale=800.0):
    """Per-row normalized loss: zero inside the declared score interval.

    This models score reporting uncertainty only, not search error/confidence.
    Resident caller certifies finite targets, nonnegative widths and source kind.
    """
    if (not all(isinstance(v, torch.Tensor) for v in (predicted_cp, target_cp, interval_halfwidth)) or
            predicted_cp.shape != target_cp.shape or target_cp.shape != interval_halfwidth.shape or
            predicted_cp.device != target_cp.device or target_cp.device != interval_halfwidth.device or
            type(scale) not in (int, float) or not math.isfinite(scale) or scale <= 0):
        raise ValueError('same-shape resident score/target/width and positive scale required')
    distance = ((predicted_cp - target_cp).abs() - interval_halfwidth).clamp_min(0) / scale
    return F.smooth_l1_loss(distance, torch.zeros_like(distance), reduction='none')
