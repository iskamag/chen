"""Version-pinned, same-architecture TorchScript static evaluator reference.

The three-logit head serves a bounded search-score utility, not calibrated WDL.
Export is an explicit CPU artifact boundary; resident inference transfers no
per-leaf payload to CPU. Only trusted generated archives may be loaded.
"""
from __future__ import annotations

import argparse
import copy
import ctypes
import hashlib
import json
from pathlib import Path
import warnings

import torch
from torch import nn

from .features import SCHEMA_VERSION, encode_fen
from .models import ChessLatent, MODELS

DEPLOYMENT_VERSION = 1
SCALE_CP = 800.0
INVERSE_EPSILON = 1e-6
SCORE_LIMIT = 29000
MAX_ARCHIVE_BYTES = 64 * 1024 * 1024
MAX_MANIFEST_BYTES = 64 * 1024
MANIFEST_KEY = "chen_manifest.json"
TARGET_KINDS = ("finite_depth_minimax_cp_utility", "material_proxy_cp_utility",
                "untrained_architecture_fixture")


def normalized_scalar(model: ChessLatent, features: torch.Tensor) -> torch.Tensor:
    """Exact existing head readout; loss/draw/win names imply no calibration."""
    probabilities = model.wdl(model.encode(features).mean(1)).softmax(-1)
    return probabilities[:, 2] - probabilities[:, 0]


def scalar_to_cp(value: torch.Tensor) -> torch.Tensor:
    bounded = value.clamp(-1 + INVERSE_EPSILON, 1 - INVERSE_EPSILON)
    return (bounded.atanh() * SCALE_CP).round().clamp(-SCORE_LIMIT, SCORE_LIMIT).to(torch.int32)


class StaticEvaluator(nn.Module):
    """Keep the full trained encoder and scalar head, omit unused training heads."""
    def __init__(self, model: ChessLatent):
        super().__init__()
        self.model = model

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return scalar_to_cp(normalized_scalar(self.model, features))


def state_sha256(model: ChessLatent) -> str:
    digest = hashlib.sha256()
    for name, value in sorted(model.state_dict().items()):
        value = value.detach().cpu().contiguous().clone()
        descriptor = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":"))
        digest.update(descriptor.encode() + b"\0")
        # This is an explicit CPU artifact boundary, never the resident path.
        digest.update(ctypes.string_at(value.data_ptr(), value.numel() * value.element_size()))
    return digest.hexdigest()


def model_config(model: ChessLatent):
    return {"name": model.kind, "width": model.width, "depth": model.depth,
            "plan_tokens": model.plans.shape[0] if model.kind == "plan" else 8}


def _fixtures(batch):
    fens = (
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 99 11",
        "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 7 12",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    )
    return torch.stack([encode_fen(fens[index % len(fens)]) for index in range(batch)])


def export_inference(model, archive, *, target_kind, provenance,
                     max_batch=256, checkpoint_sha256=None):
    """Snapshot unchanged trained tensors; trace/freeze only static evaluation.

    CPU tracing is deliberate. GPU remapping/execution requires independent
    hardware parity evidence before this artifact is used in resident search.
    """
    if not isinstance(model, ChessLatent) or model.kind not in MODELS:
        raise ValueError("export requires a current ChessLatent architecture")
    if target_kind not in TARGET_KINDS:
        raise ValueError("declare a supported utility target kind")
    if type(max_batch) is not int or not 1 <= max_batch <= 4096:
        raise ValueError("max_batch must be an integer in [1,4096]")
    if not isinstance(provenance, dict):
        raise ValueError("explicit training/fixture provenance is required")
    if checkpoint_sha256 is not None and (not isinstance(checkpoint_sha256, str) or
                                         len(checkpoint_sha256) != 64 or
                                         any(c not in "0123456789abcdef" for c in checkpoint_sha256)):
        raise ValueError("invalid checkpoint fingerprint")
    snapshot = copy.deepcopy(model).cpu().eval()
    if any(value.dtype != torch.float32 or not bool(torch.isfinite(value).all())
           for value in snapshot.state_dict().values() if value.is_floating_point()):
        raise ValueError("reference deployment requires finite float32 trained tensors")
    before = state_sha256(snapshot)
    wrapper = StaticEvaluator(snapshot).eval()
    batches = sorted({1, min(3, max_batch), min(17, max_batch)})
    with torch.inference_mode(), warnings.catch_warnings(record=True) as observed:
        warnings.simplefilter("always", torch.jit.TracerWarning)
        traced = torch.jit.trace(wrapper, _fixtures(batches[0]), strict=True,
                                 check_inputs=[(_fixtures(count),) for count in batches[1:]])
        frozen = torch.jit.freeze(traced, optimize_numerics=False)
        for count in batches:
            expected, actual = wrapper(_fixtures(count)), frozen(_fixtures(count))
            if not torch.equal(expected, actual):
                raise ValueError("integer-score trace parity failed")
    if state_sha256(snapshot) != before:
        raise ValueError("export mutated model state")
    metadata = {
        "deployment_version": DEPLOYMENT_VERSION,
        "feature_schema": SCHEMA_VERSION,
        "format": "torchscript_static_reference", "torch_version": str(torch.__version__),
        "model_config": model_config(snapshot), "state_sha256": before,
        "checkpoint_sha256": checkpoint_sha256, "target_kind": target_kind,
        "input_dtype": "float32", "input_shape": ["B", 18, 8, 8],
        "output_dtype": "int32", "output_shape": ["B"], "max_batch": max_batch,
        "perspective": "side_to_move", "head_readout": "softmax_logits_2_minus_0",
        "scale_cp": SCALE_CP, "epsilon": INVERSE_EPSILON,
        "rounding": "nearest_even", "score_limit": SCORE_LIMIT,
        "terminal_responsibility": "simulator", "strength_evidence": False,
        "provenance": provenance,
        "export_validation": {"device": "cpu", "batches": batches,
                              "integer_scores_equal": True,
                              "tracer_warnings": sorted({str(item.message) for item in observed
                                                         if issubclass(item.category, torch.jit.TracerWarning)})},
    }
    content = json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False)
    if len(content.encode()) > MAX_MANIFEST_BYTES:
        raise ValueError("deployment manifest exceeds size limit")
    archive = Path(archive)
    archive.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(frozen, str(archive), _extra_files={MANIFEST_KEY: content})
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        archive.unlink()
        raise ValueError("deployment archive exceeds size limit")
    sidecar = {"manifest": metadata, "manifest_json": content,
               "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest()}
    sidecar_text = json.dumps(sidecar, indent=2, allow_nan=False) + "\n"
    if len(sidecar_text.encode()) > MAX_MANIFEST_BYTES:
        archive.unlink()
        raise ValueError("deployment sidecar exceeds size limit")
    archive.with_suffix(archive.suffix + ".json").write_text(sidecar_text)
    return sidecar


def load_inference(archive, *, expected_sha256, device="cpu"):
    """Load a pinned trusted-generated executable graph; no pickle fallback."""
    archive = Path(archive)
    if archive.stat().st_size > MAX_ARCHIVE_BYTES:
        raise ValueError("deployment archive exceeds size limit")
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if not isinstance(expected_sha256, str) or actual != expected_sha256:
        raise ValueError("deployment archive fingerprint mismatch")
    sidecar_path = archive.with_suffix(archive.suffix + ".json")
    if sidecar_path.stat().st_size > MAX_MANIFEST_BYTES:
        raise ValueError("deployment sidecar exceeds size limit")
    sidecar = json.loads(sidecar_path.read_text())
    if (not isinstance(sidecar, dict) or sidecar.get("archive_sha256") != actual or
        not isinstance(sidecar.get("manifest_json"), str) or
        json.loads(sidecar["manifest_json"]) != sidecar.get("manifest")):
        raise ValueError("deployment sidecar envelope mismatch")
    metadata = sidecar["manifest"]
    _validate_manifest(metadata)
    device = torch.device(device)
    if device.type == "cuda" and device.index is None:
        device = torch.device("cuda", torch.cuda.current_device())
    extra = {MANIFEST_KEY: ""}
    module = torch.jit.load(str(archive), map_location=device, _extra_files=extra).eval()
    embedded = extra[MANIFEST_KEY]
    if isinstance(embedded, bytes):
        embedded = embedded.decode("utf-8")
    if embedded != sidecar["manifest_json"]:
        raise ValueError("embedded deployment manifest mismatch")
    return InferenceSession(module, metadata, device)


def _validate_manifest(metadata):
    fixed = {"deployment_version": 1, "feature_schema": 1,
             "format": "torchscript_static_reference",
             "input_dtype": "float32", "input_shape": ["B", 18, 8, 8],
             "output_dtype": "int32", "output_shape": ["B"],
             "perspective": "side_to_move", "head_readout": "softmax_logits_2_minus_0",
             "scale_cp": SCALE_CP, "epsilon": INVERSE_EPSILON,
             "rounding": "nearest_even", "score_limit": SCORE_LIMIT,
             "terminal_responsibility": "simulator", "strength_evidence": False}
    if not isinstance(metadata, dict) or any(metadata.get(key) != value for key, value in fixed.items()):
        raise ValueError("deployment schema/value contract mismatch")
    if metadata["strength_evidence"] is not False:
        raise ValueError("deployment strength status must be false")
    for key in ("deployment_version", "feature_schema", "score_limit", "max_batch"):
        if type(metadata.get(key)) is not int:
            raise ValueError("deployment integer metadata is malformed")
    if not 1 <= metadata["max_batch"] <= 4096 or metadata.get("target_kind") not in TARGET_KINDS:
        raise ValueError("deployment capacity/target contract mismatch")
    config = metadata.get("model_config")
    if (not isinstance(config, dict) or config.get("name") not in MODELS or
        any(type(config.get(key)) is not int for key in ("width", "depth", "plan_tokens")) or
        not 8 <= config["width"] <= 512 or not 1 <= config["depth"] <= 32 or
        not 1 <= config["plan_tokens"] <= 128):
        raise ValueError("deployment model configuration is malformed")
    if (not isinstance(metadata.get("provenance"), dict) or
        not isinstance(metadata.get("torch_version"), str) or
        not isinstance(metadata.get("state_sha256"), str) or
        len(metadata["state_sha256"]) != 64 or
        any(c not in "0123456789abcdef" for c in metadata["state_sha256"])):
        raise ValueError("deployment provenance/fingerprint is malformed")


class InferenceSession:
    """Metadata-only ingress guards; native score validation stays on device."""
    def __init__(self, module, metadata, device):
        self.module, self.metadata, self.device = module, metadata, device

    @torch.inference_mode()
    def __call__(self, features):
        if (not isinstance(features, torch.Tensor) or features.dtype != torch.float32 or
            features.device != self.device or features.ndim != 4 or
            tuple(features.shape[1:]) != (18, 8, 8) or not features.is_contiguous() or
            not 1 <= features.shape[0] <= self.metadata["max_batch"]):
            raise ValueError("deployment input must be contiguous float32 [B,18,8,8] on its device")
        return self.module(features)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("checkpoint")
    parser.add_argument("archive")
    parser.add_argument("--max-batch", type=int, default=256)
    parser.add_argument("--checkpoint-kind", choices=("teacher", "material-proxy"), default="teacher")
    args = parser.parse_args()
    if args.checkpoint_kind == "teacher":
        from .teacher import load_teacher_checkpoint
        model, metadata = load_teacher_checkpoint(args.checkpoint, device="cpu")
        target_kind = "finite_depth_minimax_cp_utility"
    else:
        from .train import load_checkpoint
        model, metadata = load_checkpoint(args.checkpoint, device="cpu")
        target_kind = "material_proxy_cp_utility"
    # Full teacher lineage stays in the pinned checkpoint and experiment report.
    # Embedding it again can overflow the bounded deployment manifest.
    provenance = {"kind": metadata.get("kind"),
                  "checkpoint_metadata_sha256": hashlib.sha256(json.dumps(metadata, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest(),
                  "utility_contract": metadata.get("utility_contract"),
                  "training_config": metadata.get("training_config")}
    if "teacher_manifest" in metadata:
        teacher = metadata["teacher_manifest"]
        provenance["teacher_manifest"] = {key: teacher[key] for key in
            ("teacher_kind", "tree_depth", "parent_horizon", "evaluator", "scale_cp", "records", "deduplication", "fixture_only") if key in teacher}
        provenance["teacher_manifest_sha256"] = hashlib.sha256(json.dumps(teacher, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()
    result = export_inference(model, args.archive, target_kind=target_kind,
                              provenance=provenance, max_batch=args.max_batch,
                              checkpoint_sha256=hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest())
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
