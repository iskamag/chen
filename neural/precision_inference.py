"""Owned FP16 and static graph research controls; no deployment admission.

FP16 affine control rounds operands and affine outputs to FP16. Surrounding
attention, norms, residuals and chess algebra stay FP32. Torch GEMM accumulation
is its producing backend's policy, not a promise of native FP32 accumulation.
"""
from __future__ import annotations

import ctypes
import hashlib
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .int8_inference import admit_gfx1030, loaded_hip_runtime, value_only_copy
from .models import ChessLatent
from .semantic_strategy import SemanticAnchor


class HalfLinear(nn.Module):
    def __init__(self, source):
        super().__init__()
        if (not isinstance(source, nn.Linear) or source.weight.dtype != torch.float32 or
                source.weight.device.type != 'cuda' or source.training or
                any(p.requires_grad for p in source.parameters())):
            raise ValueError('frozen admitted FP32 GPU affine required')
        if any(not bool(torch.isfinite(p).all()) for p in source.parameters()):
            raise ValueError('finite affine parameters required')
        self.in_features, self.out_features = source.in_features, source.out_features
        weight = source.weight.detach().to(torch.float16).contiguous()
        bias = None if source.bias is None else source.bias.detach().to(torch.float16).contiguous()
        if not bool(torch.isfinite(weight).all()) or (bias is not None and not bool(torch.isfinite(bias).all())):
            raise ValueError('FP16 parameter overflow')
        self.register_buffer('weight', weight)
        self.register_buffer('bias', bias)

    def forward(self, values):
        if (values.device != self.weight.device or values.device.type != 'cuda' or
                values.dtype != torch.float32 or values.ndim < 1 or
                values.shape[-1] != self.in_features):
            raise ValueError('resident FP32 affine input required')
        return F.linear(values.to(torch.float16), self.weight, self.bias).float()


class NativeHalfBackend:
    """Pinned wave32 FDOT2 affine: FP16 operands, FP32 accumulation/output."""
    def __init__(self, library, digest, *, runtime_path, runtime_sha256, device='cuda:0'):
        if hashlib.sha256(Path(library).read_bytes()).hexdigest() != digest:
            raise ValueError('FP16 native library pin mismatch')
        self.device, self.admission = admit_gfx1030(device)
        runtime = loaded_hip_runtime()
        if runtime['path'] != runtime_path or runtime['sha256'] != runtime_sha256:
            raise ValueError('actual FP16 HIP provider changed')
        self.runtime = runtime
        self.library = ctypes.CDLL(str(library))
        self.linear = self.library.chen_float_linear
        self.linear.argtypes = [ctypes.c_void_p] * 4 + [ctypes.c_uint64] * 4 + [ctypes.c_void_p]
        self.linear.restype = ctypes.c_int32

    def __call__(self, rows, weight, bias, output_width, padded_width):
        if (rows.device != self.device or rows.dtype != torch.float32 or rows.ndim != 2 or
                rows.shape[0] > 1048576 or not 1 <= rows.shape[1] <= 4096 or
                not 1 <= output_width <= 4096 or padded_width != (rows.shape[1] + 3) // 4 * 4 or
                weight.device != self.device or weight.dtype != torch.float16 or
                weight.shape != (padded_width // 2, (output_width + 31) // 32 * 64) or
                not weight.is_contiguous() or weight.data_ptr() % 4):
            raise ValueError('invalid native FP16 affine metadata')
        if bias is not None and (bias.device != self.device or bias.dtype != torch.float32 or
                                 bias.shape != (output_width,) or not bias.is_contiguous()):
            raise ValueError('native FP16 bias metadata mismatch')
        activation = F.pad(rows, (0, padded_width - rows.shape[1])).to(torch.float16).contiguous()
        output = rows.new_empty((rows.shape[0], output_width))
        if rows.shape[0]:
            stream = torch.cuda.current_stream(self.device)
            status = self.linear(activation.data_ptr(), weight.data_ptr(),
                                 None if bias is None else bias.data_ptr(), output.data_ptr(),
                                 rows.shape[0], output_width, padded_width, 1, stream.cuda_stream)
            if status:
                raise RuntimeError(f'native FP16 affine launch failed: {status}')
            for tensor in (activation, weight, bias, output):
                if tensor is not None:
                    tensor.record_stream(stream)
        return output


def pack_half_weight(weight):
    """Pure layout fixture seam: CPU permitted, no native runtime fallback."""
    if weight.dtype != torch.float32 or weight.ndim != 2 or min(weight.shape) == 0:
        raise ValueError('FP32 [N,K] weight required')
    n, k = weight.shape
    padded = F.pad(weight, (0, (-k) % 4, 0, (-n) % 32)).to(torch.float16)
    return padded.reshape(padded.shape[0], -1, 2).permute(1, 0, 2).contiguous().flatten(1)


class NativeHalfLinear(nn.Module):
    def __init__(self, source, backend):
        super().__init__()
        if (not isinstance(source, nn.Linear) or source.weight.dtype != torch.float32 or
                source.weight.device != backend.device or source.training or
                any(p.requires_grad for p in source.parameters())):
            raise ValueError('frozen native FP16 source required')
        if any(not bool(torch.isfinite(p).all()) for p in source.parameters()):
            raise ValueError('finite source required')
        self.in_features, self.out_features = source.in_features, source.out_features
        self.padded_width = (self.in_features + 3) // 4 * 4
        self.backend = backend
        packed = pack_half_weight(source.weight.detach())
        if not bool(torch.isfinite(packed).all()):
            raise ValueError('FP16 weight overflow')
        self.register_buffer('weight', packed)
        self.register_buffer('bias', None if source.bias is None else source.bias.detach().clone())

    def forward(self, values):
        if values.ndim < 1 or values.shape[-1] != self.in_features:
            raise ValueError('native FP16 input width mismatch')
        output = self.backend(values.reshape(-1, self.in_features), self.weight, self.bias,
                              self.out_features, self.padded_width)
        return output.reshape(*values.shape[:-1], self.out_features)


def mixed_fp16_copy(source, backend=None):
    """Independent frozen value-only copy, with explicit active affine ownership."""
    admit_gfx1030(next(source.parameters()).device)
    owned = value_only_copy(source)
    names = []

    def replace(parent, prefix='', anchor=False):
        for name, child in list(parent.named_children()):
            path = f'{prefix}.{name}' if prefix else name
            if anchor and name in {'board', 'maps', 'rights', 'ep', 'clock'}:
                continue
            if isinstance(parent, ChessLatent) and name not in {
                    'input', 'square_layers', 'relation_layers', 'plan_query',
                    'plan_key', 'plan_value', 'plan_output', 'latent_norm'}:
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, name, HalfLinear(child) if backend is None else NativeHalfLinear(child, backend))
                names.append(path)
            else:
                replace(child, path, isinstance(child, SemanticAnchor))

    replace(owned, anchor=isinstance(owned, SemanticAnchor))
    return owned, tuple(names)


class StaticCapturedInference:
    """Single-device, fixed-shape graph with owned resident inputs and outputs.

    Outputs are borrowed: the next replay overwrites them. Callers must consume
    or clone before another replay. Input-copy cost is a separate timed route;
    this class never constructs features or transfers CPU inputs in a forward.
    Capture failure is an error, never a hidden eager fallback.
    """
    def __init__(self, call, features, actions=None, *, warmups=3):
        device, self.admission = admit_gfx1030(features.device)
        if (features.device != device or features.dtype != torch.float32 or
                not features.is_contiguous() or features.ndim != 4 or
                tuple(features.shape[1:]) != (18, 8, 8) or features.shape[0] == 0 or
                type(warmups) is not int or not 1 <= warmups <= 16):
            raise ValueError('fixed nonempty resident feature shape required')
        if actions is not None and (actions.device != device or actions.dtype != torch.int64 or
                                    not actions.is_contiguous() or actions.ndim != 2 or
                                    actions.shape[0] != features.shape[0]):
            raise ValueError('same-device fixed complete-root actions required')
        self.features = features.clone()
        self.actions = None if actions is None else actions.clone()
        self.call = call  # Retain frozen model and all captured weight storage.
        self.device = device
        self.stream = torch.cuda.Stream(device=device)
        self.stream.wait_stream(torch.cuda.current_stream(device))
        with torch.cuda.stream(self.stream), torch.inference_mode():
            for _ in range(warmups):
                call(self.features, self.actions)
        torch.cuda.current_stream(device).wait_stream(self.stream)
        torch.cuda.synchronize(device)
        self.graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(self.graph, stream=self.stream):
            self.outputs = call(self.features, self.actions)
        torch.cuda.synchronize(device)

    def replay(self):
        if torch.cuda.current_device() != self.device.index:
            raise ValueError('captured device changed')
        self.graph.replay()
        return self.outputs

    def copy_and_replay(self, features, actions=None):
        if (features.device != self.device or features.dtype != self.features.dtype or
                features.shape != self.features.shape or not features.is_contiguous()):
            raise ValueError('captured feature identity changed')
        if self.actions is None:
            if actions is not None:
                raise ValueError('anchor graph has no actions')
        elif (actions is None or actions.device != self.device or actions.dtype != self.actions.dtype or
              actions.shape != self.actions.shape or not actions.is_contiguous()):
            raise ValueError('captured action identity changed')
        self.features.copy_(features)
        if actions is not None:
            self.actions.copy_(actions)
        return self.replay()
