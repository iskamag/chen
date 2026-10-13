"""Separate mixed INT8 inference candidate; never a CPU deployment fallback.

Dynamic row scaling is input arithmetic, not calibration on heldout examples.
Any future fixed clipping/calibration must have a TRAIN-only declaration.
Attention, normalization, nonlinearities and exact chess algebra remain FP32.
This candidate is not accepted by Chen's deployment_version1 loader.
"""
from __future__ import annotations

import copy
import ctypes
import hashlib
import os
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .semantic_strategy import SemanticAnchor, SemanticTransition
from .models import ChessLatent
from .value_heads import SCALE_CP

KIND = 'chen_semantic_mixed_int8_dynamic_rows_v1'
MAX_K = (2**31 - 1) // (127 * 127)
NATIVE_MAX_K = 4096
NATIVE_MAX_N = 4096
NATIVE_MAX_M = 1048576


def symmetric_rows(values):
    """Quantize finite FP32 rows; CPU is allowed only for arithmetic fixtures.

    Zero rows use scale one, so zero heads stay exactly zero. Round-to-even
    and the symmetric [-127,127] range are part of this variant's contract.
    Metadata/finite admission happens at ingress, outside the timed forward.
    """
    if values.dtype != torch.float32 or values.ndim != 2 or values.shape[1] == 0:
        raise ValueError('expected finite float32 [rows,K]')
    maximum = values.abs().amax(-1)
    scale = torch.where(maximum == 0, torch.ones_like(maximum), maximum * (1. / 127))
    # Tiny nonzero FP32 values can underflow on division; preserve finite math.
    scale = scale.clamp_min(torch.finfo(torch.float32).tiny)
    packed = (values / scale[:, None]).round().clamp(-127, 127).to(torch.int8)
    return F.pad(packed, (0, (-values.shape[1]) % 4)).contiguous(), scale.contiguous()


def _gpu_tensor(tensor, dtype, shape, device):
    if (not isinstance(tensor, torch.Tensor) or tensor.dtype != dtype or
            tuple(tensor.shape) != tuple(shape) or tensor.device != device or
            device.type != 'cuda' or not tensor.is_contiguous() or tensor.data_ptr() % 4):
        raise ValueError('INT8 backend requires owned contiguous CUDA tensors')


def admit_gfx1030(device='cuda:0'):
    """Enforce actual ROCm/device identity, never an override or CPU backend."""
    admitted = torch.device(device)
    if (admitted.type != 'cuda' or torch.version.hip is None or
            os.environ.get('HSA_OVERRIDE_GFX_VERSION') or not torch.cuda.is_available()):
        raise ValueError('actual non-overridden gfx1030 ROCm device required')
    if admitted.index is None:
        admitted = torch.device('cuda', torch.cuda.current_device())
    properties = torch.cuda.get_device_properties(admitted)
    if properties.gcnArchName.split(':')[0] != 'gfx1030':
        raise ValueError('INT8 candidate requires actual gfx1030')
    return admitted, dict(architecture=properties.gcnArchName, name=properties.name,
                          torch_version=torch.__version__, rocm_build=torch.version.hip,
                          override=False)


def loaded_hip_runtime():
    """Bind the HIP implementation actually mapped into this Torch process."""
    paths = {str(Path(line.split()[-1]).resolve()) for line in
             Path('/proc/self/maps').read_text().splitlines()
             if 'libamdhip64.so' in line and line.split()[-1].startswith('/')}
    if len(paths) != 1:
        raise ValueError('exactly one loaded HIP runtime implementation required')
    path = paths.pop()
    library = ctypes.CDLL(path)
    version = ctypes.c_int()
    function = library.hipRuntimeGetVersion
    function.argtypes = [ctypes.POINTER(ctypes.c_int)]
    function.restype = ctypes.c_int
    if function(ctypes.byref(version)) != 0 or version.value <= 0:
        raise ValueError('HIP runtime version admission failed')

    class DlInfo(ctypes.Structure):
        _fields_ = [('filename', ctypes.c_char_p), ('base', ctypes.c_void_p),
                    ('symbol', ctypes.c_char_p), ('address', ctypes.c_void_p)]

    dladdr = ctypes.CDLL(None).dladdr
    dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
    dladdr.restype = ctypes.c_int
    info = DlInfo()
    if not dladdr(ctypes.cast(function, ctypes.c_void_p), ctypes.byref(info)):
        raise ValueError('HIP runtime symbol owner admission failed')
    owner = str(Path(info.filename.decode()).resolve())
    if owner != path:
        raise ValueError('HIP runtime symbol owner differs from mapped implementation')
    return dict(path=path, sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                hipRuntimeGetVersion=version.value, symbol_owner=owner,
                symbol=None if info.symbol is None else info.symbol.decode())


class NativeInt8Backend:
    """Hash-pinned HIP C ABI, current Torch stream; no synchronization/fallback.

    ABI: int32 chen_int8_linear(A,W,activation_scale,weight_scale,bias,Y,
                              uint64 M,uint64 N,uint64 Kpad,hipStream_t).
    All allocations are retained on the stream via record_stream, including
    temporary quantized inputs. An immediate HIP error fails this invocation.
    Asynchronous faults are surfaced by the caller's normal completion fence.
    """
    def __init__(self, library, sha256, device='cuda:0', *, runtime_path=None, runtime_sha256=None):
        path = Path(library).resolve(strict=True)
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha256:
            raise ValueError('INT8 native library hash mismatch')
        self.device, self.device_identity = admit_gfx1030(device)
        self.sha256 = sha256
        self.library = ctypes.CDLL(str(path))
        self.runtime_identity = loaded_hip_runtime()
        if (runtime_path is None or runtime_sha256 is None or
                self.runtime_identity['path'] != str(Path(runtime_path).resolve()) or
                self.runtime_identity['sha256'] != runtime_sha256):
            raise ValueError('actual loaded HIP runtime differs from frozen admission')
        self.launch = self.library.chen_int8_linear
        self.launch.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_uint64] * 3 +
                                [ctypes.c_void_p])
        self.launch.restype = ctypes.c_int32
        self.pack = self.library.chen_int8_pack_weights
        self.pack.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_uint64,
                              ctypes.c_uint64, ctypes.c_void_p]
        self.pack.restype = ctypes.c_int32
        self.launch_packed = self.library.chen_int8_linear_packed
        self.launch_packed.argtypes = self.launch.argtypes
        self.launch_packed.restype = ctypes.c_int32

    def prepare_weight(self, weight, *, force=False):
        """One-time coalesced int32[Kpad/4,Npad] weight packing for N>=32."""
        n, k = weight.shape
        _gpu_tensor(weight, torch.int8, (n, k), weight.device)
        if weight.device != self.device:
            raise ValueError('INT8 weight device mismatch')
        if k == 0 or k % 4 or k > NATIVE_MAX_K or n > NATIVE_MAX_N:
            raise ValueError('invalid packed INT8 weight width')
        if n < 32 and not force:
            return weight
        with torch.cuda.device(weight.device):
            stream = torch.cuda.current_stream(weight.device)
            packed = torch.empty((k // 4, (n + 31) & ~31), device=weight.device,
                                 dtype=torch.int32)
            weight.record_stream(stream)
            packed.record_stream(stream)
            status = self.pack(weight.data_ptr(), packed.data_ptr(), n, k, stream.cuda_stream)
            if status != 0:
                raise RuntimeError(f'INT8 weight pack failed: {status}')
        return packed

    def __call__(self, activation, weight, activation_scale, weight_scale, bias):
        if activation.ndim != 2 or weight.ndim != 2:
            raise ValueError('INT8 GEMM requires matrices')
        m, k = activation.shape
        n = weight_scale.shape[0]
        device = activation.device
        if device != self.device or os.environ.get('HSA_OVERRIDE_GFX_VERSION'):
            raise ValueError('INT8 admitted device/override changed')
        if not (0 < m <= NATIVE_MAX_M and 0 < n <= NATIVE_MAX_N and
                0 < k <= NATIVE_MAX_K and k % 4 == 0):
            raise ValueError('invalid INT8 dimensions or int32 accumulator bound')
        packed = weight.dtype == torch.int32
        weight_shape = (k // 4, (n + 31) & ~31) if packed else (n, k)
        for value, dtype, shape in ((activation, torch.int8, (m, k)),
                                   (weight, torch.int32 if packed else torch.int8, weight_shape),
                                   (activation_scale, torch.float32, (m,)),
                                   (weight_scale, torch.float32, (n,)),
                                   (bias, torch.float32, (n,))):
            _gpu_tensor(value, dtype, shape, device)
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            output = torch.empty((m, n), dtype=torch.float32, device=device)
            for value in (activation, weight, activation_scale, weight_scale, bias, output):
                value.record_stream(stream)
            launch = self.launch_packed if packed else self.launch
            status = launch(*(value.data_ptr() for value in
                                 (activation, weight, activation_scale, weight_scale, bias, output)),
                                 m, n, k, stream.cuda_stream)
            if status != 0:
                raise RuntimeError(f'INT8 HIP launch failed: {status}')
        return output


class NativeDynamicInt8Backend(NativeInt8Backend):
    """V3 fused dynamic quantization/dot/dequant, with unpadded source rows.

    Policy is identical to symmetric_rows: zero->1, otherwise maximum times
    FP32(1/127), clamped to FP32tiny; true division then round-even and clamp.
    This API needs a separately pinned, parity-validated v3 native library.
    """
    def __init__(self, library, sha256, device='cuda:0', *, runtime_path=None, runtime_sha256=None,
                 wider_rows=None):
        super().__init__(library, sha256, device, runtime_path=runtime_path,
                         runtime_sha256=runtime_sha256)
        if wider_rows is not None and (type(wider_rows) is not int or wider_rows <= 0):
            raise ValueError('positive declared FP32 wider-row threshold required')
        self.wider_rows = wider_rows
        self.launch_rows = self.library.chen_int8_linear_dynamic_packed_rows
        self.launch_rows.argtypes = ([ctypes.c_void_p] * 6 + [ctypes.c_uint64] * 4 +
                                     [ctypes.c_void_p])
        self.launch_rows.restype = ctypes.c_int32

    def prepare_weight(self, weight):
        return super().prepare_weight(weight, force=True)

    def dynamic_linear(self, values, weight, weight_scale, bias):
        m, source_k = values.shape
        n = weight_scale.shape[0]
        k = (source_k + 3) & ~3
        device = values.device
        if (device != self.device or os.environ.get('HSA_OVERRIDE_GFX_VERSION') or
                not (0 < m <= NATIVE_MAX_M and 0 < n <= NATIVE_MAX_N and
                     0 < source_k <= NATIVE_MAX_K)):
            raise ValueError('invalid admitted INT8 rows device/dimensions')
        for value, dtype, shape in ((values, torch.float32, (m, source_k)),
                                   (weight, torch.int32, (k // 4, (n + 31) & ~31)),
                                   (weight_scale, torch.float32, (n,)),
                                   (bias, torch.float32, (n,))):
            _gpu_tensor(value, dtype, shape, device)
        with torch.cuda.device(device):
            stream = torch.cuda.current_stream(device)
            output = torch.empty((m, n), dtype=torch.float32, device=device)
            for value in (values, weight, weight_scale, bias, output):
                value.record_stream(stream)
            status = self.launch_rows(values.data_ptr(), weight.data_ptr(), weight_scale.data_ptr(),
                                      bias.data_ptr(), None, output.data_ptr(),
                                      m, n, source_k, k, stream.cuda_stream)
            if status != 0:
                raise RuntimeError(f'INT8 dynamic rows launch failed: {status}')
        return output


class PackedInt8Linear(nn.Module):
    """Frozen linear: INT8 dot, int32 accumulation, fused FP32 dequant+bias."""
    def __init__(self, source, backend):
        super().__init__()
        if (not isinstance(source, nn.Linear) or source.weight.device.type != 'cuda' or
                source.weight.dtype != torch.float32 or source.in_features > NATIVE_MAX_K or
                source.out_features > NATIVE_MAX_N):
            raise ValueError('INT8 conversion requires admitted frozen FP32 GPU linear')
        if not torch.isfinite(source.weight).all().item() or (source.bias is not None and
                not torch.isfinite(source.bias).all().item()):
            raise ValueError('nonfinite INT8 source weights')
        self.in_features, self.out_features = source.in_features, source.out_features
        self.backend = backend
        self.wider_rows = getattr(backend, 'wider_rows', None)
        self.register_buffer('fp32_weight', None if self.wider_rows is None else source.weight.detach().clone())
        weight, scale = symmetric_rows(source.weight.detach())
        weight = backend.prepare_weight(weight)
        self.register_buffer('packed_weight', weight)
        self.register_buffer('weight_scale', scale)
        self.register_buffer('bias', (source.bias.detach().clone() if source.bias is not None else
                                    source.weight.new_zeros(source.out_features)))

    def forward(self, values):
        if (not isinstance(values, torch.Tensor) or values.dtype != torch.float32 or
                values.device != self.packed_weight.device or values.ndim < 1 or
                values.shape[-1] != self.in_features):
            raise ValueError('INT8 linear input dtype/device/shape mismatch')
        rows = values.reshape(-1, self.in_features)
        if rows.shape[0] == 0:
            return values.new_empty((*values.shape[:-1], self.out_features))
        if self.wider_rows is not None and rows.shape[0] >= self.wider_rows:
            return F.linear(rows, self.fp32_weight, self.bias).reshape(*values.shape[:-1], self.out_features)
        if hasattr(self.backend, 'dynamic_linear'):
            output = self.backend.dynamic_linear(rows.contiguous(), self.packed_weight,
                                                 self.weight_scale, self.bias)
        else:
            activation, scale = symmetric_rows(rows)
            output = self.backend(activation, self.packed_weight, scale, self.weight_scale, self.bias)
        _gpu_tensor(output, torch.float32, (rows.shape[0], self.out_features), values.device)
        return output.reshape(*values.shape[:-1], self.out_features)


class ValueOnlyAnchor(SemanticAnchor):
    """Exact FP32 value readout; omits board/rule decoder work used in training."""
    def decode(self, latent, pieces):
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (8, self.width):
            raise ValueError('expected semantic latent [B,8,D]')
        material = self.diagram.material(pieces) * self.material_scale
        residual = self.cp(latent.flatten(1)).squeeze(-1) * SCALE_CP
        cp = residual + material if self.anchor_kind == 'material' else residual
        return dict(latent=latent, pieces=pieces, value_cp=cp, residual_cp=residual,
                    material_cp=material)


def value_only_copy(source):
    """Independent owned inference variant; the source checkpoint stays intact."""
    if type(source) not in (SemanticAnchor, SemanticTransition):
        raise ValueError('expected admitted semantic anchor or transition')
    if isinstance(source, SemanticTransition) and type(source.anchor) is not SemanticAnchor:
        raise ValueError('custom encoder needs a reviewed value-only adapter')
    owned = copy.deepcopy(source).eval().requires_grad_(False)
    anchor = owned.anchor if isinstance(owned, SemanticTransition) else owned
    # This class has identical storage/configuration and only changes decode.
    anchor.__class__ = ValueOnlyAnchor
    return owned


def mixed_int8_copy(source, backend):
    """Return a new variant plus explicit linear ownership, never mutate source.

    Quantizes encoder/role/value/transition linears. Unused training-only heads
    are retained in the checkpoint copy but not quantized or run. The caller
    binds source checkpoint/variant/native library pins in its declaration.
    """
    owned = value_only_copy(source)
    names = []
    unused = {'board', 'maps', 'rights', 'ep', 'clock'}

    def replace(parent, prefix='', semantic_anchor=False):
        for name, child in list(parent.named_children()):
            path = f'{prefix}.{name}' if prefix else name
            if semantic_anchor and name in unused:
                continue
            if isinstance(parent, ChessLatent) and name not in {
                    'input', 'square_layers', 'relation_layers', 'plan_query',
                    'plan_key', 'plan_value', 'plan_output', 'latent_norm'}:
                continue
            if isinstance(child, nn.Linear):
                setattr(parent, name, PackedInt8Linear(child, backend))
                names.append(path)
            else:
                replace(child, path, isinstance(child, SemanticAnchor))

    replace(owned, semantic_anchor=isinstance(owned, SemanticAnchor))
    return owned, tuple(names)
