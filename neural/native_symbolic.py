"""Hash-pinned native resident-state producer; no teacher or CPU reconstruction."""
import ctypes as c
import hashlib
import json
from pathlib import Path
import time

import torch

from .bridge import RuleArena, validate_output_tensors
from .int8_inference import admit_gfx1030, loaded_hip_runtime
from .symbolic_residual import CHANNELS, CompactSymbolicState


class SymbolicSink(c.Structure):
    _fields_ = [('abi_version', c.c_uint32), ('struct_size', c.c_uint32),
                ('device', c.c_int32), ('reserved', c.c_uint32),
                ('capacity', c.c_uint64)] + [(name, c.c_uint64) for name in
                ('features', 'channels', 'history', 'position_hash', 'node_id', 'root_epoch')]


class SymbolicExport(c.Structure):
    _fields_ = [('rows', c.c_uint64), ('root_epoch', c.c_uint64), ('kernel_ms', c.c_double)]


SCHEMA = dict(features=((18, 8, 8), torch.float32),
              channels=((64, len(CHANNELS)), torch.float32),
              history=((4,), torch.int64), position_hash=((), torch.int64),
              node_id=((), torch.int64), root_epoch=((), torch.int64))
assert c.sizeof(SymbolicSink) == 72 and c.sizeof(SymbolicExport) == 24


class NativeSymbolicProducer:
    """Serialized synchronous snapshot producer over an existing RuleArena.

    All actual board/history/channel arithmetic occurs in HIP. Initial root
    upload is the existing arena reset boundary. Source positions remain in
    the arena; the result is an owned Torch snapshot, not a live alias.
    This correctness-first bridge has a Torch pre-write barrier and two private
    stream completion/status readbacks. It is not a captured epoch integration.
    """
    def __init__(self, arena, library_sha256):
        if not isinstance(arena, RuleArena):
            raise ValueError('symbolic producer requires an owned RuleArena')
        with arena._lock:
            arena._require_open()
            if hashlib.sha256(Path(arena.library).read_bytes()).hexdigest() != library_sha256:
                raise ValueError('native symbolic library hash mismatch')
            self.device, self.device_identity = admit_gfx1030(arena.device)
            self.runtime_identity = loaded_hip_runtime()
            if self.runtime_identity['path'] != str(Path(arena._runtime._name).resolve()):
                raise ValueError('symbolic producer/Torch HIP provider mismatch')
            version = arena._lib.chen_symbolic_abi_version
            version.argtypes, version.restype = [], c.c_uint32
            if version() != 1:
                raise ValueError('unsupported symbolic producer ABI')
            fn = arena._lib.chen_rules_export_symbolic
            fn.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64,
                           c.POINTER(SymbolicSink), c.POINTER(SymbolicExport), c.c_char_p, c.c_uint64]
            fn.restype = c.c_int32
            self.arena, self.library_sha256 = arena, library_sha256

    def export(self, span):
        arena = self.arena
        with arena._lock:
            arena._require_open()
            if arena._root_manifest is None:
                raise ValueError('reset an authoritative root before symbolic export')
            source = arena._validate_span(span)
            started = time.perf_counter()
            outputs = {name: torch.empty((source.count, *shape), dtype=dtype, device=self.device)
                       for name, (shape, dtype) in SCHEMA.items()}
            validate_output_tensors(outputs, SCHEMA, source.count, self.device)
            sink = SymbolicSink(1, c.sizeof(SymbolicSink), self.device.index, 0, source.count,
                               *(outputs[name].data_ptr() for name in SCHEMA))
            result = SymbolicExport()
            torch.cuda.synchronize(self.device)
            arena._call('export_symbolic', source.start, source.count, source.epoch,
                        c.byref(sink), c.byref(result))
            if result.rows != source.count or result.root_epoch != source.epoch:
                raise RuntimeError('native symbolic completion identity mismatch')
            context = dict(arena._root_manifest)
            context.update(start=source.start, count=source.count)
            pin = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            state = CompactSymbolicState(outputs['features'], outputs['channels'], outputs['history'],
                        outputs['position_hash'], outputs['node_id'], outputs['root_epoch'],
                        pin, self.library_sha256, source.epoch)
            state.validate_metadata()
            return state, dict(rows=result.rows, root_epoch=result.root_epoch, start=source.start,
                producer_sha256=self.library_sha256, context_sha256=pin,
                kernel_event_ms=result.kernel_ms, wall_ms=(time.perf_counter()-started)*1000,
                device=self.device_identity, runtime=self.runtime_identity,
                state_download=False, teacher_input=False, graph_capture=False,
                ownership='owned_Torch_snapshot_after_native_stream_completion')
