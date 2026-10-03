"""Exact HIP-arena -> owned Torch sparse tactical observation.

Only caller root/history ingestion is host-origin. This serialized bridge
reads completion/status metadata, never board/edge payloads on the host.
"""
import ctypes as c
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import time

import torch

from .bridge import RuleArena, validate_output_tensors
from .int8_inference import admit_gfx1030, loaded_hip_runtime
from .native_symbolic import SymbolicExport
from .tactical_operator import TacticalBatch, TacticalConfig, _metadata

# Versioned ABI contract, checked against the independent CPU reference in tests.
# The native producer itself has no python-chess dependency.
FEATURE_SCHEMA = 'chen_tactical_exact_v1'
FIELDS = ('pieces', 'squares', 'actions', 'piece_mask', 'piece_squares', 'action_mask',
          'action_ids', 'edge_index', 'edge_kind', 'edge_mask', 'edge_features',
          'edge_overflow', 'value_prior_cp', 'action_prior_cp', 'position_hash',
          'node_id', 'root_epoch', 'history', 'terminal', 'slot_counts')
TENSOR_FIELDS = FIELDS[:14]


class TacticalSink(c.Structure):
    _fields_ = [('abi_version', c.c_uint32), ('struct_size', c.c_uint32),
                ('device', c.c_int32), ('reserved', c.c_uint32), ('capacity', c.c_uint64),
                ('legal_slots', c.c_uint32), ('edge_slots', c.c_uint32)] + [
                    (name, c.c_uint64) for name in FIELDS]


assert c.sizeof(TacticalSink) == 192


def sink_schema(edge_slots=6144):
    if type(edge_slots) is not int or not 4992 <= edge_slots <= 65536:
        raise ValueError('edge capacity must be an integer in [4992,65536]')
    f, i, b = torch.float32, torch.int64, torch.bool
    return dict(pieces=((32, 16), f), squares=((64, 12), f), actions=((256, 12), f),
                piece_mask=((32,), b), piece_squares=((32,), i), action_mask=((256,), b),
                action_ids=((256,), i), edge_index=((edge_slots, 2), i), edge_kind=((edge_slots,), i),
                edge_mask=((edge_slots,), b), edge_features=((edge_slots, 4), f), edge_overflow=((), b),
                value_prior_cp=((), f), action_prior_cp=((256,), f), position_hash=((), i),
                node_id=((), i), root_epoch=((), i), history=((4,), i),
                terminal=((), torch.int32), slot_counts=((3,), torch.int32))


@dataclass(frozen=True)
class NativeTacticalState:
    batch: TacticalBatch
    sidecar: dict
    manifest: dict


class NativeTacticalProducer:
    """Native emitter owns content; Torch operator owns learned arithmetic.

    Caller must obey the exact terminal sidecar before playing any model action.
    No teacher, scalar score download, CPU legality or graph reconstruction is
    performed here. No graph-capture or asynchronous handoff claim is made.
    """
    def __init__(self, arena, library_sha256, *, edge_slots=6144):
        if not isinstance(arena, RuleArena):
            raise ValueError('tactical producer requires an owned RuleArena')
        self.schema = sink_schema(edge_slots)
        with arena._lock:
            arena._require_open()
            if hashlib.sha256(Path(arena.library).read_bytes()).hexdigest() != library_sha256:
                raise ValueError('native tactical library hash mismatch')
            self.device, self.device_identity = admit_gfx1030(arena.device)
            self.runtime_identity = loaded_hip_runtime()
            if self.runtime_identity['path'] != str(Path(arena._runtime._name).resolve()):
                raise ValueError('tactical producer/Torch HIP provider mismatch')
            version = arena._lib.chen_tactical_abi_version
            version.argtypes, version.restype = [], c.c_uint32
            if version() != 1:
                raise ValueError('unsupported tactical producer ABI')
            function = arena._lib.chen_rules_export_tactical
            function.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64,
                                c.POINTER(TacticalSink), c.POINTER(SymbolicExport), c.c_char_p, c.c_uint64]
            function.restype = c.c_int32
        self.arena, self.library_sha256, self.edge_slots = arena, library_sha256, edge_slots

    def export(self, span):
        arena = self.arena
        with arena._lock:
            arena._require_open()
            if arena._root_manifest is None:
                raise ValueError('reset an authoritative root before tactical export')
            source = arena._validate_span(span)
            started = time.perf_counter()
            outputs = {name: torch.empty((source.count, *shape), dtype=dtype, device=self.device)
                       for name, (shape, dtype) in self.schema.items()}
            validate_output_tensors(outputs, self.schema, source.count, self.device)
            sink = TacticalSink(1, c.sizeof(TacticalSink), self.device.index, 0, source.count,
                                256, self.edge_slots, *(outputs[name].data_ptr() for name in FIELDS))
            completion = SymbolicExport()
            torch.cuda.synchronize(self.device)
            arena._call('export_tactical', source.start, source.count, source.epoch,
                        c.byref(sink), c.byref(completion))
            if completion.rows != source.count or completion.root_epoch != source.epoch:
                raise RuntimeError('native tactical completion identity mismatch')
            context = dict(arena._root_manifest)
            context.update(start=source.start, count=source.count)
            pin = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
            batch = TacticalBatch(**{name: outputs[name] for name in TENSOR_FIELDS},
                                  feature_schema=FEATURE_SCHEMA)
            if source.count:
                _metadata(batch, TacticalConfig(feature_schema=FEATURE_SCHEMA))
            manifest = dict(rows=completion.rows, root_epoch=completion.root_epoch, start=source.start,
                            context_sha256=pin, producer_sha256=self.library_sha256,
                            feature_schema=FEATURE_SCHEMA, edge_slots=self.edge_slots, legal_slots=256,
                            kernel_event_ms=completion.kernel_ms, wall_ms=(time.perf_counter()-started)*1000,
                            device=self.device_identity, runtime=self.runtime_identity,
                            state_download=False, teacher_input=False, graph_capture=False,
                            ownership='owned_Torch_snapshot_after_native_stream_completion')
            sidecar = {name: outputs[name] for name in FIELDS[14:]}
            return NativeTacticalState(batch, sidecar, manifest)
