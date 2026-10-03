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

from .bridge import BridgeError, RuleArena, validate_output_tensors
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


def _sink_schema(edge_slots, minimum):
    if type(edge_slots) is not int or not minimum <= edge_slots <= 65536:
        raise ValueError(f'edge capacity must be an integer in [{minimum},65536]')
    f, i, b = torch.float32, torch.int64, torch.bool
    return dict(pieces=((32, 16), f), squares=((64, 12), f), actions=((256, 12), f),
                piece_mask=((32,), b), piece_squares=((32,), i), action_mask=((256,), b),
                action_ids=((256,), i), edge_index=((edge_slots, 2), i), edge_kind=((edge_slots,), i),
                edge_mask=((edge_slots,), b), edge_features=((edge_slots, 4), f), edge_overflow=((), b),
                value_prior_cp=((), f), action_prior_cp=((256,), f), position_hash=((), i),
                node_id=((), i), root_epoch=((), i), history=((4,), i),
                terminal=((), torch.int32), slot_counts=((3,), torch.int32))


def sink_schema(edge_slots=6144):
    """Conservative ABI1 capacity policy, unchanged."""
    return _sink_schema(edge_slots, 4992)


def bucket_schema(edge_slots):
    """API2 capacity policy; successful native export must still be complete."""
    return _sink_schema(edge_slots, 1)


# An unretired native call must outlive Python ownership/destructors. These
# allocations/context are deliberately leaked until process retirement.
_UNRETIRED_BUCKETS = []


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
        self._initialize(arena, library_sha256, sink_schema(edge_slots), edge_slots,
                         abi=1, version_name='chen_tactical_abi_version', operation='export_tactical')

    def _initialize(self, arena, library_sha256, schema, edge_slots, *, abi, version_name, operation):
        if not isinstance(arena, RuleArena):
            raise ValueError('tactical producer requires an owned RuleArena')
        self.schema, self.abi, self.operation = schema, abi, operation
        with arena._lock:
            arena._require_open()
            if hashlib.sha256(Path(arena.library).read_bytes()).hexdigest() != library_sha256:
                raise ValueError('native tactical library hash mismatch')
            self.device, self.device_identity = admit_gfx1030(arena.device)
            self.runtime_identity = loaded_hip_runtime()
            if self.runtime_identity['path'] != str(Path(arena._runtime._name).resolve()):
                raise ValueError('tactical producer/Torch HIP provider mismatch')
            version = getattr(arena._lib, version_name)
            version.argtypes, version.restype = [], c.c_uint32
            if version() != abi:
                raise ValueError('unsupported tactical producer ABI')
            function = getattr(arena._lib, 'chen_rules_' + operation)
            function.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64,
                                c.POINTER(TacticalSink), c.POINTER(SymbolicExport), c.c_char_p, c.c_uint64]
            function.restype = c.c_int32
        self.arena, self.library_sha256, self.edge_slots = arena, library_sha256, edge_slots

    def _source_locked(self, span):
        self.arena._require_open()
        if self.arena._root_manifest is None:
            raise ValueError('reset an authoritative root before tactical export')
        return self.arena._validate_span(span)

    def _emit_locked(self, source, edge_slots):
        """Shared allocation/emission; outputs cannot escape before completion."""
        schema = self.schema if self.abi == 1 else bucket_schema(edge_slots)
        outputs = {name: torch.empty((source.count, *shape), dtype=dtype, device=self.device)
                   for name, (shape, dtype) in schema.items()}
        validate_output_tensors(outputs, schema, source.count, self.device)
        sink = TacticalSink(self.abi, c.sizeof(TacticalSink), self.device.index, 0, source.count,
                            256, edge_slots, *(outputs[name].data_ptr() for name in FIELDS))
        completion = SymbolicExport()
        torch.cuda.synchronize(self.device)
        try:
            self.arena._call(self.operation, source.start, source.count, source.epoch,
                             c.byref(sink), c.byref(completion))
        except BridgeError as failure:
            if self.abi == 2 and failure.code == -3:
                # Prevent RuleArena.close/__del__ from destroying a context whose
                # output writes might still be live. All arena APIs now see closed.
                context = self.arena._ctx
                self.arena._ctx = None
                self.arena._root_manifest = None
                _UNRETIRED_BUCKETS.append((self.arena, context, outputs))
            raise
        if completion.rows != source.count or completion.root_epoch != source.epoch:
            raise RuntimeError('native tactical completion identity mismatch')
        return outputs, completion

    def _snapshot_locked(self, source, outputs, completion, edge_slots, started, **extra):
        context = dict(self.arena._root_manifest)
        context.update(start=source.start, count=source.count)
        pin = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(',', ':')).encode()).hexdigest()
        batch = TacticalBatch(**{name: outputs[name] for name in TENSOR_FIELDS},
                              feature_schema=FEATURE_SCHEMA)
        if source.count:
            _metadata(batch, TacticalConfig(feature_schema=FEATURE_SCHEMA))
        manifest = dict(rows=completion.rows, root_epoch=completion.root_epoch, start=source.start,
                        context_sha256=pin, producer_sha256=self.library_sha256,
                        feature_schema=FEATURE_SCHEMA, edge_slots=edge_slots, legal_slots=256,
                        kernel_event_ms=completion.kernel_ms, wall_ms=(time.perf_counter()-started)*1000,
                        device=self.device_identity, runtime=self.runtime_identity,
                        state_download=False, teacher_input=False, graph_capture=False,
                        ownership='owned_Torch_snapshot_after_native_stream_completion', **extra)
        sidecar = {name: outputs[name] for name in FIELDS[14:]}
        return NativeTacticalState(batch, sidecar, manifest)

    def export(self, span):
        with self.arena._lock:
            source = self._source_locked(span)
            started = time.perf_counter()
            outputs, completion = self._emit_locked(source, self.edge_slots)
            return self._snapshot_locked(source, outputs, completion, self.edge_slots, started)


class NativeTacticalBucketProducer(NativeTacticalProducer):
    """Named API2 complete-graph bucket/retry producer; never returns truncation.

    Only retired edge-capacity -2 retries. -1 propagates; -3 quarantines native
    ownership and closes admission to this arena, requiring process retirement.
    The lock spans all retries, preserving root/history/epoch provenance. Failed
    tensors are never returned. No payload or slot-count download chooses E.
    """
    variant = 'native_tactical_bucket_v2'

    def __init__(self, arena, library_sha256, *, buckets=(512, 1024, 2048, 4096, 6144)):
        if not isinstance(buckets, tuple) or not buckets:
            raise ValueError('buckets must be a nonempty strictly increasing tuple')
        for size in buckets:
            bucket_schema(size)
        if any(a >= b for a, b in zip(buckets, buckets[1:])):
            raise ValueError('buckets must be strictly increasing')
        self.buckets = buckets
        self._initialize(arena, library_sha256, bucket_schema(buckets[0]), buckets[0],
                         abi=2, version_name='chen_tactical_bucket_abi_version',
                         operation='export_tactical_bucket')

    def export(self, span):
        with self.arena._lock:
            source = self._source_locked(span)
            started = time.perf_counter()
            attempts = []
            for size in self.buckets:
                attempt_started = time.perf_counter()
                try:
                    outputs, completion = self._emit_locked(source, size)
                except BridgeError as failure:
                    attempts.append(dict(edge_slots=size, status=failure.code,
                                         wall_ms=(time.perf_counter()-attempt_started)*1000))
                    if failure.code != -2 or size == self.buckets[-1]:
                        raise
                    continue
                attempts.append(dict(edge_slots=size, status=0,
                                     wall_ms=(time.perf_counter()-attempt_started)*1000))
                return self._snapshot_locked(source, outputs, completion, size, started,
                    producer_variant=self.variant, bucket_policy=self.buckets, attempts=attempts,
                    retries=len(attempts)-1, count_download=False)
            raise AssertionError('nonempty bucket policy must succeed or propagate failure')
