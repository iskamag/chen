"""Synchronous HIP-arena to Torch-owned compact replay ingress.

Only root/history enter from CPU. State payloads remain on the device after
expansion. Admission still synchronizes scalar validation; this is a state-copy
claim, not a claim that control telemetry never reaches the host.
"""
from __future__ import annotations

import ctypes as c
from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
import threading
import uuid

import torch

from .compact import CompactReplay, compact_schema, pack_fens


class RulePosition(c.Structure):
    _fields_ = [("boards", c.c_uint64 * 12), ("side", c.c_uint32),
                ("castling", c.c_uint32), ("ep_square", c.c_int32),
                ("halfmove", c.c_uint32), ("fullmove", c.c_uint32),
                ("reserved", c.c_uint32), ("hash", c.c_uint64)]


class RuleStats(c.Structure):
    _fields_ = [("arena_positions", c.c_uint64), ("classified_positions", c.c_uint64),
                ("evaluated_positions", c.c_uint64), ("next_start", c.c_uint64),
                ("next_count", c.c_uint64), ("kernel_ms", c.c_double)]


SINK_FIELDS = ("pieces", "state", "next_pieces", "next_state", "legal_actions",
               "legal_counts", "actions", "material_proxy", "terminal",
               "parent_hash", "child_hash", "parent_node", "child_node", "root_epoch")
SIDECAR_FIELDS = SINK_FIELDS[9:]
RUNTIME_APIS = ("hipPointerGetAttributes", "hipMemGetAddressRange", "hipMalloc",
                "hipFree", "hipStreamSynchronize")


class ReplaySink(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32),
                ("device", c.c_int32), ("legal_slots", c.c_uint32),
                ("capacity", c.c_uint64)] + [(name, c.c_uint64) for name in SINK_FIELDS]


class ReplayExport(c.Structure):
    _fields_ = [("rows", c.c_uint64), ("root_epoch", c.c_uint64), ("kernel_ms", c.c_double)]


def sink_schema(legal_slots):
    if type(legal_slots) is not int or not 1 <= legal_slots <= 256:
        raise ValueError("HIP sink legal_slots must be an integer in [1, 256]")
    schema = compact_schema(legal_slots)
    return {name: (schema[name] if name in schema else ((), torch.int64))
            for name in SINK_FIELDS}


@dataclass(frozen=True)
class ExportedTransitions:
    """Owned Torch snapshots; sidecar node IDs are scoped by manifest identity."""
    tensors: dict[str, torch.Tensor]
    sidecar: dict[str, torch.Tensor]
    manifest: dict[str, object]

    def append_to(self, replay: CompactReplay):
        if replay.device != self.tensors["pieces"].device:
            raise ValueError("export/replay device mismatch")
        if replay.max_slots != self.manifest["legal_slots"]:
            raise ValueError("export/replay legal slot mismatch")
        # Completed synchronous handoff, including cross-stream Torch readers
        # and previous ring writes. Source owners may retire when this returns.
        torch.cuda.synchronize(replay.device)
        try:
            replay.append(self.tensors)
        finally:
            torch.cuda.synchronize(replay.device)
        return self


@dataclass(frozen=True)
class ResidentSpan:
    context_uuid: str
    epoch: int
    start: int
    count: int

    def __post_init__(self):
        if not isinstance(self.context_uuid, str) or not self.context_uuid:
            raise ValueError("span requires context identity")
        if type(self.epoch) is not int or not 1 <= self.epoch <= MASK64:
            raise ValueError("span requires a valid uint64 epoch")
        if type(self.start) is not int or self.start < 0 or type(self.count) is not int or self.count < 0:
            raise ValueError("span range must be nonnegative integers")
        if self.start + self.count > MASK64:
            raise ValueError("span range overflow")

    def subset(self, offset, count):
        if type(offset) is not int or offset < 0 or type(count) is not int or count <= 0 or offset + count > self.count:
            raise ValueError("invalid bounded span subset")
        return ResidentSpan(self.context_uuid, self.epoch, self.start + offset, count)


MASK64 = (1 << 64) - 1


def hash_key(offset):
    """Exact simulator hash key; mirrors src/core.rs and GPU rules splitmix."""
    value = (0x4348454E00000000 + offset + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def position_from_fen(fen):
    """Explicit CPU root ingestion adapter using the optional chess oracle.

    The root preserves raw EP, while its repetition hash includes only legally
    effective EP. Callers with a simulator packet can avoid this dependency.
    """
    import chess
    board = chess.Board(fen)
    pieces, state = pack_fens([fen])
    result = RulePosition()
    for index, bits in enumerate(pieces[0].tolist()):
        result.boards[index] = bits & MASK64
    side, rights, ep, halfmove, fullmove = state[0].tolist()
    result.side, result.castling, result.ep_square = side, rights, ep
    result.halfmove, result.fullmove = halfmove & 0xFFFFFFFF, fullmove & 0xFFFFFFFF
    key = hash_key(768 + rights) ^ (hash_key(792) if side else 0)
    if board.has_legal_en_passant():
        key ^= hash_key(784 + ep % 8)
    for index, bits in enumerate(result.boards):
        while bits:
            bit = bits & -bits
            key ^= hash_key(index * 64 + bit.bit_length() - 1)
            bits ^= bit
    result.hash = key
    return result


class BridgeError(RuntimeError):
    def __init__(self, operation, code, message):
        self.operation, self.code = operation, code
        super().__init__(f"{operation}: status {code}: {message}")


def _positive_integer(name, value):
    if type(value) is not int or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _device(device):
    result = torch.device(device)
    if result.type != "cuda" or torch.version.hip is None:
        raise ValueError("bridge requires a ROCm Torch cuda device")
    index = torch.cuda.current_device() if result.index is None else result.index
    return torch.device("cuda", index)


def _torch_runtime():
    """Promote the exact Torch-bundled runtime before loading the HIP producer."""
    path = Path(torch.__file__).parent / "lib" / "libamdhip64.so"
    if not path.is_file():
        raise RuntimeError(f"Torch bundled HIP runtime missing: {path}")
    return c.CDLL(str(path.resolve()), mode=c.RTLD_GLOBAL)


def _bind(library, runtime=None):
    lib = c.CDLL(str(Path(library).resolve()))
    ptr, u64, u32, err = c.c_void_p, c.c_uint64, c.c_uint32, c.c_char_p
    signatures = {
        "create": ([c.c_int32, u64, err, u64], ptr),
        "destroy": ([ptr], None),
        "reset": ([ptr, c.POINTER(RulePosition), c.POINTER(u64), u64, err, u64], c.c_int32),
        "expand": ([ptr, u64, u64, u32, c.POINTER(RuleStats), err, u64], c.c_int32),
        "export_transitions": ([ptr, u64, u64, u64, c.POINTER(ReplaySink),
                                c.POINTER(ReplayExport), err, u64], c.c_int32),
    }
    for name, (args, restype) in signatures.items():
        fn = getattr(lib, "chen_rules_" + name)
        fn.argtypes, fn.restype = args, restype
    lib.chen_replay_epoch.argtypes = [ptr, c.POINTER(u64), err, u64]
    lib.chen_replay_epoch.restype = c.c_int32
    lib.chen_rules_abi_version.argtypes = []
    lib.chen_rules_abi_version.restype = u32
    if lib.chen_rules_abi_version() != 1:
        raise ValueError("unsupported rules ABI version")
    lib.chen_replay_abi_version.argtypes = []
    lib.chen_replay_abi_version.restype = u32
    if lib.chen_replay_abi_version() != 1:
        raise ValueError("unsupported rules export ABI version")
    if runtime is not None:
        lib.chen_replay_runtime_identity.argtypes = [u32]
        lib.chen_replay_runtime_identity.restype = u64
        for selector, name in enumerate(RUNTIME_APIS):
            expected = c.cast(getattr(runtime, name), ptr).value
            actual = lib.chen_replay_runtime_identity(selector)
            if not expected or actual != expected:
                raise RuntimeError(f"producer/Torch HIP runtime identity mismatch: {name}")
    return lib


def validate_output_tensors(tensors, schema, capacity, device):
    """Validate pointer metadata without reading any device tensor contents."""
    _positive_integer("capacity", capacity)
    device = torch.device(device)
    if set(tensors) != set(schema):
        raise ValueError("output tensor keys do not match sink schema")
    ranges = []
    for name, (shape, dtype) in schema.items():
        value = tensors[name]
        if not isinstance(value, torch.Tensor) or value.shape != (capacity, *shape):
            raise ValueError(f"{name}: output shape mismatch")
        if value.dtype != dtype or value.device != device or not value.is_contiguous():
            raise ValueError(f"{name}: output dtype/device/stride mismatch")
        if value.requires_grad or value.storage_offset() != 0:
            raise ValueError(f"{name}: output must be an untracked base allocation")
        address = value.data_ptr()
        size = value.numel() * value.element_size()
        if not address or address % value.element_size():
            raise ValueError(f"{name}: output pointer alignment mismatch")
        if value.untyped_storage().nbytes() != size:
            raise ValueError(f"{name}: output must own the complete bounded allocation")
        if any(address < end and start < address + size for start, end in ranges):
            raise ValueError(f"{name}: output allocations overlap")
        ranges.append((address, address + size))
    return tensors


class RuleArena:
    """Serialized, explicitly closed HIP context; Torch owns every export buffer.

    Operations synchronize for correctness. Fresh Torch outputs are completed
    before the HIP private stream writes them, and the HIP call completes before
    Torch consumes them. No live arena pointer is exposed to Python.
    """

    def __init__(self, library=None, capacity=4096, device="cuda"):
        self._ctx = None
        self._lock = threading.RLock()
        self.capacity = _positive_integer("capacity", capacity)
        if capacity > 0x7FFFFFFF:
            raise ValueError("capacity exceeds rules arena limit")
        self.device = _device(device)
        library = library or os.environ.get("CHEN_GPU_LIBRARY")
        if not library:
            raise ValueError("pass a HIP library path or set CHEN_GPU_LIBRARY")
        self._runtime = _torch_runtime()
        self._lib = _bind(library, self._runtime)
        self.library = str(Path(library).resolve())
        self.context_uuid = str(uuid.uuid4())
        self._root_manifest = None
        self._generation = self._used = self._epoch = 0
        error = c.create_string_buffer(1024)
        self._ctx = self._lib.chen_rules_create(self.device.index, capacity, error, len(error))
        if not self._ctx:
            raise BridgeError("create", -1, error.value.decode(errors="replace"))

    @property
    def closed(self):
        return not self._ctx

    def _require_open(self):
        if self.closed:
            raise RuntimeError("rules arena is closed")

    def _call(self, operation, *args):
        self._require_open()
        error = c.create_string_buffer(1024)
        symbol = "chen_replay_epoch" if operation == "epoch" else "chen_rules_" + operation
        code = getattr(self._lib, symbol)(self._ctx, *args, error, len(error))
        if code:
            raise BridgeError(operation, code, error.value.decode(errors="replace"))

    def close(self):
        with self._lock:
            if self._ctx:
                self._lib.chen_rules_destroy(self._ctx)
                self._ctx = None
                self._root_manifest = None
                self._used = 0

    def __enter__(self):
        self._require_open()
        return self

    def __exit__(self, *_):
        self.close()

    def __del__(self):
        if getattr(self, "_ctx", None):
            try:
                self.close()
            except Exception:
                pass  # explicit context-manager close is the supported lifecycle

    def reset(self, root: RulePosition, history=()):
        with self._lock:
            self._require_open()
            if not isinstance(root, RulePosition):
                raise TypeError("root must be a RulePosition packet")
            history = tuple(history)
            if len(history) > 65535 or any(type(key) is not int or not 0 <= key <= MASK64 for key in history):
                raise ValueError("history must contain at most 65535 previous uint64 hashes")
            packet = RulePosition.from_buffer_copy(bytes(root))
            hashes = (c.c_uint64 * len(history))(*history)
            # A failed reset invalidates the old tree; never reuse old provenance.
            self._used = 0
            self._root_manifest = None
            self._call("reset", c.byref(packet), hashes, len(history))
            epoch = c.c_uint64()
            self._call("epoch", c.byref(epoch))
            self._epoch = epoch.value
            self._generation += 1
            self._used = 1
            self._root_manifest = {
                "context_uuid": self.context_uuid, "python_reset_generation": self._generation,
                "root_epoch": self._epoch,
                "root_packet_sha256": hashlib.sha256(bytes(packet)).hexdigest(),
                "root_hash_hex": f"{packet.hash:016x}", "history_count": len(history),
                "previous_history_sha256": hashlib.sha256(bytes(hashes)).hexdigest(),
                "history_contract": "caller-supplied previous positions only, root excluded",
            }
            return dict(self._root_manifest)

    def _validate_span(self, span):
        if not isinstance(span, ResidentSpan):
            raise TypeError("use an epoch-bound ResidentSpan")
        if span.context_uuid != self.context_uuid or span.epoch != self._epoch:
            raise ValueError("span belongs to a different context or stale epoch")
        if span.count <= 0 or span.start + span.count > self._used:
            raise ValueError("invalid resident span range")
        return span

    def expand(self, span=None):
        """Expand the current root, or an epoch-bound child span from expand."""
        with self._lock:
            self._require_open()
            if self._root_manifest is None:
                raise RuntimeError("reset a valid root before expansion")
            source = ResidentSpan(self.context_uuid, self._epoch, 0, 1) if span is None else self._validate_span(span)
            stats = RuleStats()
            self._call("expand", source.start, source.count, 1, c.byref(stats))
            self._used = stats.arena_positions
            return {**{name: getattr(stats, name) for name, _ in stats._fields_},
                    "span": ResidentSpan(self.context_uuid, self._epoch, stats.next_start, stats.next_count)}

    def export(self, span, *, legal_slots=256, game_id=0):
        """Export existing contiguous child nodes into independent device tensors."""
        with self._lock:
            self._require_open()
            if self._root_manifest is None:
                raise RuntimeError("reset a valid root before export")
            span = self._validate_span(span)
            child_start, count = span.start, span.count
            if child_start <= 0:
                raise ValueError("export range must contain existing non-root child nodes")
            if type(game_id) is not int or not 0 <= game_id <= 0x7FFFFFFFFFFFFFFF:
                raise ValueError("game_id must be a nonnegative int64")
            schema = sink_schema(legal_slots)
            outputs = {name: torch.empty((count, *shape), dtype=dtype, device=self.device)
                       for name, (shape, dtype) in schema.items()}
            validate_output_tensors(outputs, schema, count, self.device)
            sink = ReplaySink(1, c.sizeof(ReplaySink), self.device.index, legal_slots, count,
                              *(outputs[name].data_ptr() for name in SINK_FIELDS))
            result = ReplayExport()
            # Torch allocator/streams finish first; HIP private stream completes
            # synchronously in export_transitions before any subsequent reads.
            torch.cuda.synchronize(self.device)
            self._call("export_transitions", child_start, count, span.epoch, c.byref(sink), c.byref(result))
            if result.rows != count or result.root_epoch != span.epoch:
                raise RuntimeError("export success returned an inconsistent row count")
            tensors = {key: outputs[key] for key in compact_schema(legal_slots) if key in outputs}
            try:
                tensors.update(
                    target_actions=outputs["actions"].clone(),
                    history_handles=torch.full((count,), -1, dtype=torch.int64, device=self.device),
                    next_history_handles=torch.full((count,), -1, dtype=torch.int64, device=self.device),
                    game_ids=torch.full((count,), game_id, dtype=torch.int64, device=self.device),
                )
            finally:
                # Complete partial publication even if a later allocation fails.
                torch.cuda.synchronize(self.device)
            manifest = {**self._root_manifest, "rules_abi": 1, "export_abi": 1,
                        "root_epoch": result.root_epoch, "rows": result.rows,
                        "child_start": child_start, "legal_slots": legal_slots,
                        "export_kernel_ms": result.kernel_ms, "device": str(self.device),
                        "policy_target": "executed legal action imitation; no best-move teacher",
                        "material_proxy": "tanh(simulator material evaluator0 centipawns clamped +/-29000 /800), current mover, toy supervision",
                        "terminal_policy": "simulator_current_position_draw_policy: mate/stalemate first; halfmove>=100, insufficient, current threefold",
                        "history_scope": "root hash prefix retained in simulator; replay handles absent; model lacks repetition features",
                        "ownership": "Torch-owned output snapshots; synchronous pre-write device barrier and HIP completion",
                        "hip_runtime": {"torch_runtime_path": self._runtime._name,
                                        "checked_bound_apis": list(RUNTIME_APIS)},
                        "state_download_api_used": False,
                        "copy_evidence": "source-level native path plus Torch dispatch audit; no traced byte-count claim",
                        "strength_evidence": False}
            return ExportedTransitions(tensors, {name: outputs[name] for name in SIDECAR_FIELDS}, manifest)


assert c.sizeof(RulePosition) == 128
assert c.sizeof(RuleStats) == 48
assert c.sizeof(ReplaySink) == 136
assert c.sizeof(ReplayExport) == 24


def smoke(*, library=None, device="cuda", capacity=4096, steps=4, batch_size=32,
          seed=20260930, output=None, producer_revision="unrecorded", gpu_source_root=None):
    """Bounded actual arena→replay→optimizer mechanics run; no strength evidence."""
    import time
    from .models import build_model
    from .train import seed_all, train_step

    _positive_integer("steps", steps)
    _positive_integer("batch_size", batch_size)
    if type(seed) is not int or not 0 <= seed <= 0x7FFFFFFFFFFFFFFF:
        raise ValueError("seed must be a nonnegative int64")
    root_fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"
    start = time.perf_counter()
    root = position_from_fen(root_fen)
    root_pack_seconds = time.perf_counter() - start
    phases = {}
    start = time.perf_counter()
    with RuleArena(library, capacity, device) as arena:
        phases["context_create_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        arena.reset(root)
        phases["initial_root_history_ingress_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        first = arena.expand()
        second = arena.expand(first["span"])
        phases["two_level_resident_expansion_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        exported = arena.export(second["span"], game_id=seed)
        phases["torch_allocation_barriers_export_and_labels_seconds"] = time.perf_counter() - start
        start = time.perf_counter()
        replay = CompactReplay(second["next_count"], arena.device)
        exported.append_to(replay)
        phases["replay_allocation_validated_admission_barriers_seconds"] = time.perf_counter() - start
        source_manifest = exported.manifest
        library_path = arena.library
        resident_manifest = replay.manifest()
        torch.cuda.synchronize(arena.device)
        resident_memory = {"torch_allocated_bytes": torch.cuda.memory_allocated(arena.device),
                           "torch_reserved_bytes": torch.cuda.memory_reserved(arena.device),
                           "driver_free_total_bytes": torch.cuda.mem_get_info(arena.device)}
        # The arena may close now: Torch owns all exported records and replay.
    seed_all(seed)
    model = build_model("relation", width=16, depth=1).to(replay.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
    generator = torch.Generator(device=replay.device).manual_seed(seed + 1)
    torch.cuda.synchronize(replay.device)
    start = time.perf_counter()
    losses = []
    identities = []
    for _ in range(steps):
        batch = replay.sample(batch_size, generator)
        losses.append(train_step(model, optimizer, batch))
        identities.append(batch["sequence_ids"])
    torch.cuda.synchronize(replay.device)
    phases["sample_decode_and_training_seconds"] = time.perf_counter() - start
    # Explicit reporting boundary: only scalar losses/sample IDs are downloaded.
    report = {
        "kind": "direct_resident_rules_to_replay_smoke", "strength_evidence": False,
        "root_cpu_adapter_seconds": root_pack_seconds, "phases": phases,
        "source": source_manifest, "replay": resident_manifest,
        "environment": {"torch": str(torch.__version__), "torch_hip": torch.version.hip,
                        "device": str(replay.device),
                        "architecture": getattr(torch.cuda.get_device_properties(replay.device), "gcnArchName", "unknown")},
        "config": {"seed": seed, "steps": steps, "batch_size": batch_size,
                   "model": "relation", "width": 16, "depth": 1, "learning_rate": 3e-4,
                   "arena_capacity": capacity, "resident_levels": 2,
                   "root_fen": root_fen, "previous_history_count": 0},
        "first_loss": {name: float(value) for name, value in losses[0].items()},
        "final_loss": {name: float(value) for name, value in losses[-1].items()},
        "sample_sequence_sha256": hashlib.sha256(json.dumps(torch.stack(identities).cpu().tolist()).encode()).hexdigest(),
        "resident_before_training": resident_memory,
        "library": {"path": library_path, "sha256": hashlib.sha256(Path(library_path).read_bytes()).hexdigest(),
                    "producer_revision": producer_revision},
        "python_source_sha256": {f"neural/{name}": hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                                 for name in ("bridge.py", "compact.py", "features.py", "models.py", "train.py")},
        "producer_source_sha256": {},
        "limitations": ["single fixed root, bounded exhaustive two-ply legal tree; not GPU self-play",
                        "executed-action imitation and material proxy targets; no outcome/strength labels",
                        "synchronous barriers and scalar validation/control transfers",
                        "short un-warmed mechanics run; timings are not throughput estimates",
                        "no native byte-count trace collected by this smoke function"],
    }
    if gpu_source_root:
        directory = Path(gpu_source_root)
        report["producer_source_sha256"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                                            for name in ("chen_replay.h", "chen_rules.h", "chen_rules.hip",
                                                         "chen_gpu.h", "chen_gpu.hip", "build.sh")}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--library", help="defaults to CHEN_GPU_LIBRARY")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--producer-revision", default="unrecorded")
    parser.add_argument("--gpu-source-root")
    parser.add_argument("--output")
    print(json.dumps(smoke(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
