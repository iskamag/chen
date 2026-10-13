"""Owned schema-1 corpus views and complete-action-group batches.

Importing this module does not import Torch. Admission first runs the CPU
semantic reader; only then are bytes copied into private Torch storage. Public
batches own their gathered tensors. These slab views are NOT native export
sinks (the native bridge requires separate complete base allocations).

GPU upload is implemented but unvalidated: it requires a separately pinned
ROCm admission record. Reader-validated schedule indices are appended before
the one packed Tensor.to call. No DMA-count or throughput claim follows.
Upload drains before publication. Later gathers obey PyTorch caller-stream
lifetimes; cross-stream handoff requires caller wait/record_stream discipline.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import stat
import struct
import sys
from typing import Any


_DTYPES = {"int64": (8, "q"), "int32": (4, "i"), "int16": (2, "h"),
           "float32": (4, "f"), "uint8": (1, "B")}
_TABLES = {
    "states.boards": ("int64", 12), "states.fields": ("int32", 5),
    "states.hash": ("int64", None), "states.terminal": ("uint8", None),
    "states.family": ("int32", None), "states.origin_context": ("int32", None),
    "states.origin_epoch": ("int64", None), "states.origin_node": ("int64", None),
    "history.offsets": ("int64", None), "history.keys": ("int64", None),
    "observations.state": ("int32", None), "observations.query": ("int32", None),
    "observations.reported_cp": ("int32", None), "observations.query_material": ("int32", None),
    "observations.midpoint": ("float32", None), "observations.halfwidth": ("float32", None),
    "observations.ordinary": ("uint8", None),
    "roots.state": ("int32", None), "roots.query": ("int32", None),
    "roots.family": ("int32", None), "roots.legal_count": ("int32", None),
    "roots.edge_offsets": ("int64", None),
    "edges.root": ("int32", None), "edges.child_state": ("int32", None),
    "edges.action": ("int16", None), "edges.reported_cp": ("int32", None),
    "edges.midpoint": ("float32", None), "edges.halfwidth": ("float32", None),
    "edges.rank": ("int16", None), "edges.ordinary": ("uint8", None),
}


def _sha_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _pin(value: str) -> str:
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("expected lowercase SHA256 pin")
    return value


def _open_regular(path: Path | str, limit: int):
    path = Path(path)
    before = path.lstat()
    if not stat.S_ISREG(before.st_mode):
        raise ValueError("pinned path must be a regular non-symlink file")
    if before.st_size > limit:
        raise ValueError("pinned file exceeds byte limit")
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    try:
        actual = os.fstat(descriptor)
        if (not stat.S_ISREG(actual.st_mode) or actual.st_size > limit
                or (actual.st_dev, actual.st_ino) != (before.st_dev, before.st_ino)):
            raise ValueError("pinned file changed identity/type/size during admission")
        return os.fdopen(descriptor, "rb")
    except BaseException:
        os.close(descriptor)
        raise


def _read_pinned(path: Path | str, expected: str, limit: int = 1 << 20) -> bytes:
    with _open_regular(path, limit) as handle:
        data = handle.read(limit + 1)
    if len(data) > limit or _sha_bytes(data) != _pin(expected):
        raise ValueError("pinned file hash/size mismatch")
    return data


def _hash_regular(path: Path | str, limit: int) -> str:
    h, total = hashlib.sha256(), 0
    with _open_regular(path, limit) as handle:
        while block := handle.read(min(1 << 20, limit - total + 1)):
            total += len(block)
            if total > limit:
                raise ValueError("pinned file grew beyond byte cap")
            h.update(block)
    return h.hexdigest()


def _json_record(path: Path | str, expected: str) -> dict:
    def unique(pairs):
        out = {}
        for key, value in pairs:
            if key in out:
                raise ValueError("duplicate JSON key")
            out[key] = value
        return out
    value = json.loads(_read_pinned(path, expected), object_pairs_hook=unique)
    if not isinstance(value, dict):
        raise ValueError("expected record object")
    return value


@dataclass(frozen=True)
class EvaluationGate:
    """Externally pinned stage seal; test additionally pins selection inputs.

    Gate JSON fields: kind ('semantic_frozen_checkpoint_v1' for validation,
    'semantic_selected_checkpoint_v1' for test), corpus_manifest_sha256,
    packing_source_sha256, checkpoint_sha256, and for test
    selection_source_sha256. Selection JSON fields: kind
    'semantic_selection_source_v1', selected_checkpoint_sha256,
    selection_splits exactly ['train','validation'], test_policy 'sealed_out'.
    SHA pins must come from the separately frozen declaration, not the files
    being admitted. This enforces stage identity, not an adversarial sandbox.
    """
    path: str | Path
    sha256: str
    checkpoint_path: str | Path
    checkpoint_sha256: str
    selection_source_path: str | Path | None = None
    selection_source_sha256: str | None = None


def _stage_gate(stage, split, manifest_sha, source_sha, gate):
    if stage == "train":
        if split != "train" or gate is not None:
            raise ValueError("training admits only train split without evaluation gate")
        return
    if stage not in ("validation_eval", "test_eval") or split != stage.removesuffix("_eval"):
        raise ValueError("explicit matching evaluation stage required")
    if not isinstance(gate, EvaluationGate):
        raise ValueError("evaluation requires pinned checkpoint/source gate, not a boolean")
    record = _json_record(gate.path, gate.sha256)
    required = {"kind", "corpus_manifest_sha256", "packing_source_sha256", "checkpoint_sha256"}
    if stage == "test_eval":
        required.add("selection_source_sha256")
    if set(record) != required:
        raise ValueError("unknown/missing evaluation gate fields")
    kind = "semantic_frozen_checkpoint_v1" if stage == "validation_eval" else "semantic_selected_checkpoint_v1"
    if record["kind"] != kind or record["corpus_manifest_sha256"] != manifest_sha or record["packing_source_sha256"] != source_sha:
        raise ValueError("evaluation gate identity mismatch")
    if record["checkpoint_sha256"] != _pin(gate.checkpoint_sha256):
        raise ValueError("checkpoint pin mismatch")
    # Stream hash regular bounded checkpoints without deserializing a model.
    if _hash_regular(gate.checkpoint_path, 128 << 20) != gate.checkpoint_sha256:
        raise ValueError("checkpoint bytes mismatch")
    if Path(gate.path).resolve() == Path(gate.checkpoint_path).resolve():
        raise ValueError("checkpoint and stage seal must be distinct files")
    if stage == "validation_eval":
        if gate.selection_source_path is not None or gate.selection_source_sha256 is not None:
            raise ValueError("validation uses frozen, not selected checkpoint gate")
    else:
        if gate.selection_source_path is None or gate.selection_source_sha256 is None:
            raise ValueError("test requires distinct sealed selection-source record")
        if record["selection_source_sha256"] != _pin(gate.selection_source_sha256):
            raise ValueError("selection-source pin mismatch")
        paths = {Path(p).resolve() for p in (gate.path, gate.checkpoint_path, gate.selection_source_path)}
        if len(paths) != 3:
            raise ValueError("stage, checkpoint, selection-source files must be distinct")
        source = _json_record(gate.selection_source_path, gate.selection_source_sha256)
        if source != {"kind": "semantic_selection_source_v1", "selected_checkpoint_sha256": gate.checkpoint_sha256,
                      "selection_splits": ["train", "validation"], "test_policy": "sealed_out"}:
            raise ValueError("test cannot participate in checkpoint selection")


def _layout(pack, max_bytes):
    """Independent metadata/bounds defense after the semantic CPU reader."""
    if sys.byteorder != "little":
        raise ValueError("little-endian host required for typed byte views")
    payload, manifest, tables = pack.payload, pack.manifest, pack.segments
    if type(payload) is not bytes or not 0 < len(payload) <= max_bytes:
        raise ValueError("reader must return bounded immutable payload bytes")
    if (type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1
            or tables != manifest.get("tables") or set(tables) != set(_TABLES)
            or manifest.get("byte_order") != "little" or manifest.get("alignment") != 64
            or manifest.get("payload_bytes") != len(payload) or manifest.get("payload_sha256") != _sha_bytes(payload)):
        raise ValueError("unexpected corpus schema/tables")
    ranges, counts = [], {}
    for name, (dtype, width) in _TABLES.items():
        info = tables[name]
        if set(info) != {"dtype", "shape", "offset", "nbytes"} or info["dtype"] != dtype:
            raise ValueError("unexpected segment metadata/dtype")
        shape = info["shape"]
        if not isinstance(shape, (list, tuple)) or len(shape) != (1 if width is None else 2) or any(type(n) is not int or n < 0 for n in shape):
            raise ValueError("invalid table shape")
        if width is not None and shape[1] != width:
            raise ValueError("invalid state width")
        offset, size = info["offset"], info["nbytes"]
        if type(offset) is not int or type(size) is not int or offset < 0 or offset % 64 or size != math.prod(shape) * _DTYPES[dtype][0]:
            raise ValueError("invalid alignment/declared table bytes")
        if offset > len(payload) or size > len(payload) - offset:
            raise ValueError("table outside payload")
        if size:
            ranges.append((offset, offset + size))
        counts[name] = shape[0]
    ranges.sort()
    if any(a[1] > b[0] for a, b in zip(ranges, ranges[1:])):
        raise ValueError("overlapping table ranges")
    for prefix, primary in (("states.", "states.boards"), ("observations.", "observations.state"),
                            ("roots.", "roots.state"), ("edges.", "edges.root")):
        for name in counts:
            expected = counts[primary] + (name == "roots.edge_offsets")
            if name.startswith(prefix) and counts[name] != expected:
                raise ValueError("joined table row count mismatch")
    if counts["history.offsets"] != counts["states.boards"] + 1:
        raise ValueError("history CSR row count mismatch")
    return payload, tables


def _integers(payload, tables, name):
    info = tables[name]
    fmt = "<" + _DTYPES[info["dtype"]][1] * math.prod(info["shape"])
    return struct.unpack_from(fmt, payload, info["offset"])


def _decode(torch, boards, fields):
    """Schema1 formula matched to frozen compact.py a52ba420... (CPU oracle tests)."""
    side = fields[:, 0].to(torch.int64)
    planes = (torch.arange(12, device=boards.device)[None] + side[:, None] * 6) % 12
    pieces = boards.gather(1, planes)
    squares = torch.arange(64, device=boards.device)[None].bitwise_xor(side[:, None] * 56)
    bits = pieces[:, :, None].bitwise_right_shift(squares[:, None]).bitwise_and(1).float()
    order = (torch.arange(4, device=boards.device)[None] + side[:, None] * 2) % 4
    rights = fields[:, 1, None].to(torch.int64).bitwise_right_shift(order).bitwise_and(1).float()
    ep = (squares == fields[:, 2, None]).float()[:, None]
    clock = (fields[:, 3].to(torch.int64).bitwise_and(0xffffffff).to(torch.float64) / 100).clamp(max=1).float()
    return torch.cat((bits, rights[:, :, None].expand(-1, -1, 64), ep,
                      clock[:, None, None].expand(-1, 1, 64)), dim=1).reshape(-1, 18, 8, 8)


@dataclass(frozen=True)
class RootPlan:
    key: int
    root_ids: tuple[int, ...]
    legal_count: int


@dataclass(frozen=True)
class ObservationPlan:
    key: int
    observation_ids: tuple[int, ...]


@dataclass(frozen=True)
class ScheduleSegment:
    plan_key: int
    vector: int
    offset: int
    count: int
    sha256: str


@dataclass(frozen=True)
class PackedIngress:
    payload_bytes: int
    auxiliary_bytes: int
    packed_bytes: int
    packed_sha256: str
    auxiliary_sha256: str
    segments: tuple[ScheduleSegment, ...]


def _history_vectors(states, offsets):
    keys, starts = [], [0]
    for state in states:
        begin, end = offsets[state:state + 2]
        keys.extend(range(begin, end))
        starts.append(len(keys))
    return tuple(keys), tuple(starts)


def _output_bytes(states, offsets, extra):
    history = sum(offsets[s + 1] - offsets[s] for s in states)
    return len(states) * (4608 + 157) + (len(states) + 1 + history) * 8 + extra


def _scheduled_blob(payload, tables, schedule, *, max_aux_bytes, max_batch_bytes):
    """CPU-only compilation of the fixed schedule into aligned index vectors."""
    counts = _integers(payload, tables, "roots.legal_count")
    roots = _integers(payload, tables, "roots.state")
    starts = _integers(payload, tables, "roots.edge_offsets")
    children = _integers(payload, tables, "edges.child_state")
    observations = _integers(payload, tables, "observations.state")
    history = _integers(payload, tables, "history.offsets")
    state_count = tables["states.boards"]["shape"][0]
    if (len(history) != state_count + 1 or history[0] != 0 or history[-1] != tables["history.keys"]["shape"][0]
            or any(a > b for a, b in zip(history, history[1:]))
            or starts[0] != 0 or starts[-1] != len(children) or any(a > b for a, b in zip(starts, starts[1:]))
            or any(s < 0 or s >= state_count for s in roots + children + observations)):
        raise ValueError("invalid structural joins/history CSR")
    specs, root_plans, observation_plans = [], [], []
    seen_roots, seen_observations = [], []
    estimated_cursor = len(payload)

    def admit_vectors(lengths):
        nonlocal estimated_cursor
        for length in lengths:
            estimated_cursor = (estimated_cursor + 63) // 64 * 64 + length * 8
        if (estimated_cursor + 63) // 64 * 64 - len(payload) > max_aux_bytes:
            raise ValueError("aligned schedule extension byte cap exceeded before vector allocation")

    def history_count(states):
        return sum(history[s + 1] - history[s] for s in states)

    for batch in schedule["root_batches"]:
        ids, k = tuple(batch["roots"]), batch["legal_count"]
        if (type(k) is not int or k < 1 or not ids or len(ids) != len(set(ids))
                or any(type(r) is not int or r < 0 or r >= len(roots) or counts[r] != k for r in ids)):
            raise ValueError("invalid exact-K structural batch")
        if (len(ids) + len(ids) * k) * 4608 > max_batch_bytes:
            raise ValueError("root feature output byte cap exceeded before index allocation")
        edges = tuple(e for root in ids for e in range(starts[root], starts[root + 1]))
        if len(edges) != len(ids) * k:
            raise ValueError("incomplete structural root")
        root_states = tuple(roots[r] for r in ids)
        child_states = tuple(children[e] for e in edges)
        if _output_bytes(root_states + child_states, history, len(ids) * 4 + len(edges) * 40 + 8) > max_batch_bytes:
            raise ValueError("declared root schedule output byte cap exceeded")
        admit_vectors((len(ids), len(edges), history_count(root_states), len(ids) + 1,
                       history_count(child_states), len(edges) + 1))
        plan = RootPlan(len(specs) + 1, ids, k)
        specs.append((plan, (ids, edges, *_history_vectors(root_states, history), *_history_vectors(child_states, history))))
        root_plans.append(plan)
        seen_roots.extend(ids)
    for batch in schedule["observation_batches"]:
        ids = tuple(batch)
        if (not ids or len(ids) != len(set(ids))
                or any(type(o) is not int or o < 0 or o >= len(observations) for o in ids)):
            raise ValueError("invalid observation schedule batch")
        if len(ids) * 4608 > max_batch_bytes:
            raise ValueError("observation feature output byte cap exceeded before index allocation")
        states = tuple(observations[o] for o in ids)
        if _output_bytes(states, history, len(ids) * 40) > max_batch_bytes:
            raise ValueError("declared observation schedule output byte cap exceeded")
        admit_vectors((len(ids), history_count(states), len(ids) + 1))
        plan = ObservationPlan(len(specs) + 1, ids)
        specs.append((plan, (ids, *_history_vectors(states, history))))
        observation_plans.append(plan)
        seen_observations.extend(ids)
    if sorted(seen_roots) != list(range(len(roots))) or sorted(seen_observations) != list(range(len(observations))):
        raise ValueError("schedule must cover each complete root/observation exactly once")
    # Size every vector before allocation. Offsets refer to the complete owned
    # blob, while the external immutable pack and its SHA remain unchanged.
    cursor, segments = len(payload), []
    for plan, vectors in specs:
        for number, vector in enumerate(vectors):
            cursor = (cursor + 63) // 64 * 64
            segments.append((plan.key, number, cursor, vector))
            cursor += len(vector) * 8
    cursor = (cursor + 63) // 64 * 64
    if cursor - len(payload) > max_aux_bytes:
        raise ValueError("aligned schedule extension byte cap exceeded")
    blob = bytearray(cursor)
    blob[:len(payload)] = payload
    descriptors = []
    for key, number, offset, vector in segments:
        raw = struct.pack("<" + "q" * len(vector), *vector)
        blob[offset:offset + len(raw)] = raw
        descriptors.append(ScheduleSegment(key, number, offset, len(vector), _sha_bytes(raw)))
    ingress = PackedIngress(len(payload), cursor - len(payload), cursor, _sha_bytes(blob),
                            _sha_bytes(blob[len(payload):]), tuple(descriptors))
    return blob, ingress, tuple(plan for plan, _ in specs), (tuple(root_plans), tuple(observation_plans))


@dataclass(frozen=True)
class StateBatch:
    """Exact simulator sidecars, row-major (children flatten B*K).

    uint64/uint32 bit patterns remain signed int64/int32 tensors. Histories are
    CSR keys preceding each current state, not schema1 feature planes. CPU
    origin0 and native epoch/node -1 remain explicit, not invented HIP IDs.
    """
    boards: Any
    fields: Any
    hashes: Any
    terminal: Any
    family_ids: Any
    origin_context: Any
    origin_epoch: Any
    origin_node: Any
    history_offsets: Any
    history_keys: Any


@dataclass(frozen=True)
class RootTargets:
    """Successor supervision, never inputs to root decision/branch forward.

    midpoint/halfwidth are child-mover common-reference intervals using the
    PARENT query's material. reported_cp retains raw parent-mover UCI CP. There
    is no second negation here. Terminal children remain present but ordinary
    score supervision is masked. Successor features support the frozen eight
    role-map/board decoder targets without making them root decision inputs.
    """
    features: Any
    midpoint: Any
    halfwidth: Any
    ordinary: Any
    terminal: Any
    reported_cp: Any
    child_state_ids: Any
    states: StateBatch


@dataclass(frozen=True)
class RootBatch:
    features: Any
    legal_actions: Any
    family_ids: Any
    root_ids: tuple[int, ...]
    targets: RootTargets
    states: StateBatch


@dataclass(frozen=True)
class ObservationBatch:
    """Standalone current-query-mover reference intervals and exact states."""
    features: Any
    midpoint: Any
    halfwidth: Any
    ordinary: Any
    terminal: Any
    reported_cp: Any
    family_ids: Any
    observation_ids: tuple[int, ...]
    states: StateBatch


class ResidentCorpus:
    """Private owned tables; create via load_resident, not unvalidated tensors."""
    def __init__(self, token, *, torch, storage, tables, payload, ingress, prepared_plans, prepared_schedule, stage, manifest_sha256,
                 max_batch_bytes, max_schedule_bytes):
        if token is not _ADMITTED:
            raise ValueError("use strict load_resident admission")
        self._torch, self._storage, self._stage = torch, storage, stage
        self._views = {}
        for name, info in tables.items():
            self._views[name] = storage.narrow(0, info["offset"], info["nbytes"]).view(getattr(torch, info["dtype"])).reshape(info["shape"])
        self._root_counts = _integers(payload, tables, "roots.legal_count")
        self._root_offsets = _integers(payload, tables, "roots.edge_offsets")
        self._root_states = _integers(payload, tables, "roots.state")
        self._child_states = _integers(payload, tables, "edges.child_state")
        self._observation_states = _integers(payload, tables, "observations.state")
        self._history_offsets = _integers(payload, tables, "history.offsets")
        self._observation_count = tables["observations.state"]["shape"][0]
        self._plans, self._next_key = {}, len(prepared_plans) + 1
        self._max_batch_bytes, self._max_schedule_bytes = max_batch_bytes, max_schedule_bytes
        self._schedule_bytes = ingress.auxiliary_bytes
        self._prepared_schedule = prepared_schedule
        self.ingress = ingress
        for plan in prepared_plans:
            segments = tuple(s for s in ingress.segments if s.plan_key == plan.key)
            tensors = tuple(storage.narrow(0, s.offset, s.count * 8).view(torch.int64) for s in segments)
            self._plans[plan.key] = (plan, tensors)
        self.manifest_sha256 = manifest_sha256

    @property
    def stage(self):
        return self._stage

    @property
    def device(self):
        return self._storage.device

    @property
    def payload_bytes(self):
        return self.ingress.payload_bytes

    def _indices(self, ids, cardinality):
        ids = tuple(ids)
        if not ids or any(type(n) is not int or n < 0 or n >= cardinality for n in ids) or len(ids) != len(set(ids)):
            raise ValueError("nonempty unique in-range structural IDs required")
        return ids

    def _register(self, plan, *indices):
        if self.device.type != "cpu":
            raise ValueError("GPU plans must be in the admitted packed structural schedule")
        needed = sum(len(index) for index in indices) * 8
        if needed > self._max_schedule_bytes - self._schedule_bytes:
            raise ValueError("prepared schedule index byte cap exceeded")
        # CPU-only supplementary mechanical plans. GPU plans are prepacked.
        tensors = tuple(self._torch.tensor(index, dtype=self._torch.int64, device=self.device) for index in indices)
        self._plans[plan.key] = (plan, tensors)
        self._next_key += 1
        self._schedule_bytes += needed
        return plan

    def _history_indices(self, states):
        return _history_vectors(states, self._history_offsets)

    def _batch_cap(self, feature_rows, states, *, extra):
        if feature_rows != len(states):
            raise ValueError("feature/state geometry mismatch")
        needed = _output_bytes(states, self._history_offsets, extra)
        if needed > self._max_batch_bytes:
            raise ValueError("gather output byte cap exceeded")

    def prepare_roots(self, root_ids):
        ids = self._indices(root_ids, len(self._root_counts))
        k = self._root_counts[ids[0]]
        if k < 1 or any(self._root_counts[n] != k for n in ids):
            raise ValueError("complete roots must share exact positive K; no padding")
        edges = []
        for root in ids:
            begin, end = self._root_offsets[root:root + 2]
            if end - begin != k:
                raise ValueError("complete root edge CSR mismatch")
            edges.extend(range(begin, end))
        roots = tuple(self._root_states[r] for r in ids)
        children = tuple(self._child_states[e] for e in edges)
        self._batch_cap(len(ids) + len(edges), roots + children, extra=len(ids) * 4 + len(edges) * 40 + 8)
        return self._register(RootPlan(self._next_key, ids, k), ids, edges,
                              *self._history_indices(roots), *self._history_indices(children))

    def root_schedule(self, *, batch_size: int, seed: int = 0, root_ids=None):
        """Deterministic structural permutations, with each family/root intact.

        Groups are bucketed by exact K, permuted using SHA256(seed,domain,row) and
        chunked without dropping the final batch. This does not implement loss
        weighting or approve a score-dependent sampling policy.
        """
        if type(batch_size) is not int or batch_size < 1 or type(seed) is not int:
            raise ValueError("positive batch size and integer seed required")
        ids = self._indices(range(len(self._root_counts)) if root_ids is None else root_ids, len(self._root_counts))
        buckets = {}
        for root in ids:
            buckets.setdefault(self._root_counts[root], []).append(root)
        plans = []
        for k in sorted(buckets):
            group = sorted(buckets[k], key=lambda i: hashlib.sha256(f"{seed}:root:{i}".encode()).hexdigest())
            plans.extend(self.prepare_roots(group[i:i + batch_size]) for i in range(0, len(group), batch_size))
        return tuple(plans)

    def prepare_structural_schedule(self):
        """Use the exact reader-validated one-epoch structural schedule.

        It remains an unsealed mechanical schedule, not permission to learn.
        Repeated calls return the same immutable plans without new ingress.
        """
        return self._prepared_schedule

    def prepare_observations(self, observation_ids):
        ids = self._indices(observation_ids, self._observation_count)
        states = tuple(self._observation_states[i] for i in ids)
        self._batch_cap(len(ids), states, extra=len(ids) * 40)
        return self._register(ObservationPlan(self._next_key, ids), ids, *self._history_indices(states))

    def _plan(self, plan, kind):
        stored = self._plans.get(getattr(plan, "key", None))
        if type(plan) is not kind or stored is None or stored[0] is not plan:
            raise ValueError("plan must belong to this corpus admission")
        return stored[1]

    def _features(self, state_ids):
        return _decode(self._torch, self._views["states.boards"].index_select(0, state_ids),
                       self._views["states.fields"].index_select(0, state_ids))

    def _states(self, states, history_keys, history_offsets):
        v = self._views
        return StateBatch(*(v[name].index_select(0, states) for name in (
            "states.boards", "states.fields", "states.hash", "states.terminal", "states.family",
            "states.origin_context", "states.origin_epoch", "states.origin_node")),
            history_offsets.clone(), v["history.keys"].index_select(0, history_keys))

    def roots(self, plan: RootPlan) -> RootBatch:
        root_ids, edge_ids, root_keys, root_offsets, child_keys, child_offsets = self._plan(plan, RootPlan)
        v, torch, b, k = self._views, self._torch, len(plan.root_ids), plan.legal_count
        states = v["roots.state"].index_select(0, root_ids).to(torch.int64)
        children = v["edges.child_state"].index_select(0, edge_ids).to(torch.int64)
        terminal = v["states.terminal"].index_select(0, children).bool().reshape(b, k)
        ordinary = v["edges.ordinary"].index_select(0, edge_ids).bool().reshape(b, k) & ~terminal
        targets = RootTargets(self._features(children).reshape(b, k, 18, 8, 8),
                              v["edges.midpoint"].index_select(0, edge_ids).reshape(b, k),
                              v["edges.halfwidth"].index_select(0, edge_ids).reshape(b, k), ordinary, terminal,
                              v["edges.reported_cp"].index_select(0, edge_ids).reshape(b, k), children.reshape(b, k),
                              self._states(children, child_keys, child_offsets))
        return RootBatch(self._features(states), v["edges.action"].index_select(0, edge_ids).to(torch.int64).reshape(b, k),
                         v["roots.family"].index_select(0, root_ids), plan.root_ids, targets,
                         self._states(states, root_keys, root_offsets))

    def observations(self, plan: ObservationPlan) -> ObservationBatch:
        indices, keys, offsets = self._plan(plan, ObservationPlan)
        v, torch = self._views, self._torch
        states = v["observations.state"].index_select(0, indices).to(torch.int64)
        terminal = v["states.terminal"].index_select(0, states).bool()
        return ObservationBatch(self._features(states), v["observations.midpoint"].index_select(0, indices),
                                v["observations.halfwidth"].index_select(0, indices),
                                v["observations.ordinary"].index_select(0, indices).bool() & ~terminal, terminal,
                                v["observations.reported_cp"].index_select(0, indices),
                                v["states.family"].index_select(0, states), plan.observation_ids,
                                self._states(states, keys, offsets))


_ADMITTED = object()


def load_resident(path, *, expected_manifest_sha256: str, expected_source_sha256: str,
                  stage: str, gate: EvaluationGate | None = None, device="cpu",
                  max_bytes=64 << 20, max_metadata_bytes=64 << 20,
                  max_batch_bytes=64 << 20, max_schedule_bytes=8 << 20,
                  rocm_admission_path=None, rocm_admission_sha256=None):
    """Load one sealed split; CPU semantic validation precedes Torch allocation.

    Caps bound payload, metadata, gathered *output* bytes and persistent schedule
    indices separately. They do not bound allocator rounding, feature-decoder
    transient tensors or the caller's retained batches. Callers must consume a
    bounded number of batches. GPU success is not currently admitted by evidence.
    """
    for cap in (max_bytes, max_metadata_bytes, max_batch_bytes, max_schedule_bytes):
        if type(cap) is not int or cap < 1:
            raise ValueError("positive integer byte caps required")
    manifest_sha, source_sha = _pin(expected_manifest_sha256), _pin(expected_source_sha256)
    if stage not in ("train", "validation_eval", "test_eval"):
        raise ValueError("explicit train/validation_eval/test_eval stage required")
    split = "train" if stage == "train" else stage.removesuffix("_eval")
    # Reject unsealed held-out access BEFORE the CPU reader opens label bytes.
    _stage_gate(stage, split, manifest_sha, source_sha, gate)
    from . import packing
    if _hash_regular(packing.__file__, 1 << 20) != source_sha:
        raise ValueError("executing semantic reader source pin mismatch")
    pack = packing.read_pack(path, max_bytes=max_bytes, max_metadata_bytes=max_metadata_bytes,
                     expected_manifest_sha256=manifest_sha, expected_source_sha256=source_sha,
                     purpose="train" if stage == "train" else "audit", expected_split=split)
    if pack.manifest.get("split") != split:
        raise ValueError("reader split does not match stage")
    payload, tables = _layout(pack, max_bytes)
    blob, ingress, prepared_plans, prepared_schedule = _scheduled_blob(
        payload, tables, pack.schedule, max_aux_bytes=max_schedule_bytes, max_batch_bytes=max_batch_bytes)
    is_cpu = str(device) == "cpu"
    if is_cpu:
        if rocm_admission_path is not None or rocm_admission_sha256 is not None:
            raise ValueError("CPU loading does not accept GPU admission")
    else:
        if not str(device).startswith("cuda:") or rocm_admission_path is None or rocm_admission_sha256 is None:
            raise ValueError("explicit indexed ROCm device and pinned admission required")
        record = _json_record(rocm_admission_path, rocm_admission_sha256)
        if record != {"kind": "semantic_resident_rocm_admission_v1", "corpus_manifest_sha256": manifest_sha,
                      "packing_source_sha256": source_sha, "resident_source_sha256": _sha_bytes(Path(__file__).read_bytes()),
                      "stage": stage, "device_arch": "gfx1030"}:
            raise ValueError("GPU admission source/corpus/stage mismatch")
        if os.environ.get("HSA_OVERRIDE_GFX_VERSION"):
            raise ValueError("hardware override cannot admit resident GPU upload")
    import torch
    target = torch.device(device)
    if not is_cpu:
        if torch.version.hip is None or not torch.cuda.is_available():
            raise ValueError("actual ROCm device required; CPU substitution forbidden")
        props = torch.cuda.get_device_properties(target)
        if getattr(props, "gcnArchName", "").split(":")[0] != "gfx1030":
            raise ValueError("ROCm architecture does not match admission")
    source = torch.frombuffer(blob, dtype=torch.uint8)
    if is_cpu:
        storage = torch.empty(len(blob), dtype=torch.uint8)
        storage.copy_(source)
    else:
        # Keep pinned host owner alive through success/failure synchronization.
        with torch.cuda.device(target):
            storage = torch.empty(len(blob), dtype=torch.uint8, pin_memory=True)
            storage.copy_(source)
            try:
                uploaded = storage.to(target, non_blocking=True)
                torch.cuda.synchronize(target)
            except BaseException:
                torch.cuda.synchronize(target)
                raise
        storage = uploaded
    if storage.data_ptr() % 64:
        raise ValueError("owned storage is not 64-byte aligned")
    return ResidentCorpus(_ADMITTED, torch=torch, storage=storage, tables=tables, payload=payload,
                          ingress=ingress, prepared_plans=prepared_plans, prepared_schedule=prepared_schedule,
                          stage=stage, manifest_sha256=manifest_sha, max_batch_bytes=max_batch_bytes,
                          max_schedule_bytes=max_schedule_bytes)
