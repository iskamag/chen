"""Compact, exact-state replay with resident feature and legal-support decoding.

Signed int64 pieces preserve the bit pattern of twelve absolute-color uint64
bitboards. State int32 columns are side, rights, raw EP (-1 absent), halfmove,
fullmove; the last two transport uint32 bit patterns. History handles are opaque
references, not repetition context. CUDA is Torch's spelling for ROCm devices.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Mapping

import torch

from .data import DeviceReplay, SyntheticCorpus, generate_synthetic
from .features import ACTION_COUNT, PIECES, SCHEMA_VERSION, encode_fen, square_index

COMPACT_VERSION = 1
STATE_FIELDS = ("side", "rights", "ep", "halfmove", "fullmove")
DEFAULT_SLOTS = 256
LABEL_FIELDS = ("actions", "target_actions", "material_proxy", "terminal", "game_ids")


def compact_schema(max_slots=DEFAULT_SLOTS):
    if type(max_slots) is not int or not 1 <= max_slots <= ACTION_COUNT:
        raise ValueError("max_slots must be an integer in [1, 20480]")
    return {
        "pieces": ((12,), torch.int64), "state": ((5,), torch.int32),
        "next_pieces": ((12,), torch.int64), "next_state": ((5,), torch.int32),
        "history_handles": ((), torch.int64), "next_history_handles": ((), torch.int64),
        "legal_actions": ((max_slots,), torch.int16), "legal_counts": ((), torch.int32),
        "actions": ((), torch.int64), "target_actions": ((), torch.int64),
        "material_proxy": ((), torch.float32), "terminal": ((), torch.bool),
        "game_ids": ((), torch.int64),
    }


def unsigned_clock(column: torch.Tensor):
    """Interpret int32 transport as uint32 without a host extraction."""
    return column.to(torch.int64).bitwise_and(0xFFFFFFFF)


def pack_fens(fens: list[str] | tuple[str, ...]):
    """CPU ingestion only; exact raw rule fields, no lossy dense-feature inversion."""
    piece_rows, state_rows = [], []
    for fen in fens:
        encode_fen(fen)  # schema syntax validation, not an independent legality proof
        placement, side, rights, ep, halfmove, fullmove = fen.split()
        halfmove, fullmove = int(halfmove), int(fullmove)
        if not 0 <= halfmove <= 65535 or not 1 <= fullmove <= 0xFFFFFFFF:
            raise ValueError("clocks exceed the exact Rust packet domain")
        boards = [0] * 12
        for rank, row in enumerate(placement.split("/")):
            file = 0
            for character in row:
                if character.isdigit():
                    file += int(character)
                else:
                    plane = PIECES.index(character.upper()) + (0 if character.isupper() else 6)
                    boards[plane] |= 1 << (file + 8 * (7 - rank))
                    file += 1
        piece_rows.append([bits if bits < 1 << 63 else bits - (1 << 64) for bits in boards])
        castling = sum(1 << index for index, right in enumerate("KQkq") if right in rights)
        state_rows.append([int(side == "b"), castling, -1 if ep == "-" else square_index(ep),
                           halfmove, fullmove if fullmove < 1 << 31 else fullmove - (1 << 32)])
    return (torch.tensor(piece_rows, dtype=torch.int64).reshape(-1, 12),
            torch.tensor(state_rows, dtype=torch.int32).reshape(-1, 5))


@dataclass(frozen=True)
class CompactCorpus:
    tensors: dict[str, torch.Tensor]
    manifest: dict[str, object]

    def to(self, device):
        return CompactCorpus({key: value.to(device) for key, value in self.tensors.items()}, self.manifest)


def compact_corpus(corpus: SyntheticCorpus, max_slots=DEFAULT_SLOTS) -> CompactCorpus:
    """Convert a bounded CPU oracle corpus; refuse an implicit device-to-host route."""
    compact_schema(max_slots)
    if any(value.device.type != "cpu" for value in corpus.tensors.values()):
        raise ValueError("compact_corpus requires explicit CPU ingestion data")
    count = len(corpus.records)
    if corpus.tensors["features"].shape[0] != count:
        raise ValueError("record/tensor count mismatch")
    pieces, state = pack_fens(tuple(str(row["fen"]) for row in corpus.records))
    next_pieces, next_state = pack_fens(tuple(str(row["next_fen"]) for row in corpus.records))
    counts = corpus.tensors["legal_mask"].sum(-1)
    if bool((counts > max_slots).any()):
        raise ValueError("legal support exceeds max_slots; record cannot be truncated")
    actions = torch.full((count, max_slots), -1, dtype=torch.int16)
    for index in range(count):
        listed = corpus.tensors["legal_mask"][index].nonzero().flatten()
        actions[index, :listed.numel()] = listed.to(torch.int16)
    tensors = {"pieces": pieces, "state": state, "next_pieces": next_pieces,
               "next_state": next_state, "legal_actions": actions,
               "legal_counts": counts.to(torch.int32),
               "history_handles": torch.full((count,), -1, dtype=torch.int64),
               "next_history_handles": torch.full((count,), -1, dtype=torch.int64),
               **{key: corpus.tensors[key].clone() for key in LABEL_FIELDS}}
    return CompactCorpus(tensors, {**corpus.manifest, "compact_version": COMPACT_VERSION,
                                 "max_slots": max_slots, "history_scope": "absent handles (-1); no repetition input",
                                 "source_schema_version": SCHEMA_VERSION,
                                 "ep": "raw FEN target, -1 absent",
                                 "clock_transport": "int32 uint32 bit pattern; halfmove<=65535"})


def decode_features(pieces: torch.Tensor, state: torch.Tensor, dtype=torch.float32):
    """Expand exact states on their device, without scalar reads or host copies.

    Content must have passed producer/replay admission. Shape/dtype checks use
    metadata only; validation is deliberately outside the asynchronous hot path.
    """
    if pieces.ndim != 2 or pieces.shape[1] != 12 or pieces.dtype != torch.int64:
        raise ValueError("expected int64 pieces [B,12]")
    if state.shape != (pieces.shape[0], 5) or state.dtype != torch.int32 or state.device != pieces.device:
        raise ValueError("expected resident int32 state [B,5]")
    if dtype not in (torch.float32, torch.float16, torch.bfloat16):
        raise ValueError("unsupported feature precision")
    device = pieces.device
    side = state[:, 0].to(torch.int64)
    planes = (torch.arange(12, device=device)[None] + side[:, None] * 6) % 12
    own_enemy = pieces.gather(1, planes)
    squares = torch.arange(64, device=device)[None].bitwise_xor(side[:, None] * 56)
    bits = own_enemy[:, :, None].bitwise_right_shift(squares[:, None]).bitwise_and(1).float()
    rights_order = (torch.arange(4, device=device)[None] + side[:, None] * 2) % 4
    rights = state[:, 1, None].to(torch.int64).bitwise_right_shift(rights_order).bitwise_and(1).float()
    rights = rights[:, :, None].expand(-1, -1, 64)
    ep = (squares == state[:, 2, None]).float()[:, None]
    # Compute in float32 first, matching encode_fen followed by a dtype conversion.
    # encode_fen computes Python's double ratio, then writes float32. Match that
    # rounding before reduced-precision conversion; GPU float reciprocal kernels
    # can round some integer/100 float32 divisions differently.
    halfmove = (unsigned_clock(state[:, 3]).to(torch.float64) / 100).clamp(max=1).float()
    clock = halfmove[:, None, None].expand(-1, 1, 64)
    return torch.cat((bits, rights, ep, clock), dim=1).reshape(-1, 18, 8, 8).to(dtype)


def decode_legal_mask(actions: torch.Tensor, counts: torch.Tensor):
    """Resident sparse-to-dense support; padded slots cannot overwrite action zero."""
    if actions.ndim != 2 or actions.dtype != torch.int16 or counts.shape != (actions.shape[0],) or counts.dtype != torch.int32 or counts.device != actions.device:
        raise ValueError("expected int16 legal actions [B,K] and int32 counts [B]")
    active = torch.arange(actions.shape[1], device=actions.device)[None] < counts[:, None]
    indexes = torch.where(active, actions.to(torch.int64), ACTION_COUNT)
    # Pads write only to a discarded sentinel column. Never scatter false pads
    # into a real index, where they could erase a valid legal action.
    mask = torch.zeros(actions.shape[0], ACTION_COUNT + 1, dtype=torch.bool, device=actions.device)
    mask.scatter_(1, indexes, active)
    return mask[:, :ACTION_COUNT]


def decode_batch(batch: Mapping[str, torch.Tensor], dtype=torch.float32):
    result = {key: batch[key] for key in LABEL_FIELDS}
    result.update(features=decode_features(batch["pieces"], batch["state"], dtype),
                  next_features=decode_features(batch["next_pieces"], batch["next_state"], dtype),
                  legal_mask=decode_legal_mask(batch["legal_actions"], batch["legal_counts"]))
    if "sequence_ids" in batch:
        result["sequence_ids"] = batch["sequence_ids"]
    return result


class CompactReplay:
    """Bounded single-owner ring with immutable gathered minibatches.

    Producer data must already be resident. Admission validates/synchronizes;
    sample_compact(), sample() and decode_batch() never read tensor data on CPU.
    Callers must order external producer streams before append. Gathered batches
    are independent allocations, so later overwrite/clear cannot mutate them.
    History handles are preserved opaque ids whose ownership belongs to caller.
    """

    def __init__(self, capacity, device, *, max_slots=DEFAULT_SLOTS,
                 feature_dtype=torch.float32, max_bytes=None):
        if type(capacity) is not int or capacity <= 0:
            raise ValueError("capacity must be a positive integer")
        if feature_dtype not in (torch.float32, torch.float16, torch.bfloat16):
            raise ValueError("unsupported feature precision")
        self.capacity, self.max_slots = capacity, max_slots
        self.schema = compact_schema(max_slots)
        self.feature_dtype = feature_dtype
        self.bytes_per_record = 8 + sum(math.prod(shape) * torch.empty((), dtype=dtype).element_size()
                                        for shape, dtype in self.schema.values())
        if max_bytes is not None and self.bytes_per_record * capacity > max_bytes:
            raise ValueError("compact replay exceeds configured memory budget")
        self.device = torch.empty(0, device=device).device
        self.storage = {key: torch.empty((capacity, *shape), dtype=dtype, device=self.device)
                        for key, (shape, dtype) in self.schema.items()}
        self.sequence_ids = torch.empty(capacity, dtype=torch.int64, device=self.device)
        self.size = self.cursor = self.total_inserted = 0

    def clear(self):
        self.size = self.cursor = 0

    @torch.no_grad()
    def append(self, batch: Mapping[str, torch.Tensor]):
        if set(batch) != set(self.schema):
            raise ValueError("batch keys must exactly match compact schema")
        if not isinstance(batch["pieces"], torch.Tensor) or batch["pieces"].ndim != 2:
            raise ValueError("pieces must be a batched tensor")
        count = batch["pieces"].shape[0]
        for key, (shape, dtype) in self.schema.items():
            value = batch[key]
            if not isinstance(value, torch.Tensor) or value.shape != (count, *shape) or value.dtype != dtype or value.device != self.device:
                raise ValueError(f"{key}: schema/dtype/device mismatch")
        if count == 0:
            return
        if self.total_inserted + count > torch.iinfo(torch.int64).max:
            raise ValueError("sequence id range exhausted; explicit new epoch required")
        for prefix in ("", "next_"):
            state, pieces = batch[prefix + "state"], batch[prefix + "pieces"]
            if not bool(((state[:, 0] >= 0) & (state[:, 0] <= 1) &
                         (state[:, 1] >= 0) & (state[:, 1] <= 15) &
                         (state[:, 2] >= -1) & (state[:, 2] <= 63) &
                         (unsigned_clock(state[:, 3]) <= 65535) &
                         (unsigned_clock(state[:, 4]) >= 1)).all()):
                raise ValueError(f"{prefix}state: invalid exact rule metadata")
            ep = state[:, 2]
            if not bool(((ep == -1) | ((ep >= 16) & (ep < 24)) | ((ep >= 40) & (ep < 48))).all()):
                raise ValueError(f"{prefix}state: invalid raw EP rank")
            squares = torch.arange(64, device=self.device)
            occupancy = pieces[:, :, None].bitwise_right_shift(squares).bitwise_and(1)
            if not bool((occupancy.sum(1) <= 1).all()):
                raise ValueError(f"{prefix}pieces: overlapping bitboards")
            if not bool((occupancy[:, [5, 11]].sum(-1) == 1).all()):
                raise ValueError(f"{prefix}pieces: expected one king of each color")
            if not bool((batch[prefix + "history_handles"] >= -1).all()):
                raise ValueError(f"{prefix}history_handles: invalid opaque id")
        counts, listed = batch["legal_counts"], batch["legal_actions"]
        if not bool(((counts > 0) & (counts <= self.max_slots)).all()):
            raise ValueError("current legal support empty or exceeds max_slots; no truncation")
        active = torch.arange(self.max_slots, device=self.device)[None] < counts[:, None]
        if not bool(torch.where(active, (listed >= 0) & (listed < ACTION_COUNT), listed == -1).all()):
            raise ValueError("legal support out of range or padding sentinel invalid")
        if self.max_slots > 1 and not bool(torch.where(active[:, 1:], listed[:, 1:] > listed[:, :-1], True).all()):
            raise ValueError("legal actions must be sorted unique active prefixes")
        for key in ("actions", "target_actions"):
            value = batch[key]
            if not bool(((value >= 0) & (value < ACTION_COUNT)).all()) or not bool(((listed == value[:, None]) & active).any(-1).all()):
                raise ValueError(f"{key}: action absent from exact legal support")
        if not bool((torch.isfinite(batch["material_proxy"]) & (batch["material_proxy"].abs() <= 1)).all()):
            raise ValueError("invalid material proxy")
        kept = min(count, self.capacity)
        skip = count - kept
        indexes = (torch.arange(kept, device=self.device) + self.cursor + skip) % self.capacity
        for key, value in batch.items():
            self.storage[key].index_copy_(0, indexes, value[skip:])
        identities = torch.arange(self.total_inserted + skip, self.total_inserted + count, device=self.device)
        self.sequence_ids.index_copy_(0, indexes, identities)
        self.cursor = (self.cursor + count) % self.capacity
        self.size = min(self.capacity, self.size + count)
        self.total_inserted += count

    def sample_compact(self, count, generator=None):
        if type(count) is not int or count <= 0 or self.size == 0:
            raise ValueError("sampling requires positive count and nonempty replay")
        indexes = torch.randint(self.size, (count,), device=self.device, generator=generator)
        batch = {key: value.index_select(0, indexes) for key, value in self.storage.items()}
        batch["sequence_ids"] = self.sequence_ids.index_select(0, indexes)
        return batch

    def sample(self, count, generator=None):
        return decode_batch(self.sample_compact(count, generator), self.feature_dtype)

    def manifest(self):
        return {"compact_version": COMPACT_VERSION, "feature_schema_version": SCHEMA_VERSION,
                "device": str(self.device), "capacity": self.capacity, "size": self.size,
                "cursor": self.cursor, "total_inserted": self.total_inserted,
                "max_slots": self.max_slots, "bytes_per_record": self.bytes_per_record,
                "allocated_payload_bytes": self.bytes_per_record * self.capacity,
                "history_scope": "opaque external handles, no repetition adjudication",
                "admission": "same-device exact tensors; validation synchronizes at ingress"}


def _synchronize(device):
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)


def benchmark(*, device="cpu", capacity=4096, source_samples=192, batch_size=32,
              repeats=25, steps=8, seed=20260930, max_slots=256, output=None):
    """Frozen tiled legal smoke: fixed-capacity memory and end-to-end train comparison.

    CPU oracle staging is bounded by source_samples. Only one ring exists at a
    time; no persistent full-capacity dense corpus is duplicated on the device.
    The benchmark measures resident sampling/decode/training, not a GPU producer.
    """
    from .models import build_model
    from .train import seed_all, train_step

    if min(capacity, source_samples, batch_size, repeats, steps) < 1:
        raise ValueError("benchmark counts must be positive")
    source_start = time.perf_counter()
    cpu_source = generate_synthetic(source_samples, seed, max_plies=48)
    source_seconds = time.perf_counter() - source_start
    source_start = time.perf_counter()
    packed = compact_corpus(cpu_source, max_slots)
    packing_seconds = time.perf_counter() - source_start
    results = []
    for mode in ("dense", "compact"):
        baseline_allocated = torch.cuda.memory_allocated(device) if torch.device(device).type == "cuda" else None
        baseline_reserved = torch.cuda.memory_reserved(device) if torch.device(device).type == "cuda" else None
        driver_before = torch.cuda.mem_get_info(device) if torch.device(device).type == "cuda" else None
        replay = (DeviceReplay(capacity, device) if mode == "dense" else
                  CompactReplay(capacity, device, max_slots=max_slots))
        _synchronize(replay.device)
        ingress_start = time.perf_counter()
        staging = {key: value.to(replay.device) for key, value in
                   (cpu_source.tensors if mode == "dense" else packed.tensors).items()}
        remaining = capacity
        while remaining:
            count = min(remaining, source_samples)
            replay.append({key: value[:count] for key, value in staging.items()})
            remaining -= count
        _synchronize(replay.device)
        ingress_seconds = time.perf_counter() - ingress_start
        del staging
        resident_allocated = torch.cuda.memory_allocated(replay.device) if replay.device.type == "cuda" else None
        resident_reserved = torch.cuda.memory_reserved(replay.device) if replay.device.type == "cuda" else None
        driver_resident = torch.cuda.mem_get_info(replay.device) if replay.device.type == "cuda" else None
        generator = torch.Generator(device=replay.device).manual_seed(seed + 1)
        replay.sample(batch_size, generator)  # warm sampling and decode kernels
        _synchronize(replay.device)
        start = time.perf_counter()
        for _ in range(repeats):
            sampled = replay.sample(batch_size, generator)
        _synchronize(replay.device)
        sample_seconds = (time.perf_counter() - start) / repeats
        del sampled
        seed_all(seed)
        model = build_model("relation", width=16, depth=1).to(replay.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        training_rng = torch.Generator(device=replay.device).manual_seed(seed + 2)
        train_step(model, optimizer, replay.sample(batch_size, training_rng))
        # Warmed optimizer/model are reset for identical measured training paths.
        seed_all(seed)
        model = build_model("relation", width=16, depth=1).to(replay.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
        training_rng.manual_seed(seed + 2)
        if replay.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(replay.device)
        _synchronize(replay.device)
        start = time.perf_counter()
        first = final = None
        identities = []
        for step in range(steps):
            batch = replay.sample(batch_size, training_rng)
            loss = train_step(model, optimizer, batch)
            if step == 0:
                first = loss
            final = loss
            identities.append(batch["sequence_ids"])
        _synchronize(replay.device)
        training_seconds = time.perf_counter() - start
        sequence_hash = hashlib.sha256(json.dumps(torch.stack(identities).cpu().tolist()).encode()).hexdigest()
        results.append({"mode": mode, "replay": replay.manifest(),
                        "source_ingress_and_ring_fill_seconds": ingress_seconds,
                        "sample_decode_seconds_per_batch": sample_seconds,
                        "training_seconds": training_seconds,
                        "training_positions_per_second": steps * batch_size / training_seconds,
                        "first_loss": {key: float(value) for key, value in first.items()},
                        "final_loss": {key: float(value) for key, value in final.items()},
                        "sample_sequence_sha256": sequence_hash,
                        "baseline_torch_allocated_bytes": baseline_allocated,
                        "baseline_torch_reserved_bytes": baseline_reserved,
                        "resident_torch_allocated_bytes": resident_allocated,
                        "resident_torch_reserved_bytes": resident_reserved,
                        "replay_torch_allocated_delta_bytes": resident_allocated - baseline_allocated if resident_allocated is not None else None,
                        "driver_before_free_total_bytes": driver_before,
                        "driver_resident_free_total_bytes": driver_resident,
                        "peak_torch_allocated_bytes": torch.cuda.max_memory_allocated(replay.device) if replay.device.type == "cuda" else None,
                        "torch_reserved_bytes": torch.cuda.memory_reserved(replay.device) if replay.device.type == "cuda" else None})
        del batch, replay, model, optimizer, identities, first, final, loss
        if torch.device(device).type == "cuda":
            torch.cuda.empty_cache()
            _synchronize(torch.device(device))
        results[-1].update({
            "post_cleanup_torch_allocated_bytes": torch.cuda.memory_allocated(device) if torch.device(device).type == "cuda" else None,
            "post_cleanup_torch_reserved_bytes": torch.cuda.memory_reserved(device) if torch.device(device).type == "cuda" else None,
            "post_cleanup_driver_free_total_bytes": torch.cuda.mem_get_info(device) if torch.device(device).type == "cuda" else None,
        })
    if results[0]["sample_sequence_sha256"] != results[1]["sample_sequence_sha256"]:
        raise RuntimeError("dense/compact comparisons consumed different replay rows")
    report = {"kind": "compact_resident_replay_smoke", "strength_evidence": False,
              "environment": {"torch": str(torch.__version__), "hip": torch.version.hip,
                              "device": str(device),
                              "architecture": getattr(torch.cuda.get_device_properties(device), "gcnArchName", "unknown") if torch.device(device).type == "cuda" else None},
              "source": cpu_source.manifest,
              "cpu_source_generation_seconds": source_seconds,
              "cpu_compact_conversion_seconds": packing_seconds,
              "config": {"capacity": capacity, "source_samples": source_samples,
                         "batch_size": batch_size, "repeats": repeats, "steps": steps,
                         "seed": seed, "max_slots": max_slots}, "results": results,
              "limitations": ["bounded CPU oracle ingestion, tiled legal source, no direct HIP producer",
                              "dense masks/logits still materialized per minibatch",
                              "first optimizer allocation remains in measured training",
                              "single owner and ordered stream; external history handles only",
                              "short mechanics timings; no chess strength or steady-state claim"],
              "source_sha256": {f"neural/{path.name}": hashlib.sha256(path.read_bytes()).hexdigest()
                                for path in (Path(__file__), Path(__file__).with_name("features.py"),
                                             Path(__file__).with_name("models.py"), Path(__file__).with_name("train.py"),
                                             Path(__file__).with_name("data.py"))}}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--capacity", type=int, default=4096)
    parser.add_argument("--source-samples", type=int, default=192)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=25)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260930)
    parser.add_argument("--max-slots", type=int, default=256)
    parser.add_argument("--output")
    print(json.dumps(benchmark(**vars(parser.parse_args())), indent=2))


if __name__ == "__main__":
    main()
