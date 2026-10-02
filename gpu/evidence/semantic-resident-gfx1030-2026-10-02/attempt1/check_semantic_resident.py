#!/usr/bin/env python3
"""TRAIN-only actual-device serialization QA; no learning or quality metrics.

QA deliberately downloads batches to compare exact bytes. Such downloads are
outside the resident training path. One instrumented loader Tensor.to(device)
call does not prove one physical DMA or zero allocator/runtime transfers.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib
import json
import os
from pathlib import Path
import sys
import time
import types
from unittest.mock import patch


FROZEN = {
    "compact.py": "a52ba420dca331edf999e539cc37ea5b9208630ed0f9d9587fe24501b2c944d3",
    "data.py": "62eefa0ec1e0ae385c8200cf11203a0bd4968f9d42597d0d00dd5d13ad9dad16",
    "features.py": "3aecc21de78d1c586543117aca78b3e8920b84b3694377d700228425a5e67cfc",
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tensors(value, prefix=""):
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from tensors(getattr(value, field.name), prefix + "." + field.name)
    elif hasattr(value, "dtype") and hasattr(value, "device"):
        yield prefix, value


def equal_batch(torch, cpu, gpu):
    require(type(cpu) is type(gpu), "batch class changed")
    left, right = dict(tensors(cpu)), dict(tensors(gpu))
    require(left.keys() == right.keys(), "batch tensor fields changed")
    checked = 0
    for name, a in left.items():
        b = right[name]
        require(b.device.type == "cuda", "batch escaped resident GPU: " + name)
        require(a.shape == b.shape and a.dtype == b.dtype and torch.equal(a, b.cpu()),
                "CPU/GPU tensor bytes differ: " + name)
        checked += a.numel() * a.element_size()
    id_field = "root_ids" if hasattr(cpu, "root_ids") else "observation_ids"
    require(getattr(cpu, id_field) == getattr(gpu, id_field), "structural IDs changed")
    return checked


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pack", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--packing-source-sha256", required=True)
    parser.add_argument("--resident-source-sha256", required=True)
    parser.add_argument("--admission", type=Path, required=True)
    parser.add_argument("--admission-sha256", required=True)
    parser.add_argument("--frozen-neural", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    require(not args.output.exists(), "output must be new")
    require(not os.environ.get("HSA_OVERRIDE_GFX_VERSION"), "hardware override forbidden")
    workspace = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(workspace))
    from experiments.semantic_corpus import packing, resident
    require(digest(packing.__file__) == args.packing_source_sha256, "packer source changed")
    require(digest(resident.__file__) == args.resident_source_sha256, "resident source changed")
    require(digest(args.pack / "manifest.json") == args.manifest_sha256, "manifest changed")
    require(digest(args.admission) == args.admission_sha256, "admission changed")
    for name, pin in FROZEN.items():
        require(digest(args.frozen_neural / name) == pin, "frozen reference changed: " + name)
    import torch
    started = time.monotonic()
    cpu = resident.load_resident(args.pack, expected_manifest_sha256=args.manifest_sha256,
                                expected_source_sha256=args.packing_source_sha256, stage="train")
    print("CPU TRAIN admission complete", flush=True)
    calls = []
    original_to = torch.Tensor.to

    def recorded_to(tensor, *positional, **keywords):
        calls.append({"shape": list(tensor.shape), "dtype": str(tensor.dtype),
                      "from_device": str(tensor.device), "args": [str(a) for a in positional],
                      "kwargs": {k: str(v) for k, v in keywords.items()}})
        return original_to(tensor, *positional, **keywords)

    torch.cuda.reset_peak_memory_stats(args.device)
    with patch.object(torch.Tensor, "to", recorded_to):
        gpu = resident.load_resident(args.pack, expected_manifest_sha256=args.manifest_sha256,
                                    expected_source_sha256=args.packing_source_sha256, stage="train",
                                    device=args.device, rocm_admission_path=args.admission,
                                    rocm_admission_sha256=args.admission_sha256)
    require(len(calls) == 1 and calls[0]["args"] == [args.device]
            and calls[0]["dtype"] == "torch.uint8"
            and calls[0]["shape"] == [gpu.ingress.packed_bytes], "loader packed upload call changed")
    require(cpu.ingress == gpu.ingress, "CPU/GPU ingress descriptors differ")
    print("gfx1030 packed TRAIN upload complete", flush=True)
    module = types.ModuleType("_resident_actual_frozen")
    module.__path__ = [str(args.frozen_neural)]
    sys.modules[module.__name__] = module
    previous_bytecode = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        compact = importlib.import_module(module.__name__ + ".compact")
    finally:
        sys.dont_write_bytecode = previous_bytecode
    cpu_schedule, gpu_schedule = cpu.prepare_structural_schedule(), gpu.prepare_structural_schedule()
    require(cpu_schedule == gpu_schedule, "prepared schedules differ")
    require(gpu.prepare_structural_schedule() is gpu_schedule, "schedule reconstruction allocated new plans")
    roots_seen, observations_seen, edge_count, checked_bytes = [], [], 0, 0
    frozen_rows = 0

    def forbidden(*args, **kwargs):
        raise AssertionError("new host index Tensor construction during prepared GPU gather")

    for cpu_plan, gpu_plan in zip(cpu_schedule[0], gpu_schedule[0]):
        expected = cpu.roots(cpu_plan)
        with patch.object(torch, "tensor", forbidden), patch.object(torch, "frombuffer", forbidden):
            actual = gpu.roots(gpu_plan)
        checked_bytes += equal_batch(torch, expected, actual)
        require(actual.legal_actions.dtype == torch.int64, "action dtype differs")
        require(actual.legal_actions.shape == (len(gpu_plan.root_ids), gpu_plan.legal_count),
                "complete root geometry differs")
        require(torch.equal(actual.features, compact.decode_features(actual.states.boards, actual.states.fields)),
                "frozen parent feature formula differs")
        require(torch.equal(actual.targets.features.flatten(0, 1),
                            compact.decode_features(actual.targets.states.boards, actual.targets.states.fields)),
                "frozen successor feature formula differs")
        roots_seen.extend(gpu_plan.root_ids)
        edge_count += actual.legal_actions.numel()
        frozen_rows += actual.states.boards.shape[0] + actual.targets.states.boards.shape[0]
        del actual, expected
    for cpu_plan, gpu_plan in zip(cpu_schedule[1], gpu_schedule[1]):
        expected = cpu.observations(cpu_plan)
        with patch.object(torch, "tensor", forbidden), patch.object(torch, "frombuffer", forbidden):
            actual = gpu.observations(gpu_plan)
        checked_bytes += equal_batch(torch, expected, actual)
        require(torch.equal(actual.features, compact.decode_features(actual.states.boards, actual.states.fields)),
                "frozen observation feature formula differs")
        observations_seen.extend(gpu_plan.observation_ids)
        frozen_rows += actual.states.boards.shape[0]
        del actual, expected
    manifest = json.loads((args.pack / "manifest.json").read_bytes())
    require(sorted(roots_seen) == list(range(manifest["counts"]["roots"])), "root coverage changed")
    require(sorted(observations_seen) == list(range(manifest["counts"]["observations"])), "observation coverage changed")
    require(edge_count == manifest["counts"]["edges"], "complete edge coverage changed")
    # Public batches must own storage. Mutate every returned tensor, then gather
    # again and compare against untouched independent CPU corpus state.
    first_cpu, first_gpu = cpu_schedule[0][0], gpu_schedule[0][0]
    changed = gpu.roots(first_gpu)
    for _, value in tensors(changed):
        value.zero_()
    require(torch.equal(cpu.roots(first_cpu).features, gpu.roots(first_gpu).features.cpu()),
            "public feature mutation damaged private owner")
    equal_batch(torch, cpu.roots(first_cpu), gpu.roots(first_gpu))
    # Supplemental GPU schedules are forbidden after the single packed ingress.
    try:
        gpu.prepare_roots([0])
    except ValueError:
        pass
    else:
        raise AssertionError("GPU accepted new host-produced schedule")
    torch.cuda.synchronize(args.device)
    props = torch.cuda.get_device_properties(args.device)
    report = {
        "kind": "semantic_resident_train_actual_gfx1030_mechanical_qa_v1", "status": "passed",
        "stage": "train", "validation_test_labels_loaded": False, "optimizers_or_model_updates": 0,
        "quality_metrics": "not_computed", "manifest_sha256": args.manifest_sha256,
        "packing_source_sha256": args.packing_source_sha256, "resident_source_sha256": args.resident_source_sha256,
        "admission_sha256": args.admission_sha256, "checker_source_sha256": digest(__file__),
        "frozen_reference_sha256": FROZEN, "counts": manifest["counts"],
        "root_batches": len(gpu_schedule[0]), "observation_batches": len(gpu_schedule[1]),
        "complete_edges_compared": edge_count, "frozen_feature_rows_compared": frozen_rows,
        "batch_tensor_bytes_compared": checked_bytes, "all_tensor_fields_equal": True,
        "owned_public_mutation_check": "passed", "prepared_gathers_without_host_index_construction": "passed",
        "new_gpu_schedule_rejected": True, "instrumented_loader_tensor_to_calls": calls,
        "upload_scope": "one loader Tensor.to(device) call; physical DMA and runtime transfer counts unmeasured",
        "qa_scope": "explicit diagnostic batch downloads; no throughput or zero-transfer training claim",
        "ingress": dataclasses.asdict(gpu.ingress), "python": sys.version, "torch": torch.__version__,
        "hip": torch.version.hip, "device": str(gpu.device), "device_name": props.name,
        "gcn_arch": props.gcnArchName, "peak_allocated_bytes": torch.cuda.max_memory_allocated(args.device),
        "wall_seconds": time.monotonic() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("xb") as handle:
        handle.write((json.dumps(report, indent=2, sort_keys=True) + "\n").encode())
    print(json.dumps({k: report[k] for k in ("status", "counts", "complete_edges_compared", "wall_seconds")}), flush=True)


if __name__ == "__main__":
    main()
