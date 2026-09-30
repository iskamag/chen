#!/usr/bin/env python3
"""Same-model native GPU inference parity; fixture weights imply no strength.

Run with PYTHONPATH containing the separate neural checkout and chess oracle.
All packet uploads, reference construction and output inspection are diagnostic
boundaries outside the synchronous native callback's marked copy region.
"""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path

import torch
from neural.bridge import RulePosition, position_from_fen
from neural.features import encode_fen
from neural.inference import StaticEvaluator, load_inference
from neural.models import build_model


class CopyTrace(c.Structure):
    _fields_ = [("calls", c.c_uint64 * 5), ("bytes", c.c_uint64 * 5), ("maxima", c.c_uint64 * 5)]


FENS = (
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 1 2",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 99 11",
    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 7 12",
    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 b - - 65535 4294967295",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 41 1",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("library")
    parser.add_argument("archives", nargs="+")
    parser.add_argument("--trace-copies", action="store_true")
    parser.add_argument("--checkpoint-directory")
    args = parser.parse_args()
    assert torch.version.hip and torch.cuda.is_available()
    assert "gfx1030" in torch.cuda.get_device_properties(0).gcnArchName
    runtime_path = Path(torch.__file__).parent / "lib/libamdhip64.so"
    runtime = c.CDLL(str(runtime_path), mode=c.RTLD_GLOBAL)
    lib = c.CDLL(args.library, mode=c.RTLD_GLOBAL)
    lib.chen_neural_create.restype = c.c_void_p
    lib.chen_neural_create.argtypes = [c.c_void_p, c.c_uint64, c.c_void_p, c.c_uint64,
                                     c.c_int32, c.c_uint64, c.c_void_p, c.c_uint64]
    lib.chen_neural_destroy.argtypes = [c.c_void_p]
    lib.chen_neural_runtime_identity.restype = c.c_uint64
    lib.chen_neural_runtime_identity.argtypes = [c.c_uint32]
    lib.chen_neural_evaluate.argtypes = [c.c_void_p, c.c_int32, c.c_void_p,
                                       c.c_uint64, c.c_void_p, c.c_void_p, c.c_uint64]
    lib.chen_neural_decode.argtypes = [c.c_void_p, c.c_uint64, c.c_void_p, c.c_void_p]
    for i, name in enumerate(("hipPointerGetAttributes", "hipMemGetAddressRange",
                              "hipMalloc", "hipFree", "hipStreamSynchronize")):
        assert lib.chen_neural_runtime_identity(i) == c.cast(getattr(runtime, name), c.c_void_p).value
    assert lib.chen_neural_runtime_identity(5) == 0
    trace = c.CDLL(None) if args.trace_copies else None
    if trace:
        trace.chen_neural_copy_trace_begin.argtypes = []
        trace.chen_neural_copy_trace_end.argtypes = [c.POINTER(CopyTrace)]
        # All three interposed APIs must detect an injected payload download.
        payload = torch.ones(128, dtype=torch.uint8, device="cuda")
        host = c.create_string_buffer(128)
        copies = c.CDLL(None)
        copies.hipMemcpy.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int]
        copies.hipMemcpyAsync.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_void_p]
        copies.hipMemcpyWithStream.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int, c.c_void_p]
        torch.cuda.synchronize()
        trace.chen_neural_copy_trace_begin()
        assert copies.hipMemcpy(host, payload.data_ptr(), 128, 2) == 0
        assert copies.hipMemcpyAsync(host, payload.data_ptr(), 128, 2, None) == 0
        assert copies.hipMemcpyWithStream(host, payload.data_ptr(), 128, 2, None) == 0
        torch.cuda.synchronize()
        control = CopyTrace()
        trace.chen_neural_copy_trace_end(control)
        assert control.calls[2] == 3 and control.bytes[2] == 384
        print(json.dumps({"record": "native_payload_copy_negative_control",
                          "D2H_bytes": control.bytes[2], "largest": control.maxima[2]}), flush=True)
    error = c.create_string_buffer(2048)
    total = 0
    for archive_name in args.archives:
        archive = Path(archive_name)
        sidecar = json.loads(archive.with_suffix(archive.suffix + ".json").read_text())
        blob = c.create_string_buffer(archive.read_bytes())
        content = sidecar["manifest_json"].encode()
        manifest = c.create_string_buffer(content)
        metadata = sidecar["manifest"]
        session = load_inference(archive, expected_sha256=sidecar["archive_sha256"], device="cuda:0")
        reference = None
        state = archive.with_suffix(".state.pt")
        if state.exists():
            reference = build_model(**metadata["model_config"])
            reference.load_state_dict(torch.load(state, weights_only=True, map_location="cpu"))
            reference = StaticEvaluator(reference.cuda().eval()).eval()
        elif args.checkpoint_directory:
            from neural.teacher import load_teacher_checkpoint
            checkpoint = Path(args.checkpoint_directory) / (metadata["model_config"]["name"] + ".pt")
            assert sha(checkpoint) == metadata["checkpoint_sha256"]
            model, _ = load_teacher_checkpoint(checkpoint, device="cuda:0")
            reference = StaticEvaluator(model.eval()).eval()
        ctx = lib.chen_neural_create(blob, len(blob) - 1, manifest, len(content), 0,
                                     metadata["max_batch"], error, len(error))
        assert ctx, error.value.decode()
        try:
            for count in (1, 3, 17, 64):
                fens = [FENS[i % len(FENS)] for i in range(count)]
                packets = (RulePosition * count)(*(position_from_fen(fen) for fen in fens))
                raw = c.string_at(c.addressof(packets), c.sizeof(packets))
                device_packets = torch.tensor(list(raw), dtype=torch.uint8, device="cuda")
                features = torch.stack([encode_fen(fen) for fen in fens]).cuda()
                decoded = torch.empty_like(features)
                rc = lib.chen_neural_decode(device_packets.data_ptr(), count, decoded.data_ptr(),
                                             torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                assert rc == 0 and torch.equal(decoded, features), "schema1 GPU decode differs"
                with torch.inference_mode():
                    expected = session(features)
                    if reference is not None:
                        eager = reference(features)
                        assert torch.equal(expected, eager), (metadata["model_config"], count,
                                                               expected.cpu().tolist(), eager.cpu().tolist())
                scores = torch.full((count,), -31000, dtype=torch.int32, device="cuda")
                torch.cuda.synchronize()
                if trace:
                    trace.chen_neural_copy_trace_begin()
                try:
                    rc = lib.chen_neural_evaluate(ctx, 0, device_packets.data_ptr(), count,
                                                  scores.data_ptr(), error, len(error))
                finally:
                    if trace:
                        report = CopyTrace()
                        trace.chen_neural_copy_trace_end(report)
                assert rc == 0, error.value.decode()
                assert torch.equal(scores, expected), (metadata["model_config"], count,
                                                       scores.cpu().tolist(), expected.cpu().tolist())
                row = {"record": "native_same_model_parity", "model": metadata["model_config"],
                       "rows": count, "archive_sha256": sidecar["archive_sha256"],
                       "target_kind": metadata["target_kind"], "decoded_inputs_equal": True,
                       "eager_gpu_equal": reference is not None, "scripted_gpu_scores_equal": True}
                if trace:
                    row["copy_trace"] = {"calls": list(report.calls), "bytes": list(report.bytes),
                                         "maxima": list(report.maxima)}
                    assert report.bytes[1] == 0, "native callback uploaded a payload"
                    assert report.maxima[2] <= 8, ("native callback downloaded an interval larger than a scalar", row["copy_trace"])
                print(json.dumps(row), flush=True)
                total += count
            assert lib.chen_neural_evaluate(ctx, 0, None, 0, None, error, len(error)) == 0
            assert lib.chen_neural_evaluate(ctx, 1, None, 0, None, error, len(error)) != 0
            bad = c.create_string_buffer(b"{}")
            assert not lib.chen_neural_create(blob, len(blob) - 1, bad, 2, 0,
                                              metadata["max_batch"], error, len(error))
        finally:
            lib.chen_neural_destroy(ctx)
    print(json.dumps({"record": "native_neural_probe_complete", "rows": total,
                      "library_sha256": sha(args.library), "runtime_sha256": sha(runtime_path),
                      "torch": torch.__version__, "hip": torch.version.hip,
                      "scope": "same-model numerical parity, no playing-strength claim"}), flush=True)


if __name__ == "__main__":
    main()
