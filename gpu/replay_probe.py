#!/usr/bin/env python3
"""Independent executed HIP->Torch replay packet/action/pointer-boundary oracle.

State downloads below are diagnostics after synchronous export, never the producer
path. The producer API uploads no boards and returns scalar status/epoch only.
"""
import argparse
import ctypes as c
import json
import hashlib
import math
from pathlib import Path

import chess
import torch

from rules_probe import Position, Stats, Timings, MASK64, bind, pack

ADDRESSES = ("pieces", "state", "next_pieces", "next_state", "legal_actions",
             "legal_counts", "actions", "material_proxy", "terminal", "parent_hash",
             "child_hash", "parent_node", "child_node", "root_epoch")


class Sink(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32),
                ("device", c.c_int32), ("legal_slots", c.c_uint32),
                ("capacity", c.c_uint64)] + [(key, c.c_uint64) for key in ADDRESSES]


class Export(c.Structure):
    _fields_ = [("rows", c.c_uint64), ("root_epoch", c.c_uint64), ("kernel_ms", c.c_double)]


class CopyTrace(c.Structure):
    _fields_ = [("calls", c.c_uint64 * 5), ("bytes", c.c_uint64 * 5), ("largest", c.c_uint64)]


def allocate(rows, slots=256):
    specs = {"pieces": ((rows, 12), torch.int64), "state": ((rows, 5), torch.int32),
             "next_pieces": ((rows, 12), torch.int64), "next_state": ((rows, 5), torch.int32),
             "legal_actions": ((rows, slots), torch.int16), "legal_counts": ((rows,), torch.int32),
             "actions": ((rows,), torch.int64), "material_proxy": ((rows,), torch.float32),
             "terminal": ((rows,), torch.bool)}
    specs.update({key: ((rows,), torch.int64) for key in ADDRESSES[9:]})
    tensors = {key: torch.empty(shape, dtype=dtype, device="cuda:0") for key, (shape, dtype) in specs.items()}
    sink = Sink(1, c.sizeof(Sink), 0, slots, rows, *(tensors[key].data_ptr() for key in ADDRESSES))
    return sink, tensors


def action(move, side):
    flip = 56 if side else 0
    return (((move.from_square ^ flip) * 64 + (move.to_square ^ flip)) * 5
            + (move.promotion - 1 if move.promotion else 0))


def unsigned(values):
    return [int(value) & MASK64 for value in values]


def check_packet(tensors, row, position, next_state=False):
    prefix = "next_" if next_state else ""
    assert unsigned(tensors[prefix + "pieces"][row]) == list(position.boards)
    fields = [position.side, position.castling, position.ep_square, position.halfmove, position.fullmove]
    got = [int(value) for value in tensors[prefix + "state"][row]]
    assert got[:3] == fields[:3]
    assert [value & 0xffffffff for value in got[3:]] == fields[3:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--trace-copies", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.version.hip
    assert "gfx1030" in torch.cuda.get_device_properties(0).gcnArchName
    assert c.sizeof(Sink) == 136 and c.sizeof(Export) == 24
    # Promote Torch's actual allocation runtime before loading the producer.
    runtime_path = Path(torch.__file__).parent / "lib" / "libamdhip64.so"
    runtime = c.CDLL(str(runtime_path), mode=c.RTLD_GLOBAL)
    lib = bind(args.library)
    lib.chen_replay_abi_version.restype = c.c_uint32
    assert lib.chen_replay_abi_version() == 1
    lib.chen_replay_runtime_identity.restype = c.c_uint64
    lib.chen_replay_runtime_identity.argtypes = [c.c_uint32]
    runtime_names = ("hipPointerGetAttributes", "hipMemGetAddressRange", "hipMalloc", "hipFree", "hipStreamSynchronize")
    class DlInfo(c.Structure):
        _fields_ = [("path", c.c_char_p), ("base", c.c_void_p), ("symbol", c.c_char_p), ("address", c.c_void_p)]
    process = c.CDLL(None)
    process.dladdr.argtypes = [c.c_void_p, c.POINTER(DlInfo)]
    bindings = {}
    for selector, name in enumerate(runtime_names):
        token = lib.chen_replay_runtime_identity(selector)
        assert token == c.cast(getattr(runtime, name), c.c_void_p).value
        info = DlInfo()
        assert process.dladdr(token, info)
        actual_path = Path(info.path.decode()).resolve()
        bindings[name] = {"actual_runtime": str(actual_path), "sha256": hashlib.sha256(actual_path.read_bytes()).hexdigest()}
    assert lib.chen_replay_runtime_identity(5) == 0
    lib.chen_replay_epoch.argtypes = [c.c_void_p, c.POINTER(c.c_uint64), c.c_char_p, c.c_uint64]
    lib.chen_replay_epoch.restype = c.c_int32
    lib.chen_rules_export_transitions.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64,
                                                  c.POINTER(Sink), c.POINTER(Export), c.c_char_p, c.c_uint64]
    lib.chen_rules_export_transitions.restype = c.c_int32
    error = c.create_string_buffer(1024)
    ctx = lib.chen_rules_create(0, 12000, error, len(error))
    assert ctx, error.value.decode()
    exported_rows, kernel_ms = 0, 0.0
    successful_traces = []
    trace = c.CDLL(None) if args.trace_copies else None
    if trace:
        trace.chen_copy_trace_begin.argtypes = []
        trace.chen_copy_trace_end.argtypes = [c.POINTER(CopyTrace)]
        trace.hipMemcpy.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int32]
        trace.hipMemcpyAsync.argtypes = [c.c_void_p, c.c_void_p, c.c_size_t, c.c_int32, c.c_void_p]
        # Intentional payload-copy negative control, outside any producer call.
        payload = torch.arange(16, dtype=torch.int64, device="cuda:0")
        destination = c.create_string_buffer(128)
        torch.cuda.synchronize(0)
        trace.chen_copy_trace_begin()
        assert trace.hipMemcpy(destination, payload.data_ptr(), 128, 2) == 0
        assert trace.hipMemcpyAsync(destination, payload.data_ptr(), 128, 2, None) == 0
        torch.cuda.synchronize(0)
        control = CopyTrace()
        trace.chen_copy_trace_end(control)
        assert list(control.calls) == [0, 0, 2, 0, 0]
        assert list(control.bytes) == [0, 0, 256, 0, 0] and control.largest == 128
        assert list(c.cast(destination, c.POINTER(c.c_int64 * 16)).contents) == list(range(16))

    def call(name, *values, expected=0):
        status = getattr(lib, "chen_rules_" + name)(ctx, *values, error, len(error))
        assert status == expected, (name, status, error.value.decode())

    def epoch():
        value = c.c_uint64()
        status = lib.chen_replay_epoch(ctx, value, error, len(error))
        assert status == 0, error.value.decode()
        return value.value

    def export(start, count, sink, expected=0, expected_epoch=None):
        torch.cuda.synchronize(0)
        result = Export()
        selected_epoch = epoch() if expected_epoch is None else expected_epoch
        if trace:
            trace.chen_copy_trace_begin()
        try:
            call("export_transitions", start, count, selected_epoch, sink, result, expected=expected)
        finally:
            if trace:
                report = CopyTrace()
                trace.chen_copy_trace_end(report)
        if trace and expected == 0 and count:
            assert list(report.calls) == [0, 0, 2, 0, 0], list(report.calls)
            assert list(report.bytes) == [0, 0, 32, 0, 0], list(report.bytes)
            assert report.largest == 16
            successful_traces.append(report)
        return result

    def reset(board, history=()):
        position = pack(board)
        hashes = (c.c_uint64 * len(history))(*history)
        call("reset", position, hashes, len(history))
        stats = Stats()
        call("expand", 0, 1, 1, stats)
        return stats

    try:
        assert epoch() == 0
        fixtures = [chess.Board(),
                    chess.Board("4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12"),
                    chess.Board("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 99 1"),
                    chess.Board("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 99 4294967295"),
                    chess.Board("4k3/P7/8/8/8/8/7p/4K3 w - - 0 1"),
                    chess.Board("7k/5Q2/6K1/8/8/8/8/8 w - - 99 1")]
        # Retained predecessor history makes c6b8 a third occurrence of start.
        repeated = chess.Board()
        history = []
        for uci in ("b1c3", "b8c6", "c3b1", "c6b8", "b1c3", "b8c6", "c3b1"):
            history.append(pack(repeated).hash)
            repeated.push_uci(uci)
        fixtures.append(repeated)
        for index, board in enumerate(fixtures):
            stats = reset(board, history if index == len(fixtures) - 1 else ())
            assert stats.next_count == board.legal_moves.count()
            sink, tensors = allocate(stats.next_count)
            result = export(stats.next_start, stats.next_count, sink)
            assert result.rows == stats.next_count and result.root_epoch == epoch()
            exported_rows += result.rows
            kernel_ms += result.kernel_ms
            cpu = {key: tensor.cpu() for key, tensor in tensors.items()}
            legal = sorted(action(move, pack(board).side) for move in board.legal_moves)
            actions = list(map(int, cpu["actions"]))
            assert sorted(actions) == legal
            for row, action_id in enumerate(actions):
                raw, promotion = divmod(action_id, 5)
                source, target = divmod(raw, 64)
                flip = 56 if not board.turn else 0
                move = chess.Move(source ^ flip, target ^ flip, promotion + 1 if promotion else None)
                successor = board.copy()
                successor.push(move)
                current, child = pack(board), pack(successor)
                check_packet(cpu, row, current)
                check_packet(cpu, row, child, next_state=True)
                assert int(cpu["legal_counts"][row]) == len(legal)
                assert list(map(int, cpu["legal_actions"][row, :len(legal)])) == legal
                assert bool((cpu["legal_actions"][row, len(legal):] == -1).all())
                assert (int(cpu["parent_hash"][row]) & MASK64) == current.hash
                assert (int(cpu["child_hash"][row]) & MASK64) == child.hash
                assert int(cpu["parent_node"][row]) == 0
                assert int(cpu["child_node"][row]) == stats.next_start + row
                assert int(cpu["root_epoch"][row]) == epoch()
                terminal = (not any(successor.legal_moves) or successor.halfmove_clock >= 100
                            or successor.is_insufficient_material() or successor.is_repetition(3))
                assert bool(cpu["terminal"][row]) == terminal
                values = (100, 320, 330, 500, 900, 0)
                white = sum(values[piece - 1] * (len(board.pieces(piece, chess.WHITE)) - len(board.pieces(piece, chess.BLACK)))
                            for piece in range(1, 7))
                proxy = math.tanh((white if board.turn else -white) / 800)
                assert abs(float(cpu["material_proxy"][row]) - proxy) < 1e-6
        # Output is an owned snapshot and may select already-expanded successors.
        stats = reset(chess.Board())
        source_start, source_count, original_epoch = stats.next_start, stats.next_count, epoch()
        call("expand", source_start, source_count, 1, stats)
        sink, tensors = allocate(source_count)
        export(source_start, source_count, sink)
        before = tensors["pieces"].clone()
        # All preflight errors leave the sentinel output untouched.
        def reject(**changes):
            tensors["pieces"].fill_(0x12345)
            altered = Sink.from_buffer_copy(sink)
            for key, value in changes.items():
                setattr(altered, key, value)
            export(source_start, source_count, altered, expected=-1)
            assert bool((tensors["pieces"] == 0x12345).all())
        reject(abi_version=2)
        reject(struct_size=144)
        reject(device=1)
        reject(pieces=0)
        reject(pieces=sink.pieces + 1)
        reject(state=sink.pieces)
        host = torch.empty((source_count, 12), dtype=torch.int64)
        reject(pieces=host.data_ptr())
        tiny = Sink.from_buffer_copy(sink)
        tiny.capacity = source_count - 1
        export(source_start, source_count, tiny, expected=-2)
        tiny = Sink.from_buffer_copy(sink)
        tiny.legal_slots = 1
        export(source_start, source_count, tiny, expected=-2)
        export(0, 1, sink, expected=-1)
        # Query allocation range on a raw HIP allocation independent of Torch's cache.
        hip = runtime
        hip.hipMalloc.argtypes = [c.POINTER(c.c_void_p), c.c_size_t]
        hip.hipFree.argtypes = [c.c_void_p]
        hip.hipMemGetAddressRange.argtypes = [c.POINTER(c.c_void_p), c.POINTER(c.c_size_t), c.c_void_p]
        allocation = c.c_void_p()
        assert hip.hipMalloc(allocation, 128) == 0
        base, size = c.c_void_p(), c.c_size_t()
        assert hip.hipMemGetAddressRange(base, size, allocation) == 0
        assert size.value >= 128 and size.value % 8 == 0
        reject(pieces=base.value + size.value - 8)
        old_address = allocation.value
        assert hip.hipFree(allocation) == 0
        reject(pieces=old_address)
        # Same numeric child IDs after reset must not revive stale selectors.
        reset(chess.Board())
        export(source_start, source_count, sink, expected=-1, expected_epoch=original_epoch)
        assert epoch() > original_epoch
        original_epoch = epoch()
        call("legal", None, 0, None, None, None, Timings())
        assert epoch() == original_epoch + 1
        export(source_start, source_count, sink, expected=-1, expected_epoch=original_epoch)
        stats = reset(chess.Board())
        current_epoch = epoch()
        # Empty zero-capacity C export and metadata query do not allocate or change epoch.
        empty = Sink(1, c.sizeof(Sink), 0, 256, 0)
        assert export(1, 0, empty).rows == 0 and epoch() == current_epoch
        export(stats.next_start, stats.next_count, sink)
        snapshot = tensors["pieces"].clone()
        root, nodes = pack(chess.Board()), c.c_uint64()
        call("perft", root, 1, nodes, stats)
        assert epoch() > current_epoch
        export(1, 20, sink, expected=-1)
        assert torch.equal(tensors["pieces"], snapshot)
        lib.chen_rules_destroy(ctx)
        ctx = None
        assert torch.equal(tensors["pieces"], snapshot)
        print(json.dumps({"record": "resident_replay_oracle", "device": "gfx1030", "fixtures": len(fixtures),
                          "transitions": exported_rows, "exact_boards_fields_hashes": True,
                          "sorted_canonical_support": True, "retained_history_terminal": True,
                          "expanded_successor": True, "kernel_ms": kernel_ms}))
        print(json.dumps({"record": "resident_replay_runtime_bindings", "producer_consumer_tokens_match": True,
                          "Torch_hip_build_version": torch.version.hip, "runtime_bindings": bindings}))
        print(json.dumps({"record": "resident_replay_boundaries", "host_null_misaligned_freed_pointer": True,
                          "actual_allocation_bounds": True, "advertised_output_overlap": True,
                          "schema_and_device_rejection": True, "row_and_legal_capacity": True,
                          "stale_reset_and_generic_epochs": True, "perft_mode_rejection": True,
                          "empty_export": True, "snapshot_survives_reset_and_destroy": True,
                          "limitations": ["One physical GPU; foreign-device allocation rejection not exercised.",
                                          "HIP allocation bounds cannot prove Torch logical-view ownership."]}))
        if trace:
            print(json.dumps({"record": "resident_replay_copy_boundary",
                              "interposed_APIs": ["hipMemcpy", "hipMemcpyAsync"],
                              "marked_successful_calls": len(successful_traces),
                              "per_call_D2H_copies": 2, "per_call_D2H_bytes": 32,
                              "largest_copy_bytes": 16, "H2D_bytes": 0, "D2D_bytes": 0,
                              "payload_negative_control": {"APIs": 2, "D2H_bytes": 256, "largest_copy_bytes": 128},
                              "scope": "Synchronous export C call only; diagnostic Torch output inspection excluded."}))
    finally:
        if ctx:
            lib.chen_rules_destroy(ctx)


if __name__ == "__main__":
    main()
