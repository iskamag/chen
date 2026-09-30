#!/usr/bin/env python3
"""Executed GPU search-teacher oracle against Rust cpu-resident minimax.

Output/state inspection is diagnostic only, outside both export hot paths.
"""
import argparse
import ctypes as c
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile

import chess
import torch

from replay_probe import CopyTrace, action, allocate as replay_allocate
from rules_probe import Move, Position, Stats, Result, bind, pack


class Sink(c.Structure):
    _fields_ = [("abi_version", c.c_uint32), ("struct_size", c.c_uint32), ("device", c.c_int32),
                ("reserved", c.c_uint32), ("capacity", c.c_uint64),
                ("scores", c.c_uint64), ("best_actions", c.c_uint64), ("parent_ply", c.c_uint64)]


class Export(c.Structure):
    _fields_ = [("rows", c.c_uint64), ("root_epoch", c.c_uint64), ("kernel_ms", c.c_double),
                ("evaluator", c.c_int32), ("label_kind", c.c_uint32)]


def allocate(rows):
    tensors = {"scores": torch.full((rows,), 0x12345, dtype=torch.int32, device="cuda:0"),
               "best_actions": torch.full((rows,), -1, dtype=torch.int64, device="cuda:0"),
               "parent_ply": torch.full((rows,), -1, dtype=torch.int32, device="cuda:0")}
    return Sink(1, c.sizeof(Sink), 0, 0, rows, *(tensors[k].data_ptr() for k in tensors)), tensors


def board_from_packet(p):
    board = chess.Board(None)
    for kind, bb in enumerate(p.boards):
        while bb:
            bit = bb & -bb
            board.set_piece_at(bit.bit_length() - 1, chess.Piece(kind % 6 + 1, kind < 6))
            bb ^= bit
    board.turn = p.side == 0
    board.castling_rights = sum(1 << square for bit, square in [(1, 7), (2, 0), (4, 63), (8, 56)] if p.castling & bit)
    board.ep_square = p.ep_square if p.ep_square >= 0 else None
    board.halfmove_clock, board.fullmove_number = p.halfmove, p.fullmove
    return board


def rust_oracle(binary, positions, depth, evaluator):
    with tempfile.TemporaryDirectory(prefix="chen-teacher-") as directory:
        suite = Path(directory) / "suite.txt"
        suite.write_text("".join(f"{index} | {board.fen(en_passant='fen')}\n" for index, board in enumerate(positions)))
        command = [binary, "bench", "--backend", "cpu-resident", "--eval", evaluator,
                   "--depth", str(depth), "--suite", str(suite), "--max-positions", "250000"]
        result = subprocess.run(command, check=True, capture_output=True, text=True)
    return [json.loads(line) for line in result.stdout.splitlines()]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--rust-engine", required=True)
    parser.add_argument("--trace-copies", action="store_true")
    args = parser.parse_args()
    assert torch.cuda.is_available() and torch.version.hip and c.sizeof(Sink) == 48 and c.sizeof(Export) == 32
    runtime = c.CDLL(str(Path(torch.__file__).parent / "lib" / "libamdhip64.so"), mode=c.RTLD_GLOBAL)
    lib = bind(args.library)
    lib.chen_replay_runtime_identity.argtypes, lib.chen_replay_runtime_identity.restype = [c.c_uint32], c.c_uint64
    for index, name in enumerate(("hipPointerGetAttributes", "hipMemGetAddressRange", "hipMalloc", "hipFree", "hipStreamSynchronize")):
        assert lib.chen_replay_runtime_identity(index) == c.cast(getattr(runtime, name), c.c_void_p).value
    lib.chen_teacher_abi_version.restype = c.c_uint32
    assert lib.chen_teacher_abi_version() == 1
    lib.chen_replay_epoch.argtypes = [c.c_void_p, c.POINTER(c.c_uint64), c.c_char_p, c.c_uint64]
    lib.chen_rules_export_teacher.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64,
                                             c.POINTER(Sink), c.POINTER(Export), c.c_char_p, c.c_uint64]
    error = c.create_string_buffer(1024)
    ctx = lib.chen_rules_create(0, 250000, error, len(error))
    assert ctx, error.value.decode()
    trace = c.CDLL(None) if args.trace_copies else None
    if trace:
        trace.chen_copy_trace_end.argtypes = [c.POINTER(CopyTrace)]
    successful_copies, comparisons, rows, mate_rows, tied_rows = 0, 0, 0, 0, 0

    def call(name, *values, expected=0):
        code = getattr(lib, "chen_rules_" + name)(ctx, *values, error, len(error))
        assert code == expected, (name, code, error.value.decode())

    def epoch():
        value = c.c_uint64()
        assert lib.chen_replay_epoch(ctx, value, error, len(error)) == 0
        return value.value

    def export(start, count, sink, expected=0, expected_epoch=None):
        nonlocal successful_copies
        torch.cuda.synchronize(0)
        result = Export()
        chosen_epoch = epoch() if expected_epoch is None else expected_epoch
        if trace:
            trace.chen_copy_trace_begin()
        try:
            call("export_teacher", start, count, chosen_epoch, sink, result, expected=expected)
        finally:
            if trace:
                copies = CopyTrace()
                trace.chen_copy_trace_end(copies)
        if trace and expected == 0 and count:
            assert list(copies.calls) == [0, 0, 2, 0, 0]
            assert list(copies.bytes) == [0, 0, 32, 0, 0] and copies.largest == 16
            successful_copies += 1
        return result

    fixtures = [chess.STARTING_FEN,
                "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1",
                "8/8/8/8/8/6k1/5q2/7K b - - 0 1",
                "7k/8/5K2/8/8/8/8/6Q1 b - - 0 1",
                "4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12",
                "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
                "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 99 4294967295"]
    try:
        for evaluator in (0, 1):
            for fixture in fixtures:
                board = chess.Board(fixture)
                root = pack(board)
                call("reset", root, None, 0)
                levels = [(0, 1)]
                stats = Stats()
                for depth in range(3):
                    call("expand", *levels[-1], 1, stats)
                    levels.append((stats.next_start, stats.next_count))
                root_sink, root_tensors = allocate(levels[1][1])
                export(*levels[1], root_sink, expected=-1)
                call("evaluate", *levels[-1], evaluator, stats)
                # Scored leaves alone never authorize unreduced parent labels.
                export(*levels[1], root_sink, expected=-1)
                for span in reversed(levels[:-1]):
                    call("reduce", *span, stats)
                for source_level in (0, 1):
                    selected = levels[source_level + 1]
                    if not selected[1]:
                        continue
                    sink, tensors = allocate(selected[1])
                    result = export(*selected, sink)
                    assert (result.rows, result.root_epoch, result.evaluator, result.label_kind) == (selected[1], epoch(), evaluator, 1)
                    # Replay sidecars identify the exact same source parents.
                    replay_sink, replay_tensors = replay_allocate(selected[1])
                    from replay_probe import Export as ReplayExport, Sink as ReplaySink
                    lib.chen_rules_export_transitions.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64,
                                                                 c.POINTER(ReplaySink), c.POINTER(ReplayExport), c.c_char_p, c.c_uint64]
                    torch.cuda.synchronize(0)
                    call("export_transitions", *selected, epoch(), replay_sink, ReplayExport())
                    cpu = {key: tensor.cpu() for key, tensor in tensors.items()}
                    parent_ids = list(map(int, replay_tensors["parent_node"].cpu()))
                    unique = list(dict.fromkeys(parent_ids))
                    packets = (Position * levels[source_level][1])()
                    call("download", *levels[source_level], packets)
                    parents = [board_from_packet(packets[node - levels[source_level][0]]) for node in unique]
                    expected = rust_oracle(args.rust_engine, parents, 3 - source_level, "material" if evaluator == 0 else "tapered")
                    oracle = dict(zip(unique, expected))
                    for row, parent_id in enumerate(parent_ids):
                        answer = oracle[parent_id]
                        score = answer["score_cp"]
                        if abs(score) >= 30000 - 128:
                            score += -source_level if score > 0 else source_level
                            mate_rows += 1
                        assert int(cpu["scores"][row]) == score, (fixture, evaluator, source_level, parent_id, int(cpu["scores"][row]), answer)
                        parent = parents[unique.index(parent_id)]
                        chosen = chess.Move.from_uci(answer["bestmove"])
                        assert int(cpu["best_actions"][row]) == action(chosen, int(not parent.turn))
                        assert int(cpu["parent_ply"][row]) == source_level
                        if fixture == chess.STARTING_FEN and evaluator == 0:
                            tied_rows += 1
                    comparisons += len(unique)
                    rows += len(parent_ids)
                # Invalid buffers/capacity/schema/stale epoch cannot publish labels.
                for key, value in [("abi_version", 2), ("reserved", 1), ("device", 1),
                                   ("scores", 0), ("scores", root_sink.scores + 1),
                                   ("parent_ply", root_sink.scores)]:
                    invalid = Sink.from_buffer_copy(root_sink)
                    setattr(invalid, key, value)
                    root_tensors["scores"].fill_(0x12345)
                    export(*levels[1], invalid, expected=-1)
                    assert bool((root_tensors["scores"] == 0x12345).all())
                invalid = Sink.from_buffer_copy(root_sink)
                invalid.capacity -= 1
                export(*levels[1], invalid, expected=-2)
                host = torch.empty((levels[1][1],), dtype=torch.int32)
                invalid = Sink.from_buffer_copy(root_sink)
                invalid.scores = host.data_ptr()
                export(*levels[1], invalid, expected=-1)
                old_epoch = epoch()
                call("reset", root, None, 0)
                call("expand", 0, 1, 1, stats)
                export(*levels[1], root_sink, expected=-1, expected_epoch=old_epoch)
        assert comparisons and rows and mate_rows and tied_rows
        print(json.dumps({"record": "resident_teacher_oracle", "fixtures": len(fixtures), "evaluators": ["material", "tapered"],
                          "Rust_unique_parent_comparisons": comparisons, "edge_rows": rows, "mate_distance_rows": mate_rows,
                          "Material_tied_start_rows": tied_rows, "root_and_source_ply1": True,
                          "exact_scores_actions_parent_ply": True,
                          "Rust_engine_sha256": hashlib.sha256(Path(args.rust_engine).read_bytes()).hexdigest()}))
        print(json.dumps({"record": "resident_teacher_boundaries", "unscored_and_unreduced_rejected": True,
                          "invalid_schema_reserved_device_pointer_overlap": True, "row_capacity": True,
                          "stale_epoch_alias_rejected": True, "same_replay_edge_parents": True}))
        if trace:
            print(json.dumps({"record": "resident_teacher_copy_scope", "successful_marked_calls": successful_copies,
                              "interposed_APIs": ["hipMemcpy", "hipMemcpyAsync"], "per_call_D2H_bytes": 32,
                              "per_call_D2H_copies": 2, "largest_copy_bytes": 16, "H2D_bytes": 0, "D2D_bytes": 0,
                              "scope": "Teacher export C calls only; Rust oracle and diagnostic Tensor/packet downloads excluded."}))
    finally:
        lib.chen_rules_destroy(ctx)


if __name__ == "__main__":
    main()
