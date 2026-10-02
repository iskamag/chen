#!/usr/bin/env python3
"""Actual HIP legal/transition/hash/resident-perft oracle against python-chess."""
import argparse
import ctypes as c
import json
import random
import time

import chess


class Position(c.Structure):
    _fields_ = [("boards", c.c_uint64 * 12), ("side", c.c_uint32),
                ("castling", c.c_uint32), ("ep_square", c.c_int32),
                ("halfmove", c.c_uint32), ("fullmove", c.c_uint32),
                ("reserved", c.c_uint32), ("hash", c.c_uint64)]


class Move(c.Structure):
    _fields_ = [("from_square", c.c_uint8), ("to_square", c.c_uint8),
                ("promotion", c.c_uint8), ("reserved", c.c_uint8)]

    @classmethod
    def pack(cls, move):
        return cls(move.from_square, move.to_square,
                   move.promotion - 1 if move.promotion else 0, 0)

    def uci(self):
        return chess.Move(self.from_square, self.to_square,
                          self.promotion + 1 if self.promotion else None).uci()


class Timings(c.Structure):
    _fields_ = [("h2d_ms", c.c_double), ("kernel_ms", c.c_double),
                ("d2h_ms", c.c_double), ("count", c.c_uint64), ("capacity", c.c_uint64)]


class Stats(c.Structure):
    _fields_ = [("arena_positions", c.c_uint64), ("classified_positions", c.c_uint64),
                ("evaluated_positions", c.c_uint64), ("next_start", c.c_uint64),
                ("next_count", c.c_uint64), ("kernel_ms", c.c_double)]


class Result(c.Structure):
    _fields_ = [("score", c.c_int32), ("pv_length", c.c_uint32)]


MASK64 = (1 << 64) - 1


def hash_key(offset):
    value = (0x4348454E00000000 + offset + 0x9E3779B97F4A7C15) & MASK64
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & MASK64
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & MASK64
    return value ^ (value >> 31)


def pack(board):
    p = Position()
    for color in (chess.WHITE, chess.BLACK):
        base = 0 if color else 6
        for piece in range(1, 7):
            p.boards[base + piece - 1] = board.pieces_mask(piece, color)
    p.side = int(board.turn == chess.BLACK)
    p.castling = sum(bit for color, king, bit in [(chess.WHITE, True, 1), (chess.WHITE, False, 2),
                                                (chess.BLACK, True, 4), (chess.BLACK, False, 8)]
                     if (board.has_kingside_castling_rights(color) if king
                         else board.has_queenside_castling_rights(color)))
    p.ep_square = board.ep_square if board.ep_square is not None else -1
    p.halfmove = min(65535, board.halfmove_clock)
    p.fullmove = min((1 << 32) - 1, board.fullmove_number)
    key = hash_key(768 + p.castling) ^ (hash_key(792) if p.side else 0)
    if board.has_legal_en_passant():
        key ^= hash_key(784 + board.ep_square % 8)
    for piece, bb in enumerate(p.boards):
        while bb:
            bit = bb & -bb
            key ^= hash_key(piece * 64 + bit.bit_length() - 1)
            bb ^= bit
    p.hash = key
    return p


FIXTURES = [
    chess.STARTING_FEN,
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "r3k2r/Pppp1ppp/1b3nbN/nP6/BBP1P3/q4N2/Pp1P2PP/R2Q1RK1 w kq - 0 1",
    "rnbq1k1r/pp1Pbppp/2p5/8/2B5/8/PPP1NnPP/RNBQK2R w KQ - 1 8",
    "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10",
    "8/6bb/8/8/R1pP2k1/4P3/P7/K7 b - d3 0 1",
    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12",
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 99 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 65535 4294967295",
    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
    "7k/6Q1/6K1/8/8/8/8/8 b - - 100 1",
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",
]
PERFT = [(FIXTURES[0], [1, 20, 400, 8902, 197281]),
         (FIXTURES[1], [1, 48, 2039, 97862]),
         (FIXTURES[2], [1, 14, 191, 2812]),
         (FIXTURES[3], [1, 6, 264, 9467]),
         (FIXTURES[4], [1, 44, 1486, 62379]),
         (FIXTURES[5], [1, 46, 2079, 89890])]


def bind(library):
    lib = c.CDLL(library)
    ptr, u64, u32, err = c.c_void_p, c.c_uint64, c.c_uint32, c.c_char_p
    position, move, stats, times = c.POINTER(Position), c.POINTER(Move), c.POINTER(Stats), c.POINTER(Timings)
    signatures = {
        "create": [c.c_int32, u64, err, u64], "destroy": [ptr],
        "legal": [ptr, position, u64, move, c.POINTER(u32), c.POINTER(u32), times, err, u64],
        "make": [ptr, position, move, u64, position, times, err, u64],
        "reset": [ptr, position, c.POINTER(u64), u64, err, u64],
        "classify": [ptr, u64, u64, u32, stats, err, u64],
        "expand": [ptr, u64, u64, u32, stats, err, u64],
        "evaluate": [ptr, u64, u64, u32, stats, err, u64],
        "reduce": [ptr, u64, u64, stats, err, u64],
        "finish": [ptr, c.POINTER(Result), move, u64, stats, err, u64],
        "download": [ptr, u64, u64, position, err, u64],
        "perft": [ptr, position, u32, c.POINTER(u64), stats, err, u64],
    }
    for name, args in signatures.items():
        fn = getattr(lib, "chen_rules_" + name)
        fn.argtypes = args
        fn.restype = ptr if name == "create" else None if name == "destroy" else c.c_int32
    lib.chen_rules_abi_version.restype = u32
    assert lib.chen_rules_abi_version() == 1
    return lib


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("library")
    parser.add_argument("--positions", type=int, default=256)
    parser.add_argument("--capacity", type=int, default=250000)
    parser.add_argument("--max-depth", type=int, default=4)
    args = parser.parse_args()
    assert args.positions >= len(FIXTURES) and args.capacity >= args.positions
    assert c.sizeof(Position) == 128 and c.sizeof(Move) == 4 and c.sizeof(Stats) == 48
    lib = bind(args.library)
    error = c.create_string_buffer(1024)
    ctx = lib.chen_rules_create(0, args.capacity, error, len(error))
    assert ctx, error.value.decode()
    def call(name, *values, expected=0):
        code = getattr(lib, "chen_rules_" + name)(ctx, *values, error, len(error))
        assert code == expected, (name, code, error.value.decode())
    try:
        boards = [chess.Board(fen) for fen in FIXTURES]
        rng = random.Random(0xC4E1030)
        board = chess.Board()
        while len(boards) < args.positions:
            legal = list(board.legal_moves)
            if not legal or board.halfmove_clock >= 100:
                board = chess.Board()
            else:
                board.push(rng.choice(legal))
            boards.append(board.copy())
        inputs = (Position * len(boards))(*(pack(b) for b in boards))
        moves = (Move * (len(boards) * 256))()
        counts, flags = (c.c_uint32 * len(boards))(), (c.c_uint32 * len(boards))()
        timing = Timings()
        call("legal", inputs, len(inputs), moves, counts, flags, c.byref(timing))
        transitions = []
        for i, b in enumerate(boards):
            expected = sorted(m.uci() for m in b.legal_moves)
            actual = sorted(moves[i * 256 + j].uci() for j in range(counts[i]))
            assert actual == expected, (i, b.fen(), actual, expected)
            assert flags[i] == int(b.is_check()) + 2 * int(b.is_insufficient_material())
            for m in b.legal_moves:
                successor = b.copy(); successor.push(m)
                transitions.append((pack(b), Move.pack(m), pack(successor)))
        before = (Position * len(transitions))(*(p for p, _, _ in transitions))
        requested = (Move * len(transitions))(*(m for _, m, _ in transitions))
        after = (Position * len(transitions))()
        call("make", before, requested, len(transitions), after, c.byref(timing))
        for i, (_, _, expected) in enumerate(transitions):
            assert bytes(after[i]) == bytes(expected), (i, requested[i].uci(), list(after[i].boards), after[i].hash, expected.hash)
        print(json.dumps({"record": "rules_oracle", "positions": len(boards),
                          "legal_transitions": len(transitions), "python_chess": chess.__version__,
                          "seed": 0xC4E1030, "exact_packet_and_hash": True}))
        root = pack(chess.Board())
        call("legal", None, 0, None, None, None, c.byref(timing))
        call("make", None, None, 0, None, c.byref(timing))
        invalid = Position.from_buffer_copy(root); invalid.hash ^= 1
        call("legal", c.byref(invalid), 1, moves, counts, flags, c.byref(timing), expected=-1)
        illegal_move = Move(12, 36, 0, 0)  # e2e5
        call("make", c.byref(root), c.byref(illegal_move), 1, after, c.byref(timing), expected=-1)
        stats = Stats()
        for fen, expected_counts in PERFT:
            position = pack(chess.Board(fen))
            for depth, expected in enumerate(expected_counts[:args.max_depth + 1]):
                count = c.c_uint64()
                start = time.perf_counter()
                call("perft", c.byref(position), depth, c.byref(count), c.byref(stats))
                assert count.value == expected, (fen, depth, count.value, expected)
                print(json.dumps({"record": "resident_perft", "fen": fen, "depth": depth,
                                  "nodes": count.value, "arena_positions": stats.arena_positions,
                                  "classified_positions": stats.classified_positions,
                                  "kernel_ms": stats.kernel_ms,
                                  "wall_ms": (time.perf_counter() - start) * 1000}))
        # Real repetition context: same board after eight reversible plies.
        b, history = chess.Board(), []
        for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
            history.append(pack(b).hash); b.push_uci(uci)
        repeated_root = pack(b)
        hashes = (c.c_uint64 * len(history))(*history)
        call("reset", c.byref(repeated_root), hashes, len(history))
        call("evaluate", 0, 1, 1, c.byref(stats))
        result, pv = Result(), (Move * 64)()
        call("finish", c.byref(result), pv, 64, c.byref(stats))
        assert result.score == 0 and result.pv_length == 0 and stats.evaluated_positions == 0
        call("reset", c.byref(root), None, 0)
        call("evaluate", 0, 1, 1, c.byref(stats))
        first_count = stats.evaluated_positions
        call("evaluate", 0, 1, 1, c.byref(stats))
        assert stats.evaluated_positions == first_count == 1
        call("expand", 0, 1, 1, c.byref(stats), expected=-1)
        call("evaluate", 0, 1, 0, c.byref(stats), expected=-1)
        call("classify", 0, 1, 0, c.byref(stats), expected=-1)
        # Full resident depth-three minimax: only root/history and scalar/PV leave VRAM.
        call("reset", c.byref(root), None, 0)
        levels, start, count = [], 0, 1
        for _ in range(3):
            levels.append((start, count))
            call("expand", start, count, 1, c.byref(stats))
            start, count = stats.next_start, stats.next_count
        call("evaluate", start, count, 1, c.byref(stats))
        call("reduce", 0, 21, c.byref(stats), expected=-1)
        for start, count in reversed(levels):
            call("reduce", start, count, c.byref(stats))
        call("finish", c.byref(result), pv, 64, c.byref(stats))
        assert result.score == 36 and result.pv_length == 3, (result.score, result.pv_length)
        test = chess.Board()
        for i in range(result.pv_length): test.push_uci(pv[i].uci())
        print(json.dumps({"record": "resident_minimax", "depth": 3, "score": result.score,
                          "pv": [pv[i].uci() for i in range(result.pv_length)],
                          "arena_positions": stats.arena_positions,
                          "classified_positions": stats.classified_positions,
                          "evaluated_positions": stats.evaluated_positions,
                          "kernel_ms": stats.kernel_ms}))
        call("finish", c.byref(result), pv, 1, c.byref(stats), expected=-2)
        call("classify", 0, 1, 1, c.byref(stats), expected=-1)
        print(json.dumps({"record": "rules_boundaries", "empty_batches": True,
                          "hash_and_illegal_move_rejected": True,
                          "repetition": True, "idempotent_evaluation": True,
                          "mode_and_evaluator_lock": True, "PV_capacity": True}))
    finally:
        lib.chen_rules_destroy(ctx)
    small = lib.chen_rules_create(0, 20, error, len(error))
    assert small, error.value.decode()
    try:
        root, stats = pack(chess.Board()), Stats()
        assert lib.chen_rules_reset(small, c.byref(root), None, 0, error, len(error)) == 0
        assert lib.chen_rules_expand(small, 0, 1, 1, c.byref(stats), error, len(error)) == -2
        downloaded = Position()
        assert lib.chen_rules_download(small, 0, 1, c.byref(downloaded), error, len(error)) == 0
        assert bytes(root) == bytes(downloaded)
        assert lib.chen_rules_evaluate(small, 0, 1, 1, c.byref(stats), error, len(error)) == 0
        assert stats.arena_positions == 1 and stats.classified_positions == 1
        print(json.dumps({"record": "rules_capacity", "capacity": 20,
                          "overflow_explicit": True, "root_preserved": True,
                          "classify_and_evaluate_recovery": True}))
    finally:
        lib.chen_rules_destroy(small)


if __name__ == "__main__":
    main()
