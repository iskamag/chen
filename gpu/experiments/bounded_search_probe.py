#!/usr/bin/env python3
"""Independent python-chess traversal, reused Tapered formula; no speed claim.

CPU-only preparation never loads HIP. Runtime compares complete packets with a
same-policy alpha-beta oracle and checks bounds against exhaustive small trees.
"""
import argparse
import ctypes as c
import hashlib
import json
import os
import platform
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
from rules_probe import Position, Move, pack


class Job(c.Structure):
    _fields_ = [("root", Position), ("node_budget", c.c_uint64),
                ("history_offset", c.c_uint64), ("history_count", c.c_uint64)] + [
                    (name, c.c_uint32) for name in (
                        "abi_version", "struct_size", "depth", "max_qply",
                        "evaluator", "quiescence", "root_ply", "reserved")] + [
                    ("alpha", c.c_int32), ("beta", c.c_int32)]


class Result(c.Structure):
    _fields_ = [(name, c.c_uint32) for name in (
        "abi_version", "struct_size", "kind", "abort_reason")] + [
        ("score", c.c_int32), ("completed_depth", c.c_uint32),
        ("pv_length", c.c_uint32), ("reserved", c.c_uint32),
        ("nodes", c.c_uint64), ("qnodes", c.c_uint64), ("pv", Move * 64)]


assert c.sizeof(Job) == 192 and c.sizeof(Result) == 304


def evaluate(board, mode):
    """Reused existing engine formula; independently traversed board storage."""
    values, phases = (100, 320, 330, 500, 900, 0), (0, 1, 1, 2, 4, 0)
    mg = eg = phase = 0
    for color in (chess.WHITE, chess.BLACK):
        sign = 1 if color else -1
        for piece in range(1, 7):
            squares = board.pieces(piece, color)
            if not mode:
                mg += sign * values[piece - 1] * len(squares)
                continue
            phase += phases[piece - 1] * len(squares)
            for square in squares:
                file = square % 8
                rank = square // 8 if color else 7 - square // 8
                center = 14 - abs(2 * file - 7) - abs(2 * rank - 7)
                m, e = ((rank * 8 + center * 2, rank * 14 + center),
                        (center * 6, center * 4),
                        (center * 3 + rank * 2, center * 3),
                        (rank * 2 + (18 if rank == 6 else 0), center),
                        (center, center * 2),
                        (-center * 5 - rank * 4 + (35 if rank == 0 and file in (2, 6) else 0),
                         center * 6))[piece - 1]
                mg += sign * (values[piece - 1] + m)
                eg += sign * (values[piece - 1] + e)
        if mode and len(board.pieces(chess.BISHOP, color)) >= 2:
            mg += sign * 25
            eg += sign * 40
    phase = min(phase, 24)
    numerator = mg * phase + eg * (24 - phase)
    score = (abs(numerator) // 24) * (1 if numerator >= 0 else -1) if mode else mg
    return max(-29000, min(29000, score if board.turn else -score))


def canonical(board):
    def key(move):
        if board.is_castling(move):
            return (6, 0, int(move.to_square < move.from_square), 0, 0)
        piece = board.piece_type_at(move.from_square)
        if piece == chess.PAWN:
            capture = board.is_capture(move)
            target = move.to_square if capture else abs(move.to_square - move.from_square)
            return (0, move.from_square, int(capture), target, -(move.promotion or 0))
        return (piece - 1, move.from_square, 0, move.to_square, 0)
    return sorted(board.legal_moves, key=key)


class Abort(Exception):
    def __init__(self, reason):
        self.reason = reason


def oracle(case, exhaustive=False):
    board = chess.Board(case["fen"])
    history = case.get("history", [])
    path, nodes, qnodes = [], 0, 0
    optimal = []

    def visit(depth, qply, alpha, beta, ply):
        nonlocal nodes, qnodes, optimal
        if nodes >= case["node_budget"]:
            raise Abort(1)
        nodes += 1
        qsearch = bool(case["quiescence"] and depth <= 0)
        qnodes += int(qsearch)
        moves, check = canonical(board), board.is_check()
        if not moves:
            return (-30000 + case["root_ply"] + ply if check else 0), []
        key = pack(board).hash
        recent = (history + path + [key])[-(board.halfmove_clock + 1):]
        if board.halfmove_clock >= 100 or board.is_insufficient_material() or recent.count(key) >= 3:
            return 0, []
        if depth <= 0 and not qsearch:
            return evaluate(board, case["evaluator"]), []
        best, pv = -32000, []
        if qsearch:
            if not check:
                best = evaluate(board, case["evaluator"])
                if not exhaustive and best >= beta:
                    return best, []
                alpha = max(alpha, best)
                moves = [move for move in moves if board.is_capture(move) or move.promotion]
            if moves and qply >= case["max_qply"]:
                raise Abort(2)
        root_scores = []
        for move in moves:
            path.append(key)
            board.push(move)
            try:
                value, child_pv = visit(0 if qsearch else depth - 1,
                                        qply + 1 if qsearch else 0,
                                        -beta, -alpha, ply + 1)
            finally:
                board.pop()
                path.pop()
            score = -value
            if ply == 0:
                root_scores.append((move.uci(), score))
            if score > best:
                best, pv = score, [move.uci()] + child_pv
            alpha = max(alpha, score)
            if not exhaustive and alpha >= beta:
                break
        if ply == 0 and exhaustive:
            optimal = [move for move, score in root_scores if score == best]
        return best, pv

    try:
        score, pv = visit(case["depth"], 0, case["alpha"], case["beta"], 0)
        kind = 2 if score <= case["alpha"] else 1 if score >= case["beta"] else 0
        return dict(kind=kind, abort_reason=0, score=score, completed_depth=case["depth"],
                    pv=pv, nodes=nodes, qnodes=qnodes, optimal_moves=optimal)
    except Abort as abort:
        return dict(kind=3, abort_reason=abort.reason, score=0, completed_depth=0,
                    pv=[], nodes=nodes, qnodes=qnodes, optimal_moves=[])


def fixtures():
    cases = []

    def add(name, fen, **kwargs):
        case = dict(name=name, fen=fen, depth=1, max_qply=16, node_budget=50000,
                    evaluator=1, quiescence=1, root_ply=0, alpha=-32000, beta=32000)
        case.update(kwargs)
        assert chess.Board(fen).is_valid(), name
        cases.append(case)
        return case

    add("start-d1", chess.STARTING_FEN)
    add("start-d2", chess.STARTING_FEN, depth=2)
    add("material-static-d2", chess.STARTING_FEN, depth=2, evaluator=0, quiescence=0)
    add("tapered-static-d2", chess.STARTING_FEN, depth=2, quiescence=0)
    authored = [
        ("recapture", "8/7k/4p3/3r4/3Q4/8/8/K7 w - - 0 1", 1),
        ("quiet-evasion", "7k/8/8/8/8/8/r7/K7 w - - 0 1", 0),
        ("ep", "4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12", 1),
        ("ep-pin", "8/6bb/8/8/R1pP2k1/4P3/P7/K7 b - d3 0 1", 1),
        ("promotion", "7k/P7/8/8/8/8/8/7K w - - 0 1", 1),
        ("mate", "7k/6Q1/6K1/8/8/8/8/8 b - - 100 1", 2),
        ("stalemate", "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1", 2),
        ("insufficient", "4k3/8/8/8/8/8/8/4K3 w - - 0 1", 2),
        ("fifty", "7k/8/8/8/8/8/7P/K7 w - - 100 1", 2),
    ]
    for name, fen, depth in authored:
        base = add(name, fen, depth=depth, exhaustive=name != "ep-pin")
        if name in ("recapture", "quiet-evasion", "ep", "promotion"):
            exact = oracle(base, exhaustive=True)
            assert exact["kind"] == 0, (name, exact)
            base["exhaustive_reference"] = exact
            # Narrow windows independently anchored to a no-cutoff tree value.
            add(name + "-lower", fen, depth=depth, alpha=exact["score"] - 2,
                beta=exact["score"] - 1, exhaustive_reference=exact)
            add(name + "-upper", fen, depth=depth, alpha=exact["score"] + 1,
                beta=exact["score"] + 2, exhaustive_reference=exact)
    add("mate-root-ply7", authored[5][1], depth=2, root_ply=7, exhaustive=True)
    quiet = "7k/8/8/8/8/8/7P/K7 w - - 0 1"
    capture = authored[0][1]
    check = authored[1][1]
    add("qcap-quiet", quiet, depth=0, max_qply=0, exhaustive=True)
    add("qcap-capture", capture, depth=0, max_qply=0)
    add("qcap-check", check, depth=0, max_qply=0)
    stand = evaluate(chess.Board(capture), 1)
    add("qcap-standpat-cutoff", capture, depth=0, max_qply=0, alpha=stand - 2, beta=stand - 1)
    add("zero-budget", quiet, node_budget=0)
    add("one-budget-abort", quiet, node_budget=1)
    completed = oracle(cases[0])
    add("exact-budget", chess.STARTING_FEN, node_budget=completed["nodes"])
    add("one-short-budget", chess.STARTING_FEN, node_budget=completed["nodes"] - 1)
    # Previous history excludes current root. Test both root and descendant draw.
    board, history = chess.Board(), []
    # Black is losing, so the repeated descendant's draw changes the best move
    # and root value. Returning to an equal-eval start board would hide a missed
    # draw adjudication in this particular shallow fixture.
    board.remove_piece_at(chess.D8)
    for uci in ["g1f3", "g8f6", "f3g1", "f6g8"] * 2:
        history.append(pack(board).hash)
        board.push_uci(uci)
    add("threefold-root", board.fen(), history=history, depth=2, exhaustive=True)
    board.pop()
    descendant = add("threefold-descendant", board.fen(), history=history[:-1], depth=1)
    drawn = oracle(descendant)
    undrawn = oracle(dict(descendant, history=[]))
    assert drawn["score"] == 0 and drawn["pv"] == ["f6g8"] and undrawn["score"] < 0
    invalid = add("invalid-root-hash", quiet, depth=0)
    invalid["corrupt_hash"] = True
    for case in cases:
        if case.get("exhaustive") and "exhaustive_reference" not in case:
            case["exhaustive_reference"] = oracle(case, exhaustive=True)
        expected = oracle(case)
        if case.get("corrupt_hash"):
            expected = dict(kind=3, abort_reason=4, score=0, completed_depth=0,
                            pv=[], nodes=0, qnodes=0, optimal_moves=[])
        case["expected"] = expected
    return cases


def packet(result):
    assert result.abi_version == 1 and result.struct_size == 304 and result.reserved == 0
    assert result.pv_length <= 64
    return dict(kind=result.kind, abort_reason=result.abort_reason, score=result.score,
                completed_depth=result.completed_depth, pv=[m.uci() for m in result.pv[:result.pv_length]],
                nodes=result.nodes, qnodes=result.qnodes)


def validate(case, observed):
    expected = {k: v for k, v in case["expected"].items() if k != "optimal_moves"}
    assert observed == expected, (case["name"], expected, observed)
    board = chess.Board(case["fen"])
    for uci in observed["pv"]:
        move = chess.Move.from_uci(uci)
        assert move in board.legal_moves, (case["name"], uci)
        board.push(move)
    exact = case.get("exhaustive_reference")
    if exact and exact["kind"] == 0 and observed["kind"] != 3:
        score, true = observed["score"], exact["score"]
        assert (observed["kind"] == 0 and score == true or
                observed["kind"] == 1 and case["beta"] <= score <= true or
                observed["kind"] == 2 and true <= score <= case["alpha"]), (case["name"], exact, observed)
        if observed["kind"] == 0 and observed["pv"]:
            assert observed["pv"][0] in exact["optimal_moves"], case["name"]


def runtime(library, cases):
    lib = c.CDLL(library)
    fn = lib.chen_bounded_search_run
    fn.argtypes = [c.c_int32, c.POINTER(Job), c.c_uint32, c.POINTER(c.c_uint64),
                   c.c_uint64, c.POINTER(Result), c.c_char_p, c.c_uint64]
    fn.restype = c.c_int32
    class DeviceInfo(c.Structure):
        _fields_ = [("name", c.c_char * 128), ("architecture", c.c_char * 64),
                    ("total_memory", c.c_uint64)] + [(name, c.c_uint32) for name in (
                        "compute_units", "wave_size", "async_engines", "runtime_version")]
    lib.chen_gpu_create.argtypes = [c.c_int32, c.c_uint64, c.c_char_p, c.c_uint64]
    lib.chen_gpu_create.restype = c.c_void_p
    lib.chen_gpu_device_info.argtypes = [c.c_void_p, c.POINTER(DeviceInfo), c.c_char_p, c.c_uint64]
    lib.chen_gpu_device_info.restype = c.c_int32
    lib.chen_gpu_destroy.argtypes = [c.c_void_p]
    lib.chen_gpu_destroy.restype = None
    info_error = c.create_string_buffer(1024)
    context = lib.chen_gpu_create(0, 1, info_error, len(info_error))
    assert context, info_error.value.decode()
    info = DeviceInfo()
    try:
        assert lib.chen_gpu_device_info(context, c.byref(info), info_error, len(info_error)) == 0, info_error.value.decode()
    finally:
        lib.chen_gpu_destroy(context)
    device_info = {name: (value.decode() if isinstance(value, bytes) else value)
                   for name, _ in info._fields_ for value in [getattr(info, name)]}
    assert device_info["architecture"].split(":")[0] == "gfx1030"
    all_history = []
    jobs = (Job * len(cases))()
    for job, case in zip(jobs, cases):
        job.root = pack(chess.Board(case["fen"]))
        if case.get("corrupt_hash"):
            job.root.hash ^= 1
        job.abi_version, job.struct_size = 1, c.sizeof(Job)
        job.history_offset, job.history_count = len(all_history), len(case.get("history", []))
        all_history.extend(case.get("history", []))
        for field in ("node_budget", "depth", "max_qply", "evaluator", "quiescence", "root_ply", "alpha", "beta"):
            setattr(job, field, case[field])
    history = (c.c_uint64 * max(1, len(all_history)))(*all_history)
    saved_jobs, saved_history = bytes(jobs), bytes(history)
    error, observed_rows = c.create_string_buffer(1024), []
    # Fresh calls exercise singleton and maximum-size batches; duplicated jobs
    # must remain independent and reusable across device/thread assignments.
    for label, indices in [("single", [0]), ("mixed", list(range(len(cases)))),
                           ("max64", [i % len(cases) for i in range(64)]),
                           ("reverse-reuse", list(reversed(range(len(cases)))) )]:
        batch = (Job * len(indices))(*[jobs[i] for i in indices])
        output = (Result * len(indices))()
        before = bytes(batch)
        started = time.monotonic()
        status = fn(0, batch, len(indices), history, len(all_history), output, error, len(error))
        assert status == 0, (label, error.value.decode())
        assert bytes(batch) == before and bytes(history) == saved_history
        rows = []
        for index, result in zip(indices, output):
            got = packet(result)
            validate(cases[index], got)
            if got["kind"] == 3:
                assert bytes(result.pv) == bytes(c.sizeof(result.pv))
            rows.append(dict(case=cases[index]["name"], result=got))
        observed_rows.append(dict(batch=label, seconds=time.monotonic()-started, rows=rows))
    assert bytes(jobs) == saved_jobs
    rejection_rows = []
    for label, changes, count, history_count in [
        ("depth17", {"depth": 17}, 1, len(all_history)),
        ("empty-window", {"alpha": 0, "beta": 0}, 1, len(all_history)),
        ("qply33", {"max_qply": 33}, 1, len(all_history)),
        ("stack-sum", {"root_ply": 63}, 1, len(all_history)),
        ("history-range", {"history_offset": len(all_history)+1}, 1, len(all_history)),
        ("reserved", {"reserved": 1}, 1, len(all_history)),
        ("budget", {"node_budget": 1000001}, 1, len(all_history)),
        ("batch65", {}, 65, len(all_history)),
        ("history65536", {}, 1, 65536),
    ]:
        job = Job.from_buffer_copy(jobs[0])
        for field, value in changes.items():
            setattr(job, field, value)
        output = (Result * 65)()
        c.memset(output, 0xA5, c.sizeof(output))
        sentinel, job_before = bytes(output), bytes(job)
        status = fn(0, c.byref(job), count, history, history_count, output, error, len(error))
        assert status == -1 and bytes(output) == sentinel and bytes(job) == job_before
        rejection_rows.append(dict(case=label, status=status, error=error.value.decode()))
    return dict(batches=observed_rows, rejections=rejection_rows, device_info=device_info,
                python=platform.python_version(), cpu_affinity=sorted(os.sched_getaffinity(0)))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--cases", required=True)
    parser.add_argument("--library")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.prepare:
        cases = fixtures()
        Path(args.cases).write_text(json.dumps(cases, indent=2, sort_keys=True) + "\n")
        print(json.dumps(dict(cpu_preparation="PASS", cases=len(cases),
                              exhaustive=sum("exhaustive_reference" in case for case in cases),
                              python_chess=chess.__version__)))
        return
    assert args.library and args.output
    raw = Path(args.cases).read_bytes()
    cases = json.loads(raw)
    # Recompute CPU oracle from frozen declaration rather than trusting results.
    for case in cases:
        if not case.get("corrupt_hash"):
            assert oracle(case) == case["expected"], case["name"]
        if "exhaustive_reference" in case:
            full = dict(case, alpha=-32000, beta=32000)
            assert oracle(full, exhaustive=True) == case["exhaustive_reference"], case["name"]
    results = runtime(args.library, cases)
    results.update(status="PASS", python_chess=chess.__version__,
                   cases_sha256=hashlib.sha256(raw).hexdigest(),
                   library_sha256=hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),
                   scope="Rules/traversal independent; Tapered formula reused; no speed/strength claim")
    Path(args.output).write_text(json.dumps(results, indent=2, sort_keys=True) + "\n")
    print(json.dumps(dict(status="PASS", cases=len(cases),
                         job_trajectories=sum(len(batch["rows"]) for batch in results["batches"]),
                         host_rejections=len(results["rejections"]))))


if __name__ == "__main__":
    main()
