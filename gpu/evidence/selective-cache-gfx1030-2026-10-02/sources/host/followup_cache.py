#!/usr/bin/env python3
"""Pinned native step parity and separate 1s sparse classical trace workloads."""
import argparse
import ctypes as c
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
from profile_engine import digest, run, write
from rules_probe import FIXTURES, Move, Position, Result, chess, pack


class Config(c.Structure):
    _fields_ = [(x, c.c_uint32) for x in ("abi", "size", "max_ply", "policy", "batch", "candidates")] + [("reserved", c.c_uint64)]


class Stats(c.Structure):
    _fields_ = [(x, c.c_uint32) for x in ("abi", "size")] + [(x, c.c_uint64) for x in
        ("arena", "classified", "evaluated", "steps", "expanded", "open")] + [
        ("selected", c.c_uint32), ("added", c.c_uint32), ("kernel_ms", c.c_double)] + [
        (x, c.c_uint32) for x in ("minimum", "maximum", "pv_depth", "root_flags", "stop", "reserved")]


def bind(path):
    lib = c.CDLL(str(path), mode=c.RTLD_LOCAL)
    p, u64, err = c.c_void_p, c.c_uint64, c.c_char_p
    signatures = {
        "create": [c.c_int32, u64, err, u64], "destroy": [p],
        "reset": [p, c.POINTER(Position), c.POINTER(u64), u64, c.POINTER(Config), c.POINTER(Stats), err, u64],
        "advance": [p, u64, c.c_uint32, p, c.POINTER(Stats), err, u64],
        "finish": [p, c.POINTER(Result), c.POINTER(Move), u64, c.POINTER(Stats), err, u64],
    }
    for name, signature in signatures.items():
        f = getattr(lib, "chen_selective_" + name)
        f.argtypes = signature
        f.restype = p if name == "create" else None if name == "destroy" else c.c_int32
    lib.chen_selective_abi_version.restype = c.c_uint32
    assert lib.chen_selective_abi_version() == 1
    return lib


def comparable(stats):
    return {key: getattr(stats, key) for key, _ in stats._fields_ if key != "kernel_ms"}


def deep(args):
    assert (c.sizeof(Config), c.sizeof(Stats), c.sizeof(Position), c.sizeof(Move)) == (32, 96, 128, 4)
    libs = [bind(args.baseline), bind(args.candidate)]
    error = c.create_string_buffer(1024)
    contexts = [lib.chen_selective_create(0, 6000, error, len(error)) for lib in libs]
    assert all(contexts), error.value
    records = []
    try:
        # Reuse both contexts through deep/high-marker, shallow, terminal and
        # history resets. This comparison never downloads private arena state.
        cases = [(i, 64, batch, evaluator, 5000, False) for i in range(3)
                 for batch in (1, 32, 256) for evaluator in (0, 1)]
        cases += [(i, 3, 256, 1, 1000, False) for i in range(6, len(FIXTURES))]
        cases += [(0, 64, 256, 1, limit, False) for limit in (0, 1, 20, 21, 5999, 10000)]
        cases += [(0, 64, 256, 1, 5000, True)]
        for index, (fen_index, depth, batch, evaluator, limit, history_draw) in enumerate(cases):
            board = chess.Board(FIXTURES[fen_index])
            if history_draw:
                board.halfmove_clock = 2
            root = pack(board)
            config = Config(1, 32, depth, 0, batch, 1024, 0)
            history = (c.c_uint64 * 2)(root.hash, root.hash) if history_draw else None
            for lib, context in zip(libs, contexts):
                stats = Stats()
                assert lib.chen_selective_reset(context, c.byref(root), history, 2 if history_draw else 0,
                    c.byref(config), c.byref(stats), error, len(error)) == 0, error.value
                assert stats.root_flags == 0 and stats.stop == 5
            snapshots = []
            for step in range(512):
                arms = []
                for lib, context in zip(libs, contexts):
                    stats = Stats()
                    assert lib.chen_selective_advance(context, limit, evaluator, None, c.byref(stats), error, len(error)) == 0, error.value
                    assert stats.classified <= limit and stats.arena <= 6000
                    state = comparable(stats)
                    if stats.root_flags & 1:
                        result, pv = Result(), (Move * 64)()
                        assert lib.chen_selective_finish(context, c.byref(result), pv, 64, c.byref(stats), error, len(error)) == 0, error.value
                        moves = [pv[i].uci() for i in range(result.pv_length)]
                        legal = board.copy(stack=False)
                        for text in moves:
                            move = chess.Move.from_uci(text)
                            assert move in legal.legal_moves, (index, step, text)
                            legal.push(move)
                        state.update(score=result.score, pv=moves)
                    arms.append(state)
                assert arms[0] == arms[1], (index, step, arms)
                snapshots.append(arms[0])
                if history_draw:
                    assert arms[0]["root_flags"] & 2 and arms[0]["score"] == 0 and arms[0]["stop"] == 1
                if arms[0]["stop"] != 0:
                    break
            else:
                raise RuntimeError("native step bound exceeded")
            record = {"case": index, "fen": board.fen(), "max_ply": depth, "batch": batch,
                      "evaluator": evaluator, "absolute_limit": limit, "repeated_root_history": history_draw,
                      "snapshots": snapshots, "exact_per_step_parity": True}
            records.append(record)
            print(json.dumps({"case": index, "steps": len(snapshots), "last": snapshots[-1]}), flush=True)
        write(args.output / "result.json", {"status": "passed", "cases": records,
              "scope": "Native pinned-reference per-step score/PV/public metadata parity plus independent legal PV; synthetic maximum bounds are a separate fixture."})
    finally:
        for lib, context in zip(libs, contexts):
            if context:
                lib.chen_selective_destroy(context)


def uci(args):
    suite = json.loads(args.suite.read_text())
    prefix = ["/opt/rocm/bin/rocprofv3", "--runtime-trace", "--stats", "--output-format", "csv",
              "--output-directory", "{stem}-trace", "--output-file", "trace", "--"] if args.mode == "trace" else []
    order = ("baseline", "candidate") if prefix else ("baseline", "candidate", "candidate", "baseline")
    records, legal_pvs = [], 0
    for repeat, arm in enumerate(order, 1):
        library = getattr(args, arm)
        directory = args.output / (arm if prefix else str(repeat) + "-" + arm)
        directory.mkdir()
        if prefix:
            directory = directory / "gpu-trace"
            directory.mkdir()
        runs = []
        for item in suite:
            assert digest(library) == getattr(args, arm + "_sha256")
            stem = directory.resolve() / ("1-" + item["name"])
            env = dict(os.environ, CHEN_GPU_LIBRARY=str(library))
            argv = ["taskset", "-c", "5", *[x.replace("{stem}", str(stem)) for x in prefix], str(args.engine), "uci"]
            result = run(argv, env, stem, item["fen"], "hip-selective", 1000, 32, 250000)
            result.update(name=item["name"], category=item["category"], repeat=1, arm=arm)
            board = chess.Board(item["fen"])
            assert chess.Move.from_uci(result["bestmove"]) in board.legal_moves
            for line in stem.with_suffix(".stdout").read_text().splitlines():
                if line.startswith("info depth ") and " pv " in line:
                    b = board.copy(stack=False)
                    for text in line.split(" pv ", 1)[1].split():
                        move = chess.Move.from_uci(text)
                        assert move in b.legal_moves
                        b.push(move)
                    legal_pvs += 1
            runs.append(result)
            print(json.dumps(result), flush=True)
        write(directory / "result.json", {"status": "passed", "runs": runs})
        records.extend(runs)
    write(args.output / "result.json", {"status": "passed", "runs": records, "legal_reported_pvs": legal_pvs,
          "scope": "One-second observations with unequal completed work; isolated instrumentation is not a latency or strength comparison."})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode", choices=("deep", "uci", "trace"))
    for name in ("baseline", "candidate", "engine", "suite", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline", "candidate", "engine"):
        p.add_argument("--" + name + "-sha256", required=True)
    args = p.parse_args()
    for name in ("baseline", "candidate", "engine", "suite"):
        setattr(args, name, getattr(args, name).resolve(strict=True))
    for name in ("baseline", "candidate", "engine"):
        assert digest(getattr(args, name)) == getattr(args, name + "_sha256")
    for key in ("HSA_OVERRIDE_GFX_VERSION", "CHEN_NEURAL_MODEL", "CHEN_NEURAL_LIBRARY", "LD_PRELOAD"):
        assert not os.environ.get(key), key
    args.output.mkdir(parents=True, exist_ok=False)
    write(args.output / "declaration.json", {"mode": args.mode, "source_sha256": digest(__file__),
        "artifacts": {name: {"path": str(getattr(args, name)), "sha256": digest(getattr(args, name))}
                      for name in ("baseline", "candidate", "engine", "suite")},
        "cpu_affinity": [5], "concurrent_load": "Root CPU0 strength match has ended; GPU exclusively reserved. Shared system cache/power uncontrolled.",
        "variant": "packed_best_open_child_cache_v1"})
    if args.mode == "deep":
        deep(args)
    else:
        uci(args)
    for name in ("baseline", "candidate", "engine"):
        assert digest(getattr(args, name)) == getattr(args, name + "_sha256")


if __name__ == "__main__":
    main()
