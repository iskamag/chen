#!/usr/bin/env python3
"""Bounded, explicitly pinned UCI workloads for separate sampling/trace runs.

Instrumentation is an argv prefix, never a shell command. Profiling results
are not clean latency comparisons or strength measurements.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import select
import signal
import subprocess
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    with Path(path).open("x") as f:
        json.dump(value, f, indent=2, allow_nan=False)
        f.write("\n")


def run(argv, env, stem, fen, backend, movetime_ms, batch, capacity):
    started = time.monotonic()
    commands, lines, pending = [], [], bytearray()
    matched = None
    go_started = None
    with Path(str(stem) + ".stderr").open("xb") as err, Path(str(stem) + ".stdout").open("xb") as out:
        proc = subprocess.Popen(argv, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                stderr=err, env=env, start_new_session=True)
        def send(line):
            commands.append(line)
            proc.stdin.write((line + "\n").encode())
            proc.stdin.flush()

        def wait_for(predicate, timeout):
            deadline = time.monotonic() + timeout
            while True:
                while b"\n" in pending:
                    line, _, tail = pending.partition(b"\n")
                    pending[:] = tail
                    text = line.decode("utf-8", errors="strict").rstrip("\r")
                    lines.append(text)
                    if "backend failed" in text or "info string invalid" in text:
                        raise RuntimeError(text)
                    if predicate(text):
                        return text
                remain = deadline - time.monotonic()
                if remain <= 0:
                    raise TimeoutError("UCI response deadline exceeded")
                ready, _, _ = select.select([proc.stdout], [], [], remain)
                if not ready:
                    continue
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    raise RuntimeError("UCI stdout closed before required response")
                out.write(chunk)
                out.flush()
                pending.extend(chunk)

        try:
            send("uci")
            wait_for(lambda x: x == "uciok", 10)
            for option in ("Backend " + backend, "Eval tapered", "Hash 16", "Level 20",
                           "SelectiveBatch " + str(batch), "MaxPositions " + str(capacity)):
                name, value = option.rsplit(" ", 1)
                send("setoption name " + name + " value " + value)
            send("ucinewgame")
            send("isready")
            wait_for(lambda x: x == "readyok", 10)
            send("position fen " + fen)
            go_started = time.monotonic()
            send("go depth 64 movetime " + str(movetime_ms))
            matched = wait_for(lambda x: x.startswith("bestmove "), movetime_ms / 1000 + 10)
            search_wall = time.monotonic() - go_started
            if matched.split()[1] == "0000":
                raise RuntimeError("nonterminal workload returned null move")
            send("quit")
            proc.stdin.close()
            proc.wait(timeout=10)
            if proc.returncode != 0:
                raise RuntimeError("profiled engine exited " + str(proc.returncode))
            # Drain profiler output after engine retirement, preserving exact bytes.
            drain_deadline = time.monotonic() + 5
            while True:
                remain = drain_deadline - time.monotonic()
                if remain <= 0 or not select.select([proc.stdout], [], [], max(0, remain))[0]:
                    raise TimeoutError("retired leader stdout drain deadline exceeded")
                chunk = os.read(proc.stdout.fileno(), 65536)
                if not chunk:
                    break
                out.write(chunk)
                pending.extend(chunk)
            os.killpg(proc.pid, 0)
            raise RuntimeError("owned process group remains after leader retirement")
        except ProcessLookupError:
            pass
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            proc.wait(timeout=5)
            raise
    infos = [x for x in lines if x.startswith("info depth ")]
    result = {"argv": argv, "pid": proc.pid, "backend": backend, "fen": fen,
              "commands": commands, "movetime_ms": movetime_ms, "depth_limit": 64,
              "selective_batch": batch, "max_positions": capacity,
              "bestmove": matched.split()[1], "search_response_wall_seconds": search_wall,
              "last_info": infos[-1] if infos else None, "exit_code": proc.returncode,
              "process_group_absent": True, "process_wall_seconds": time.monotonic() - started,
              "scope": "Observed instrumented or uninstrumented UCI workload; no strength claim"}
    write(str(stem) + ".json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", type=Path, required=True)
    parser.add_argument("--engine-sha256", required=True)
    parser.add_argument("--library", type=Path)
    parser.add_argument("--library-sha256")
    parser.add_argument("--suite", type=Path, required=True, help="JSON list of {name,category,fen}")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--backend", choices=("alphabeta", "hip-selective"), required=True)
    parser.add_argument("--cpu", type=int, default=5)
    parser.add_argument("--movetime-ms", type=int, default=1000)
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--capacity", type=int, default=250000)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--prefix-json", default="[]", help="argv before engine, with optional {stem} substitution")
    args = parser.parse_args()
    args.engine = args.engine.resolve(strict=True)
    if digest(args.engine) != args.engine_sha256:
        parser.error("engine fingerprint mismatch")
    if not 1 <= args.movetime_ms <= 2000 or not 1 <= args.repeats <= 3:
        parser.error("bounded profile budget exceeded")
    if args.cpu not in os.sched_getaffinity(0):
        parser.error("CPU outside permitted affinity")
    prefix = json.loads(args.prefix_json)
    if not isinstance(prefix, list) or any(not isinstance(x, str) for x in prefix):
        parser.error("prefix must be a JSON argv list")
    env = dict(os.environ)
    for key in ("HSA_OVERRIDE_GFX_VERSION", "CHEN_NEURAL_MODEL", "CHEN_NEURAL_LIBRARY", "LD_PRELOAD"):
        if env.get(key):
            parser.error("unexpected inherited " + key)
    if args.backend == "hip-selective":
        if args.library is None or args.library_sha256 is None:
            parser.error("GPU requires external library pin")
        args.library = args.library.resolve(strict=True)
        if digest(args.library) != args.library_sha256:
            parser.error("native library fingerprint mismatch")
        env["CHEN_GPU_LIBRARY"] = str(args.library)
    suite = json.loads(args.suite.read_text())
    if not isinstance(suite, list) or not 1 <= len(suite) <= 6:
        parser.error("bounded profile suite requires 1..6 positions")
    for item in suite:
        if set(item) != {"name", "category", "fen"} or not all(isinstance(x, str) for x in item.values()):
            parser.error("invalid suite item")
        if not item["name"].replace("-", "").isalnum() or "\n" in item["fen"]:
            parser.error("invalid workload name or FEN")
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = {"kind": "bounded_separate_uci_profiling_workloads", "source_sha256": digest(__file__),
                   "engine_sha256": args.engine_sha256, "library_sha256": args.library_sha256,
                   "suite_sha256": digest(args.suite), "suite": suite, "backend": args.backend,
                   "cpu_affinity": [args.cpu], "movetime_ms": args.movetime_ms, "depth_limit": 64,
                   "batch": args.batch, "capacity": args.capacity, "repeats": args.repeats,
                   "instrumentation_prefix": prefix, "concurrent_load": "Root strength match on CPU0; avoid SMT siblings6/11. Shared cache/power remain uncontrolled.",
                   "scope": "Profiling and independent UCI workload observation; algorithms have different work and are not a speedup comparison"}
    write(args.output / "declaration.json", declaration)
    results = []
    try:
        for repeat in range(args.repeats):
            for item in suite:
                if digest(args.engine) != args.engine_sha256 or (args.library and digest(args.library) != args.library_sha256):
                    raise RuntimeError("artifact changed after declaration")
                stem = args.output.resolve() / (str(repeat + 1) + "-" + item["name"])
                argv = ["taskset", "-c", str(args.cpu), *[x.replace("{stem}", str(stem)) for x in prefix], str(args.engine), "uci"]
                result = run(argv, env, stem, item["fen"], args.backend,
                             args.movetime_ms, args.batch, args.capacity)
                result.update(name=item["name"], category=item["category"], repeat=repeat + 1)
                results.append(result)
                print(json.dumps(result), flush=True)
        write(args.output / "result.json", {"status": "passed", "runs": results})
    except Exception as error:
        write(args.output / "result.json", {"status": "failed_or_partial", "error": str(error), "runs": results})
        raise


if __name__ == "__main__":
    main()
