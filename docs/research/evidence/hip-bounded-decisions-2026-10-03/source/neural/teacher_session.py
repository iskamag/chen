"""CPU UCI transport extracted unchanged from the audited packet-v2 teacher.

Original source SHA256: b5bdc1b2df754673504c656c53cd37389a7989ed7c67644fa39d8bac606c7df5
StockfishSession, GamePosition and sha256_file blocks are verbatim. No model
input or target-selection policy lives in this transport module.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import re
import selectors
import subprocess
import time

def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


@dataclass(frozen=True)
class GamePosition:
    game_id: str
    ply: int
    fen: str
    moves: tuple[str, ...]
    previous_hashes: tuple[int, ...]


class StockfishSession:
    """Bounded serialized UCI subprocess; constructing starts no search."""
    def __init__(self, binary, *, expected_sha256, timeout=20, transcript_bytes=8 * 1024**2):
        if type(timeout) not in (int, float) or not 0 < timeout <= 60 or type(transcript_bytes) is not int or transcript_bytes < 1024:
            raise ValueError("bounded positive UCI deadlines and transcript bytes required")
        if sha256_file(binary) != expected_sha256:
            raise ValueError("teacher binary differs from declared SHA")
        self.binary_sha256 = expected_sha256
        self.timeout, self.transcript_bytes = timeout, transcript_bytes
        self.buffer = b""; self.trace = []
        self.process = subprocess.Popen([str(Path(binary).resolve())], cwd=Path(binary).resolve().parent,
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        os.set_blocking(self.process.stdout.fileno(), False)
        self.selector = selectors.DefaultSelector(); self.selector.register(self.process.stdout, selectors.EVENT_READ)
        try:
            self.send("uci"); self.identity = self.receive(lambda line: line == "uciok")
            if not any(line.startswith("id name Stockfish 19") for line in self.identity):
                raise ValueError("unexpected teacher UCI identity")
            for command in ("setoption name Threads value 1", "setoption name Hash value 64",
                            "setoption name NumaPolicy value none", "setoption name Ponder value false",
                            "setoption name UCI_LimitStrength value false", "setoption name Skill Level value 20",
                            "setoption name UCI_ShowWDL value false", "setoption name UCI_Chess960 value false",
                            "setoption name SyzygyPath value", "setoption name SyzygyProbeLimit value 0"):
                self.send(command)
            self.send("isready"); self.receive(lambda line: line == "readyok")
            self.setup_trace = tuple(self.take_trace())
        except BaseException:
            self.close(); raise

    def send(self, command):
        if "\n" in command or "\r" in command:
            raise ValueError("UCI command newline rejected")
        self.trace.append("> " + command)
        self.process.stdin.write((command + "\n").encode()); self.process.stdin.flush()

    def take_trace(self):
        trace, self.trace = self.trace, []
        return trace

    def receive(self, done):
        started, lines, size = time.monotonic(), [], 0
        while True:
            while b"\n" in self.buffer:
                raw, self.buffer = self.buffer.split(b"\n", 1)
                line = raw.decode("utf-8", errors="strict").rstrip("\r")
                lines.append(line); self.trace.append("< " + line)
                if "CRITICAL ERROR" in line:
                    raise RuntimeError("teacher rejected position: " + line)
                if done(line): return lines
            remaining = self.timeout - (time.monotonic() - started)
            if remaining <= 0:
                raise TimeoutError("teacher UCI deadline exceeded")
            events = self.selector.select(min(remaining, .1))
            if not events:
                if self.process.poll() is not None:
                    raise RuntimeError("teacher exited before completion")
                continue
            block = os.read(self.process.stdout.fileno(), 65536)
            if not block:
                raise RuntimeError("teacher EOF before completion")
            size += len(block)
            if size > self.transcript_bytes:
                raise ValueError("teacher transcript exceeds admission bound")
            self.buffer += block

    def analyze(self, position, *, nodes, all_legal=False, declaration_sha256):
        """Caller must pin/validate declaration; this API refuses an implicit search."""
        import chess
        if type(nodes) is not int or nodes < 1 or not re.fullmatch(r"[0-9a-f]{64}", declaration_sha256):
            raise ValueError("positive node budget and sealed declaration SHA required")
        board = chess.Board()
        for uci in position.moves:
            move = chess.Move.from_uci(uci)
            if move not in board.legal_moves: raise ValueError("illegal teacher ingress history")
            board.push(move)
        if board.fen(en_passant="fen") != position.fen:
            raise ValueError("teacher history/FEN mismatch")
        width = len(list(board.legal_moves)) if all_legal else 1
        if not 1 <= width <= 256: raise ValueError("teacher MultiPV support exceeds bounded schema")
        if self.trace:
            raise ValueError("previous request transcript must be drained before another search")
        try:
            self.send("setoption name MultiPV value " + str(width))
            self.send("ucinewgame"); self.send("isready"); self.receive(lambda line: line == "readyok")
            self.send("position startpos" + (" moves " + " ".join(position.moves) if position.moves else ""))
            self.send("go nodes " + str(nodes))
            lines = self.receive(lambda line: line.startswith("bestmove "))
            best = lines[-1].split()
            if len(best) < 2 or best[1] not in {move.uci() for move in board.legal_moves}:
                raise ValueError("teacher bestmove is absent or illegal")
        except BaseException:
            # A timeout/partial response cannot be recycled as the next query.
            self.close()
            raise
        return lines

    def close(self):
        if getattr(self, "process", None) is not None:
            if self.process.poll() is None:
                try: self.send("quit"); self.process.wait(timeout=2)
                except (OSError, subprocess.TimeoutExpired): self.process.kill(); self.process.wait(timeout=5)
            self.process.stdin.close(); self.process.stdout.close()
        if getattr(self, "selector", None) is not None: self.selector.close()

    def __enter__(self): return self
    def __exit__(self, *error): self.close()
