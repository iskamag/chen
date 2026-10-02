#!/usr/bin/env python3
"""Special-case CPU certification of the Rust corpus oracle, no teacher metrics."""
import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess

import chess


def check(oracle, source, output):
    spec = importlib.util.spec_from_file_location("independent_hashes", Path(__file__).with_name("audit_semantic_admission.py"))
    hashes = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(hashes)
    fens = [
        chess.STARTING_FEN,
        "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 4294967295",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
        "4k3/P7/8/8/8/8/7p/4K3 b - - 0 1",
        "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4r1k1/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
        "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1",
        "7k/6Q1/6K1/8/8/8/8/8 b - - 100 1",
        "7k/5Q2/6K1/8/8/8/8/8 b - - 100 1",
        "7k/8/8/8/8/8/8/K7 w - - 0 1",
        "7k/8/8/8/8/8/8/KR6 w - - 100 1",
        "7k/8/8/8/8/8/8/KR6 w - - 99 1",
        "7k/8/8/8/8/8/8/KR6 w - - 65535 4294967295",
        "7k/8/8/8/8/8/8/KN6 w - - 0 1",
        "7k/8/8/8/8/8/8/KNN5 w - - 0 1",
    ]
    cases = [(fen, []) for fen in fens]
    board = chess.Board("7k/8/8/8/8/8/8/KR6 w - - 4 1")
    key = hashes.repetition_hash(board)
    cases += [(board.fen(), [key, 0, key]), (board.fen(), [key, key, 0, 0, 0, 0])]
    inputs = "".join(fen + "\t" + ",".join(format(h, "x") for h in history) + "\tchildren\n" for fen, history in cases)
    run = subprocess.run([str(Path(oracle).resolve())], input=inputs, text=True, capture_output=True, timeout=20, check=True)
    snapshots = 0

    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def compare(position, history, snapshot):
        nonlocal snapshots
        pieces = [int(position.pieces(piece, color)) for color in (chess.WHITE, chess.BLACK) for piece in range(1, 7)]
        rights = sum(1 << i for i, (color, kingside) in enumerate(((True, True), (True, False), (False, True), (False, False)))
                     if (position.has_kingside_castling_rights(color) if kingside else position.has_queenside_castling_rights(color)))
        require(snapshot["boards"] == pieces, "piece snapshot differs")
        fields = [int(not position.turn), rights, -1 if position.ep_square is None else position.ep_square,
                  min(position.halfmove_clock, 65535), min(position.fullmove_number, 4294967295)]
        require(snapshot["fields"] == fields, "exact rule fields differ")
        key = hashes.repetition_hash(position)
        require(snapshot["hash"] == key, "legally effective EP hash differs")
        previous = history[-position.halfmove_clock:] if position.halfmove_clock else []
        terminal = (not any(position.legal_moves) or position.halfmove_clock >= 100 or
                    position.is_insufficient_material() or 1 + sum(h == key for h in previous) >= 3)
        require(snapshot["terminal"] is terminal, "current terminal classification differs")
        legal = snapshot["legal"]
        require({row["uci"] for row in legal} == {move.uci() for move in position.legal_moves}, "legal support differs")
        for row in legal:
            move = chess.Move.from_uci(row["uci"])
            flip = 0 if position.turn else 56
            promotion = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]
            action = (((move.from_square ^ flip) * 64 + (move.to_square ^ flip)) * 5 + promotion)
            require(row["action"] == action, "mover-relative canonical action differs")
        require([row["action"] for row in legal] == sorted(row["action"] for row in legal), "legal support is not canonical-sorted")
        snapshots += 1

    for index, (row, (fen, history)) in enumerate(zip(map(json.loads, run.stdout.splitlines()), cases, strict=True)):
        require(row["row"] == index, "oracle row sequence differs")
        board = chess.Board(fen)
        compare(board, history, row["state"])
        require([(child["action"], child["uci"]) for child in row["children"]] ==
                [(move["action"], move["uci"]) for move in row["state"]["legal"]], "successor support differs")
        for edge in row["children"]:
            child = board.copy()
            child.push_uci(edge["uci"])
            child.halfmove_clock = min(child.halfmove_clock, 65535)
            child.fullmove_number = min(child.fullmove_number, 4294967295)
            compare(child, history + [hashes.repetition_hash(board)], edge["state"])
    sha = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
    result = dict(kind="Rust_CPU_corpus_oracle_special_case_check", status="pass", root_cases=len(cases),
                  total_root_and_child_snapshots=snapshots, source_sha256=sha(source), binary_sha256=sha(oracle),
                  checker_sha256=sha(__file__), hash_oracle_sha256=sha(Path(__file__).with_name("audit_semantic_admission.py")),
                  scope="both_movers_castling_promotions_raw_and_pinned_EP_terminal_states_clock_saturation_unsigned_fullmove_history_window_canonical_actions",
                  GPU_runtime_calls=0, teacher_quality_metrics=False)
    path = Path(output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--oracle", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    print(json.dumps(check(args.oracle, args.source, args.output), indent=2))
