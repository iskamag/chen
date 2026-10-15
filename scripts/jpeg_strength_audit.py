#!/usr/bin/env python3
"""Replay every PGN/JSON move, exact outcome and engine timing independently.

This is a separate post-match audit using python-chess. It does not estimate
human/Lichess Elo or adjudicate positions from engine scores.
"""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
import statistics

import chess
import chess.pgn


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def timing(values, seconds_per_move=1.0):
    ordered = sorted(values)
    return {
        "moves": len(values),
        "median_ms": statistics.median(values) / 1e6,
        "p95_ms": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] / 1e6,
        "max_ms": max(values) / 1e6,
        "over_1100ms": sum(x > 1.1e9 for x in values),
        "over_1500ms": sum(x > 1.5e9 for x in values),
        "over_budget_plus_100ms": sum(x > (seconds_per_move + 0.1) * 1e9 for x in values),
    }


def audit(directory, seconds_per_move=1.0):
    metadata = json.loads((directory / "metadata.json").read_text())
    budget = metadata["budgets"]
    assert seconds_per_move > 0
    assert budget["movetime"] == seconds_per_move, "collector budget differs from declared audit budget"
    assert budget["max_plies"] == 512, "512-ply safety cap must remain unscored"
    rows = [json.loads(x) for x in (directory / "games.jsonl").read_text().splitlines() if x.strip()]
    games = []
    with (directory / "games.pgn").open() as source:
        while (game := chess.pgn.read_game(source)) is not None:
            games.append(game)
    assert len(games) == len(rows) and rows, "PGN/JSON count disagreement or no completed game records"
    opening_path = Path(budget["openings"])
    if not opening_path.exists():
        opening_path = directory.parent / opening_path.name
    assert sha(opening_path) == metadata["opening_sha256"]
    opening_map = {}
    for line in opening_path.read_text().splitlines():
        if line.strip() and not line.lstrip().startswith("#"):
            name, fen = line.split("|", 1)
            opening_map[name.strip()] = chess.Board(fen.strip()).fen(en_passant="fen")
    expected_games = 2 * budget["rounds"] * len(opening_map)
    elapsed = defaultdict(list)
    pairs = defaultdict(list)
    score = Counter()
    terminations = Counter()
    special = Counter()
    audited_moves = 0
    per_game = []
    for index, (game, row) in enumerate(zip(games, rows)):
        assert not game.errors, (index, game.errors)
        board = game.board()
        assert row["opening"] in opening_map
        assert 0 <= row["round"] < budget["rounds"]
        assert game.headers["Opening"] == row["opening"]
        assert board.fen(en_passant="fen") == opening_map[row["opening"]]
        pgn_moves = list(game.mainline_moves())
        trace = row["move_trace"]
        assert len(pgn_moves) == len(trace) == row["plies"]
        initial_ply = board.ply()
        for ply, (mv, move) in enumerate(zip(pgn_moves, trace)):
            assert board.outcome(claim_draw=True) is None, (index, ply, "played beyond protocol rule outcome")
            assert move["ply"] == ply
            assert move["fen"] == board.fen(en_passant="fen"), (index, ply, "pre-move FEN")
            assert mv.uci() == move["move"], (index, ply, "PGN/JSON move")
            assert mv in board.legal_moves, (index, ply, "illegal move", mv.uci())
            expected_engine = game.headers["White" if board.turn else "Black"]
            assert move["engine"] == expected_engine
            assert move["wall_ns"] > 0
            elapsed[expected_engine].append(move["wall_ns"])
            special["castles"] += board.is_castling(mv)
            special["en_passant"] += board.is_en_passant(mv)
            special["promotions"] += mv.promotion is not None
            special["captures"] += board.is_capture(mv)
            board.push(mv)
            audited_moves += 1
        assert board.ply() - initial_ply == len(trace)
        assert row["final_fen"] == board.fen(en_passant="fen")
        assert game.headers["Result"] == row["result"]
        assert game.headers["Termination"] == row["termination"]
        outcome = board.outcome(claim_draw=True)
        candidate_color = chess.WHITE if row["candidate_color"] == "white" else chess.BLACK
        assert row["candidate_color"] in ["white", "black"]
        assert game.headers["White" if candidate_color else "Black"] == "candidate"
        if outcome is None:
            assert row["result"] == "*" and row["candidate_score"] is None
            if row["failure"]:
                assert row["termination"] == "technical_failure"
            else:
                assert len(trace) == 512 and row["termination"] == "ply_cap_inconclusive"
            score["unscored"] += 1
        else:
            assert not row["failure"]
            assert outcome.result() == row["result"]
            assert outcome.termination.name == row["termination"]
            expected_score = 0.5 if outcome.winner is None else float(outcome.winner == candidate_color)
            assert row["candidate_score"] == expected_score
            score[{1.0: "wins", 0.5: "draws", 0.0: "losses"}[expected_score]] += 1
        terminations[row["termination"]] += 1
        pairs[(row["round"], row["opening"])].append(row["candidate_color"])
        per_game.append({key: row[key] for key in ["opening", "candidate_color", "result", "candidate_score", "plies", "termination", "failure"]})
    for name, colors in pairs.items():
        assert len(colors) <= 2 and len(set(colors)) == len(colors), (name, "duplicate color")
    # No loaded files may silently change after the run: check every launcher.
    for label, entry in metadata["config"].items():
        assert sha(entry["command"][0]) == metadata["launcher_sha256"][label]
    return {
        "status": "PASS",
        "completed_recorded_games": len(rows),
        "expected_games": expected_games,
        "all_expected_games_recorded": len(rows) == expected_games,
        "audited_legal_moves": audited_moves,
        "candidate_wdl": dict(score),
        "terminations": dict(terminations),
        "special_moves": dict(special),
        "requested_seconds_per_move": seconds_per_move,
        "transport_allowance_ms": 100,
        "timing": {key: timing(values, seconds_per_move) for key, values in elapsed.items()},
        "timing_within_100ms_transport_allowance": all(max(values) <= (seconds_per_move + 0.1) * 1e9 for values in elapsed.values()),
        "games": per_game,
        "engines": metadata["config"],
        "python_chess": chess.__version__,
        "auditor_sha256": sha(__file__),
        "sha256": {name: sha(directory / name) for name in ["metadata.json", "games.pgn", "games.jsonl"]},
        "interpretation": "Exact trace/rule-outcome audit only. Small paired sample selects observed behavior; no Elo estimate or target-parity claim.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--seconds-per-move", type=float, default=1.0,
                        help="expected fixed budget; defaults to the original one-second protocol")
    args = parser.parse_args()
    result = audit(args.directory, args.seconds_per_move)
    text = json.dumps(result, indent=2) + "\n"
    if args.out:
        args.out.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
