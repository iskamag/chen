#!/usr/bin/env python3
"""Replay mechanics shards using scalar attack rays independent of producer.

python-chess remains the shared legal-move oracle. Hash/feature identity formula
is shared; this is not an independent chess implementation or strength test.
"""
import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path

import chess
from audit_semantic_admission import feature_key, repetition_hash


def attacks(board, mover):
    output = [0] * 64
    flip = 0 if mover else 56
    for square, piece in board.piece_map().items():
        file, rank = square % 8, square // 8
        if piece.piece_type == chess.PAWN:
            directions, slider = [(-1, 1 if piece.color else -1), (1, 1 if piece.color else -1)], False
        elif piece.piece_type == chess.KNIGHT:
            directions, slider = [(1, 2), (2, 1), (2, -1), (1, -2), (-1, -2), (-2, -1), (-2, 1), (-1, 2)], False
        else:
            directions = []
            if piece.piece_type in (chess.BISHOP, chess.QUEEN, chess.KING):
                directions += [(1, 1), (1, -1), (-1, 1), (-1, -1)]
            if piece.piece_type in (chess.ROOK, chess.QUEEN, chess.KING):
                directions += [(1, 0), (-1, 0), (0, 1), (0, -1)]
            slider = piece.piece_type != chess.KING
        for df, dr in directions:
            f, r = file + df, rank + dr
            while 0 <= f < 8 and 0 <= r < 8:
                target = r * 8 + f
                output[square ^ flip] |= 1 << (target ^ flip)
                if not slider or board.piece_at(target) is not None:
                    break
                f, r = f + df, r + dr
    return output


def require(condition, message):
    if not condition:
        raise ValueError(message)


def audit(path):
    path = Path(path)
    declaration = json.loads((path / "declaration.json").read_text())
    report = json.loads((path / "report.json").read_text())
    require(hashlib.sha256((path / "families.jsonl.gz").read_bytes()).hexdigest() == report["shard_sha256"], "shard hash")
    require(hashlib.sha256((path / "declaration.json").read_bytes()).hexdigest() == report["declaration_sha256"], "declaration hash")
    require(declaration["split"] == "train" and not declaration["teacher_labels"], "pilot scope")
    exclusion_payload = gzip.decompress((path / "exclusions.json.gz").read_bytes())
    require(hashlib.sha256(exclusion_payload).hexdigest() == declaration["exclusion_sha256"], "exclusion binding")
    excluded = set(json.loads(exclusion_payload))
    require(len(excluded) == declaration["exclusion_key_count"], "exclusion count")
    counts, ids, prior_keys = Counter(), set(), set()
    with gzip.open(path / "families.jsonl.gz", "rt") as stream:
        for line in stream:
            family = json.loads(line)
            require(family["split"] == "train" and family["family_id"] not in ids, "family split/identity")
            require(family["family_id"] == hashlib.sha256(json.dumps([family["schema"], family["moves"]], sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "family content identity")
            ids.add(family["family_id"])
            roots = {root["ply"]: root for root in family["roots"]}
            require(len(roots) == len(family["roots"]) <= 8, "root support")
            board, previous, family_keys = chess.Board(), [], set()
            for ply in range(len(family["moves"]) + 1):
                if ply in roots:
                    root = roots[ply]
                    require(root["fen"] == board.fen(en_passant="fen"), "root witness")
                    mover, flip = board.turn, 0 if board.turn else 56
                    before = attacks(board, mover)
                    require(before == [int(x, 16) for x in root["attacks"]], "root pseudoattack rays")
                    retained = previous[-board.halfmove_clock:] if board.halfmove_clock else []
                    current = repetition_hash(board)
                    require(root["previous_hashes"] == [f"{x:016x}" for x in retained], "retained history")
                    require(root["current_hash"] == f"{current:016x}", "current identity")
                    require(root["history_counters"] == [board.halfmove_clock, (retained + [current]).count(current), len(retained) + 1, ply], "history counters")
                    require(root["feature_key"] == feature_key(board), "root feature identity")
                    identity = [feature_key(board), retained + [current], ply]
                    require(root["identity"] == hashlib.sha256(json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()).hexdigest(), "history identity")
                    pins = [sum(1 << (s ^ flip) for s, p in board.piece_map().items() if p.color == c and board.is_pinned(c, s)) for c in (mover, not mover)]
                    require(pins == [int(x, 16) for x in root["pins"]], "pins")
                    checkers = sum(1 << (s ^ flip) for s in board.checkers())
                    require(checkers == int(root["checkers"], 16), "checkers")
                    legal = {m.uci() for m in board.legal_moves}
                    require(len(root["actions"]) == len(legal) and {a["uci"] for a in root["actions"]} == legal, "complete legal support")
                    ids_at_root = []
                    family_keys.add(feature_key(board))
                    for action in root["actions"]:
                        move = chess.Move.from_uci(action["uci"])
                        promo = {None: 0, chess.KNIGHT: 1, chess.BISHOP: 2, chess.ROOK: 3, chess.QUEEN: 4}[move.promotion]
                        expected_id = ((move.from_square ^ flip) * 64 + (move.to_square ^ flip)) * 5 + promo
                        require(action["id"] == expected_id, "action identity/frame")
                        ids_at_root.append(expected_id)
                        ep, capture, castle = board.is_en_passant(move), board.is_capture(move), board.is_castling(move)
                        victim = move.to_square - (8 if mover else -8) if ep else move.to_square if capture else None
                        require((action["en_passant"], action["capture"], action["castling"], action["promotion"], action["victim_square"]) == (ep, capture, castle, move.promotion or 0, -1 if victim is None else victim ^ flip), "action flags/victim")
                        old = [board.piece_at(s) for s in range(64)]
                        board.push(move)
                        after = attacks(board, mover)
                        def code(piece):
                            return 0 if piece is None else piece.piece_type + (0 if piece.color == mover else 6)
                        expected_edits = [[s ^ flip, code(old[s]), code(board.piece_at(s))] for s in range(64) if old[s] != board.piece_at(s)]
                        require(action["edits"] == expected_edits, "exact action edits")
                        delta = [[s, f"{a ^ b:016x}"] for s, (a, b) in enumerate(zip(before, after)) if a != b]
                        require(action["attack_xor"] == delta, "complete changed/unchanged attack sources")
                        require(action["child_fen"] == board.fen(en_passant="fen") and action["child_feature_key"] == feature_key(board), "child state identity")
                        require(action["child_hash"] == f"{repetition_hash(board):016x}", "child hash")
                        expected_cp = sum((0, 100, 320, 330, 500, 900, 0)[p.piece_type] * (1 if p.color == mover else -1) for p in board.piece_map().values())
                        require(action["child_material_parent_cp"] == expected_cp, "parent-frame material")
                        family_keys.add(feature_key(board))
                        board.pop()
                        counts["actions"] += 1
                        counts["changed_attack_sources"] += len(delta)
                    require(ids_at_root == sorted(set(ids_at_root)), "canonical complete actions")
                    counts["roots"] += 1
                if ply < len(family["moves"]):
                    previous.append(repetition_hash(board))
                    move = chess.Move.from_uci(family["moves"][ply])
                    require(move in board.legal_moves, "legal standard-start reachability")
                    board.push(move)
            require(board.fen(en_passant="fen") == family["final_fen"], "final witness")
            require(not family_keys & prior_keys, "cross-family observed/planned-child overlap")
            require(not family_keys & excluded, "supplied spent-input overlap")
            prior_keys.update(family_keys)
            counts["families"] += 1
    for key, value in counts.items():
        require(report["counts"][key] == value, "reported count " + key)
    return dict(kind="scalar_ray_legal_replay_audit", counts=dict(counts), passed=True,
                independent_attack_rays=True, shared_python_chess_legality=True, shared_hash_formula=True,
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), strength_evidence=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path")
    args = parser.parse_args()
    print(json.dumps(audit(args.path), indent=2))
