#!/usr/bin/env python3
"""Reachable TRAIN-only mechanics pilot. No teacher, training or HIP dependency.

python-chess 1.11.2 is the optional rules oracle. All children stay in their
source family. Compact effects/topology deltas are supervision, not inputs.
"""
import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import random
import time

import chess

from audit_semantic_admission import feature_key, repetition_hash

SCHEMA = "chen_reachable_mechanics_v1"
VALUES = (0, 100, 320, 330, 500, 900, 0)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def canonical_square(square, mover):
    return square ^ (0 if mover else 56)


def canonical_mask(mask, mover):
    if mover:
        return int(mask)
    return int.from_bytes(int(mask).to_bytes(8, "little"), "big")


def action_id(move, mover):
    promotion = 0 if move.promotion is None else " nbrq".index(chess.piece_symbol(move.promotion))
    return (canonical_square(move.from_square, mover) * 64 +
            canonical_square(move.to_square, mover)) * 5 + promotion


def piece_code(piece, mover):
    return 0 if piece is None else piece.piece_type + (0 if piece.color == mover else 6)


def topology(board, mover=None):
    mover = board.turn if mover is None else mover
    result = [0] * 64
    for square in board.piece_map():
        result[canonical_square(square, mover)] = canonical_mask(board.attacks_mask(square), mover)
    return result


def terminal(board):
    # Match the engine contract: mate first, then claimable current-state draws.
    if not any(board.legal_moves):
        return "checkmate" if board.is_check() else "stalemate"
    if board.halfmove_clock >= 100:
        return "fifty_move"
    if board.is_repetition(3):
        return "threefold"
    if board.is_insufficient_material():
        return "insufficient_material"
    return None


def material(board, mover):
    return sum(VALUES[p.piece_type] * (1 if p.color == mover else -1)
               for p in board.piece_map().values())


def motifs(board):
    moves = list(board.legal_moves)
    result = ["check"] if board.is_check() else []
    for label, predicate in (("capture", board.is_capture), ("en_passant", board.is_en_passant),
                             ("castling", board.is_castling), ("promotion", lambda m: m.promotion is not None)):
        if any(predicate(move) for move in moves):
            result.append(label)
    if any(board.is_pinned(piece.color, square) for square, piece in board.piece_map().items()):
        result.append("pin")
    if len(board.piece_map()) <= 10:
        result.append("low_material")
    if board.halfmove_clock >= 80:
        result.append("high_clock")
    return result


def exact_root(board, ply, previous):
    mover = board.turn
    before = topology(board)
    current_hash = repetition_hash(board)
    retained = previous[-board.halfmove_clock:] if board.halfmove_clock else []
    history = retained + [current_hash]
    pins = [sum(1 << canonical_square(s, mover) for s in board.piece_map()
                if board.color_at(s) == color and board.is_pinned(color, s))
            for color in (mover, not mover)]
    actions = []
    for move in sorted(board.legal_moves, key=lambda m: action_id(m, mover)):
        victim = (move.to_square + (-8 if mover else 8) if board.is_en_passant(move)
                  else move.to_square if board.is_capture(move) else None)
        record = dict(id=action_id(move, mover), uci=move.uci(),
                      capture=board.is_capture(move), en_passant=board.is_en_passant(move),
                      castling=board.is_castling(move), promotion=move.promotion or 0,
                      victim_square=-1 if victim is None else canonical_square(victim, mover))
        old = board.piece_map()
        board.push(move)
        try:
            new = board.piece_map()
            changed = sorted(s for s in set(old) | set(new) if old.get(s) != new.get(s))
            after = topology(board, mover)
            record.update(edits=[[canonical_square(s, mover), piece_code(old.get(s), mover),
                                  piece_code(new.get(s), mover)] for s in changed],
                          attack_xor=[[s, f"{a ^ b:016x}"] for s, (a, b) in enumerate(zip(before, after)) if a != b],
                          child_feature_key=feature_key(board), child_hash=f"{repetition_hash(board):016x}",
                          child_fen=board.fen(en_passant="fen"), child_terminal=terminal(board),
                          child_material_parent_cp=material(board, mover))
        finally:
            board.pop()
        actions.append(record)
    return dict(ply=ply, fen=board.fen(en_passant="fen"), feature_key=feature_key(board),
                identity=digest([feature_key(board), history, ply]),
                current_hash=f"{current_hash:016x}", previous_hashes=[f"{h:016x}" for h in retained],
                history_counters=[board.halfmove_clock, history.count(current_hash), len(history), ply],
                mover="white" if mover else "black", terminal=terminal(board), motifs=motifs(board),
                attacks=[f"{x:016x}" for x in before], pins=[f"{x:016x}" for x in pins],
                checkers=f"{canonical_mask(board.checkers_mask(), mover):016x}",
                material_mover_cp=material(board, mover), actions=actions)


@dataclass(frozen=True)
class Config:
    seed: int = 20261003
    families: int = 256
    max_plies: int = 192
    roots_per_family: int = 8
    min_ply: int = 12
    min_separation: int = 8
    max_attempts: int = 1024

    def validate(self):
        if not (1 <= self.families <= self.max_attempts and 1 <= self.max_plies <= 256 and
                1 <= self.roots_per_family <= 8 and 1 <= self.min_ply <= self.max_plies and
                1 <= self.min_separation <= self.max_plies):
            raise ValueError("invalid bounded pilot configuration")


def choose_move(board, rng, policy):
    moves = sorted(board.legal_moves, key=lambda m: action_id(m, board.turn))
    if policy == "uniform" or rng.random() < 0.25:
        return rng.choice(moves)
    # A declared weak, current-state causal policy. No search/teacher claim.
    scored = []
    mover = board.turn
    for move in moves:
        score = 40 * int(board.is_castling(move))
        board.push(move)
        score += material(board, mover) + 20 * int(board.is_check())
        if board.is_checkmate():
            score += 30000
        board.pop()
        scored.append((score + rng.uniform(-100, 100), move))
    return max(scored, key=lambda pair: pair[0])[1]


def family(config, attempt):
    seed = int(digest([SCHEMA, config.seed, attempt])[:16], 16)
    rng = random.Random(seed)
    policy = "uniform" if attempt % 2 == 0 else "noisy_one_ply_material"
    board, moves, candidates = chess.Board(), [], []
    for ply in range(config.max_plies + 1):
        end = terminal(board)
        if ply >= config.min_ply and end is None:
            candidates.append((ply, motifs(board)))
        if end is not None or ply == config.max_plies:
            break
        move = choose_move(board, rng, policy)
        moves.append(move.uci())
        board.push(move)
    # Half spread across phase; half event candidates, always separated.
    selected = []
    if candidates:
        count = min((config.roots_per_family + 1) // 2, len(candidates))
        indices = [int(i * (len(candidates) - 1) / max(1, count - 1)) for i in range(count)]
        # Even spacing can silently select one mover for every phase anchor.
        # Alternate desired parity, with a per-family randomized first mover.
        priority = [min((row for row in candidates if row[0] % 2 == (seed + i) % 2),
                        key=lambda row: abs(row[0] - candidates[index][0]),
                        default=candidates[index]) for i, index in enumerate(indices)]
        events = candidates[:]
        rng.shuffle(events)
        events.sort(key=lambda row: sum(label in row[1] for label in
                    ("promotion", "en_passant", "castling", "pin", "check", "low_material")), reverse=True)
        for ply, _ in priority:
            if all(abs(ply - old) >= config.min_separation for old in selected):
                selected.append(ply)
        while len(selected) < config.roots_per_family:
            eligible = [row for row in events if all(abs(row[0] - old) >= config.min_separation for old in selected)]
            if not eligible:
                break
            wanted = 0 if sum(p % 2 == 0 for p in selected) < sum(p % 2 == 1 for p in selected) else 1
            ply, _ = next((row for row in eligible if row[0] % 2 == wanted), eligible[0])
            selected.append(ply)
    replay, previous, roots = chess.Board(), [], []
    for ply in range(len(moves) + 1):
        if ply in selected:
            roots.append(exact_root(replay, ply, previous))
        if ply < len(moves):
            previous.append(repetition_hash(replay))
            replay.push_uci(moves[ply])
    return dict(schema=SCHEMA, family_id=digest([SCHEMA, moves]), split="train", attempt=attempt,
                seed=seed, policy=policy, moves=moves, roots=roots, final_fen=replay.fen(en_passant="fen"),
                termination=terminal(replay) or "capped_no_outcome", outcome_label=None)


def exclusion_keys(paths):
    """Read input fields only; never select using labels/metrics in old reports.

    Caller supplies admissions and/or spent reports. Expand explicit roots two
    plies to match previous spent-family protection; child_fen fields count too.
    Coverage is precisely the bound files, not a claim of a universal registry.
    """
    keys, bindings = set(), []
    def add(fen, depth):
        board = chess.Board(fen)
        def visit(left):
            keys.add(feature_key(board))
            if left:
                for move in list(board.legal_moves):
                    board.push(move)
                    visit(left - 1)
                    board.pop()
        visit(depth)
    def walk(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key in ("fen", "root_fen", "child_fen") and isinstance(item, str):
                    add(item, 2 if key != "child_fen" else 0)
                elif key == "canonical_position_fens":
                    for fen in item:
                        add(fen, 0)
                elif key in ("source_records", "split", "roots", "positions", "runs", "families"):
                    walk(item)
        elif isinstance(value, list):
            for item in value:
                walk(item)
    for path in paths:
        raw = Path(path).read_bytes()
        data = gzip.decompress(raw) if str(path).endswith(".gz") else raw
        # Root admission split names are containers, not score fields.
        parsed = json.loads(data)
        sources = parsed.get("source_records") or parsed.get("split", {}).get("source_records")
        if sources:
            for rows in sources.values():
                for row in rows:
                    if "positions" in row and "action_root_index" in row:
                        for position in row["positions"]:
                            add(position["fen"], 0)
                        add(row["positions"][row["action_root_index"]]["fen"], 1)
                    else:
                        walk(row)
        else:
            walk(parsed)
        bindings.append(dict(path=str(Path(path).resolve()), sha256=hashlib.sha256(raw).hexdigest()))
    return keys, bindings


def generate(config, output, excluded=(), bindings=()):
    config.validate()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    keys, ids = set(excluded), set()
    exclusion_payload = json.dumps(sorted(keys), separators=(",", ":")).encode()
    (output / "exclusions.json.gz").write_bytes(gzip.compress(exclusion_payload, mtime=0))
    stats, policies, motif_counts, phases, movers = (Counter() for _ in range(5))
    declaration = dict(schema=SCHEMA, config=asdict(config), split="train", teacher_labels=False,
                       selected_using_teacher=False, generated_on="cpu_python_chess", oracle_version=chess.__version__,
                       source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                       exclusion_sources=list(bindings), exclusion_key_count=len(keys),
                       exclusion_sha256=hashlib.sha256(exclusion_payload).hexdigest(),
                       effect_frame="parent_mover", root_frame="root_mover", outcomes=False,
                       selection="alternating_mover_phase_spread_then_balanced_event_priority_min_separation", inclusion_probability=None)
    (output / "declaration.json").write_text(json.dumps(declaration, indent=2) + "\n")
    shard = output / "families.jsonl.gz"
    with shard.open("wb") as raw, gzip.GzipFile(filename="", fileobj=raw, mode="wb", mtime=0) as stream:
        for attempt in range(config.max_attempts):
            row = family(config, attempt)
            stats["attempted_families"] += 1
            if not row["roots"] or row["family_id"] in ids:
                stats["empty_or_duplicate_family"] += 1
                continue
            family_keys = {root["feature_key"] for root in row["roots"]}
            family_keys.update(action["child_feature_key"] for root in row["roots"] for action in root["actions"])
            if family_keys & keys:
                stats["exclusion_or_prior_family_collision"] += 1
                continue
            ids.add(row["family_id"])
            keys.update(family_keys)
            stats["families"] += 1
            policies[row["policy"]] += 1
            for root in row["roots"]:
                stats["roots"] += 1
                stats["actions"] += len(root["actions"])
                stats["changed_attack_sources"] += sum(len(a["attack_xor"]) for a in root["actions"])
                movers[root["mover"]] += 1
                motif_counts.update(root["motifs"])
                pieces = len(chess.Board(root["fen"]).piece_map())
                phases["low_10" if pieces <= 10 else "middle_11_24" if pieces <= 24 else "high_25_32"] += 1
            stream.write(json.dumps(row, separators=(",", ":")).encode() + b"\n")
            if stats["families"] >= config.families:
                break
    elapsed = time.perf_counter() - started
    report = dict(kind="synthetic_mechanics_train_pilot", complete=stats["families"] == config.families,
                  counts=dict(stats), policies=dict(policies), root_motifs=dict(motif_counts),
                  material_buckets=dict(phases), movers=dict(movers), wall_seconds=elapsed,
                  roots_per_second=stats["roots"] / elapsed, actions_per_second=stats["actions"] / elapsed,
                  compressed_bytes=shard.stat().st_size, bytes_per_root=shard.stat().st_size / max(1, stats["roots"]),
                  shard_sha256=hashlib.sha256(shard.read_bytes()).hexdigest(),
                  declaration_sha256=hashlib.sha256((output / "declaration.json").read_bytes()).hexdigest(),
                  strength_evidence=False, neural_training=False, closed_test_access=False)
    (output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--families", type=int, default=256)
    parser.add_argument("--seed", type=int, default=20261003)
    parser.add_argument("--max-plies", type=int, default=192)
    parser.add_argument("--max-attempts", type=int, default=1024)
    parser.add_argument("--exclude-admission", action="append", default=[])
    args = parser.parse_args()
    excluded, bindings = exclusion_keys(args.exclude_admission)
    report = generate(Config(seed=args.seed, families=args.families, max_plies=args.max_plies,
                             max_attempts=args.max_attempts), args.output, excluded, bindings)
    print(json.dumps(report, indent=2))
    return 0 if report["complete"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
