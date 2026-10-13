"""Licensed game-family admission and explicitly typed strong-teacher labels.

CPU ingestion/provenance only. Teacher search is opt-in after a sealed declaration;
no module import starts an engine, downloads data or produces scores.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import re
import selectors
import subprocess
import time
import urllib.request

ARCHIVE_NAME = "lichess_db_standard_rated_2013-01.pgn.zst"
ARCHIVE_URL = "https://database.lichess.org/standard/" + ARCHIVE_NAME
ARCHIVE_SHA256 = "aa40b3671fa3cf1072eb182892cd90b0e1e003a4a5943492f64b77e7f3fd1635"
# Locally derived from the publisher-pinned compressed bytes; not a second
# publisher checksum. Prevent accepting an unrelated pre-existing decode.
PGN_SHA256 = "8963b6a1620a0e9c77e5515a0744ec133e86869487188af047bb0a74400dee37"
LICENSE_URL = "https://database.lichess.org/"
CHECKSUM_URL = "https://database.lichess.org/standard/sha256sums.txt"
STANDALONE_CP = "stockfish_standalone_root_normalized_cp"
PARENT_ACTION_CP = "stockfish_parent_root_normalized_action_cp"


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def acquire_shard(directory, *, max_compressed_bytes=32 * 1024**2, max_pgn_bytes=256 * 1024**2):
    """Full small official shard; bounded download/decode, exact publisher hash."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    archive = directory / ARCHIVE_NAME
    if not archive.exists():
        temporary = archive.with_name(archive.name + ".part")
        with urllib.request.urlopen(ARCHIVE_URL, timeout=30) as response, temporary.open("wb") as sink:
            count = 0
            while block := response.read(1024 * 1024):
                count += len(block)
                if count > max_compressed_bytes:
                    raise ValueError("compressed shard exceeds acquisition bound")
                sink.write(block)
        if sha256_file(temporary) != ARCHIVE_SHA256:
            raise ValueError("official archive SHA mismatch; partial retained for diagnosis")
        temporary.replace(archive)
    if archive.stat().st_size > max_compressed_bytes or sha256_file(archive) != ARCHIVE_SHA256:
        raise ValueError("existing archive differs from publisher pin")
    pgn = directory / ARCHIVE_NAME.removesuffix(".zst")
    if not pgn.exists():
        temporary = pgn.with_name(pgn.name + ".part")
        process = subprocess.Popen(["zstd", "-q", "-d", "-c", str(archive)], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        try:
            with temporary.open("wb") as sink:
                count = 0
                while block := process.stdout.read(1024 * 1024):
                    count += len(block)
                    if count > max_pgn_bytes:
                        raise ValueError("decompressed shard exceeds acquisition bound")
                    sink.write(block)
            if process.wait(timeout=10) != 0:
                raise ValueError("zstd rejected archive: " + process.stderr.read(4096).decode(errors="replace"))
        finally:
            if process.poll() is None:
                process.kill(); process.wait(timeout=5)
            process.stdout.close(); process.stderr.close()
        temporary.replace(pgn)
    if pgn.stat().st_size > max_pgn_bytes:
        raise ValueError("existing PGN exceeds decompression bound")
    if sha256_file(pgn) != PGN_SHA256:
        raise ValueError("decoded PGN differs from the validated publisher archive decode")
    manifest = {"kind": "official_cc0_lichess_full_month_shard", "url": ARCHIVE_URL,
                "license": "CC0-1.0", "license_source": LICENSE_URL, "checksum_source": CHECKSUM_URL,
                "archive_path": str(archive), "archive_sha256": ARCHIVE_SHA256,
                "compressed_bytes": archive.stat().st_size, "pgn_path": str(pgn),
                "pgn_sha256": sha256_file(pgn), "decompressed_bytes": pgn.stat().st_size,
                "temporal_scope": "January2013; old rating/timecontrol population, not contemporary strength evidence",
                "limits": {"compressed_bytes": max_compressed_bytes, "pgn_bytes": max_pgn_bytes}}
    atomic_json(directory / "source.json", manifest)
    return manifest


@dataclass(frozen=True)
class GamePosition:
    game_id: str
    ply: int
    fen: str
    moves: tuple[str, ...]
    previous_hashes: tuple[int, ...]


@dataclass(frozen=True)
class GameFamily:
    game_id: str
    game_sha256: str
    moves_sha256: str
    source_offset: int
    source_end_offset: int
    ratings: tuple[int, int]
    time_control: str
    positions: tuple[GamePosition, ...]
    action_root_index: int
    all_moves: tuple[str, ...]


def _header_reason(headers, min_rating, estimated_seconds):
    if headers.get("Variant", "Standard").casefold() not in ("standard", "chess") or headers.get("SetUp") == "1" or "FEN" in headers:
        return "nonstandard_origin_or_variant"
    if "rated" not in headers.get("Event", "").casefold() or "casual" in headers.get("Event", "").casefold():
        return "nonrated_header"
    try:
        ratings = tuple(int(headers[name]) for name in ("WhiteElo", "BlackElo"))
    except (KeyError, ValueError):
        return "missing_rating"
    if min(ratings) < min_rating:
        return "rating_filter"
    control = re.fullmatch(r"(\d+)\+(\d+)", headers.get("TimeControl", ""))
    if not control or int(control[1]) + 40 * int(control[2]) < estimated_seconds:
        return "estimated_bullet_or_unbounded_control"
    if not re.fullmatch(r"https://lichess\.org/[A-Za-z0-9]{8}", headers.get("Site", "")):
        return "missing_canonical_game_site"
    return None


def index_games(pgn, *, min_rating=1800, estimated_seconds=180, seed=20261007, max_games=130000):
    import chess.pgn
    if any(type(x) is not int or x < 1 for x in (min_rating, estimated_seconds, max_games)) or type(seed) is not int:
        raise ValueError("positive admission bounds and integer seed required")
    selected, counts, seen_sites = [], Counter(), set()
    with Path(pgn).open(encoding="utf-8", newline="") as stream:
        for _ in range(max_games):
            offset = stream.tell(); headers = chess.pgn.read_headers(stream)
            if headers is None:
                break
            counts["games_scanned"] += 1
            reason = _header_reason(headers, min_rating, estimated_seconds)
            if reason:
                counts[reason] += 1; continue
            site = headers["Site"]
            if site in seen_sites:
                counts["duplicate_game_site"] += 1; continue
            seen_sites.add(site)
            game_id = hashlib.sha256(site.encode()).hexdigest()
            priority = hashlib.sha256(f"{seed}:{game_id}".encode()).hexdigest()
            selected.append({"offset": offset, "end_offset": stream.tell(), "game_id": game_id,
                             "priority": priority, "ratings": [int(headers["WhiteElo"]), int(headers["BlackElo"])],
                             "time_control": headers["TimeControl"]})
        else:
            if chess.pgn.read_headers(stream) is not None:
                raise ValueError("PGN game scan cap reached before EOF")
    counts["header_candidates"] = len(selected)
    return sorted(selected, key=lambda row: (row["priority"], row["game_id"])), dict(counts)


def family_at(pgn, record, *, plies=(16, 24, 32, 40, 48, 56, 64, 72), action_root_ply=32, max_plies=512,
              min_game_plies=None):
    import chess
    import chess.pgn
    from .bridge import position_from_fen
    if (not plies or any(type(ply) is not int or ply < 1 for ply in plies) or tuple(sorted(set(plies))) != tuple(plies)
            or action_root_ply not in plies or max(plies) >= max_plies):
        raise ValueError("increasing observation plies and retained root ply required")
    with Path(pgn).open(encoding="utf-8", newline="") as stream:
        stream.seek(record["offset"]); game = chess.pgn.read_game(stream); end = stream.tell()
    if game is None or game.errors or end != record["end_offset"] or game.board().fen() != chess.STARTING_FEN:
        return None, "parse_error_or_origin"
    moves = tuple(move.uci() for move in game.mainline_moves())
    minimum = max(plies) + 1 if min_game_plies is None else min_game_plies
    if type(minimum) is not int or minimum < max(plies):
        raise ValueError('game length must cover every selected observation')
    if len(moves) < minimum or len(moves) > max_plies:
        return None, "game_length"
    board, positions, hashes, prefix = game.board(), [], [], []
    for ply in range(max(plies) + 1):
        if ply in plies:
            if (not list(board.legal_moves) or board.halfmove_clock >= 100 or
                    board.is_insufficient_material() or board.is_repetition(3)):
                return None, "terminal_observation"
            positions.append(GamePosition(record["game_id"], ply, board.fen(en_passant="fen"), tuple(prefix), tuple(hashes)))
        if ply == max(plies):
            break
        move = chess.Move.from_uci(moves[ply])
        hashes.append(position_from_fen(board.fen(en_passant="fen")).hash)
        prefix.append(move.uci()); board.push(move)
    if len(positions) != len(plies):
        return None, "missing_observation"
    raw = Path(pgn).open("rb")
    try:
        raw.seek(record["offset"]); source_bytes = raw.read(end - record["offset"])
    finally:
        raw.close()
    return GameFamily(record["game_id"], hashlib.sha256(source_bytes).hexdigest(),
                      hashlib.sha256(" ".join(moves).encode()).hexdigest(), record["offset"], end,
                      tuple(record["ratings"]), record["time_control"], tuple(positions), plies.index(action_root_ply), moves), None


def family_positions(family):
    """Exactly the planned model inputs: eight observations plus all root children."""
    import chess
    from .teacher import canonical_position_key
    keys = {canonical_position_key(position.fen) for position in family.positions}
    if len(keys) != len(family.positions):
        return None
    root = chess.Board(family.positions[family.action_root_index].fen)
    for move in list(root.legal_moves):
        root.push(move); keys.add(canonical_position_key(root)); root.pop()
    return keys


def spent_positions(paths):
    from .teacher import RootFamily, RootSample, _family_positions, canonical_position_key
    keys, sources = set(), []
    for path in paths:
        raw = Path(path).read_bytes(); data = json.loads(raw)
        sources.append({"path": str(path), "sha256": hashlib.sha256(raw).hexdigest()})
        if "split" in data and "source_records" in data["split"]:
            for families in data["split"]["source_records"].values():
                for family in families:
                    roots = tuple(RootSample(row["family_id"], row["ply"], row["fen"], tuple(row["previous_hashes"])) for row in family["roots"])
                    keys.update(_family_positions(RootFamily(family["family_id"], family["origin_fen"], tuple(family["moves"]), roots)))
        elif "canonical_position_fens" in data:
            keys.update(canonical_position_key(fen) for fen in data["canonical_position_fens"])
        elif data.get("kind") == "cpu_only_frozen_plan_actionblind_batch_geometry_diagnostic":
            keys.update(canonical_position_key(row["root_fen"]) for run in data["runs"] for row in run["roots"])
        else:
            raise ValueError("unknown spent-input provenance schema: " + str(path))
    return keys, sources


def admit_families(pgn, *, counts=None, seed=20261007, min_rating=1800, estimated_seconds=180,
                   spent_reports=(), plies=(16, 24, 32, 40, 48, 56, 64, 72), action_root_ply=32):
    counts = counts or {"train": 256, "validation": 64, "test": 64}
    if set(counts) != {"train", "validation", "test"} or any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError("positive three-way family counts required")
    if action_root_ply not in plies or plies.index(action_root_ply) % 2:
        raise ValueError('alternating mover schedule requires an even-index root observation')
    records, index_stats = index_games(pgn, min_rating=min_rating, estimated_seconds=estimated_seconds, seed=seed)
    old, spent = spent_positions(spent_reports)
    groups = {name: [] for name in ("train", "validation", "test")}
    keys = {name: set() for name in groups}; rejected = Counter(); move_ids = set()
    for record in records:
        parity = int(record['game_id'][-1], 16) % 2
        selected_plies = tuple(ply + ((index + parity) % 2) for index, ply in enumerate(plies))
        family, reason = family_at(pgn, record, plies=selected_plies, action_root_ply=action_root_ply+parity,
                                   min_game_plies=max(plies)+1)
        if family is None:
            rejected[reason] += 1; continue
        observations = family_positions(family)
        if observations is None:
            rejected["repeated_canonical_observation"] += 1; continue
        if family.moves_sha256 in move_ids:
            rejected["duplicate_mainline"] += 1; continue
        if observations & old:
            rejected["spent_input_collision"] += 1; continue
        if observations & (keys["train"] | keys["validation"] | keys["test"]):
            rejected["cross_game_canonical_collision"] += 1; continue
        name = next(name for name in groups if len(groups[name]) < counts[name])
        groups[name].append(family); keys[name].update(observations); move_ids.add(family.moves_sha256)
        if all(len(groups[name]) == counts[name] for name in groups):
            break
    if any(len(groups[name]) != counts[name] for name in groups):
        raise ValueError("insufficient fresh eligible game families: " + str({name: len(values) for name, values in groups.items()}))
    source = {name: [asdict(family) for family in values] for name, values in groups.items()}
    import chess
    mover_counts = {}
    for name, families in groups.items():
        mover_counts[name] = dict(observations=dict(Counter('white' if chess.Board(p.fen).turn else 'black'
                                    for family in families for p in family.positions)),
                                 action_roots=dict(Counter('white' if chess.Board(f.positions[f.action_root_index].fen).turn else 'black'
                                    for f in families)),
                                 all_legal_child_rows=sum(chess.Board(f.positions[f.action_root_index].fen).legal_moves.count()
                                    for f in families))
    manifest = {"kind": "fresh_licensed_real_game_family_admission", "pgn_sha256": sha256_file(pgn),
                "seed": seed, "families": counts, "plies": list(plies), "action_root_ply": action_root_ply,
                "admission_revision": 2, "mover_schedule": "base_ply+((observation_index+game_id_last_hex_parity)%2)",
                "mover_counts": mover_counts,
                "filter": {"both_min_rating": min_rating, "estimated_base_plus40increments_min_seconds": estimated_seconds,
                           "standard_start_rated_only": True, "game_min_plies_inclusive": max(plies)+1},
                "index_counts": index_stats, "rejections": dict(rejected), "source_records": source,
                "canonical_counts": {name: len(value) for name, value in keys.items()},
                "cross_split_overlap": 0, "spent_input_overlap": 0, "spent_position_count": len(old), "spent_reports": spent,
                "source_sha256": hashlib.sha256(json.dumps(source, sort_keys=True).encode()).hexdigest(),
                "outcomes_comments_embedded_evaluations_used_as_labels": False,
                "scope": "whole game families and all planned root/child model inputs; shared players/unobserved opening ancestry may remain"}
    return {name: tuple(families) for name, families in groups.items()}, manifest


@dataclass(frozen=True)
class PVLine:
    depth: int
    seldepth: int
    multipv: int
    score_kind: str
    score: int
    bound: str | None
    nodes: int
    pv: tuple[str, ...]
    tablebase_hits: int = 0


def parse_pv(line):
    tokens = line.split()
    if not tokens or tokens[0] != "info" or "string" in tokens or "score" not in tokens or "pv" not in tokens:
        return None
    def integer(key, default=None):
        if key not in tokens:
            if default is not None: return default
            raise ValueError("teacher score line missing " + key)
        return int(tokens[tokens.index(key) + 1])
    score_index, pv_index = tokens.index("score"), tokens.index("pv")
    kind, score = tokens[score_index + 1], int(tokens[score_index + 2])
    if kind not in ("cp", "mate"):
        raise ValueError("unknown teacher score kind")
    flags = [flag for flag in ("lowerbound", "upperbound") if flag in tokens[:pv_index]]
    if len(flags) > 1:
        raise ValueError("conflicting teacher score bounds")
    depth, rank, nodes = integer("depth"), integer("multipv", 1), integer("nodes")
    if depth < 1 or rank < 1 or nodes < 1:
        raise ValueError("invalid teacher telemetry")
    return PVLine(depth, integer("seldepth", depth), rank, kind, score, flags[0] if flags else None,
                  nodes, tuple(tokens[pv_index + 1:]), integer('tbhits', 0))


def common_cp_labels(lines, position, *, all_legal=False, min_depth=6):
    """Last complete common-depth PV packet, never mixed latest rank depths."""
    import chess
    from .features import move_to_index
    if type(all_legal) is not bool or type(min_depth) is not int or min_depth < 1:
        raise ValueError('explicit query kind and positive reported-depth floor required')
    board = chess.Board()
    for uci in position.moves:
        move = chess.Move.from_uci(uci)
        if move not in board.legal_moves:
            raise ValueError("teacher history contains illegal move")
        board.push(move)
    if board.fen(en_passant="fen") != position.fen:
        raise ValueError("teacher position/history mismatch")
    legal = {move.uci() for move in board.legal_moves}
    wanted = len(legal) if all_legal else 1
    if not wanted:
        return None, "terminal_root"
    by_depth, packet, packet_depth = {}, {}, None
    for line in lines:
        parsed = parse_pv(line)
        if parsed is None:
            continue
        # Stockfish emits ordered rank1..K packets. Do not join an incomplete
        # new rank1 update with stale ranks from a preceding same-depth packet.
        if parsed.multipv == 1:
            packet, packet_depth = {}, parsed.depth
        if parsed.depth != packet_depth or parsed.multipv != len(packet) + 1:
            packet, packet_depth = {}, None
            continue
        packet[parsed.multipv] = parsed
        if len(packet) == wanted:
            by_depth[parsed.depth] = dict(packet)
    complete = list(by_depth)
    if not complete:
        return None, "incomplete_common_depth_support"
    depth = max(complete)
    if depth < min_depth:
        return None, "common_depth_below_floor"
    selected = [by_depth[depth][rank] for rank in range(1, wanted + 1)]
    if any(row.score_kind != "cp" or row.bound is not None for row in selected):
        return None, "mate_or_bound_common_packet"
    # SF19 formats decisive tablebase scores as CP near +/-20000 with up to
    # MAX_PLY=246 plies distance. Tables are disabled in the producer too. Reject
    # the entire reserved region rather than guessing from a cp token alone.
    if any(abs(row.score) >= 20000 - 246 or row.tablebase_hits != 0 for row in selected):
        return None, "decisive_tablebase_or_extreme_cp_region"
    first = [row.pv[0] if row.pv else "" for row in selected]
    if len(set(first)) != wanted or any(move not in legal for move in first) or (all_legal and set(first) != legal):
        return None, "nonunique_or_incomplete_legal_actions"
    for row in selected:
        copy = board.copy(stack=True)
        for uci in row.pv:
            move = chess.Move.from_uci(uci)
            if move not in copy.legal_moves:
                raise ValueError("teacher PV contains illegal continuation")
            copy.push(move)
    if any(selected[index].score < selected[index + 1].score for index in range(len(selected) - 1)):
        return None, "nonmonotonic_common_rank_scores"
    mover = "w" if board.turn else "b"
    counts = [len(board.pieces(piece, color)) for color in (chess.WHITE, chess.BLACK) for piece in range(1, 7)]
    from .score_frames import PHASE_WEIGHTS, sf19_reference_score, score_frame_manifest
    root_material = sum(count * weight for count, weight in zip(counts, PHASE_WEIGHTS * 2))
    rows = [{"canonical_action": move_to_index(row.pv[0], mover), "uci": row.pv[0],
             "root_score_cp": row.score, "negated_action_score_cp": -row.score,
             "converted_score": sf19_reference_score(row.score, root_material,
                 score_kind="ordinary_unbounded_cp_no_tablebase", negate=all_legal),
             "pv": list(row.pv), "reported_depth": row.depth, "seldepth": row.seldepth, "nodes": row.nodes} for row in selected]
    result = {"target_kind": PARENT_ACTION_CP if all_legal else STANDALONE_CP,
              "game_id": position.game_id, "ply": position.ply, "reference_fen": position.fen,
              "reference_piece_counts_absolute": counts, "normalization_reference": "parent/root material; SF19 win_rate_params/to_cp",
              "normalization_root_material": root_material,
              "normalization_conversion_performed": True, "score_frame": score_frame_manifest(),
              "converted_perspective": "child_mover" if all_legal else "queried_root_mover",
              "tablebase_policy": "disabled_and_reserved_decisive_cp_region_rejected",
              "common_reported_depth": depth,
              "root_side_to_move": mover, "rows": rows,
              "exact_minimax_or_uniform_horizon_claim": False, "outcome_or_wdl_label": False}
    return result, None


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
            # A dead teacher may leave buffered requests in stdin. Closing it
            # can raise BrokenPipeError; preserve the original EOF/timeout and
            # always retire the output pipe and selector.
            try:
                self.process.stdin.close()
            except BrokenPipeError:
                pass
            self.process.stdout.close()
        if getattr(self, "selector", None) is not None: self.selector.close()

    def __enter__(self): return self
    def __exit__(self, *error): self.close()
