#!/usr/bin/env python3
"""Admit direct policy labels; split whole opening/ancestry components first."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import chess

SNAPSHOTS = {}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def complete_lines(path):
    """Snapshot only complete, newline-terminated records from an append stream."""
    contents = Path(path).read_bytes()
    prefix = contents[: contents.rfind(b"\n") + 1]
    SNAPSHOTS[str(path)] = prefix
    for line in prefix.splitlines():
        if line.strip():
            yield json.loads(line)


def old_train(path):
    # Only this explicitly admitted TRAIN provenance is opened. Historical
    # validation/TEST labels and bulky archives are neither opened nor unpacked.
    provenance = json.loads(Path(path).read_text())
    assert provenance["split"] == "train"
    for item in provenance["queries"]:
        query = item["query"]
        assert query["split"] == "train"
        position = query["position"]
        rows = [
            {"uci": x["uci"], "cp": x["reported_cp"], "mate": None,
             "rank": x["rank"], "depth": x["depth"], "nodes": x["nodes"]}
            for x in item["rows"]
        ]
        prefix = " ".join(position["moves"][:12])
        yield {
            "id": "legacy-" + query["id"], "game_id": position["game_id"],
            "book_group": hashlib.sha256(prefix.encode()).hexdigest()[:16],
            "ply": position["ply"], "fen": position["fen"],
            "history": position["moves"], "moves": rows,
            "hard_best_uci": min(rows, key=lambda x: x["rank"])["uci"],
            "all_legal": query["all_legal"], "soft_eligible": bool(query["all_legal"]),
            "source": "audited-old-TRAIN-only", "teacher_nodes_budget": query["nodes"],
            "score_frame": "raw_stockfish_root_normalized_cp_current_mover",
            "source_record_sha256": item["source_record_sha256"],
        }


class Components:
    def __init__(self):
        self.parent = {}

    def find(self, key):
        if key not in self.parent:
            self.parent[key] = key
        if self.parent[key] != key:
            self.parent[key] = self.find(self.parent[key])
        return self.parent[key]

    def join(self, a, b):
        a, b = self.find(a), self.find(b)
        if a != b:
            self.parent[max(a, b)] = min(a, b)


def validate(row):
    board = chess.Board(row["fen"])
    assert board.is_valid() and not board.is_game_over(claim_draw=True)
    history = row.get("history", [])
    if history:
        replay = chess.Board()
        for move in history:
            mv = chess.Move.from_uci(move)
            assert mv in replay.legal_moves
            replay.push(mv)
        assert replay.fen(en_passant="fen") == board.fen(en_passant="fen")
    legal = {m.uci() for m in board.legal_moves}
    labeled = [x["uci"] for x in row["moves"]]
    assert len(labeled) == len(set(labeled)) and set(labeled) <= legal
    hard = row.get("hard_best_uci", min(row["moves"], key=lambda x: x["rank"])["uci"])
    assert hard in legal
    row["hard_best_uci"] = hard
    if row.get("all_legal"):
        assert set(labeled) == legal
    for item in row["moves"]:
        assert (item.get("cp") is None) != (item.get("mate") is None)
        assert item.get("mate") != 0
    if row.get("actor_choice"):
        assert row["actor_choice"] in legal
    for item in row.get("extra_moves", []):
        assert item["uci"] in legal
        assert (item.get("cp") is None) != (item.get("mate") is None)
        assert item.get("mate") != 0
    row["regret_pairs"] = []
    if row.get("soft_eligible"):
        reference = min(row["moves"], key=lambda x: x["rank"])
        for bad in row["moves"] + row.get("extra_moves", []):
            if bad["uci"] == reference["uci"] or min(bad.get("depth", 0), reference.get("depth", 0)) < 8:
                continue
            if any(x.get("upperbound") or x.get("lowerbound") for x in [reference, bad]):
                continue
            if reference.get("mate") is None and bad.get("mate") is None:
                gap = reference["cp"] - bad["cp"]
                if gap < 200:
                    continue
                margin = min(6.0, gap / 100.0)
                kind = "ordinary_cp_gap"
            elif ((reference.get("mate") or 0) > 0 and (bad.get("mate") or 0) <= 0) or (reference.get("mate") is None and (bad.get("mate") or 0) < 0):
                gap, margin, kind = None, 6.0, "mate_outcome_class"
            else:
                continue
            row["regret_pairs"].append({"good": reference["uci"], "bad": bad["uci"],
                "margin": margin, "gap_cp": gap, "kind": kind,
                "status": "approximate_cross_budget_teacher_estimate" if bad.get("kind") == "forced_actor_choice" else "coherent_packet_teacher_estimate",
                "good_depth": reference["depth"], "bad_depth": bad["depth"]})
    if row.get("soft_eligible"):
        assert len({x["depth"] for x in row["moves"]}) == 1
    # Ignore clocks conservatively when excluding transposition leakage. Raw
    # FEN/history are retained for actual exact feature ingress.
    key = " ".join(board.fen(en_passant="legal").split()[:4])
    return key


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-train", type=Path)
    parser.add_argument("--hard-bank", nargs="*", default=[], type=Path)
    parser.add_argument("--soft-bank", nargs="*", default=[], type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--anchor", type=Path, help="Preserve an earlier admitted split when extending teacher data")
    parser.add_argument("--seed", default=20261015, type=int)
    args = parser.parse_args()
    anchor_rows = {}
    anchor_groups = {}
    if args.anchor:
        # Neutral admission reads identities, not evaluation metrics. The fitter
        # never opens TEST. Existing roots/labels remain byte-for-byte records;
        # new ancestry components bridging old splits are rejected, not moved.
        for split in ["train", "dev", "test"]:
            for row in complete_lines(args.anchor / f"{split}.jsonl"):
                anchor_rows[row["canonical_fen_key"]] = row
                for key in ["book:" + row["book_group"], "game:" + row["game_id"]]:
                    assert key not in anchor_groups or anchor_groups[key] == split
                    anchor_groups[key] = split
    rows = list(old_train(args.old_train)) if args.old_train else []
    inputs = [args.old_train] if args.old_train else []
    for paths, soft in [(args.hard_bank, False), (args.soft_bank, True)]:
        for path in paths:
            inputs.append(path)
            for row in complete_lines(path):
                row = dict(row)
                row["id"] = f"new-{row['game_id']}-{row['ply']}"
                row["soft_eligible"] = bool(soft and row.get("soft_eligible", False))
                row["source"] = "coherent-new-teacher" if soft else "mixed-depth-hard-only-new-teacher"
                row["score_frame"] = "raw_stockfish_root_normalized_cp_current_mover"
                rows.append(row)
    components = Components()
    fen_group = {}
    best_by_fen = {}
    discarded = Counter()
    for row in rows:
        key = validate(row)
        group = "book:" + row["book_group"]
        components.join(group, "game:" + row["game_id"])
        if key in fen_group:
            components.join(group, fen_group[key])
        fen_group[key] = group
        priority = (bool(row.get("soft_eligible")), row.get("teacher_nodes_budget", 0), len(row["moves"]))
        if key not in best_by_fen or priority > best_by_fen[key][0]:
            if key in best_by_fen:
                discarded["replaced_duplicate_fen"] += 1
            best_by_fen[key] = (priority, row)
        else:
            discarded["weaker_duplicate_fen"] += 1
    prepared = {name: [] for name in ["train", "dev", "test"]}
    anchored_components = {}
    for key, split in anchor_groups.items():
        anchored_components.setdefault(components.find(key), set()).add(split)
    for key, (_, row) in best_by_fen.items():
        group = components.find("book:" + row["book_group"])
        anchors = anchored_components.get(group, set())
        key_hash = hashlib.sha256(key.encode()).hexdigest()
        if key_hash in anchor_rows:
            row = dict(anchor_rows[key_hash])
            split = row["split"]
            if len(anchors) > 1:
                prepared[split].append(row)
                continue
        elif len(anchors) > 1:
            discarded["new_component_bridges_frozen_splits"] += 1
            continue
        elif anchors:
            split = next(iter(anchors))
        else:
            bucket = int(hashlib.sha256(f"{args.seed}:{group}".encode()).hexdigest()[:8], 16) % 100
            split = "train" if bucket < 70 else "dev" if bucket < 85 else "test"
        row["split"] = split
        row["split_component"] = hashlib.sha256(group.encode()).hexdigest()[:24]
        row["canonical_fen_key"] = key_hash
        prepared[split].append(row)
    args.out.mkdir(parents=True, exist_ok=False)
    feature_rows = []
    for split, subset in prepared.items():
        subset.sort(key=lambda x: x["id"])
        (args.out / f"{split}.jsonl").write_text("".join(json.dumps(r, separators=(",", ":")) + "\n" for r in subset))
        # Closed TEST FEN/labels are retained separately, not admitted to fitting.
        if split != "test":
            feature_rows.extend(f"{row['id']}|{row['fen']}\n" for row in subset)
    (args.out / "positions.tsv").write_text("".join(feature_rows))
    report = {
        "version": 1, "input_records": len(rows), "deduplicated_positions": len(best_by_fen),
        "discarded": dict(discarded), "seed": args.seed,
        "split_rule": "70/15/15 by transitive opening-prefix/game/canonical-FEN components",
        "splits": {name: {"positions": len(subset), "games": len({r['game_id'] for r in subset}),
                          "components": len({r['split_component'] for r in subset}),
                          "soft_eligible": sum(r['soft_eligible'] for r in subset),
                          "all_legal": sum(r['all_legal'] for r in subset),
                          "regret_pairs": sum(len(r.get('regret_pairs', [])) for r in subset)} for name, subset in prepared.items()},
        "input_sha256": {str(p): hashlib.sha256(SNAPSHOTS[str(p)]).hexdigest() if str(p) in SNAPSHOTS else sha(p) for p in inputs},
        "output_sha256": {p.name: sha(p) for p in args.out.iterdir()},
        "old_test_opened": False, "producer_sha256": sha(__file__),
        "split_anchor": str(args.anchor) if args.anchor else None,
    }
    snapshots = args.out / "input-snapshots"
    snapshots.mkdir()
    report["consumed_prefixes"] = {}
    for i, path in enumerate(args.hard_bank + args.soft_bank):
        prefix = SNAPSHOTS[str(path)]
        dest = snapshots / f"{i:02d}-{path.parent.name}-{path.name}"
        dest.write_bytes(prefix)
        report["consumed_prefixes"][str(path)] = {"bytes": len(prefix), "records": prefix.count(b"\n"),
            "sha256": hashlib.sha256(prefix).hexdigest(), "snapshot": str(dest.relative_to(args.out))}
    (args.out / "manifest.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
