import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import chess
from generate_synthetic import Config, action_id, exact_root, family, generate, topology
from audit_synthetic import attacks, audit


class SyntheticTests(unittest.TestCase):
    def test_special_effects_and_parent_frames(self):
        cases = [("r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1", "e1g1", 4),
                 ("4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1", "e5d6", 3),
                 ("7k/P7/8/8/8/8/8/7K w - - 0 1", "a7a8n", 2),
                 ("7K/8/8/8/8/8/p7/7k b - - 0 1", "a2a1q", 2)]
        for fen, uci, edited in cases:
            board = chess.Board(fen)
            self.assertEqual(topology(board), attacks(board, board.turn))
            row = exact_root(board, 0, [])
            action = next(a for a in row["actions"] if a["uci"] == uci)
            self.assertEqual(len(action["edits"]), edited)
            self.assertEqual(action["id"], action_id(chess.Move.from_uci(uci), board.turn))
            board.push_uci(uci)
            parent = topology(board, not board.turn)
            reconstructed = [int(x, 16) for x in row["attacks"]]
            for source, delta in action["attack_xor"]:
                reconstructed[source] ^= int(delta, 16)
            self.assertEqual(reconstructed, parent)

    def test_seeded_reachability_and_separation(self):
        config = Config(families=2, max_plies=64, max_attempts=8)
        first = family(config, 0)
        self.assertEqual(first, family(config, 0))
        board = chess.Board()
        for uci in first["moves"]:
            move = chess.Move.from_uci(uci)
            self.assertIn(move, board.legal_moves)
            board.push(move)
        plies = [r["ply"] for r in first["roots"]]
        self.assertTrue(all(b - a >= config.min_separation for a, b in zip(plies, plies[1:])))
        full = family(Config(families=1, max_plies=192), 0)
        parity = [root["ply"] % 2 for root in full["roots"]]
        self.assertLessEqual(abs(parity.count(0) - parity.count(1)), 1)

    def test_streaming_pilot_independent_rays_and_corruption(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot"
            report = generate(Config(families=2, max_plies=40, max_attempts=8), path)
            self.assertTrue(report["complete"])
            self.assertTrue(audit(path)["passed"])
            shard = path / "families.jsonl.gz"
            with gzip.open(shard, "rt") as source:
                rows = [json.loads(line) for line in source]
            rows[0]["roots"][0]["attacks"][0] = "ffffffffffffffff"
            with gzip.open(shard, "wt") as target:
                for row in rows:
                    target.write(json.dumps(row) + "\n")
            report["shard_sha256"] = hashlib.sha256(shard.read_bytes()).hexdigest()
            (path / "report.json").write_text(json.dumps(report))
            with self.assertRaisesRegex(ValueError, "pseudoattack"):
                audit(path)

    def test_exclusions_reject_whole_family_and_no_overwrite(self):
        config = Config(families=1, max_plies=24, max_attempts=1)
        key = family(config, 0)["roots"][0]["feature_key"]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pilot"
            report = generate(config, path, {key})
            self.assertFalse(report["complete"])
            self.assertEqual(report["counts"]["exclusion_or_prior_family_collision"], 1)
            with self.assertRaises(FileExistsError):
                generate(config, path)


if __name__ == "__main__":
    unittest.main()
