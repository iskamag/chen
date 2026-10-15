"""Admission and runtime boundary regressions, including all-mate packets."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from fit import Dataset, teacher_value
from model import ChannelPolicy, exported_forward
from prepare import SNAPSHOTS, complete_lines, validate


class Boundaries(unittest.TestCase):
    def test_append_snapshot_ignores_partial_tail(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "stream.jsonl"
            path.write_bytes(b'{"n":1}\n{"n":')
            self.assertEqual(list(complete_lines(path)), [{"n": 1}])
            self.assertEqual(SNAPSHOTS[str(path)], b'{"n":1}\n')

    def test_all_mate_packet_batch_is_float(self):
        data = Dataset.__new__(Dataset)
        data.features = torch.zeros((2, 1))
        row = {"count": 2, "offset": 0, "soft_eligible": True, "hard_index": 0,
               "labeled_indices": [0, 1], "labeled_values": [teacher_value({"mate": 2}), teacher_value({"mate": -3})], "pairs": []}
        values = data.batch([row], 100)[5]
        self.assertEqual(values.dtype, torch.float32)
        self.assertGreater(values[0, 0], values[0, 1])

    def test_forced_unknown_not_inserted_in_soft_packet(self):
        import chess
        row = {"fen": chess.STARTING_FEN, "history": [], "hard_best_uci": "e2e4",
               "soft_eligible": True, "moves": [{"uci": "e2e4", "cp": 10, "mate": None, "rank": 1, "depth": 10}],
               "actor_choice": "a2a3", "extra_moves": [{"uci": "a2a3", "cp": -390, "mate": None, "depth": 9, "kind": "forced_actor_choice"}]}
        validate(row)
        self.assertEqual([m["uci"] for m in row["moves"]], ["e2e4"])
        self.assertEqual(row["regret_pairs"][0]["gap_cp"], 400)
        row["extra_moves"][0]["depth"] = 7
        validate(row)
        self.assertEqual(row["regret_pairs"], [])

    def test_320_deployment_serialization(self):
        channels = [{"name": name, "start": start, "end": end} for name, start, end in [
            ("action", 0, 32), ("material", 32, 48), ("pressure", 48, 112),
            ("destination_balance", 112, 144), ("slider_lines", 144, 192),
            ("knight_pawn", 192, 224), ("king_blocks", 224, 256), ("support_obligations", 256, 320)]]
        torch.manual_seed(4)
        for architecture in ["interaction", "additive", "flat"]:
            model = ChannelPolicy(channels, 320, architecture)
            if model.residual is not None:
                torch.nn.init.normal_(model.residual[2].weight, std=0.1)
            inputs = torch.randn(17, 320)
            deployment = json.loads(json.dumps(model.export({"feature_schema_sha256": "test"})))
            actual = model(inputs).detach().numpy()
            expected, _ = exported_forward(deployment, inputs.numpy())
            np.testing.assert_allclose(actual, expected, atol=2e-6, rtol=2e-6)


if __name__ == "__main__":
    torch.set_num_threads(1)
    unittest.main()
