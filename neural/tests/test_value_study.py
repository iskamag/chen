"""Value-only semantics, exact grouping, supervision masks and device hot path."""
import os
import unittest

import torch

from neural.features import encode_fen
from neural.teacher import SearchReplay, comparable_model, material_cp
from neural.tests.test_teacher import ring_fixture
from neural.tests.test_bridge import HotResident, PayloadResident
from neural.value_heads import ValueAdapter
from neural.value_study import (HEADS, gather_values, value_loss, value_step,
                                metrics, _ordering, paired_family_interval)


class ValueStudyCPU(unittest.TestCase):
    def test_exact_current_gather_alignment_without_dense_or_next_observations(self):
        tensors, labels = ring_fixture()
        replay = SearchReplay(8, "cpu"); replay.append(tensors, labels)
        with HotResident():
            batch = gather_values(replay, torch.tensor([7, 1, 7, 3]))
        self.assertEqual(set(batch), {"features", "game_ids", "sequence_ids", "search_score_cp", "teacher_root_id", "parent_node"})
        self.assertTrue(torch.equal(batch["search_score_cp"], labels["search_score_cp"][[7, 1, 7, 3]]))
        self.assertTrue(torch.equal(batch["sequence_ids"], torch.tensor([7, 1, 7, 3])))
        old = batch["features"].clone()
        replay.append(tensors, labels)
        self.assertTrue(torch.equal(batch["features"], old))

    def test_common_cp_loss_uses_unrounded_outputs_and_excludes_mates(self):
        class Fixed(torch.nn.Module):
            def __init__(self):
                super().__init__(); self.score = torch.nn.Parameter(torch.tensor([100.25, -50.75, 999.]))
            def forward(self, features):
                return {"value_cp": self.score, "utility": (self.score / 800).tanh()}
        adapter = Fixed()
        batch = {"features": torch.zeros(3, 18, 8, 8), "search_score_cp": torch.tensor([0, 0, -29995], dtype=torch.int32)}
        with HotResident():
            loss, _ = value_loss(adapter, batch)
            loss.backward()
        self.assertAlmostEqual(float(loss), ((100.25 / 800) ** 2 + (50.75 / 800) ** 2) / 4, places=7)
        self.assertEqual(float(adapter.score.grad[2]), 0)
        adapter.zero_grad(); batch["search_score_cp"].fill_(29999)
        loss, _ = value_loss(adapter, batch); loss.backward()
        self.assertEqual(float(loss), 0)
        self.assertTrue(torch.equal(adapter.score.grad, torch.zeros(3)))

    def test_grouped_child_order_current_mover_sign_ties_and_singletons(self):
        # Root0: actual best child -100, wrong predictor picks +50 =>150 regret.
        # Root1: tied teacher best -20, any tied prediction first choice is optimal.
        # Root2 singleton is excluded from top-choice metrics.
        teacher = torch.tensor([-100., 50., -20., -20., 77.])
        prediction = torch.tensor([4., -4., 1., 1., 0.])
        roots = torch.tensor([0, 0, 1, 1, 2])
        groups, regret, correct, concordant, pairs = _ordering(prediction, teacher, roots).tolist()
        self.assertEqual((groups, regret, correct, concordant, pairs), (2, 150, 1, 0, 1))
        tied = _ordering(torch.zeros(5), teacher, roots).tolist()
        self.assertEqual(tied[-2:], [.5, 1])

    def test_metrics_bins_and_paired_family_interval_match_independent_arithmetic(self):
        teacher = torch.tensor([0, 100, 1000, 3000, 29990], dtype=torch.int32)
        cp = torch.tensor([10., 110., 900., 2900., -9999.])
        anchor = torch.tensor([0., 0., 0., 0., 100.])
        result = metrics(cp, (cp / 800).tanh(), teacher, anchor,
                         torch.tensor([0, 0, 1, 1, 2]), torch.tensor([0, 0, 1, 1, 2]))
        self.assertEqual(result["ordinary_records"], 4)
        self.assertEqual(result["mate_band_records"], 1)
        self.assertEqual(result["ordinary_centipawn_mae"], 55)
        self.assertEqual(result["material_centipawn_mae"], 1025)
        self.assertEqual([r["records"] for r in result["teacher_magnitude_bins"]], [2, 0, 1, 1])
        self.assertEqual(len(result["family_rows"]), 2)
        rows = [{"network_mae": 20, "material_mae": 30} for _ in range(5)]
        interval = paired_family_interval(rows, draws=100)
        self.assertEqual(interval["ci95"], [-10, -10])

    def test_value_only_training_updates_readout_then_encoder_and_keeps_policy_fixed(self):
        tensors, labels = ring_fixture()
        replay = SearchReplay(8, "cpu"); replay.append(tensors, labels)
        batch = gather_values(replay, torch.arange(8))
        for head in HEADS:
            adapter = ValueAdapter(comparable_model("square", width=8, depth=1, seed=123), head)
            initial = adapter.base.input.weight.detach().clone()
            policy = adapter.base.policy_source.weight.detach().clone()
            optimizer = torch.optim.AdamW(adapter.active_parameters(), lr=.003)
            for _ in range(3):
                with PayloadResident():
                    loss, norm = value_step(adapter, optimizer, batch)
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(bool(torch.isfinite(norm)))
            self.assertFalse(torch.equal(initial, adapter.base.input.weight))
            self.assertTrue(torch.equal(policy, adapter.base.policy_source.weight))


@unittest.skipUnless(os.environ.get("CHEN_VALUE_STUDY_TEST_GPU") == "1", "set CHEN_VALUE_STUDY_TEST_GPU=1")
class ValueStudyGPU(unittest.TestCase):
    def test_actual_gpu_gather_loss_backward_and_update_without_payload_host_copy(self):
        tensors, labels = ring_fixture()
        tensors = {key: value.cuda() for key, value in tensors.items()}
        labels = {key: value.cuda() for key, value in labels.items()}
        replay = SearchReplay(8, "cuda:0")
        with PayloadResident():
            replay.append(tensors, labels)
        indexes = torch.tensor([1, 3, 6, 7], device="cuda")
        for architecture in ("square", "relation", "plan"):
            for head in HEADS:
                adapter = ValueAdapter(comparable_model(architecture, width=8, depth=1, seed=2), head).cuda()
                optimizer = torch.optim.AdamW(adapter.active_parameters(), lr=.003)
                with HotResident():
                    batch = gather_values(replay, indexes)
                    loss, _ = value_loss(adapter, batch)
                    loss.backward()
                with PayloadResident():
                    loss, norm = value_step(adapter, optimizer, batch)
                self.assertEqual(batch["features"].device.type, "cuda")
                self.assertEqual(batch["features"].dtype, torch.float32)
                self.assertTrue(bool(torch.isfinite(loss)))
                self.assertTrue(bool(torch.isfinite(norm)))


if __name__ == "__main__":
    unittest.main()
