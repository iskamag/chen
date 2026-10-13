"""Replay lifecycle, legality and learning mechanics; CPU and explicit GPU modes."""
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.data import DeviceReplay, generate_synthetic, sample_legal
from neural.features import ACTION_COUNT
from neural.models import build_model
from neural.train import load_checkpoint, run_comparison, save_checkpoint, seed_all, train_step


DEVICE = os.environ.get("CHEN_TEST_DEVICE", "cpu")


class NoHostBatch(TorchDispatchMode):
    """Sampling must not request scalar extraction or a device-to-host copy."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if "_local_scalar_dense" in str(func):
            raise AssertionError("sampling extracted a scalar to the host")
        if "_to_copy" in str(func) and str(kwargs.get("device", "")) == "cpu":
            raise AssertionError("sampling requested a host copy")
        return func(*args, **kwargs)


class ReplayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.corpus = generate_synthetic(24, seed=17, max_plies=8, device=DEVICE)

    def subset(self, first, last):
        return {key: value[first:last] for key, value in self.corpus.tensors.items()}

    def test_wraparound_and_oversized_append_keep_newest(self):
        replay = DeviceReplay(7, DEVICE)
        replay.append(self.subset(0, 5))
        replay.append(self.subset(5, 10))
        self.assertEqual((replay.size, replay.cursor, replay.total_inserted), (7, 3, 10))
        self.assertEqual(sorted(replay.sequence_ids.tolist()), list(range(3, 10)))
        replay.append(self.subset(10, 24))
        self.assertEqual((replay.size, replay.cursor, replay.total_inserted), (7, 3, 24))
        self.assertEqual(sorted(replay.sequence_ids.tolist()), list(range(17, 24)))
        for row, identity in enumerate(replay.sequence_ids.tolist()):
            self.assertTrue(torch.equal(replay.storage["features"][row], self.corpus.tensors["features"][identity]))

    def test_clear_invalidation_without_realloc_or_identity_reuse(self):
        replay = DeviceReplay(7, DEVICE)
        replay.append(self.subset(0, 7))
        pointers = {key: value.data_ptr() for key, value in replay.storage.items()}
        replay.clear()
        with self.assertRaises(ValueError):
            replay.sample(1)
        replay.append(self.subset(7, 9))
        self.assertEqual(replay.total_inserted, 9)
        self.assertEqual(replay.sequence_ids[:2].tolist(), [7, 8])
        self.assertEqual(pointers, {key: value.data_ptr() for key, value in replay.storage.items()})

    def test_gather_is_owned_and_preserves_device_dtype(self):
        replay = DeviceReplay(7, DEVICE)
        replay.append(self.subset(0, 7))
        generator = torch.Generator(device=replay.device).manual_seed(5)
        with NoHostBatch():
            batch = replay.sample(4, generator)
        preserved = batch["features"].clone()
        replay.append(self.subset(7, 14))
        self.assertTrue(torch.equal(batch["features"], preserved))
        for key, (_, dtype) in replay.schema.items():
            self.assertEqual(batch[key].device, replay.device)
            self.assertEqual(batch[key].dtype, dtype)
            self.assertNotEqual(batch[key].data_ptr(), replay.storage[key].data_ptr())

    def test_ingress_drops_autograd_graph(self):
        replay = DeviceReplay(7, DEVICE)
        batch = {key: value.clone() for key, value in self.subset(0, 2).items()}
        batch["features"].requires_grad_()
        batch["next_features"].requires_grad_()
        replay.append(batch)
        self.assertFalse(replay.storage["features"].requires_grad)
        self.assertIsNone(replay.storage["features"].grad_fn)

    def test_ingress_rejects_dtype_shape_illegality_and_nonfinite_atomically(self):
        replay = DeviceReplay(7, DEVICE)
        for mutate in (
            lambda b: b.update(features=b["features"].flatten()),
            lambda b: b.update(actions=b["actions"].float()),
            lambda b: b.update(actions=b["actions"][:, None]),
            lambda b: b["legal_mask"].zero_(),
            lambda b: b["actions"].fill_(ACTION_COUNT),
            lambda b: b["features"].fill_(float("nan")),
        ):
            batch = {key: value.clone() for key, value in self.subset(0, 2).items()}
            mutate(batch)
            with self.assertRaises(ValueError):
                replay.append(batch)
            self.assertEqual((replay.size, replay.total_inserted), (0, 0))
        with self.assertRaises(ValueError):
            DeviceReplay(100, DEVICE, max_bytes=1)

    def test_sampling_reproducible_and_empty_errors(self):
        replay = DeviceReplay(7, DEVICE)
        with self.assertRaises(ValueError):
            replay.sample(2)
        replay.append(self.subset(0, 7))
        a = replay.sample(12, torch.Generator(device=replay.device).manual_seed(42))
        b = replay.sample(12, torch.Generator(device=replay.device).manual_seed(42))
        self.assertTrue(torch.equal(a["sequence_ids"], b["sequence_ids"]))
        with self.assertRaises(ValueError):
            replay.sample(0)

    def test_strict_legal_sampling_terminal_and_nan(self):
        masks = self.corpus.tensors["legal_mask"][:4]
        logits = torch.zeros(masks.shape, device=DEVICE)
        logits[~masks] = float("nan")  # ignored outside exact legal support
        generator = torch.Generator(device=logits.device).manual_seed(1)
        actions = sample_legal(logits, masks, generator)
        self.assertTrue(bool(masks.gather(1, actions[:, None]).all()))
        terminal = masks.clone()
        terminal[0].zero_()
        with self.assertRaises(ValueError):
            sample_legal(logits, terminal)
        logits[masks] = float("nan")
        with self.assertRaises(ValueError):
            sample_legal(logits, masks)

    def test_generated_transitions_legal_and_same_seed_identical(self):
        import chess
        other = generate_synthetic(24, seed=17, max_plies=8, device=DEVICE)
        self.assertEqual(self.corpus.manifest["fingerprint_sha256"], other.manifest["fingerprint_sha256"])
        self.assertEqual(self.corpus.manifest["kind"], "synthetic_legal_smoke")
        for record in self.corpus.records:
            board = chess.Board(record["fen"])
            move = chess.Move.from_uci(record["action"])
            self.assertIn(move, board.legal_moves)
            self.assertIn(chess.Move.from_uci(record["target_action"]), board.legal_moves)
            board.push(move)
            self.assertEqual(board.fen(en_passant="fen"), record["next_fen"])


class TrainingTests(unittest.TestCase):
    def test_loss_finite_update_and_reproducible_on_all_models(self):
        batch = generate_synthetic(4, seed=8, device=DEVICE).tensors
        for name in ("square", "relation", "plan"):
            seed_all(12)
            first = build_model(name, width=8, depth=1).to(DEVICE)
            seed_all(12)
            second = build_model(name, width=8, depth=1).to(DEVICE)
            before = {key: value.clone() for key, value in first.state_dict().items()}
            a = train_step(first, torch.optim.AdamW(first.parameters(), lr=1e-3), batch)
            b = train_step(second, torch.optim.AdamW(second.parameters(), lr=1e-3), batch)
            self.assertTrue(all(bool(torch.isfinite(value)) for value in a.values()))
            self.assertTrue(any(not torch.equal(before[key], value) for key, value in first.state_dict().items()))
            for key, value in first.state_dict().items():
                self.assertTrue(torch.allclose(value, second.state_dict()[key], atol=1e-6, rtol=1e-5))
            self.assertTrue(torch.allclose(a["total"], b["total"], atol=1e-6, rtol=1e-5))

    def test_comparison_identical_data_and_batch_schedule(self):
        report = run_comparison(device=DEVICE, samples=8, steps=1, batch_size=2,
                                width=8, depth=1, seed=5, no_checkpoints=True)
        self.assertFalse(report["strength_evidence"])
        rows = report["results"]
        self.assertEqual(len({row["sample_sequence_sha256"] for row in rows}), 1)
        self.assertTrue(all(row["greedy_moves_legal"] for row in rows))

    def test_checkpoint_roundtrip_all_architectures(self):
        corpus = generate_synthetic(2, seed=4, device=DEVICE)
        with tempfile.TemporaryDirectory() as directory:
            for name in ("square", "relation", "plan"):
                model = build_model(name, width=8, depth=1).to(DEVICE).eval()
                path = Path(directory) / f"{name}.pt"
                artifact = save_checkpoint(model, path, corpus_manifest=corpus.manifest,
                                           training_config={"seed": 4, "steps": 0})
                restored, metadata = load_checkpoint(path, device=DEVICE)
                self.assertEqual(metadata["model_config"]["name"], name)
                self.assertFalse(metadata["strength_evidence"])
                self.assertEqual(metadata["corpus_manifest"], corpus.manifest)
                self.assertEqual(len(artifact["sha256"]), 64)
                with torch.no_grad():
                    expected = model(corpus.tensors["features"], corpus.tensors["legal_mask"])
                    actual = restored(corpus.tensors["features"], corpus.tensors["legal_mask"])
                for key, value in expected.items():
                    self.assertTrue(torch.equal(value, actual[key]), (name, key))

    def test_checkpoint_rejects_schema_weights_and_arbitrary_pickle(self):
        model = build_model("square", width=8, depth=1)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(model, path, corpus_manifest={"kind": "synthetic_legal_smoke"},
                            training_config={"steps": 0})
            good = torch.load(path, weights_only=True)
            for mutate in (
                lambda data: data.update(schema_version=999),
                lambda data: data.update(schema_version=True),
                lambda data: data.update(checkpoint_version=True),
                lambda data: data["model_config"].update(width=100000),
                lambda data: data["state_dict"].pop("position"),
                lambda data: data["state_dict"]["position"].fill_(float("nan")),
            ):
                data = {**good, "model_config": good["model_config"].copy(),
                        "state_dict": {key: value.clone() for key, value in good["state_dict"].items()}}
                mutate(data)
                torch.save(data, path)
                with self.assertRaises(ValueError):
                    load_checkpoint(path)
            torch.save(SimpleNamespace(unexpected="object payload"), path)
            with self.assertRaises(ValueError):
                load_checkpoint(path)

    def test_checkpoint_rejects_corrupted_deterministic_buffers(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            for name in ("square", "relation", "plan"):
                save_checkpoint(build_model(name, width=8, depth=1), path,
                                corpus_manifest={"kind": "synthetic_legal_smoke"},
                                training_config={"steps": 0})
                good = torch.load(path, weights_only=True)
                for buffer_name, corrupted in (
                    ("next_orientation", good["state_dict"]["next_orientation"].clone()),
                    ("geometry", good["state_dict"]["geometry"].clone()),
                    ("attack_patterns", good["state_dict"]["attack_patterns"].clone()),
                    ("between", good["state_dict"]["between"].clone()),
                ):
                    if buffer_name == "next_orientation":
                        corrupted[0] = 64  # valid dtype/shape, later transition index failure
                    else:
                        corrupted.flatten()[0] += 1  # finite and valid-shaped, wrong definition
                    bad = {**good, "state_dict": {**good["state_dict"], buffer_name: corrupted}}
                    torch.save(bad, path)
                    with self.assertRaisesRegex(ValueError, "deterministic buffer mismatch"):
                        load_checkpoint(path, device=DEVICE)
                bad = {**good, "state_dict": {**good["state_dict"],
                                              "next_orientation": good["state_dict"]["next_orientation"].int()}}
                torch.save(bad, path)
                with self.assertRaisesRegex(ValueError, "deterministic buffer mismatch"):
                    load_checkpoint(path, device=DEVICE)


if __name__ == "__main__":
    unittest.main()
