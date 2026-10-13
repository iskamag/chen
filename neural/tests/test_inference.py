"""Same-model static deployment parity; CPU only unless explicitly requested."""
import copy
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from neural.features import encode_fen
from neural.inference import (StaticEvaluator, export_inference, load_inference,
                              normalized_scalar, scalar_to_cp, state_sha256)
from neural.models import MODELS, build_model


FENS = ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 99 11",
        "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 7 12",
        "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1")


class InferenceCPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.previous_threads)

    def test_trained_architecture_integer_export_reload_and_dynamic_batch(self):
        features = torch.stack([encode_fen(fen) for fen in FENS])
        with tempfile.TemporaryDirectory() as directory:
            for name in MODELS:
                for width, depth in ((8, 1), (16, 2)):
                    with self.subTest(name=name, width=width, depth=depth):
                        torch.manual_seed(703)
                        model = build_model(name, width=width, depth=depth)
                        initial = copy.deepcopy(model.state_dict())
                        optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
                        for _ in range(2):
                            optimizer.zero_grad()
                            loss = (normalized_scalar(model, features) - torch.tensor([0.1, -0.2, 0.3, -0.4])).square().mean()
                            loss.backward(); optimizer.step()
                        self.assertTrue(any(not torch.equal(initial[key], value)
                                            for key, value in model.state_dict().items()))
                        before = state_sha256(model)
                        archive = Path(directory) / f"{name}-{width}-{depth}.pt"
                        metadata = export_inference(model, archive,
                            target_kind="untrained_architecture_fixture",
                            provenance={"kind": "trained_tensor_fixture", "teacher_strength_claim": False},
                            max_batch=17)
                        self.assertEqual(state_sha256(model), before)
                        self.assertTrue(model.training)
                        session = load_inference(archive, expected_sha256=metadata["archive_sha256"])
                        with torch.inference_mode():
                            for batch in (1, 3, 17):
                                sample = features[torch.arange(batch) % len(features)]
                                expected = StaticEvaluator(model.eval())(sample)
                                self.assertTrue(torch.equal(session(sample), expected))
                                self.assertEqual(session(sample).dtype, torch.int32)
                        with self.assertRaises(ValueError):
                            session(features.double())
                        with self.assertRaises(ValueError):
                            session(features[torch.arange(18) % len(features)])

    def test_conversion_is_symmetric_finite_static_cp_not_mate(self):
        utilities = torch.tensor([-1., -0.8, 0., 0.8, 1.])
        scores = scalar_to_cp(utilities)
        self.assertTrue(torch.equal(scores, -scores.flip(0)))
        self.assertEqual(scores[2], 0)
        self.assertTrue(bool((scores.abs() < 29000).all()))
        self.assertEqual(scores.dtype, torch.int32)
        self.assertTrue(torch.equal(scalar_to_cp(torch.tanh(torch.tensor([-800., 0., 800.]) / 800)),
                                    torch.tensor([-800, 0, 800], dtype=torch.int32)))

    def test_explicit_teacher_checkpoint_cli_preserves_contract_and_weights(self):
        from neural.teacher import save_teacher_checkpoint
        from neural.inference import main
        model = build_model("plan", width=8, depth=1)
        teacher = {"teacher_kind": "finite_depth_classical_minimax_search_targets",
                   "tree_depth": 3, "parent_horizon": 2, "evaluator": 1,
                   "scale_cp": 800.0, "fixture_only": True,
                   "lineage_size_fixture": "x" * 70000}
        with tempfile.TemporaryDirectory() as directory:
            checkpoint, archive = Path(directory) / "teacher.pt", Path(directory) / "deploy.pt"
            save_teacher_checkpoint(model, checkpoint, teacher_manifest=teacher,
                                    training_config={"fixture_only": True})
            with patch("sys.argv", ["inference", str(checkpoint), str(archive)]), patch("sys.stdout", new_callable=io.StringIO):
                main()
            sidecar = json.loads(archive.with_suffix(".pt.json").read_text())
            manifest = sidecar["manifest"]
            self.assertEqual(manifest["target_kind"], "finite_depth_minimax_cp_utility")
            self.assertEqual(manifest["checkpoint_sha256"], hashlib.sha256(checkpoint.read_bytes()).hexdigest())
            self.assertEqual(manifest["state_sha256"], state_sha256(model))
            self.assertTrue(manifest["provenance"]["teacher_manifest"]["fixture_only"])
            self.assertLess(archive.with_suffix(".pt.json").stat().st_size, 65536)
            self.assertEqual(manifest["provenance"]["teacher_manifest_sha256"],
                             hashlib.sha256(json.dumps(teacher, sort_keys=True, separators=(",", ":")).encode()).hexdigest())
            session = load_inference(archive, expected_sha256=sidecar["archive_sha256"])
            features = torch.stack([encode_fen(fen) for fen in FENS])
            with torch.inference_mode():
                self.assertTrue(torch.equal(session(features), StaticEvaluator(model.eval())(features)))

    def test_integrity_schema_and_size_guards(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "fixture.pt"
            meta = export_inference(build_model("square", width=8, depth=1), path,
                target_kind="untrained_architecture_fixture", provenance={"fixture": True})
            with self.assertRaises(ValueError):
                load_inference(path, expected_sha256="0" * 64)
            sidecar = path.with_suffix(".pt.json")
            for field, value in (("scale_cp", 400.0), ("deployment_version", True),
                                 ("model_config", {"name": "square", "width": True,
                                                   "depth": 1, "plan_tokens": 8}),
                                 ("provenance", {"changed": True})):
                changed = {**meta["manifest"], field: value}
                sidecar.write_text(json.dumps({**meta, "manifest": changed,
                                              "manifest_json": json.dumps(changed, sort_keys=True, separators=(",", ":"))}))
                with self.assertRaises(ValueError):
                    load_inference(path, expected_sha256=meta["archive_sha256"])
            for target, bound in (("outcome_wdl", 16), ("untrained_architecture_fixture", True)):
                with self.assertRaises(ValueError):
                    export_inference(build_model("square", width=8, depth=1), path,
                        target_kind=target, provenance={}, max_batch=bound)


@unittest.skipUnless(os.environ.get("CHEN_TEST_DEVICE") == "cuda",
                     "GPU kernels require an explicitly scheduled test slot")
class InferenceGPU(unittest.TestCase):
    evidence = []

    def test_all_architectures_remap_preserves_gpu_integer_scores(self):
        type(self).evidence = []
        previous_threads = torch.get_num_threads()
        torch.set_num_threads(2)
        try:
            cpu_features = torch.stack([encode_fen(fen) for fen in FENS])
            features = cpu_features.to("cuda")
            with tempfile.TemporaryDirectory() as directory:
                for name in MODELS:
                    for width, depth in ((8, 1), (16, 2)):
                        with self.subTest(name=name, width=width, depth=depth):
                            torch.manual_seed(703)
                            model = build_model(name, width=width, depth=depth)
                            optimizer = torch.optim.SGD(model.parameters(), lr=0.04)
                            optimizer.zero_grad()
                            (normalized_scalar(model, cpu_features) - 0.1).square().mean().backward()
                            optimizer.step()
                            archive = Path(directory) / f"{name}-{width}-{depth}.pt"
                            sidecar = export_inference(model, archive, target_kind="untrained_architecture_fixture",
                                provenance={"trained_tensor_fixture": True}, max_batch=17)
                            session = load_inference(archive, expected_sha256=sidecar["archive_sha256"], device="cuda")
                            reference = StaticEvaluator(copy.deepcopy(model).to("cuda").eval())
                            with torch.inference_mode():
                                for batch in (1, 3, 17):
                                    sample = features[torch.arange(batch, device="cuda") % len(features)]
                                    expected = reference(sample)
                                    equal = torch.equal(session(sample), expected)
                                    self.assertTrue(equal)
                                    cpu_expected = StaticEvaluator(model.eval())(cpu_features[torch.arange(batch) % len(features)])
                                    difference = (expected.cpu().to(torch.int64) - cpu_expected).abs()
                                    self.assertTrue(bool(difference.le(1).all()))
                                    type(self).evidence.append({"model": name, "width": width, "depth": depth,
                                        "batch": batch, "gpu_integer_scores_equal": equal,
                                        "cpu_gpu_max_difference_cp": int(difference.max())})
            torch.cuda.synchronize()
        finally:
            torch.cuda.synchronize()
            torch.set_num_threads(previous_threads)


if __name__ == "__main__":
    unittest.main()
