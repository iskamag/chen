"""CPU-only mechanical fixtures: no real teacher corpus, training or GPU calls."""
from __future__ import annotations

import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

from . import packing, resident


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fixture(split="train"):
    """Full legal root groups, terminal children, black orientation, raw EP.

    Scores are invented CP protocol witnesses; they carry no quality claim.
    The production reader still reconstructs states/history/support/frame joins.
    """
    import chess
    roots = [(chess.STARTING_FEN, ["e2e4"]),
             ("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR b KQkq - 17 4294967295", []),
             ("7k/8/5KQ1/8/8/8/8/8 w - - 99 4294967295", [])]
    families, queries = [], []
    for i, (origin, moves) in enumerate(roots):
        board, history = chess.Board(origin), []
        for uci in moves:
            history.append(packing.repetition_hash(board))
            packing._push(board, chess.Move.from_uci(uci))
        game = hashlib.sha256(f"synthetic-family-{i}".encode()).hexdigest()
        row = dict(game_id=game, ply=len(moves), fen=board.fen(en_passant="fen"), moves=moves, previous_hashes=history)
        family = dict(game_id=game, game_sha256=game, moves_sha256=hashlib.sha256(" ".join(moves).encode()).hexdigest(),
                      source_offset=i * 100, source_end_offset=i * 100 + 50, ratings=[2000, 2000],
                      time_control="synthetic", positions=[row], action_root_index=0, all_moves=moves)
        families.append(dict(source=family, origin_fen=origin))
        legal = sorted(board.legal_moves, key=lambda m: packing.canonical_action(m, board.turn))
        material = packing._material(board)
        for all_legal in (False, True):
            query = dict(split=split, position=row, all_legal=all_legal, nodes=2000)
            query["id"] = hashlib.sha256(json.dumps(query, sort_keys=True).encode()).hexdigest()
            rows = []
            for rank, move in enumerate(legal if all_legal else legal[:1], 1):
                cp = 70 - rank
                rows.append(dict(uci=move.uci(), action=packing.canonical_action(move, board.turn), reported_cp=cp,
                                 rank=rank, depth=6, seldepth=7, nodes=2000, pv=[move.uci()],
                                 converted_score=packing.source_score_record(cp, material, negate=all_legal)))
            queries.append(dict(query=query, source_record_sha256=game, transcript_sha256=game,
                                packet_provenance=dict(protocol_version=2, query_rejection=None,
                                                       selected_emission_ordinal=1, selected_common_depth=6),
                                target_kind="stockfish_parent_root_normalized_action_cp" if all_legal else "stockfish_standalone_root_normalized_cp",
                                query_material=material, rows=rows))
    return dict(version=1, split=split, source_kind="fixture_protocol_only",
                pins={k: "a" * 64 for k in packing.SOURCE_KEYS}, families=families, queries=queries)


def frozen_compact():
    root = Path(__file__).resolve().parents[2] / "docs/research/evidence/semantic-packet-v2-preparation-2026-10-01/neural"
    if digest(root / "compact.py") != "a52ba420dca331edf999e539cc37ea5b9208630ed0f9d9587fe24501b2c944d3":
        raise RuntimeError("frozen compact oracle changed")
    name = "_resident_frozen_neural"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(root)]
        sys.modules[name] = module
    before = sys.dont_write_bytecode
    try:
        sys.dont_write_bytecode = True
        return importlib.import_module(name + ".compact")
    finally:
        sys.dont_write_bytecode = before


class ResidentTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import torch
        cls.torch = torch
        cls.tmp = tempfile.TemporaryDirectory(prefix="chen-resident-test-")
        cls.base = Path(cls.tmp.name)
        cls.packs = {}
        for split in ("train", "validation", "test"):
            folder = cls.base / split
            packing._publish(fixture(split), folder)
            cls.packs[split] = folder

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def load(self, split="train", **kwargs):
        path = self.packs[split]
        return resident.load_resident(path, expected_manifest_sha256=digest(path / "manifest.json"),
                                      expected_source_sha256=packing.source_sha256(),
                                      stage="train" if split == "train" else split + "_eval", **kwargs)

    def gate(self, split):
        checkpoint = self.base / (split + "-checkpoint.bin")
        checkpoint.write_bytes(b"synthetic checkpoint bytes only")
        cp = digest(checkpoint)
        record = dict(kind="semantic_frozen_checkpoint_v1" if split == "validation" else "semantic_selected_checkpoint_v1",
                      corpus_manifest_sha256=digest(self.packs[split] / "manifest.json"),
                      packing_source_sha256=packing.source_sha256(), checkpoint_sha256=cp)
        extras = {}
        if split == "test":
            source = self.base / "selection-source.json"
            source.write_text(json.dumps(dict(kind="semantic_selection_source_v1", selected_checkpoint_sha256=cp,
                                             selection_splits=["train", "validation"], test_policy="sealed_out")))
            record["selection_source_sha256"] = digest(source)
            extras = dict(selection_source_path=source, selection_source_sha256=digest(source))
        path = self.base / (split + "-gate.json")
        path.write_text(json.dumps(record))
        return resident.EvaluationGate(path, digest(path), checkpoint, cp, **extras)

    def test_module_import_is_torch_free(self):
        code = "import sys; from experiments.semantic_corpus import resident; assert 'torch' not in sys.modules"
        result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2],
                                capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_frozen_import_never_writes_bytecode_and_restores_flag(self):
        before = sys.dont_write_bytecode
        seen = []
        def guarded(_name):
            seen.append(sys.dont_write_bytecode)
            raise RuntimeError("synthetic failed dependency import")
        try:
            sys.dont_write_bytecode = False
            with patch.object(importlib, "import_module", side_effect=guarded):
                with self.assertRaises(RuntimeError):
                    frozen_compact()
            self.assertEqual(seen, [True])
            self.assertFalse(sys.dont_write_bytecode)
        finally:
            sys.dont_write_bytecode = before

    def test_integrated_load_and_owned_aligned_storage(self):
        corpus = self.load()
        self.assertEqual(str(corpus.device), "cpu")
        self.assertEqual(corpus._storage.data_ptr() % 64, 0)
        self.assertGreater(corpus.payload_bytes, 0)
        self.assertFalse(hasattr(corpus, "views"))
        self.assertIsNone(corpus._storage._base)

    def test_complete_sorted_int64_support_and_terminal_masks(self):
        corpus = self.load()
        pack = packing.read_pack(self.packs["train"])
        for root in range(3):
            batch = corpus.roots(corpus.prepare_roots([root]))
            k = pack.values("roots.legal_count")[root]
            self.assertEqual(batch.legal_actions.shape, (1, k))
            self.assertEqual(batch.legal_actions.dtype, self.torch.int64)
            actions = batch.legal_actions[0].tolist()
            self.assertEqual(actions, sorted(set(actions)))
            self.assertEqual(batch.targets.features.shape, (1, k, 18, 8, 8))
            self.assertEqual(batch.targets.ordinary.dtype, self.torch.bool)
            if root == 2:
                self.assertTrue(batch.targets.terminal.all().item())
                self.assertFalse(batch.targets.ordinary.any().item())
                self.assertGreater(k, 1)  # Terminal children remain in support.

    def test_byte_exact_frozen_schema_decode(self):
        corpus, compact = self.load(), frozen_compact()
        for root in range(3):
            batch = corpus.roots(corpus.prepare_roots([root]))
            self.assertTrue(self.torch.equal(batch.features, compact.decode_features(batch.states.boards, batch.states.fields)))
            child = compact.decode_features(batch.targets.states.boards, batch.targets.states.fields)
            self.assertTrue(self.torch.equal(batch.targets.features.flatten(0, 1), child))
        # Raw EP after e4, black orientation, and upper-u32 fullmove survive.
        ep = corpus.roots(corpus.prepare_roots([0])).states.fields[0, 2].item()
        self.assertEqual(ep, 20)
        self.assertEqual(corpus.roots(corpus.prepare_roots([1])).states.fields[0, 4].item(), -1)

    def test_exact_history_and_cpu_origin_sidecars(self):
        corpus = self.load()
        batch = corpus.roots(corpus.prepare_roots([0]))
        self.assertEqual(batch.states.history_offsets.tolist(), [0, 1])
        self.assertEqual(batch.targets.states.history_offsets.tolist(), list(range(0, 2 * batch.legal_actions.shape[1] + 1, 2)))
        self.assertTrue((batch.states.origin_context == 0).all().item())
        self.assertTrue((batch.targets.states.origin_epoch == -1).all().item())
        self.assertTrue((batch.targets.states.origin_node == -1).all().item())

    def test_gathers_cannot_mutate_corpus_or_schedule(self):
        corpus = self.load()
        plan = corpus.prepare_roots([0, 1])
        original = corpus.roots(plan)
        altered = corpus.roots(plan)
        altered.features.zero_()
        altered.legal_actions.fill_(-1)
        altered.targets.features.zero_()
        altered.targets.states.boards.zero_()
        altered.states.history_offsets.fill_(99)
        current = corpus.roots(plan)
        for a, b in ((current.features, original.features), (current.legal_actions, original.legal_actions),
                     (current.targets.features, original.targets.features),
                     (current.targets.states.boards, original.targets.states.boards),
                     (current.states.history_offsets, original.states.history_offsets)):
            self.assertTrue(self.torch.equal(a, b))

    def test_exact_k_rejects_ragged_empty_duplicates_and_foreign_plan(self):
        corpus = self.load()
        for roots in ([0, 2], [], [0, 0], [-1], [3], [True]):
            with self.assertRaises(ValueError):
                corpus.prepare_roots(roots)
        foreign = self.load().prepare_roots([0])
        with self.assertRaises(ValueError):
            corpus.roots(foreign)
        owned = corpus.prepare_roots([0])
        with self.assertRaises(ValueError):
            corpus.roots(resident.RootPlan(owned.key, owned.root_ids, owned.legal_count))

    def test_reader_structural_schedule_repeats_without_ingress(self):
        corpus = self.load()
        roots, observations = corpus.prepare_structural_schedule()
        before = corpus._schedule_bytes
        self.assertIs(corpus.prepare_structural_schedule(), corpus._prepared_schedule)
        self.assertEqual(corpus._schedule_bytes, before)
        self.assertEqual(sorted(r for p in roots for r in p.root_ids), [0, 1, 2])
        self.assertEqual(sorted(r for p in observations for r in p.observation_ids), [0, 1, 2])
        for plan in roots:
            self.assertEqual(corpus.roots(plan).legal_actions.shape[1], plan.legal_count)

    def test_schedule_extension_offsets_hashes_and_shared_owner(self):
        corpus = self.load()
        ingress = corpus.ingress
        self.assertEqual(ingress.packed_bytes, corpus._storage.numel())
        self.assertGreater(ingress.auxiliary_bytes, 0)
        # CPU inspection is solely a test oracle, not the training batch path.
        raw = bytes(corpus._storage.tolist())
        self.assertEqual(hashlib.sha256(raw).hexdigest(), ingress.packed_sha256)
        self.assertEqual(hashlib.sha256(raw[ingress.payload_bytes:]).hexdigest(), ingress.auxiliary_sha256)
        for segment in ingress.segments:
            self.assertEqual(segment.offset % 64, 0)
            data = raw[segment.offset:segment.offset + segment.count * 8]
            self.assertEqual(hashlib.sha256(data).hexdigest(), segment.sha256)
        for plan, tensors in corpus._plans.values():
            for tensor in tensors:
                self.assertEqual(tensor.untyped_storage().data_ptr(), corpus._storage.untyped_storage().data_ptr())

    def test_prepared_batches_need_no_new_index_tensor_construction(self):
        corpus = self.load()
        roots, observations = corpus.prepare_structural_schedule()
        with patch.object(self.torch, "tensor", side_effect=AssertionError("new CPU-list index ingress")):
            for _ in range(2):
                for plan in roots:
                    corpus.roots(plan)
                for plan in observations:
                    corpus.observations(plan)

    def test_permutation_determinism_and_complete_tails(self):
        a = self.load().root_schedule(batch_size=1, seed=9)
        b = self.load().root_schedule(batch_size=1, seed=9)
        self.assertEqual([p.root_ids for p in a], [p.root_ids for p in b])
        self.assertEqual(sorted(r for p in a for r in p.root_ids), [0, 1, 2])

    def test_observation_batch_frame_and_ownership(self):
        corpus = self.load()
        plan = corpus.prepare_observations([0, 1, 2])
        batch = corpus.observations(plan)
        self.assertEqual(batch.features.shape, (3, 18, 8, 8))
        self.assertTrue(batch.ordinary.all().item())
        self.assertTrue((batch.midpoint > 0).all().item())
        roots = corpus.roots(corpus.prepare_roots([0, 1]))
        self.assertTrue((roots.targets.midpoint < 0).all().item())
        original = batch.midpoint.clone()
        batch.midpoint.zero_()
        self.assertTrue(self.torch.equal(corpus.observations(plan).midpoint, original))

    def test_caps_and_bad_pins_reject(self):
        for kwargs in (dict(max_bytes=1), dict(max_metadata_bytes=1), dict(max_bytes=True), dict(max_batch_bytes=0)):
            with self.assertRaises(ValueError):
                self.load(**kwargs)
        with self.assertRaises(ValueError):
            self.load(max_batch_bytes=1).prepare_roots([0])
        with self.assertRaises(ValueError):
            self.load(max_schedule_bytes=1).prepare_observations([0])
        with self.assertRaises(ValueError):
            resident.load_resident(self.packs["train"], expected_manifest_sha256="0" * 64,
                                   expected_source_sha256=packing.source_sha256(), stage="train")

    def test_held_out_gate_precedes_reader_access(self):
        with patch.object(packing, "read_pack") as reader:
            for split in ("validation", "test"):
                for gate in (None, True):
                    with self.assertRaises(ValueError):
                        self.load(split, gate=gate)
            reader.assert_not_called()

    def test_train_cannot_admit_heldout(self):
        for split in ("validation", "test"):
            with self.assertRaises(ValueError):
                resident.load_resident(self.packs[split], expected_manifest_sha256=digest(self.packs[split] / "manifest.json"),
                                       expected_source_sha256=packing.source_sha256(), stage="train")

    def test_validation_and_test_require_exact_distinct_seals(self):
        for split in ("validation", "test"):
            gate = self.gate(split)
            self.assertEqual(self.load(split, gate=gate).stage, split + "_eval")
            Path(gate.checkpoint_path).write_bytes(b"changed")
            with self.assertRaises(ValueError):
                self.load(split, gate=gate)
        gate = self.gate("test")
        source = Path(gate.selection_source_path)
        value = json.loads(source.read_text())
        value["selection_splits"].append("test")
        source.write_text(json.dumps(value))
        seal = Path(gate.path)
        record = json.loads(seal.read_text())
        record["selection_source_sha256"] = digest(source)
        seal.write_text(json.dumps(record))
        gate = resident.EvaluationGate(seal, digest(seal), gate.checkpoint_path, gate.checkpoint_sha256, source, digest(source))
        with self.assertRaises(ValueError):
            self.load("test", gate=gate)

    def test_gate_special_files_and_symlinks_reject_before_open(self):
        fifo = self.base / "gate-fifo"
        os.mkfifo(fifo)
        target = self.base / "regular-gate-target"
        target.write_bytes(b"bytes")
        link = self.base / "gate-link"
        link.symlink_to(target)
        for path in (fifo, Path("/dev/null"), link, self.base):
            with patch.object(resident.os, "open", side_effect=AssertionError("special path opened")):
                with self.assertRaises(ValueError):
                    resident._read_pinned(path, "0" * 64)
                with self.assertRaises(ValueError):
                    resident._hash_regular(path, 128 << 20)

    def test_checkpoint_cap_rejects_before_open(self):
        checkpoint = self.base / "oversize-checkpoint"
        with checkpoint.open("wb") as handle:
            handle.truncate((128 << 20) + 1)
        with patch.object(resident.os, "open", side_effect=AssertionError("oversize file opened")):
            with self.assertRaises(ValueError):
                resident._hash_regular(checkpoint, 128 << 20)

    def test_growing_checkpoint_stream_is_bounded(self):
        import io
        with patch.object(resident, "_open_regular", return_value=io.BytesIO(b"more-than-cap")):
            with self.assertRaises(ValueError):
                resident._hash_regular("mock-regular-file", 3)

    def test_gpu_gate_denied_before_any_cuda_call(self):
        with patch.object(self.torch.cuda, "is_available", side_effect=AssertionError("GPU API reached")):
            for device in ("cuda", "cuda:0", "mps"):
                with self.assertRaises(ValueError):
                    self.load(device=device)

    def test_malformed_layout_rejected_before_torch_allocation(self):
        pack = packing.read_pack(self.packs["train"])
        manifest = json.loads((self.packs["train"] / "manifest.json").read_text())
        for mutate in (
            lambda m: m["tables"]["states.boards"].update(offset=1),
            lambda m: m["tables"]["states.boards"].update(dtype="float32"),
            lambda m: m["tables"]["states.boards"].update(shape=[999, 12]),
            lambda m: m["tables"]["states.fields"].update(offset=m["tables"]["states.boards"]["offset"]),
            lambda m: m.update(payload_sha256="0" * 64),
        ):
            bad = copy.deepcopy(manifest)
            mutate(bad)
            fake = types.SimpleNamespace(payload=pack.payload, manifest=bad, segments=bad["tables"], schedule=pack.schedule)
            with patch.object(packing, "read_pack", return_value=fake), patch.object(self.torch, "empty", side_effect=AssertionError("allocation before rejection")):
                with self.assertRaises(ValueError):
                    self.load()

    def test_semantically_corrupted_history_rejected_before_allocations(self):
        directory = self.base / "corrupt-history"
        import shutil
        shutil.copytree(self.packs["train"], directory)
        meta = json.loads((directory / "provenance.json").read_text())
        meta["families"][0]["source"]["positions"][0]["previous_hashes"][0] ^= 1
        raw = (json.dumps(meta, sort_keys=True, indent=2) + "\n").encode()
        (directory / "provenance.json").write_bytes(raw)
        m = json.loads((directory / "manifest.json").read_text())
        m["provenance_sha256"], m["provenance_bytes"] = hashlib.sha256(raw).hexdigest(), len(raw)
        (directory / "manifest.json").write_text(json.dumps(m))
        with patch.object(self.torch, "empty", side_effect=AssertionError("semantic rejection after allocation")):
            with self.assertRaises(ValueError):
                resident.load_resident(directory, expected_manifest_sha256=digest(directory / "manifest.json"),
                                       expected_source_sha256=packing.source_sha256(), stage="train")

    def test_rehashed_terminal_corruption_fails_semantic_reader_before_torch(self):
        directory = self.base / "corrupt-terminal"
        import shutil
        shutil.copytree(self.packs["train"], directory)
        manifest = json.loads((directory / "manifest.json").read_text())
        payload = bytearray((directory / "payload.bin").read_bytes())
        payload[manifest["tables"]["states.terminal"]["offset"]] ^= 1
        (directory / "payload.bin").write_bytes(payload)
        manifest["payload_sha256"] = hashlib.sha256(payload).hexdigest()
        (directory / "manifest.json").write_text(json.dumps(manifest))
        with patch.object(self.torch, "empty", side_effect=AssertionError("semantic rejection after allocation")):
            with self.assertRaises(ValueError):
                resident.load_resident(directory, expected_manifest_sha256=digest(directory / "manifest.json"),
                                       expected_source_sha256=packing.source_sha256(), stage="train")


if __name__ == "__main__":
    unittest.main()
