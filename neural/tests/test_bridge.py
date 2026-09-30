"""Exact producer/consumer mechanics; independent oracle downloads are explicit."""
import ctypes as c
import gc
import os
import unittest
from unittest.mock import patch

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.bridge import (BridgeError, ReplayExport, ReplaySink, RuleArena, _bind,
                           RulePosition, RuleStats, ResidentSpan, SINK_FIELDS, hash_key,
                           RUNTIME_APIS, position_from_fen, sink_schema, validate_output_tensors)
from neural.compact import CompactReplay, decode_batch, pack_fens
from neural.features import encode_fen, index_to_move, move_to_index
from neural.models import MODELS, build_model
from neural.train import loss_terms, train_step


class PayloadResident(TorchDispatchMode):
    """Permit ingress scalar control, reject any tensor payload CPU conversion."""
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if "_to_copy" in str(func) and str(kwargs.get("device", "")) == "cpu":
            raise AssertionError("state-payload device-to-host transfer")
        return func(*args, **kwargs)


class HotResident(PayloadResident):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        if "_local_scalar_dense" in str(func):
            raise AssertionError("hot-path scalar host read")
        return super().__torch_dispatch__(func, types, args, kwargs)


class BridgeCPU(unittest.TestCase):
    def test_abi_layout_and_exact_root_unsigned_fields(self):
        self.assertEqual([c.sizeof(kind) for kind in (RulePosition, RuleStats, ReplaySink, ReplayExport)],
                         [128, 48, 136, 24])
        self.assertEqual(ReplaySink.pieces.offset, 24)
        self.assertEqual(ReplaySink.root_epoch.offset, 128)
        p = position_from_fen("r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 65535 4294967295")
        self.assertEqual(p.boards[9] >> 63, 1)
        self.assertEqual((p.side, p.castling, p.ep_square, p.halfmove, p.fullmove, p.reserved),
                         (1, 15, -1, 65535, 0xFFFFFFFF, 0))
        self.assertGreaterEqual(p.hash, 0)

    def test_root_raw_ep_and_effective_ep_hash_differ(self):
        fen = "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        ep, none = position_from_fen(fen), position_from_fen(fen.replace("d6", "-"))
        self.assertEqual(ep.ep_square, 43)
        self.assertEqual(ep.hash ^ none.hash, hash_key(784 + 3))
        pinned = "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1"
        a, b = position_from_fen(pinned), position_from_fen(pinned.replace("d6", "-"))
        self.assertEqual(a.ep_square, 43)
        self.assertEqual(a.hash, b.hash)
        with self.assertRaises(ValueError):
            position_from_fen(fen.replace("0 1", "65536 1"))

    def test_pointer_metadata_validator_rejects_shape_stride_type_alias(self):
        schema = sink_schema(256)
        outputs = {key: torch.empty((2, *shape), dtype=dtype) for key, (shape, dtype) in schema.items()}
        self.assertIs(validate_output_tensors(outputs, schema, 2, "cpu"), outputs)
        for key, replacement in (
            ("pieces", torch.empty(12, 2, dtype=torch.int64).t()),
            ("state", torch.empty(2, 5, dtype=torch.int64)),
            ("pieces", torch.empty(3, 12, dtype=torch.int64)[1:]),
            ("pieces", torch.empty(3, 12, dtype=torch.int64)[:2]),
            ("next_pieces", outputs["pieces"]),
            ("material_proxy", torch.empty(2, requires_grad=True)),
        ):
            bad = {**outputs, key: replacement}
            with self.assertRaises(ValueError, msg=key):
                validate_output_tensors(bad, schema, 2, "cpu")
        with self.assertRaises(ValueError):
            validate_output_tensors(outputs, schema, True, "cpu")
        with self.assertRaises(ValueError):
            RuleArena(capacity=4, device="cpu")
        for slots in (0, 257, True):
            with self.assertRaises(ValueError):
                sink_schema(slots)

    def test_copy_audit_rejects_deliberate_scalar_violation(self):
        with self.assertRaises(AssertionError):
            with HotResident():
                torch.tensor(1).item()

    def test_epoch_bound_span_subsets_preserve_identity(self):
        source = ResidentSpan("context-a", 12, 21, 400)
        self.assertEqual(source.subset(30, 64), ResidentSpan("context-a", 12, 51, 64))
        for offset, count in [(-1, 1), (0, 0), (399, 2), (True, 1)]:
            with self.assertRaises(ValueError):
                source.subset(offset, count)
        with self.assertRaises(ValueError):
            ResidentSpan("context-a", 0, 0, 1)

    def test_unsupported_library_abis_fail_without_fallback(self):
        class Function:
            def __init__(self, result=0):
                self.result = result
            def __call__(self, *args):
                return self.result
        class Library:
            pass
        for rules, replay in ((2, 1), (1, 2)):
            lib = Library()
            for name in ("create", "destroy", "reset", "expand", "export_transitions"):
                setattr(lib, "chen_rules_" + name, Function())
            lib.chen_replay_epoch = Function()
            lib.chen_rules_abi_version = Function(rules)
            lib.chen_replay_abi_version = Function(replay)
            with patch("neural.bridge.c.CDLL", return_value=lib):
                with self.assertRaises(ValueError):
                    _bind("unsupported.so")

    def test_runtime_api_mismatch_fails_before_context_create(self):
        class Function:
            def __init__(self, result=0):
                self.result, self.calls = result, 0
            def __call__(self, *args):
                self.calls += 1
                return self.result(*args) if callable(self.result) else self.result
        class Object:
            pass
        runtime = Object()
        for name in RUNTIME_APIS:
            setattr(runtime, name, c.CFUNCTYPE(c.c_int32)(lambda: 0))
        expected = [c.cast(getattr(runtime, name), c.c_void_p).value for name in RUNTIME_APIS]
        for mismatch in range(len(RUNTIME_APIS)):
            lib = Object()
            for name in ("create", "destroy", "reset", "expand", "export_transitions"):
                setattr(lib, "chen_rules_" + name, Function())
            lib.chen_replay_epoch = Function()
            lib.chen_rules_abi_version = Function(1)
            lib.chen_replay_abi_version = Function(1)
            lib.chen_replay_runtime_identity = Function(
                lambda selector: expected[selector] + int(selector == mismatch))
            with patch("neural.bridge._device", return_value=torch.device("cuda", 0)), \
                 patch("neural.bridge._torch_runtime", return_value=runtime), \
                 patch("neural.bridge.c.CDLL", return_value=lib):
                with self.assertRaisesRegex(RuntimeError, RUNTIME_APIS[mismatch]):
                    RuleArena("mismatch.so", capacity=8)
            self.assertEqual(lib.chen_rules_create.calls, 0)


GPU_ENABLED = os.environ.get("CHEN_BRIDGE_TEST_GPU") == "1"


@unittest.skipUnless(GPU_ENABLED, "set CHEN_BRIDGE_TEST_GPU=1 with CHEN_GPU_LIBRARY for actual HIP tests")
class BridgeGPU(unittest.TestCase):
    def arena(self, capacity=4096):
        return RuleArena(capacity=capacity, device="cuda")

    def verify_export(self, exported, parent_fens):
        # Explicit oracle region: CPU downloads here are outside the data path.
        data = {key: value.cpu() for key, value in exported.tensors.items()}
        side = {key: value.cpu() for key, value in exported.sidecar.items()}
        for i in range(data["pieces"].shape[0]):
            parent_id = int(side["parent_node"][i])
            fen = parent_fens[parent_id]
            board = chess.Board(fen)
            action = int(data["actions"][i])
            move = chess.Move.from_uci(index_to_move(action, "w" if board.turn == chess.WHITE else "b"))
            self.assertIn(move, board.legal_moves)
            self.assertEqual(action, int(data["target_actions"][i]))
            count = int(data["legal_counts"][i])
            expected = sorted(move_to_index(move.uci(), "w" if board.turn == chess.WHITE else "b") for move in board.legal_moves)
            self.assertEqual(data["legal_actions"][i, :count].tolist(), expected)
            self.assertTrue(bool((data["legal_actions"][i, count:] == -1).all()))
            before = position_from_fen(fen)
            successor = board.copy(); successor.push(move)
            successor.halfmove_clock = min(65535, successor.halfmove_clock)
            successor.fullmove_number = min(0xFFFFFFFF, successor.fullmove_number)
            next_fen = successor.fen(en_passant="fen")
            after = position_from_fen(next_fen)
            pieces, state = pack_fens([fen, next_fen])
            self.assertTrue(torch.equal(data["pieces"][i], pieces[0]))
            self.assertTrue(torch.equal(data["state"][i], state[0]))
            self.assertTrue(torch.equal(data["next_pieces"][i], pieces[1]))
            self.assertTrue(torch.equal(data["next_state"][i], state[1]))
            self.assertEqual(int(side["parent_hash"][i]) & ((1 << 64) - 1), before.hash)
            self.assertEqual(int(side["child_hash"][i]) & ((1 << 64) - 1), after.hash)
            legal = list(successor.legal_moves)
            # Current-position draw policy; no prospective draw-claim lookahead.
            terminal = not legal or successor.halfmove_clock >= 100 or successor.is_insufficient_material() or successor.is_repetition(3)
            self.assertEqual(bool(data["terminal"][i]), terminal)
            expected_material = sum((1 if color == board.turn else -1) *
                                    len(board.pieces(piece, color)) * weight
                                    for color in chess.COLORS
                                    for piece, weight in zip(range(1, 7), [100, 320, 330, 500, 900, 0]))
            self.assertAlmostEqual(float(data["material_proxy"][i]),
                                   float(torch.tanh(torch.tensor(max(-29000, min(29000, expected_material)) / 800))), places=6)

    def test_sparse_fixtures_exact_transition_parity(self):
        fixtures = [chess.STARTING_FEN,
                    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 99 1",
                    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 4294967295",
                    "4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12",
                    "k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1",
                    "8/6bb/8/8/R1pP2k1/4P3/P7/K7 b - d3 0 1",
                    "4k3/P7/8/8/8/8/7p/4K3 w - - 0 1",
                    "4k3/7P/8/8/8/8/p7/4K3 b - - 0 1",
                    "4k3/8/8/8/8/2q5/2R5/4K3 w - - 0 1",
                    "7k/5Q2/6K1/8/8/8/8/8 w - - 0 1"]
        with self.arena() as arena:
            for fen in fixtures:
                arena.reset(position_from_fen(fen))
                stats = arena.expand()
                self.assertGreater(stats["next_count"], 0)
                with PayloadResident():
                    exported = arena.export(stats["span"])
                self.verify_export(exported, {0: fen})

    def test_direct_ingress_sample_loss_gradient_parity_and_lifecycle(self):
        with self.arena() as arena:
            arena.reset(position_from_fen(chess.STARTING_FEN))
            stats = arena.expand()
            with PayloadResident():
                exported = arena.export(stats["span"], game_id=51)
                replay = CompactReplay(8, "cuda")
                exported.append_to(replay)
            with HotResident():
                sample = replay.sample(4, torch.Generator(device="cuda").manual_seed(2026))
            self.assertEqual(replay.total_inserted, 20)
            epoch = exported.manifest["root_epoch"]
            snapshot = {key: value.clone() for key, value in exported.tensors.items()}
            arena.reset(position_from_fen(chess.STARTING_FEN))
            stats = arena.expand()
            later = arena.export(stats["span"].subset(0, 1))
            self.assertGreater(later.manifest["root_epoch"], epoch)
            for key, value in snapshot.items():
                self.assertTrue(torch.equal(value, exported.tensors[key]))
            arena.close()
            arena.close()
            with self.assertRaises(RuntimeError):
                arena.expand()
            # Independent expected features/support are packed after the guarded
            # producer path, then used to compare training for all architectures.
            selected = sample["sequence_ids"].cpu().tolist()
            reference = {key: value.clone() for key, value in sample.items()}
            actions = snapshot["actions"].cpu().tolist()
            fens = []
            for index in selected:
                board = chess.Board(); board.push_uci(index_to_move(actions[index], "w"))
                fens.append(board.fen(en_passant="fen"))
            reference["features"] = torch.stack([encode_fen(chess.STARTING_FEN)] * 4).to("cuda")
            reference["next_features"] = torch.stack([encode_fen(fen) for fen in fens]).to("cuda")
            legal = torch.zeros_like(reference["legal_mask"])
            legal[:, sorted(move_to_index(move.uci(), "w") for move in chess.Board().legal_moves)] = True
            reference["legal_mask"] = legal
            for name in MODELS:
                torch.manual_seed(9)
                a = build_model(name, width=8, depth=1).to("cuda")
                b = build_model(name, width=8, depth=1).to("cuda")
                b.load_state_dict(a.state_dict())
                with HotResident():
                    x, y = loss_terms(a, sample), loss_terms(b, reference)
                    x["total"].backward(); y["total"].backward()
                for key in x:
                    torch.testing.assert_close(x[key], y[key], atol=1e-6, rtol=1e-5)
                for left, right in zip(a.parameters(), b.parameters()):
                    if left.grad is not None:
                        torch.testing.assert_close(left.grad, right.grad, atol=1e-6, rtol=1e-5)
                opt = torch.optim.AdamW(a.parameters(), lr=0.001)
                before = [p.detach().clone() for p in a.parameters()]
                with PayloadResident():
                    losses = train_step(a, opt, sample)
                self.assertTrue(bool(torch.isfinite(losses["total"])))
                self.assertTrue(any(not torch.equal(old, p) for old, p in zip(before, a.parameters())))
            with self.assertRaises(AssertionError):
                with PayloadResident():
                    sample["features"].cpu()

    def test_range_capacity_epoch_and_terminal_failures(self):
        with self.arena(capacity=64) as arena:
            with self.assertRaises(RuntimeError):
                arena.export(ResidentSpan(arena.context_uuid, 1, 1, 1))
            arena.reset(position_from_fen(chess.STARTING_FEN))
            stats = arena.expand()
            for start, count in [(0, 1), (1, 21), (-1, 1), (1, 0)]:
                with self.assertRaises(ValueError):
                    arena.export(ResidentSpan(arena.context_uuid, stats["span"].epoch, start, count))
            with self.assertRaises(BridgeError) as caught:
                arena.export(stats["span"].subset(0, 1), legal_slots=1)
            self.assertEqual(caught.exception.code, -2)
            with self.assertRaises(BridgeError):
                arena.expand(stats["span"])
            # Failed capacity expansion did not commit children; old edges work.
            self.assertEqual(arena.export(stats["span"].subset(0, 1)).manifest["rows"], 1)
            arena.reset(position_from_fen("7k/6Q1/6K1/8/8/8/8/8 b - - 100 1"))
            stats = arena.expand()
            self.assertEqual(stats["next_count"], 0)
            with self.assertRaises(ValueError):
                arena.export(ResidentSpan(arena.context_uuid, 1, 1, 1))

    def test_multilevel_parent_parity_expanded_successors_and_stale_selectors(self):
        with self.arena() as arena:
            arena.reset(position_from_fen(chess.STARTING_FEN))
            first = arena.expand()
            first_export = arena.export(first["span"])
            parent_fens = {}
            for child, action in zip(first_export.sidecar["child_node"].cpu().tolist(),
                                     first_export.tensors["actions"].cpu().tolist()):
                board = chess.Board(); board.push_uci(index_to_move(action, "w"))
                parent_fens[child] = board.fen(en_passant="fen")
            second = arena.expand(first["span"])
            self.assertEqual(second["next_count"], 400)
            with PayloadResident():
                later = arena.export(second["span"])
                expanded_successors = arena.export(first["span"])
            self.verify_export(later, parent_fens)
            self.verify_export(expanded_successors, {0: chess.STARTING_FEN})
            old_span = first["span"]
            old_outputs = {**first_export.tensors, **first_export.sidecar}
            before = {name: old_outputs[name].clone() for name in SINK_FIELDS}
            arena.reset(position_from_fen(chess.STARTING_FEN))
            new = arena.expand()
            self.assertEqual(new["span"].start, old_span.start)  # numeric alias exists
            with self.assertRaises(ValueError):
                arena.export(old_span)
            with self.assertRaises(ValueError):
                arena.export(ResidentSpan("other-context", new["span"].epoch, 1, 1))
            # Also exercise native expected_epoch preflight, bypassing the
            # Python selector check; output payload must remain unchanged.
            sink = ReplaySink(1, c.sizeof(ReplaySink), arena.device.index, 256, 20,
                              *(old_outputs[name].data_ptr() for name in SINK_FIELDS))
            result = ReplayExport()
            torch.cuda.synchronize()
            with self.assertRaises(BridgeError):
                arena._call("export_transitions", 1, 20, old_span.epoch, c.byref(sink), c.byref(result))
            for name in SINK_FIELDS:
                self.assertTrue(torch.equal(before[name], old_outputs[name]))

    def test_retained_legal_history_third_repetition_successor(self):
        board, history = chess.Board(), []
        for move in ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6"]:
            history.append(position_from_fen(board.fen(en_passant="fen")).hash)
            board.push_uci(move)
        self.assertFalse(board.is_repetition(3))
        successor = board.copy(); successor.push_uci("f3g1"); successor.push_uci("f6g8")
        self.assertTrue(successor.is_repetition(3))
        with self.arena() as arena:
            arena.reset(position_from_fen(board.fen(en_passant="fen")), history)
            first = arena.expand()
            root_edges = arena.export(first["span"])
            parent_offset = root_edges.tensors["actions"].cpu().tolist().index(move_to_index("f3g1", "w"))
            stats = arena.expand(first["span"].subset(parent_offset, 1))
            with PayloadResident():
                exported = arena.export(stats["span"])
                replay = CompactReplay(64, "cuda")
                exported.append_to(replay)
            target = move_to_index("f6g8", "b")
            index = exported.tensors["actions"].cpu().tolist().index(target)
            self.assertTrue(bool(exported.tensors["terminal"][index]))
            self.assertEqual(exported.manifest["history_count"], 6)
            self.assertTrue(bool((exported.tensors["history_handles"] == -1).all()))
            self.assertIn("model lacks repetition", exported.manifest["history_scope"])

    def test_cross_stream_publication_admission_and_source_retirement(self):
        with self.arena() as arena:
            arena.reset(position_from_fen(chess.STARTING_FEN))
            stats = arena.expand()
            producer, consumer = torch.cuda.Stream(), torch.cuda.Stream()
            replay = CompactReplay(32, "cuda")
            with torch.cuda.stream(producer), PayloadResident():
                exported = arena.export(stats["span"], game_id=73)
            with torch.cuda.stream(consumer), PayloadResident():
                exported.append_to(replay)
            del exported
            gc.collect()
            arena.reset(position_from_fen(chess.STARTING_FEN))
            with HotResident():
                sample = replay.sample(16, torch.Generator(device="cuda").manual_seed(17))
            self.assertTrue(bool((sample["game_ids"] == 73).all()))
            self.assertTrue(torch.equal(sample["actions"], sample["target_actions"]))
            self.assertTrue(bool((sample["features"] == encode_fen(chess.STARTING_FEN).to("cuda")).all()))

    def test_failed_publication_and_admission_drain_queued_source_reads(self):
        with self.arena() as arena:
            arena.reset(position_from_fen(chess.STARTING_FEN))
            stats = arena.expand()
            stream = torch.cuda.Stream()
            with torch.cuda.stream(stream):
                with patch("neural.bridge.torch.full", side_effect=RuntimeError("injected auxiliary allocation failure")):
                    with self.assertRaisesRegex(RuntimeError, "injected auxiliary"):
                        arena.export(stats["span"])
                self.assertTrue(stream.query())
                exported = arena.export(stats["span"])
                replay = CompactReplay(32, "cuda")
                queued = []
                def enqueue_then_fail(batch):
                    queued.append(batch["pieces"].clone())
                    raise RuntimeError("injected admission failure")
                with patch.object(replay, "append", side_effect=enqueue_then_fail):
                    with self.assertRaisesRegex(RuntimeError, "injected admission"):
                        exported.append_to(replay)
                self.assertTrue(stream.query())
                self.assertEqual(replay.size, 0)
                source = exported.tensors["pieces"]
                del exported
                gc.collect()
            self.assertTrue(torch.equal(queued[0], source))


if __name__ == "__main__":
    unittest.main()
