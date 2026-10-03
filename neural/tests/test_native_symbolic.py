"""Actual GPU producer against independent python-chess, not strength evidence."""
import ctypes as c
import hashlib
import json
import os
from pathlib import Path
import random
import unittest

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.bridge import BridgeError, ResidentSpan, RuleArena, position_from_fen
from neural.features import encode_fen
from neural.native_symbolic import NativeSymbolicProducer, SCHEMA, SymbolicSink, SymbolicExport
from neural.semantic_strategy import SemanticAnchor
from neural.symbolic_residual import SymbolicResidualAnchor
from neural.transformer_semantics import TransformerConfig, TransformerSemantics


class HotResident(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if '_local_scalar_dense' in str(function):
            raise AssertionError('host scalar extraction in producer/model hot path')
        if '_to_copy' in str(function) and str((kwargs or {}).get('device', '')) == 'cpu':
            raise AssertionError('device payload copied to CPU')
        return function(*args, **(kwargs or {}))


def signed(value):
    return value if value < 2**63 else value - 2**64


def oracle(board, previous, ply):
    """Independent board APIs; never call the native attack/pin helpers."""
    canonical = board if board.turn else board.mirror()
    result = torch.zeros(64, 12)
    for color, side in enumerate((chess.WHITE, chess.BLACK)):
        for square, piece in canonical.piece_map().items():
            if piece.color != side:
                continue
            targets = list(canonical.attacks(square))
            result[square, 2 + color] = min(len(targets), 8) / 8
            for target in targets:
                result[target, color] += 1
            result[square, 4 + color] = canonical.is_pinned(side, square)
    result[:, :2] = result[:, :2].clamp_max(8) / 8
    for source in canonical.checkers():
        result[source, 6] = 1
    weights = {chess.KNIGHT: 1, chess.BISHOP: 1, chess.ROOK: 2, chess.QUEEN: 4}
    phase = sum(weights.get(p.piece_type, 0) for p in canonical.piece_map().values())
    result[:, 7] = min(phase, 24) / 24
    key = position_from_fen(board.fen(en_passant='fen')).hash
    window = [*previous, key][-board.halfmove_clock-1:]
    history = torch.tensor([board.halfmove_clock, window.count(key), len(window), ply], dtype=torch.int64)
    result[:, 8:] = (history.float() / torch.tensor([100., 3., 256., 256.])).clamp(0, 1)
    return result, history


class NativeSymbolicGPU(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A requested hardware run must fail rather than silently skip.
        if not torch.cuda.is_available() or torch.version.hip is None:
            raise RuntimeError('actual ROCm GPU required for this suite')
        cls.library = os.environ['CHEN_GPU_LIBRARY']
        cls.library_hash = os.environ['CHEN_GPU_LIBRARY_SHA256']
        assert hashlib.sha256(Path(cls.library).read_bytes()).hexdigest() == cls.library_hash
        torch.set_num_threads(1)
        torch.manual_seed(98123)

    def setUp(self):
        self.arena = RuleArena(self.library, capacity=6000, device='cuda:0')
        self.producer = NativeSymbolicProducer(self.arena, self.library_hash)
        self.addCleanup(self.arena.close)
        self.oracle_rows = 0

    def tearDown(self):
        print(json.dumps(dict(kind='symbolic_oracle_test', test=self._testMethodName,
                              oracle_rows=self.oracle_rows)), flush=True)

    def reset(self, board, history=()):
        manifest = self.arena.reset(position_from_fen(board.fen(en_passant='fen')), history)
        return ResidentSpan(self.arena.context_uuid, manifest['root_epoch'], 0, 1)

    def compare(self, state, boards, previous, ply):
        # Explicit independent oracle downloads, outside the audited hot path.
        f, channels, history = (state.features.cpu(), state.channels().cpu(), state.history.cpu())
        hashes, ids, epochs = (state.position_hash.cpu(), state.node_id.cpu(), state.root_epoch.cpu())
        self.oracle_rows += len(boards)
        for row, board in enumerate(boards):
            expected, counters = oracle(board, previous[row], ply)
            torch.testing.assert_close(f[row], encode_fen(board.fen(en_passant='fen')), rtol=0, atol=1e-7)
            torch.testing.assert_close(channels[row], expected, rtol=0, atol=1e-7)
            torch.testing.assert_close(history[row], counters, rtol=0, atol=0)
            self.assertEqual(int(hashes[row]), signed(position_from_fen(board.fen(en_passant='fen')).hash))
            self.assertEqual(int(epochs[row]), state.epoch)
            self.assertGreaterEqual(int(ids[row]), 0)

    def test_special_rules_pins_checkers_colors_and_clock_saturation(self):
        fixtures = [chess.STARTING_FEN,
            'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 65535 4294967295',
            '4r1k1/8/8/8/8/8/4N3/4K3 w - - 17 1',
            '4r1k1/8/8/8/8/8/8/4K3 w - - 17 1',
            'k7/8/8/8/7b/8/5N2/4K3 w - - 0 1',
            'k3r3/8/8/8/1b6/8/3BN3/4K3 w - - 0 1',
            'k3r3/8/8/8/1b6/8/3BN3/4K3 b - - 0 1',
            '4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1',
            'k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
            '4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1',
            '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
            '7k/6Q1/6K1/8/8/8/8/8 b - - 100 1',
            '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
            '4k3/8/8/8/8/8/8/4K3 w - - 0 1']
        for fen in fixtures:
            for board in (chess.Board(fen), chess.Board(fen).mirror()):
                with self.subTest(fen=board.fen()):
                    span = self.reset(board)
                    with HotResident():
                        state, _ = self.producer.export(span)
                    self.compare(state, [board], [[]], 0)

    def test_deep_ancestor_and_root_prefix_history(self):
        board = chess.Board();prefix = []
        for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
            prefix.append(position_from_fen(board.fen(en_passant='fen')).hash)
            board.push_uci(uci)
        root = self.reset(board, prefix)
        children = self.arena.expand(root)['span']
        state, _ = self.producer.export(children)
        parent = board.copy();parent.push_uci('g1f3')
        key = signed(position_from_fen(parent.fen(en_passant='fen')).hash)
        row = state.position_hash.cpu().tolist().index(key)
        parent_span = ResidentSpan(self.arena.context_uuid, root.epoch, int(state.node_id[row]), 1)
        parent_state, _ = self.producer.export(parent_span)
        root_key = position_from_fen(board.fen(en_passant='fen')).hash
        self.compare(parent_state, [parent], [[*prefix, root_key]], 1)
        grandchildren = self.arena.expand(parent_span)['span']
        state, _ = self.producer.export(grandchildren)
        child = parent.copy();child.push_uci('g8f6')
        key = signed(position_from_fen(child.fen(en_passant='fen')).hash)
        row = state.position_hash.cpu().tolist().index(key)
        selected = ResidentSpan(self.arena.context_uuid, root.epoch, int(state.node_id[row]), 1)
        state, _ = self.producer.export(selected)
        parent_key = position_from_fen(parent.fen(en_passant='fen')).hash
        self.compare(state, [child], [[*prefix, root_key, parent_key]], 2)
        self.assertEqual(state.history.cpu().tolist(), [[6, 2, 7, 2]])

    def test_random_legal_roots_and_complete_children(self):
        rng = random.Random(7112)
        board = chess.Board()
        for step in range(32):
            span = self.reset(board)
            with HotResident():
                state, _ = self.producer.export(span)
            self.compare(state, [board], [[]], 0)
            children = self.arena.expand(span)['span']
            if children.count:
                expected = {}
                root_hash = position_from_fen(board.fen(en_passant='fen')).hash
                for move in board.legal_moves:
                    child = board.copy();child.push(move)
                    key = signed(position_from_fen(child.fen(en_passant='fen')).hash)
                    self.assertNotIn(key, expected)
                    expected[key] = child
                with HotResident():
                    state, _ = self.producer.export(children)
                ordered = [expected[key] for key in state.position_hash.cpu().tolist()]
                self.assertEqual(len(ordered), len(expected))
                self.compare(state, ordered, [[root_hash] for _ in ordered], 1)
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))

    def test_repetition_window_and_owned_snapshot_after_reset_close(self):
        board = chess.Board();past = []
        for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 2:
            past.append(position_from_fen(board.fen(en_passant='fen')).hash)
            board.push_uci(uci)
        span = self.reset(board, past)
        state, _ = self.producer.export(span)
        self.compare(state, [board], [past], 0)
        self.assertEqual(int(state.history[0, 1]), 3)
        saved = state.channels().clone()
        self.reset(chess.Board())
        with self.assertRaisesRegex(ValueError, 'stale'):
            self.producer.export(span)
        self.arena.close()
        torch.testing.assert_close(state.channels(), saved, rtol=0, atol=0)

    def test_native_admission_failure_leaves_output_and_epoch_intact(self):
        span = self.reset(chess.Board())
        self.arena.expand(span)
        outputs = {n: torch.full((1, *shape), -17, dtype=dtype, device='cuda:0') for n, (shape, dtype) in SCHEMA.items()}
        sink = SymbolicSink(1, c.sizeof(SymbolicSink), 0, 0, 1, *(outputs[n].data_ptr() for n in SCHEMA))
        result = SymbolicExport(999, 999, -17)
        for start, count, epoch in [(0, 1, span.epoch+1), (0, 2, span.epoch), (9999, 1, span.epoch)]:
            with self.assertRaises(BridgeError):
                torch.cuda.synchronize()
                self.arena._call('export_symbolic', start, count, epoch, c.byref(sink), c.byref(result))
            self.assertEqual((result.rows, result.root_epoch, result.kernel_ms), (999, 999, -17))
            for value in outputs.values():
                self.assertTrue(bool((value == -17).all()))
        sink.channels = sink.features
        with self.assertRaises(BridgeError):
            self.arena._call('export_symbolic', 0, 1, span.epoch, c.byref(sink), c.byref(result))
        with self.assertRaisesRegex(ValueError, 'hash mismatch'):
            NativeSymbolicProducer(self.arena, '0' * 64)
        # Admission failures do not corrupt the authoritative reset/history.
        state, _ = self.producer.export(span)
        self.compare(state, [chess.Board()], [[]], 0)

    def test_native_channels_feed_both_learned_residual_models(self):
        state, _ = self.producer.export(self.reset(chess.Board()))
        anchor = SemanticAnchor('threat', width=8).cuda()
        with torch.no_grad():
            anchor.cp.weight.normal_(0, .03)
        model = SymbolicResidualAnchor(anchor)
        with HotResident():
            before = model(state)
            baseline = model.anchor(state.features)
            before['value_cp'].sum().backward()
        torch.testing.assert_close(before['value_cp'], baseline['value_cp'], rtol=0, atol=0)
        self.assertGreater(float(model.value.weight.grad.abs().sum()), 0)
        transformer = TransformerSemantics(TransformerConfig(width=8, heads=2, depth=1, symbolic_channels=12)).cuda()
        with HotResident():
            output = transformer.forward_symbolic(state)
            output['board_logits'].square().mean().backward()
        self.assertGreater(float(transformer.symbolic_square.weight.grad.abs().sum()), 0)
        self.assertGreater(float(transformer.symbolic_value.weight.grad.abs().sum()), 0)


if __name__ == '__main__':
    unittest.main()
