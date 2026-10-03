"""Focused actual-gfx1030 topology oracle; no training or performance claim.

Run with the pinned neural bridge on PYTHONPATH and CHEN_GPU_LIBRARY/_SHA256.
Oracle downloads are explicit and outside the audited native hot path.
"""
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
from neural.native_symbolic import NativeSymbolicProducer, SymbolicExport
from neural.int8_inference import admit_gfx1030, loaded_hip_runtime


FIELDS = ('source_attacks', 'pinned', 'checkers', 'pinner_by_square',
          'position_hash', 'node_id', 'root_epoch')
SHAPES = ((64,), (2,), (), (64,), (), (), ())


class TopologySink(c.Structure):
    _fields_ = [('abi_version', c.c_uint32), ('struct_size', c.c_uint32),
                ('device', c.c_int32), ('reserved', c.c_uint32), ('capacity', c.c_uint64)] + [
                    (name, c.c_uint64) for name in FIELDS]


assert c.sizeof(TopologySink) == 80 and c.sizeof(SymbolicExport) == 24


def signed(value):
    return value if value < 2**63 else value - 2**64


class HotResident(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if '_local_scalar_dense' in str(function):
            raise AssertionError('host scalar extraction in topology hot path')
        if '_to_copy' in str(function) and str((kwargs or {}).get('device', '')) == 'cpu':
            raise AssertionError('device payload copied to CPU')
        return function(*args, **(kwargs or {}))


def oracle(board):
    canonical = board if board.turn else board.mirror()
    attacks, pinners, pins = [0] * 64, [-1] * 64, [0, 0]
    for source, piece in canonical.piece_map().items():
        attacks[source] = int(canonical.attacks(source))
        if canonical.is_pinned(piece.color, source):
            pins[int(not piece.color)] |= 1 << source
            candidates = [square for square, enemy in canonical.piece_map().items()
                          if enemy.color != piece.color and
                          enemy.piece_type in (chess.BISHOP, chess.ROOK, chess.QUEEN) and
                          square in canonical.pin(piece.color, source) and
                          source in canonical.attacks(square)]
            assert len(candidates) == 1, (board.fen(), source, candidates)
            pinners[source] = candidates[0]
    return dict(source_attacks=[signed(value) for value in attacks],
                pinned=[signed(value) for value in pins], checkers=signed(int(canonical.checkers())),
                pinner_by_square=pinners)


class NativeTopologyGPU(unittest.TestCase):
    rows = 0

    @classmethod
    def setUpClass(cls):
        cls.library = os.environ['CHEN_GPU_LIBRARY']
        cls.library_hash = os.environ['CHEN_GPU_LIBRARY_SHA256']
        assert hashlib.sha256(Path(cls.library).read_bytes()).hexdigest() == cls.library_hash
        torch.set_num_threads(1)
        cls.device, device = admit_gfx1030('cuda:0')
        runtime = loaded_hip_runtime()
        print(json.dumps(dict(kind='topology_admission', library_sha256=cls.library_hash,
                              torch_version=torch.__version__, torch_hip=torch.version.hip,
                              device=device, runtime=runtime)), flush=True)

    def setUp(self):
        self.arena = RuleArena(self.library, capacity=6000, device=self.device)
        self.addCleanup(self.arena.close)
        version = self.arena._lib.chen_topology_abi_version
        version.argtypes, version.restype = [], c.c_uint32
        self.assertEqual(version(), 1)
        fn = self.arena._lib.chen_rules_export_topology
        fn.argtypes = [c.c_void_p, c.c_uint64, c.c_uint64, c.c_uint64,
                       c.POINTER(TopologySink), c.POINTER(SymbolicExport), c.c_char_p, c.c_uint64]
        fn.restype = c.c_int32
        self.symbolic = NativeSymbolicProducer(self.arena, self.library_hash)
        self.test_rows = 0

    def tearDown(self):
        print(json.dumps(dict(kind='topology_test', test=self._testMethodName,
                              oracle_rows=self.test_rows)), flush=True)

    def reset(self, board, history=()):
        manifest = self.arena.reset(position_from_fen(board.fen(en_passant='fen')), history)
        return ResidentSpan(self.arena.context_uuid, manifest['root_epoch'], 0, 1)

    def allocations(self, count, fill=None):
        return {name: (torch.empty((count, *shape), dtype=torch.int64, device=self.device)
                       if fill is None else torch.full((count, *shape), fill,
                           dtype=torch.int64, device=self.device)) for name, shape in zip(FIELDS, SHAPES)}

    def sink(self, outputs, capacity):
        return TopologySink(1, 80, self.device.index, 0, capacity,
                            *(outputs[name].data_ptr() for name in FIELDS))

    def export(self, span):
        outputs = self.allocations(span.count)
        sink, result = self.sink(outputs, span.count), SymbolicExport()
        torch.cuda.synchronize(self.device)
        with HotResident():
            self.arena._call('export_topology', span.start, span.count, span.epoch,
                             c.byref(sink), c.byref(result))
        self.assertEqual((result.rows, result.root_epoch), (span.count, span.epoch))
        self.assertGreaterEqual(result.kernel_ms, 0)
        return outputs

    def compare(self, outputs, boards, span):
        actual = {name: value.cpu().tolist() for name, value in outputs.items()}
        self.test_rows += len(boards)
        for row, board in enumerate(boards):
            expected = oracle(board)
            for name, value in expected.items():
                self.assertEqual(actual[name][row], value, (board.fen(), name))
            self.assertEqual(actual['position_hash'][row],
                             signed(position_from_fen(board.fen(en_passant='fen')).hash))
            self.assertEqual(actual['node_id'][row], span.start + row)
            self.assertEqual(actual['root_epoch'][row], span.epoch)

    def test_special_and_mirrored_full_masks_pinners_blockers_and_abi1(self):
        fens = [chess.STARTING_FEN,
                '4r1k1/8/8/8/8/8/4N3/4K3 w - - 17 1',
                'k7/8/8/8/7b/8/5N2/4K3 w - - 0 1',
                'k3r3/8/8/8/1b6/8/3BN3/4K3 w - - 0 1',
                '4r1k1/8/8/8/1b6/8/8/4K3 w - - 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 65535 4294967295',
                'k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
                '4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1',
                '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1',
                '7k/6Q1/6K1/8/8/8/8/8 b - - 100 1',
                '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
                '4k3/8/8/8/8/8/8/4K3 w - - 0 1']
        for fen in fens:
            for board in (chess.Board(fen), chess.Board(fen).mirror()):
                with self.subTest(fen=board.fen()):
                    span = self.reset(board)
                    output = self.export(span)
                    self.compare(output, [board], span)
                    state, _ = self.symbolic.export(span)
                    channels = state.channels().cpu()[0]
                    canonical = board if board.turn else board.mirror()
                    expected = torch.zeros(64, 7)
                    for source, piece in canonical.piece_map().items():
                        color = int(not piece.color)
                        targets = canonical.attacks(source)
                        expected[source, 2 + color] = min(len(targets), 8) / 8
                        expected[source, 4 + color] = canonical.is_pinned(piece.color, source)
                        for target in targets:
                            expected[target, color] += 1
                    expected[:, :2] = expected[:, :2].clamp_max(8) / 8
                    for checker in canonical.checkers():
                        expected[checker, 6] = 1
                    torch.testing.assert_close(channels[:, :7], expected, rtol=0, atol=0)

    def test_reachable_roots_and_complete_children_with_special_transitions(self):
        rng, board = random.Random(314159), chess.Board()
        boards = []
        for _ in range(24):
            boards.append(board.copy())
            legal = list(board.legal_moves)
            if not legal:
                break
            board.push(rng.choice(legal))
        boards.extend(chess.Board(fen) for fen in [
            'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
            '4k3/8/8/3pP3/8/8/8/4K3 w - d6 7 12',
            '4k3/P7/8/8/8/8/7p/4K3 w - - 0 1'])
        for board in boards:
            root = self.reset(board)
            self.compare(self.export(root), [board], root)
            children = self.arena.expand(root)['span']
            if not children.count:
                continue
            expected = {}
            for move in board.legal_moves:
                child = board.copy();child.push(move)
                key = signed(position_from_fen(child.fen(en_passant='fen')).hash)
                self.assertNotIn(key, expected)
                expected[key] = child
            output = self.export(children)
            ordered = [expected[key] for key in output['position_hash'].cpu().tolist()]
            self.assertEqual(len(ordered), len(expected))
            self.compare(output, ordered, children)

    def test_deep_history_snapshot_reuse_and_unchanged_search_work(self):
        board, history = chess.Board(), []
        for uci in ('g1f3', 'g8f6', 'f3g1', 'f6g8'):
            history.append(position_from_fen(board.fen(en_passant='fen')).hash)
            board.push_uci(uci)
        span = self.reset(board, history)
        for uci in ('g1f3', 'g8f6'):
            child = board.copy();child.push_uci(uci)
            children = self.arena.expand(span)['span']
            output = self.export(children)
            key = signed(position_from_fen(child.fen(en_passant='fen')).hash)
            row = output['position_hash'].cpu().tolist().index(key)
            span = ResidentSpan(self.arena.context_uuid, span.epoch, children.start + row, 1)
            board = child
        before = self.arena.evaluate(span, 1)
        snapshots = [self.export(span), self.export(span)]
        after = self.arena.evaluate(span, 1)
        for name in ('arena_positions', 'classified_positions', 'evaluated_positions', 'next_start', 'next_count'):
            self.assertEqual(before[name], after[name])
        for output in snapshots:
            self.compare(output, [board], span)
        saved = {name: value.clone() for name, value in snapshots[0].items()}
        self.reset(chess.Board())
        self.arena.close()
        for name, value in snapshots[0].items():
            torch.testing.assert_close(value, saved[name], rtol=0, atol=0)

    def test_admission_output_capacity_and_empty_range(self):
        span = self.reset(chess.Board())
        self.arena.expand(span)
        outputs = self.allocations(1, -17)
        base = self.sink(outputs, 1)
        failures = [(0, 1, span.epoch+1, {}), (0, 2, span.epoch, {}),
                    (9999, 1, span.epoch, {}), (0, 1, span.epoch, {'abi_version': 2}),
                    (0, 1, span.epoch, {'struct_size': 72}), (0, 1, span.epoch, {'reserved': 1}),
                    (0, 1, span.epoch, {'device': 1}),
                    (0, 1, span.epoch, {'source_attacks': base.source_attacks + 4}),
                    (0, 1, span.epoch, {'checkers': base.pinned}),
                    (0, 1, span.epoch, {'pinner_by_square': base.source_attacks}),
                    (0, 1, span.epoch, {'source_attacks': 0}),
                    (0, 1, span.epoch, {'capacity': 100000})]
        for start, count, epoch, changes in failures:
            sink = TopologySink.from_buffer_copy(base)
            for name, value in changes.items():
                setattr(sink, name, value)
            result = SymbolicExport(999, 999, -17)
            torch.cuda.synchronize(self.device)
            with self.assertRaises(BridgeError):
                self.arena._call('export_topology', start, count, epoch, c.byref(sink), c.byref(result))
            self.assertEqual((result.rows, result.root_epoch, result.kernel_ms), (999, 999, -17))
            for value in outputs.values():
                self.assertTrue(bool((value == -17).all()))
        empty = TopologySink(1, 80, self.device.index, 0, 0, *([0] * len(FIELDS)))
        result = SymbolicExport()
        self.arena._call('export_topology', 0, 0, span.epoch, c.byref(empty), c.byref(result))
        self.assertEqual((result.rows, result.root_epoch, result.kernel_ms), (0, span.epoch, 0))
        self.compare(self.export(span), [chess.Board()], span)


if __name__ == '__main__':
    unittest.main(verbosity=2)
