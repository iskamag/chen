"""Actual resident graph versus exact CPU sparse reference; no training/rating."""
import ctypes as c
from dataclasses import fields
import hashlib
import json
import os
from pathlib import Path
import unittest

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.bridge import BridgeError, ResidentSpan, RuleArena, RulePosition, hash_key, position_from_fen
from neural.native_symbolic import NativeSymbolicProducer, SymbolicExport
from neural.native_tactical import FIELDS, NativeTacticalProducer, TacticalSink
from neural.tactical_inputs import ReplaySpec, compact_from_replay, tactical_config, to_tactical_batch
from neural.tactical_operator import TacticalOperator, admit_contents


class ResidentOnly(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if '_local_scalar_dense' in str(function):
            raise AssertionError('device scalar read in observation/inference path')
        if '_to_copy' in str(function) and str((kwargs or {}).get('device', '')) == 'cpu':
            raise AssertionError('device payload downloaded in observation/inference path')
        return function(*args, **(kwargs or {}))


def signed(value):
    return value if value < 1 << 63 else value - (1 << 64)


class TacticalGPU(unittest.TestCase):
    rows = 0

    @classmethod
    def setUpClass(cls):
        cls.library = os.environ['CHEN_GPU_LIBRARY']
        cls.pin = os.environ['CHEN_GPU_LIBRARY_SHA256']
        assert hashlib.sha256(Path(cls.library).read_bytes()).hexdigest() == cls.pin
        torch.set_num_threads(1)

    def setUp(self):
        self.arena = RuleArena(self.library, capacity=4096, device='cuda:0')
        self.addCleanup(self.arena.close)
        self.producer = NativeTacticalProducer(self.arena, self.pin)
        self.test_rows = 0

    def tearDown(self):
        print(json.dumps(dict(kind='tactical_test', test=self._testMethodName,
                              oracle_rows=self.test_rows, runtime=self.producer.runtime_identity,
                              device=self.producer.device_identity)), flush=True)

    def test_abi1_symbolic_channels_composition(self):
        old = NativeSymbolicProducer(self.arena, self.pin)
        specs = [ReplaySpec(), ReplaySpec(moves=('e2e4',)),
                 ReplaySpec('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', (), True),
                 ReplaySpec('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', (), True)]
        for spec in specs:
            _, span = self.reset(spec)
            with ResidentOnly():
                previous, _ = old.export(span)
                current = self.producer.export(span)
            torch.testing.assert_close(previous.channels(), current.batch.squares, rtol=0, atol=0)
            torch.testing.assert_close(previous.history, current.sidecar['history'], rtol=0, atol=0)

    def reset(self, spec):
        state = compact_from_replay(spec)
        manifest = self.arena.reset(position_from_fen(state.fen), state.previous_hashes)
        span = ResidentSpan(self.arena.context_uuid, manifest['root_epoch'], 0, 1)
        return state, span

    def export(self, span):
        with ResidentOnly():
            snapshot = self.producer.export(span)
        self.assertFalse(snapshot.manifest['state_download'])
        self.assertFalse(snapshot.manifest['teacher_input'])
        return snapshot

    def compare(self, snapshot, states, span):
        expected = to_tactical_batch(states)
        actual = {f.name: getattr(snapshot.batch, f.name).cpu() for f in fields(snapshot.batch)
                  if isinstance(getattr(snapshot.batch, f.name), torch.Tensor)}
        # Cold downloads are deliberately outside ResidentOnly.
        cpu = type(snapshot.batch)(**actual, feature_schema=snapshot.batch.feature_schema)
        admit_contents(cpu, tactical_config(), expected_legal_actions=[[a.id for a in s.actions] for s in states])
        for row, state in enumerate(states):
            self.test_rows += 1
            n, a = int(expected.piece_mask[row].sum()), len(state.actions)
            e = int(expected.edge_mask[row].sum())
            for name in ('pieces', 'squares', 'piece_mask', 'piece_squares', 'value_prior_cp', 'edge_overflow'):
                torch.testing.assert_close(actual[name][row], getattr(expected, name)[row], rtol=0, atol=1e-7)
            for name in ('actions', 'action_mask', 'action_ids', 'action_prior_cp'):
                torch.testing.assert_close(actual[name][row, :a], getattr(expected, name)[row, :a], rtol=0, atol=1e-7)
            self.assertTrue(bool((actual['action_ids'][row, a:] == -1).all()))
            self.assertFalse(bool(actual['action_mask'][row, a:].any()))
            self.assertEqual(int(actual['edge_mask'][row].sum()), e)
            for name in ('edge_index', 'edge_kind', 'edge_features'):
                torch.testing.assert_close(actual[name][row, :e], getattr(expected, name)[row, :e], rtol=0, atol=1e-7)
            self.assertFalse(bool(actual['edge_mask'][row, e:].any()))
            side = {name: value[row].cpu() for name, value in snapshot.sidecar.items()}
            self.assertEqual(side['slot_counts'].tolist(), [n, a, e])
            self.assertEqual(side['history'].tolist(), list(state.history_counters))
            self.assertEqual(side['position_hash'].item(), signed(state.position_hash))
            self.assertEqual(side['node_id'].item(), span.start + row)
            self.assertEqual(side['root_epoch'].item(), span.epoch)
            terminal = {None: 0, 'checkmate': 1, 'stalemate': 2, 'fifty_move': 3,
                        'threefold': 4, 'insufficient_material': 5}[state.terminal]
            self.assertEqual(side['terminal'].item(), terminal)

    def test_all_relations_special_moves_and_mover_frames(self):
        fens = [chess.STARTING_FEN, '4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1',
                '4k3/8/8/8/8/8/4r3/4K3 w - - 0 1',
                'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
                'k3r3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
                '7k/P7/8/8/8/8/8/7K w - - 0 1',
                '7k/6Q1/5K2/8/8/8/8/8 b - - 0 1',
                '7k/5Q2/6K1/8/8/8/8/8 b - - 0 1',
                '4k3/8/8/8/8/8/8/4K3 w - - 0 1']
        relations = set()
        for fen in fens:
            for board in (chess.Board(fen), chess.Board(fen).mirror()):
                spec = ReplaySpec(board.fen(en_passant='fen'), (), True)
                state, span = self.reset(spec)
                snapshot = self.export(span)
                self.compare(snapshot, [state], span)
                relations.update(snapshot.batch.edge_kind[snapshot.batch.edge_mask].cpu().tolist())
        self.assertEqual(relations, set(range(14)))

    def test_complete_native_children_history_and_snapshot_lifetime(self):
        prefix = ('g1f3', 'g8f6', 'f3g1', 'f6g8', 'g1f3', 'g8f6', 'f3g1')
        state, span = self.reset(ReplaySpec(moves=prefix))
        old = self.export(span)
        self.compare(old, [state], span)
        children = self.arena.expand(span)['span']
        snapshot = self.export(children)
        root = chess.Board()
        for uci in prefix:
            root.push_uci(uci)
        candidates = {}
        for move in root.legal_moves:
            candidate = compact_from_replay(ReplaySpec(moves=prefix+(move.uci(),)), ply_from_search_root=1)
            candidates[signed(candidate.position_hash)] = candidate
        states = [candidates[key] for key in snapshot.sidecar['position_hash'].cpu().tolist()]
        self.compare(snapshot, states, children)
        self.arena.reset(position_from_fen(chess.STARTING_FEN))
        self.compare(old, [state], span)  # owned snapshot survives its source epoch
        with self.assertRaises(ValueError):
            self.export(children)

    def test_resident_model_forward_backward_and_terminal_authority(self):
        state, span = self.reset(ReplaySpec(moves=('e2e4',)))
        snapshot = self.export(span)
        model = TacticalOperator(tactical_config(width=8, steps=2)).to(self.arena.device)
        with torch.no_grad():
            model.value.weight.normal_(std=.01)
            model.action.weight.normal_(std=.01)
        with ResidentOnly():
            out = model(snapshot.batch)
            loss = out['value_cp'].square().mean() + torch.where(snapshot.batch.action_mask,
                out['action_scores_cp'], torch.zeros_like(out['action_scores_cp'])).square().sum()
            loss.backward()
        self.assertTrue(bool(torch.isfinite(out['value_cp']).all()))
        for parameter in (model.piece_input.weight, model.cell.source.weight):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        # A geometric action set on a draw must not override the symbolic sidecar.
        moves = ('g1f3', 'g8f6', 'f3g1', 'f6g8') * 2
        state, span = self.reset(ReplaySpec(moves=moves))
        draw = self.export(span)
        self.compare(draw, [state], span)
        self.assertTrue(bool(draw.batch.action_mask.any()))
        self.assertEqual(draw.sidecar['terminal'].item(), 4)
        mate = position_from_fen('7k/6Q1/5K2/8/8/8/8/8 b - - 100 1')
        manifest = self.arena.reset(mate)
        snapshot = self.export(ResidentSpan(self.arena.context_uuid, manifest['root_epoch'], 0, 1))
        self.assertEqual(snapshot.sidecar['terminal'].item(), 1)
        with ResidentOnly():
            out = model(snapshot.batch)
        self.assertEqual(out['canonical_action'].item(), -1)

    def test_preflight_preserves_outputs(self):
        _, span = self.reset(ReplaySpec())
        outputs = {name: torch.full((1, *shape), 1, dtype=dtype, device=self.arena.device)
                   for name, (shape, dtype) in self.producer.schema.items()}
        sink = TacticalSink(1, 192, 0, 0, 1, 256, 6144, *(outputs[name].data_ptr() for name in FIELDS))
        for field, value in (('abi_version', 2), ('struct_size', 8), ('reserved', 1),
                             ('device', 1), ('legal_slots', 255), ('edge_slots', 1),
                             ('capacity', 0), ('actions', sink.pieces), ('edge_kind', sink.edge_kind + 1)):
            bad = TacticalSink.from_buffer_copy(sink)
            setattr(bad, field, value)
            metadata = SymbolicExport(123, 123, 123)
            with self.assertRaises(BridgeError):
                self.arena._call('export_tactical', 0, 1, span.epoch, c.byref(bad), c.byref(metadata))
            self.assertEqual((metadata.rows, metadata.root_epoch, metadata.kernel_ms), (123, 123, 123))
            for output in outputs.values():
                self.assertTrue(bool((output == 1).all()))
        # Native rules accepts locally valid >32-piece packets; the object schema
        # rejects these before writing any observation, instead of overflowing.
        packet = RulePosition()
        packet.boards[5], packet.boards[11] = 1, 1 << 63
        packet.boards[1] = ((1 << 36) - 1) ^ 1
        packet.ep_square, packet.fullmove = -1, 1
        packet.hash = hash_key(768)
        for piece in range(12):
            for square in chess.scan_forward(packet.boards[piece]):
                packet.hash ^= hash_key(piece*64+square)
        manifest = self.arena.reset(packet)
        metadata = SymbolicExport(123, 123, 123)
        with self.assertRaises(BridgeError):
            self.arena._call('export_tactical', 0, 1, manifest['root_epoch'], c.byref(sink), c.byref(metadata))
        for output in outputs.values():
            self.assertTrue(bool((output == 1).all()))
        self.assertEqual(metadata.rows, 123)


if __name__ == '__main__':
    unittest.main()
