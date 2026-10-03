"""CPU chess-oracle, compact ingress and neural mechanics fixtures; no training."""
from dataclasses import replace
import gzip
import json
import os
import unittest

import chess
import torch

from neural.features import move_to_index
from neural.tactical_inputs import (ReplaySpec, admit_compact, admit_native_topology,
    canonical_square, compact_from_replay, tactical_config, to_tactical_batch, topology_packet)
from neural.tactical_operator import Relation, TacticalOperator


def fresh(fen, **kwargs):
    return compact_from_replay(ReplaySpec(fen, (), True), **kwargs)


def ray_attacks(board, source):
    """Independent scalar geometry, not Board.attacks/attackers/pin masks."""
    piece = board.piece_at(source)
    if piece is None:
        return 0
    x, y = source % 8, source // 8
    if piece.piece_type == chess.PAWN:
        offsets, slide = ((-1, 1 if piece.color else -1), (1, 1 if piece.color else -1)), False
    elif piece.piece_type == chess.KNIGHT:
        offsets, slide = ((-2,-1),(-2,1),(-1,-2),(-1,2),(1,-2),(1,2),(2,-1),(2,1)), False
    elif piece.piece_type == chess.KING:
        offsets, slide = ((-1,-1),(-1,0),(-1,1),(0,-1),(0,1),(1,-1),(1,0),(1,1)), False
    else:
        offsets, slide = [], True
        if piece.piece_type in (chess.BISHOP, chess.QUEEN):
            offsets += [(-1,-1),(-1,1),(1,-1),(1,1)]
        if piece.piece_type in (chess.ROOK, chess.QUEEN):
            offsets += [(-1,0),(1,0),(0,-1),(0,1)]
    result = 0
    for dx, dy in offsets:
        xx, yy = x + dx, y + dy
        while 0 <= xx < 8 and 0 <= yy < 8:
            target = yy * 8 + xx
            result |= 1 << canonical_square(target, board.turn)
            if not slide or board.piece_at(target):
                break
            xx, yy = xx + dx, yy + dy
    return result


class TacticalInputsTests(unittest.TestCase):
    def test_replay_action_encoding_and_orientation(self):
        for moves in ((), ('e2e4',), ('e2e4','e7e5','g1f3')):
            state = compact_from_replay(ReplaySpec(moves=moves))
            board = chess.Board()
            for move in moves:
                board.push_uci(move)
            self.assertEqual([a.id for a in state.actions], sorted(move_to_index(m.uci(), 'w' if board.turn else 'b') for m in board.legal_moves))
            for source in range(64):
                self.assertEqual(state.source_attacks[canonical_square(source, board.turn)], ray_attacks(board, source))
            self.assertEqual(state.board[4], 6)
            self.assertEqual(state.history_counters[3], 0)  # independent new search root

    def test_pin_checker_and_blocker_relations(self):
        states = [fresh(fen) for fen in (
            '4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1',
            '4k3/4r3/8/8/8/8/8/4R1K1 b - - 0 1',
            '4k3/8/8/8/8/8/4r3/4K3 w - - 0 1')]
        for state in states[:2]:
            self.assertEqual(state.pinner_by_square[chess.E2], chess.E8)
            self.assertTrue(state.source_attacks[chess.E2] >> chess.A2 & 1)  # pinned still pseudoattacks
            self.assertTrue(state.source_attacks[chess.E2] >> chess.E1 & 1)  # own king first blocker
        self.assertEqual(states[2].checkers, 1 << chess.E2)
        batch = to_tactical_batch(states)
        for row in (0, 1):
            locations = batch.piece_squares[row].tolist()
            edges = batch.edge_index[row, batch.edge_mask[row]].tolist()
            kinds = batch.edge_kind[row, batch.edge_mask[row]].tolist()
            self.assertIn([locations.index(chess.E2), locations.index(chess.E1)],
                          [edge for edge, kind in zip(edges, kinds) if kind == Relation.PINNED_TO_KING])
            self.assertIn([locations.index(chess.E8), locations.index(chess.E2)],
                          [edge for edge, kind in zip(edges, kinds) if kind == Relation.PINNER_TO_PINNED])
        self.assertIn(int(Relation.CHECKER_TO_KING), batch.edge_kind[2, batch.edge_mask[2]].tolist())

    def test_special_effects_and_parent_priors(self):
        ep = fresh('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')
        castle = fresh('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1')
        promotion = fresh('7k/P7/8/8/8/8/8/7K w - - 0 1')
        a = next(a for a in ep.actions if a.en_passant)
        self.assertEqual(a.victim_square, chess.D5)
        self.assertEqual(a.edits, ((chess.D5, 7, 0), (chess.E5, 1, 0), (chess.D6, 0, 1)))
        self.assertEqual(a.material_parent_cp, 100)
        a = next(a for a in castle.actions if a.uci == 'e1g1')
        self.assertEqual(a.edits, ((4,6,0),(5,0,4),(6,0,6),(7,4,0)))
        self.assertEqual({a.promotion for a in promotion.actions if a.promotion}, {1,2,3,4})
        batch = to_tactical_batch((ep, castle, promotion))
        for row, state in enumerate((ep, castle, promotion)):
            for slot, a in enumerate(state.actions):
                target = 96 + slot
                kinds = batch.edge_kind[row, batch.edge_mask[row]].tolist()
                edges = batch.edge_index[row, batch.edge_mask[row]].tolist()
                effects = [edge[1] - 32 for edge, kind in zip(edges, kinds)
                           if kind == Relation.ACTION_EFFECT and edge[0] == target]
                self.assertEqual(effects, [s for s, _, _ in a.edits])
                self.assertEqual(batch.action_prior_cp[row, slot].item(), a.material_parent_cp)
        # Includes occupancy, all four colored piece relations, pin/check and all action effects.
        pins = to_tactical_batch((fresh('4r1k1/8/8/8/8/8/4R3/4K3 w - - 0 1'),
                                 fresh('4k3/8/8/8/8/8/4r3/4K3 w - - 0 1'), compact_from_replay()))
        self.assertEqual(set(batch.edge_kind[batch.edge_mask].tolist()) |
                         set(pins.edge_kind[pins.edge_mask].tolist()), set(range(14)))

    def test_history_and_adjudication(self):
        moves = ('g1f3','g8f6','f3g1','f6g8') * 2
        state = compact_from_replay(ReplaySpec(moves=moves), ply_from_search_root=8)
        self.assertEqual(state.history_counters, (8,3,9,8))
        self.assertEqual(state.terminal, 'threefold')
        self.assertTrue(state.actions)  # geometric legal set retained; terminal sidecar is authoritative
        with self.assertRaises(ValueError):
            compact_from_replay(ReplaySpec(moves=moves + ('e2e4',)))
        mate = fresh('7k/6Q1/5K2/8/8/8/8/8 b - - 0 1')
        self.assertEqual(mate.terminal, 'checkmate')
        batch = to_tactical_batch((mate,))
        out = TacticalOperator(tactical_config(width=8, steps=1))(batch)
        self.assertEqual(out['canonical_action'].item(), -1)
        self.assertEqual(batch.actions.shape[1], 0)

    def test_reject_unknown_history_illegal_and_forged_packets(self):
        for spec in (ReplaySpec('4k3/8/8/8/8/8/8/4K3 w - - 0 1'),
                     ReplaySpec('4k3/8/8/8/8/8/8/4K3 w - - 7 1', (), True),
                     ReplaySpec(moves=('e2e5',)), ReplaySpec(moves=('0000',))):
            with self.assertRaises(ValueError):
                compact_from_replay(spec)
        state = compact_from_replay()
        for forged in (replace(state, checkers=1), replace(state, source_attacks=(0,) * 64),
                       replace(state, actions=state.actions[:-1]), replace(state, context_sha256='0' * 64),
                       replace(state, actions=(replace(state.actions[0], material_parent_cp=999),) + state.actions[1:])):
            with self.assertRaises(ValueError):
                admit_compact(forged)
        with self.assertRaisesRegex(ValueError, 'overflow'):
            to_tactical_batch((state,), max_edges=1)

    def test_topology_abi_words_and_identity(self):
        states = (fresh('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', node_id=23, root_epoch=19),
                  compact_from_replay(ReplaySpec(moves=('e2e4',)), node_id=24, root_epoch=19))
        packet = topology_packet(states)
        admit_native_topology(packet, states)
        self.assertEqual(packet['source_attacks'].shape, (2,64))
        self.assertTrue(bool((packet['source_attacks'] < 0).any()))  # bit63 uses signed storage
        bad = {key: value.clone() for key, value in packet.items()}
        bad['node_id'][0] += 1
        with self.assertRaises(ValueError):
            admit_native_topology(bad, states)

    def test_real_adapter_forward_and_gradient(self):
        torch.manual_seed(51)
        states = (compact_from_replay(), compact_from_replay(ReplaySpec(moves=('e2e4',))))
        batch = to_tactical_batch(states)
        model = TacticalOperator(tactical_config(width=16, steps=2))
        output = model(batch)
        torch.testing.assert_close(output['value_cp'], batch.value_prior_cp, rtol=0, atol=0)
        torch.testing.assert_close(output['action_scores_cp'][batch.action_mask],
                                   batch.action_prior_cp[batch.action_mask], rtol=0, atol=0)
        with torch.no_grad():
            model.value.weight.normal_(std=.01)
            model.action.weight.normal_(std=.01)
        output = model(batch)
        (output['value_cp'].square().mean() + output['action_scores_cp'][batch.action_mask].square().mean()).backward()
        for parameter in (model.piece_input.weight, model.cell.source.weight, model.cell.relation.weight):
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()))
            self.assertGreater(parameter.grad.abs().sum().item(), 0)
        self.assertEqual(output['score_frame'], 'current_node_mover_cp')

    @unittest.skipUnless(os.environ.get('CHEN_SYNTHETIC_SHARD'), 'explicit TRAIN pilot path not supplied')
    def test_fresh_train_shard_three_root_ingress(self):
        with gzip.open(os.environ['CHEN_SYNTHETIC_SHARD'], 'rt') as stream:
            family = json.loads(next(stream))
        self.assertEqual(family['split'], 'train')
        self.assertIsNone(family['outcome_label'])
        admitted = []
        for root in family['roots'][:3]:
            state = compact_from_replay(ReplaySpec(moves=tuple(family['moves'][:root['ply']])),
                                        ply_from_search_root=root['ply'])
            self.assertEqual(state.fen, root['fen'])
            self.assertEqual(state.source_attacks, tuple(int(x,16) for x in root['attacks']))
            self.assertEqual(state.pinned, tuple(int(x,16) for x in root['pins']))
            self.assertEqual(state.checkers, int(root['checkers'],16))
            self.assertEqual(state.position_hash, int(root['current_hash'],16))
            self.assertEqual(state.previous_hashes, tuple(int(x,16) for x in root['previous_hashes']))
            self.assertEqual(state.history_counters, tuple(root['history_counters']))
            self.assertEqual([a.id for a in state.actions], [a['id'] for a in root['actions']])
            for action, saved in zip(state.actions, root['actions']):
                self.assertEqual(action.edits, tuple(sorted(tuple(edit) for edit in saved['edits'])))
                for name in ('uci','capture','en_passant','castling','victim_square'):
                    self.assertEqual(getattr(action, name), saved[name])
                self.assertEqual(action.material_parent_cp, saved['child_material_parent_cp'])
            admitted.append(state)
        self.assertEqual(len(admitted), 3)
        batch = to_tactical_batch(admitted)
        result = TacticalOperator(tactical_config(width=8, steps=1))(batch)
        self.assertTrue(bool(torch.isfinite(result['value_cp']).all()))
        self.assertEqual(batch.action_mask.sum(1).tolist(), [len(state.actions) for state in admitted])


if __name__ == '__main__':
    unittest.main()
