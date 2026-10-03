"""Independent small unpruned exhaustive exchange oracle; CPU-only fixtures."""
from copy import deepcopy
import gzip
import json
from pathlib import Path
import tempfile
import time
import unittest

import chess

from neural.exchange_targets import (Limits, complete_initial_actions, exchange_target, generate)


def exhaustive(board, first):
    """Separate scalar minimax; no generator backup, cap or material helper."""
    origin=board.turn
    values={chess.PAWN:100,chess.KNIGHT:320,chess.BISHOP:330,chess.ROOK:500,chess.QUEEN:900,chess.KING:0}
    total=lambda b:sum(values[p.piece_type]*(1 if p.color==origin else -1) for p in b.piece_map().values())
    reference=total(board)
    target=first.to_square
    child=board.copy(stack=True)
    child.push(first)
    def walk(b):
        if b.is_checkmate() or b.is_stalemate() or b.is_insufficient_material() or b.is_repetition(3) or b.halfmove_clock>=100:
            return None
        legal=list(b.legal_moves)
        captures=[m for m in legal if m.to_square==target and b.is_capture(m)]
        checked=b.is_check()
        if checked and any(m not in captures for m in legal):return None
        outcomes=[] if checked else [total(b)-reference]
        for m in captures:
            successor=b.copy(stack=True)
            successor.push(m)
            value=walk(successor)
            if value is None:return None
            outcomes.append(value)
        return (max if b.turn==origin else min)(outcomes)
    return walk(child)


class ExchangeTargetsTests(unittest.TestCase):
    def test_exact_small_exhaustive_and_order_invariance(self):
        fixtures=[
            ('4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1','e4d5'),
            ('4k3/8/3r4/3p4/3R4/8/8/4K3 w - - 0 1','d4d5'),
            ('4k3/8/3r4/3p4/3R4/8/8/3RK3 w - - 0 1','d4d5'),
            ('4k3/4n3/8/3p4/2P5/8/8/4R1K1 w - - 0 1','c4d5'),
            ('8/8/2k5/3p4/3R4/8/8/4K2B w - - 0 1','d4d5'),
            ('rr6/P3k3/8/8/8/8/8/7K w - - 0 1','a7b8q')]
        for fen,uci in fixtures:
            for mirror in (False,True):
                board=chess.Board(fen)
                move=chess.Move.from_uci(uci)
                if mirror:
                    board=board.mirror()
                    move=chess.Move(move.from_square^56,move.to_square^56,promotion=move.promotion)
                before=board.fen(en_passant='fen')
                result=exchange_target(board,move)
                self.assertEqual(result['target_cp'],exhaustive(board,move))
                self.assertTrue(result['exact'])
                self.assertEqual(result['target_cp'],exchange_target(board,move,reverse_order=True)['target_cp'])
                self.assertEqual(board.fen(en_passant='fen'),before)
                for node in result['trace']:
                    self.assertTrue(node['closed'])
                    self.assertEqual(node['material_return_current_cp'],node['material_return_parent_cp']*(1 if node['mover_is_parent'] else -1))

    def test_recapture_loss_and_xray_backup(self):
        board=chess.Board('4k3/8/3r4/3p4/3R4/8/8/4K3 w - - 0 1')
        result=exchange_target(board,chess.Move.from_uci('d4d5'))
        self.assertEqual(result['simple_material_delta_cp'],100)
        self.assertEqual(result['target_cp'],-400)
        self.assertEqual(result['correction_vs_immediate_cp'],-500)
        board=chess.Board('4k3/8/3r4/3p4/3R4/8/8/3RK3 w - - 0 1')
        result=exchange_target(board,chess.Move.from_uci('d4d5'))
        self.assertEqual(result['target_cp'],100)
        self.assertEqual(result['dependency_rounds'],3)
        self.assertEqual(len(result['trace']),3)
        self.assertEqual(result['trace'][0]['selected_option'],'decline')  # canonical tie policy

    def test_pin_and_king_legality_not_geometric_see(self):
        for fen,uci,illegal in [
            ('4k3/4n3/8/3p4/2P5/8/8/4R1K1 w - - 0 1','c4d5','e7d5'),
            ('8/8/2k5/3p4/3R4/8/8/4K2B w - - 0 1','d4d5','c6d5')]:
            board=chess.Board(fen)
            first=chess.Move.from_uci(uci)
            result=exchange_target(board,first)
            board.push(first)
            self.assertNotIn(chess.Move.from_uci(illegal),board.legal_moves)
            self.assertEqual(result['trace'][0]['recapture_action_ids_parent'],[])
            self.assertEqual(result['target_cp'],100)

    def test_EP_victim_and_all_underpromotions(self):
        board=chess.Board('4k3/2p5/8/3pP3/8/8/8/4K3 w - d6 0 1')
        result=exchange_target(board,chess.Move.from_uci('e5d6'))
        self.assertEqual(result['victim_square'],chess.D5)
        self.assertEqual(result['occupied_destination'],chess.D6)
        self.assertEqual(result['target_cp'],0)
        self.assertEqual([x[0] for x in result['initial_edits']],[chess.D5,chess.E5,chess.D6])
        board=chess.Board('rr6/P3k3/8/8/8/8/8/7K w - - 0 1')
        actions=[m for m in complete_initial_actions(board) if m.to_square==chess.B8]
        self.assertEqual({m.promotion for m in actions},{chess.KNIGHT,chess.BISHOP,chess.ROOK,chess.QUEEN})
        for move in actions:self.assertEqual(exchange_target(board,move)['target_cp'],400)

    def test_required_external_evasion_and_real_terminal_no_point(self):
        board=chess.Board('4k3/8/8/8/8/8/4r3/4K3 w - - 0 1')
        result=exchange_target(board,chess.Move.from_uci('e1e2'))
        self.assertEqual(result['missing_reason'],'typed_terminal')
        self.assertEqual(result['missing_detail']['terminal'],'insufficient_material')
        board=chess.Board('7k/P7/8/8/8/8/8/7K w - - 0 1')
        result=exchange_target(board,chess.Move.from_uci('a7a8q'))
        self.assertEqual(result['missing_reason'],'unsupported_check_evasion')
        self.assertTrue(result['missing_detail']['outside_action_ids_current'])
        self.assertIsNone(result['target_cp'])
        self.assertTrue(all('material_return_parent_cp' not in node for node in result['trace']))

    def test_check_with_only_recapture_forbids_decline_and_mate_is_typed(self):
        board=chess.Board('5Rnk/6pp/8/8/8/8/8/5K2 w - - 0 1')
        result=exchange_target(board,chess.Move.from_uci('f8g8'))
        self.assertTrue(result['exact'])
        self.assertEqual(result['target_cp'],-180)
        self.assertTrue(result['trace'][0]['checked'])
        self.assertFalse(result['trace'][0]['decline_allowed'])
        self.assertEqual(result['trace'][0]['selected_option'],'recapture')
        self.assertEqual(result['target_cp'],exhaustive(board,chess.Move.from_uci('f8g8')))
        board=chess.Board('5Rnk/6pp/4B3/8/8/8/8/5K2 w - - 0 1')
        result=exchange_target(board,chess.Move.from_uci('f8g8'))
        self.assertIsNone(result['target_cp'])
        self.assertEqual(result['missing_detail']['terminal'],'checkmate')
        self.assertEqual(result['missing_detail']['result_parent'],1)

    def test_all_caps_remove_point_labels_and_preserve_board(self):
        board=chess.Board('4k3/8/2p5/3p4/4P3/8/8/4K3 w - - 0 1')
        before=board.fen()
        move=chess.Move.from_uci('e4d5')
        for kwargs,reason in [({'limits':Limits(max_plies=1)},'ply_cap'),
                               ({'limits':Limits(nodes_per_action=1)},'node_cap'),
                               ({'deadline':time.monotonic()-1},'wall_cap')]:
            result=exchange_target(board,move,**kwargs)
            self.assertFalse(result['exact'])
            self.assertIsNone(result['target_cp'])
            self.assertEqual(result['missing_reason'],reason)
            self.assertTrue(all('closed' not in node for node in result['trace']))
            self.assertEqual(board.fen(),before)
        with self.assertRaises(ValueError):exchange_target(chess.Board(),chess.Move.from_uci('e2e4'))

    def test_complete_family_train_pilot_tiny_fixture(self):
        moves=['e2e4','d7d5']
        board=chess.Board()
        for uci in moves:board.push_uci(uci)
        from neural.tactical_inputs import action_id
        root=dict(ply=2,fen=board.fen(en_passant='fen'),mover='white',motifs=['capture'],
                  actions=[dict(id=action_id(m,board.turn)) for m in sorted(board.legal_moves,key=lambda m:action_id(m,board.turn))])
        family=dict(schema='chen_reachable_mechanics_v1',family_id='fixture',split='train',outcome_label=None,moves=moves,roots=[root])
        with tempfile.TemporaryDirectory() as tmp:
            shard=Path(tmp)/'in.gz'
            with gzip.open(shard,'wt') as stream:stream.write(json.dumps(family)+'\n')
            report=generate(shard,Path(tmp)/'out')
            self.assertEqual(report['counts']['initial_actions'],1)
            self.assertEqual(report['counts']['closed_exact'],1)
            with gzip.open(Path(tmp)/'out/targets.jsonl.gz','rt') as stream:record=json.loads(next(stream))
            self.assertEqual(record['family_id'],'fixture')
            self.assertEqual(record['roots'][0]['complete_initial_action_ids'],[action_id(chess.Move.from_uci('e4d5'),True)])
            capped=generate(shard,Path(tmp)/'capped',limits=Limits(global_nodes=1))
            self.assertEqual(capped['counts']['entered_states'],1)
            self.assertEqual(capped['missing'],{'global_node_cap':1})
            family['split']='test'
            with gzip.open(shard,'wt') as stream:stream.write(json.dumps(family)+'\n')
            with self.assertRaises(ValueError):generate(shard,Path(tmp)/'bad')


if __name__=='__main__':unittest.main()
