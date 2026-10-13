"""Fixed-plan/strict-label fabricated fixtures. Starts no teacher searches."""
import gzip
import json
from pathlib import Path
import tempfile
import unittest

import chess

from neural.synthetic_teacher import (make_plan, ordinary_labels, compare_budgets,
                                      BASE_NODES,CALIBRATION_NODES)
from neural.tactical_inputs import ReplaySpec, compact_from_replay


def root(moves=()):
    state=compact_from_replay(ReplaySpec(moves=tuple(moves)),ply_from_search_root=len(moves))
    return dict(moves=list(moves),fen=state.fen,canonical_action_ids=[a.id for a in state.actions])


def packet(observed,depth=7,nodes=BASE_NODES,bounded=False):
    board=chess.Board(observed['fen'])
    moves=sorted(board.legal_moves,key=lambda m:m.uci())
    return [f'info depth {depth} seldepth {depth+2} multipv {i+1} score cp {100-i}' +
            (' upperbound' if bounded else '') + f' nodes {nodes} tbhits 0 pv {m.uci()}'
            for i,m in enumerate(moves)]


class SyntheticTeacherTests(unittest.TestCase):
    def test_complete_parent_frame_and_bound_last_packet(self):
        for moves in ((),('e2e4',)):
            observed=root(moves)
            rows=packet(observed)
            last=packet(observed,depth=8,bounded=True)
            final=sorted(chess.Board(observed['fen']).legal_moves,key=lambda m:m.uci())[-1].uci()
            labels,meta,reason,evidence=ordinary_labels(rows+last+[f'bestmove {final}'],observed)
            self.assertIsNone(reason)
            self.assertEqual(labels['perspective'],'parent_root_mover')
            self.assertEqual(meta['selected_emission_ordinal'],1)
            self.assertTrue(meta['final_bestmove_differs_from_selected_PV1'])
            self.assertEqual(meta['last_emitted_packet_classification'],'bounded')
            self.assertEqual([r['canonical_action'] for r in labels['rows']],observed['canonical_action_ids'])
            self.assertTrue(all(not r['converted_score']['negated'] for r in labels['rows']))
            self.assertTrue(all(r['converted_score']['value_cp']>0 for r in labels['rows']))
            self.assertIn('bounded_cp',{r['type'] for r in evidence})

    def test_no_splicing_decisive_or_corrupt_packet_fallback(self):
        observed=root()
        rows=packet(observed)
        mixed=packet(observed,depth=8)
        mixed[-1]=mixed[-1].replace('depth 8','depth 7')
        labels,meta,reason,_=ordinary_labels(rows+mixed,observed)
        self.assertIsNone(reason)
        self.assertEqual(labels['common_reported_depth'],7)
        self.assertEqual(meta['last_emitted_packet_classification'],'mixed_depth')
        for extra,expected in (
            ([rows[-1]],'malformed_emission_ranks'),
            ([rows[0].replace('score cp 100','score mate 3')],'decisive_mate_tablebase_or_extreme_evidence'),
            (['info depth 8 score cp nope nodes 2 pv a2a3'],'malformed_scored_grammar')):
            labels,_,reason,evidence=ordinary_labels(rows+extra,observed)
            self.assertIsNone(labels)
            self.assertEqual(reason,expected)
            self.assertTrue(evidence)

    def test_budget_calibration_ties_reversals_and_missing(self):
        labels=lambda values:dict(rows=[dict(canonical_action=a,reported_uci_cp=v) for a,v in values])
        base=labels([(1,20),(2,20),(3,-10)])
        high=labels([(1,0),(2,30),(3,10)])
        result=compare_budgets(base,high)
        self.assertEqual(result['base_choice'],1)
        self.assertEqual(result['high_choice'],2)
        self.assertTrue(result['choice_changed'])
        self.assertEqual(result['pairs'],3)
        self.assertEqual(result['ordering_or_tie_changed'],2)
        self.assertEqual(result['strict_reversal'],1)
        self.assertFalse(compare_budgets(None,high)['comparable'])
        with self.assertRaises(ValueError):compare_budgets(base,labels([(1,20)]))

    def test_plan_preselection_family_balance_order_invariance(self):
        families=[]
        for index in range(8):
            moves=['e2e4','e7e5']
            roots=[]
            for ply in (1,2):
                state=compact_from_replay(ReplaySpec(moves=tuple(moves[:ply])),ply_from_search_root=ply)
                roots.append(dict(ply=ply,mover='white' if state.mover_white else 'black',
                    fen=state.fen,terminal=None,actions=[dict(id=a.id) for a in state.actions],
                    attacks=[f'{x:016x}' for x in state.source_attacks],
                    history_counters=list(state.history_counters),motifs=[]))
            families.append(dict(schema='chen_reachable_mechanics_v1',split='train',outcome_label=None,
                                 family_id=str(index),moves=moves,roots=roots))
        with tempfile.TemporaryDirectory() as tmp:
            write=lambda path,rows:gzip.open(path,'wt').write(''.join(json.dumps(row)+'\n' for row in rows))
            a,b=Path(tmp)/'a.gz',Path(tmp)/'b.gz'
            write(a,families);write(b,list(reversed(families)))
            first=make_plan(a,roots=4,calibration=2)
            other=make_plan(b,roots=4,calibration=2)
            self.assertEqual(first['roots'],other['roots'])
            self.assertEqual(first['calibration_root_ids'],other['calibration_root_ids'])
            self.assertEqual(len({r['family_id'] for r in first['roots']}),4)
            self.assertEqual([r['mover'] for r in first['roots']],['white','black']*2)
            self.assertEqual([j['nodes'] for j in first['jobs']],[BASE_NODES]*4+[CALIBRATION_NODES]*2)
            families[0]['split']='validation';write(a,families)
            with self.assertRaises(ValueError):make_plan(a,roots=4,calibration=2)


if __name__=='__main__':unittest.main()
