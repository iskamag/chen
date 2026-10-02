"""Fabricated packets only; no Stockfish search or learned score evaluation."""
import unittest
import chess
from neural.ordinary_packets import select_packet, parse_pv
from neural.semantic_data import common_cp_labels
from neural.tests.test_semantic_data import packet, position

class PacketTests(unittest.TestCase):
    def select(self,lines,all_legal=True):
        return select_packet(lines,chess.Board(),all_legal=all_legal,min_depth=6)

    def test_bound_final_is_ineligible_not_latest_target(self):
        first=packet(position(),depth=7)
        last=[s.replace('score cp','score cp').replace(' nodes',' upperbound nodes') for s in packet(position(),depth=8)]
        rows,meta,error=self.select(first+last+['bestmove h2h4'])
        self.assertIsNone(error); self.assertEqual(rows[0].depth,7)
        self.assertEqual(meta['last_emitted_packet_classification'],'bounded')
        self.assertEqual(meta['selected_emission_ordinal'],1)
        self.assertTrue(meta['final_bestmove_differs_from_selected_PV1'])
        self.assertEqual(meta['selected_line_end'],19)

    def test_highest_depth_then_latest_equal_depth_and_ties(self):
        a=packet(position(),depth=8);b=packet(position(),depth=7,score=800)
        rows,meta,error=self.select(a+b)
        self.assertIsNone(error);self.assertEqual(meta['selected_emission_ordinal'],1)
        c=packet(position(),depth=8,score=800)
        rows,meta,error=self.select(a+b+c)
        self.assertIsNone(error);self.assertEqual(meta['selected_emission_ordinal'],3)
        tied=[line.replace(f'score cp {799-i}', 'score cp 300') for i,line in enumerate(c)]
        rows,meta,error=self.select(tied)
        self.assertIsNone(error);self.assertEqual(len(set(r.score for r in rows)),1)

    def test_incomplete_mixed_depth_emissions_never_splice(self):
        a=packet(position());b=packet(position(),depth=8)
        mixed=b[:];mixed[-1]=mixed[-1].replace('depth 8','depth 7')
        rows,meta,error=self.select(a+b[:3]+mixed)
        self.assertIsNone(error);self.assertEqual(meta['selected_emission_ordinal'],1)
        self.assertEqual([e['classification'] for e in meta['emissions']],['eligible','incomplete','mixed_depth'])
        _,_,error=self.select(a[:3]+a[3:])
        self.assertIsNone(error) # uninterrupted complete emission
        _,_,error=self.select(a[:3]+b[:1]+a[3:])
        self.assertEqual(error,'malformed_emission_ranks')

    def test_no_fallback_around_corruption(self):
        a=packet(position());bad=a[:];bad[-1]=bad[-1].rsplit(' pv ',1)[0]+' pv '+a[0].split(' pv ')[1]
        self.assertEqual(self.select(a+bad)[2],'nonunique_or_incomplete_legal_actions')
        for malformed in [a+[a[-1]], a[:2]+[a[1]]+a[2:], a[:2]+a[3:]]:
            self.assertEqual(self.select(malformed)[2],'malformed_emission_ranks')
        bad=a[:];bad[-1]=bad[-1].replace('nodes 32000','nodes 32001')
        self.assertEqual(self.select(a+bad)[2],'inconsistent_emission_nodes')
        bad=a[:];bad[-1]=bad[-1].replace('score cp 180','score cp 900')
        self.assertEqual(self.select(a+bad)[2],'nonmonotonic_common_rank_scores')
        bad=a[:];bad[-1]+=' a1a8'
        self.assertEqual(self.select(a+bad)[2],'illegal_packet_PV')

    def test_any_decisive_line_rejects_even_incomplete_or_bounded(self):
        a=packet(position())
        for suffix in ['score mate 3','score cp 20000 upperbound','score cp 100 tbhits 1']:
            line='info depth 9 seldepth 11 multipv 1 '+suffix+('' if 'tbhits' in suffix else ' tbhits 0')+' nodes 33000 pv a2a3'
            self.assertEqual(self.select(a+[line])[2],'decisive_mate_tablebase_or_extreme_evidence')
        self.assertEqual(self.select(packet(position(),depth=5))[2],'no_eligible_ordinary_complete_packet')

    def test_strict_scored_grammar_and_final_bestmove(self):
        for line in ['info depth 7 score cp 2 nodes 3', 'info depth 7 score cp x nodes 3 pv a2a3',
                     'info depth 7 depth 8 score cp 2 nodes 3 pv a2a3',
                     'info depth 7 score cp 2 lowerbound lowerbound nodes 3 pv a2a3']:
            with self.assertRaises(ValueError):parse_pv(line)
            self.assertEqual(self.select(packet(position())+[line])[2],'malformed_scored_grammar')
        self.assertIsNone(parse_pv('info string score cp 2 pv a2a3'))
        self.assertEqual(self.select(packet(position())+['bestmove a1a8'])[2],'illegal_final_bestmove')
        self.assertEqual(self.select(['bestmove a2a3']+packet(position()))[2],'scored_output_after_bestmove')

    def test_mandatory_fields_and_rejected_query_trace(self):
        complete=packet(position())
        for key in ('depth','seldepth','multipv','nodes','tbhits'):
            tokens=complete[0].split(); i=tokens.index(key); del tokens[i:i+2]
            with self.assertRaises(ValueError):parse_pv(' '.join(tokens))
        for suffix in ('depth 9','score cp 2','multipv 1','tbhits 0'):
            malformed=complete[0].replace(' pv ', ' '+suffix+' pv ')
            with self.assertRaises(ValueError):parse_pv(malformed)
        bad=complete+[complete[0].replace('score cp 199','score mate 2')]+['bestmove h2h4']
        _,meta,error=self.select(bad)
        self.assertEqual(error,'decisive_mate_tablebase_or_extreme_evidence')
        self.assertEqual(meta['final_bestmove'],'h2h4')
        self.assertEqual(meta['emission_trace'][-1]['ordinal'],2)
        self.assertEqual(meta['emission_trace'][-1]['classification'],'decisive')
        diagnostics={}; label,error=common_cp_labels(bad,position(),all_legal=True,diagnostics=diagnostics)
        self.assertIsNone(label);self.assertEqual(diagnostics['query_rejection'],error)
        self.assertEqual(diagnostics['final_bestmove'],'h2h4')

    def test_label_provenance_and_parent_sign(self):
        lines=packet(position())+['bestmove h2h4']
        label,error=common_cp_labels(lines,position(),all_legal=True)
        self.assertIsNone(error);self.assertEqual(label['packet_provenance']['protocol_version'],2)
        self.assertTrue(label['packet_provenance']['final_bestmove_differs_from_selected_PV1'])
        self.assertTrue(label['rows'][0]['converted_score']['negated'])

if __name__=='__main__':unittest.main()
