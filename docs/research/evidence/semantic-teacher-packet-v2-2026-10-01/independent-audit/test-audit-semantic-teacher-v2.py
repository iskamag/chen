"""Independent auditor negatives: fabricated packets only, no subprocesses."""
import importlib.util
from pathlib import Path
import unittest

import chess
from neural.ordinary_packets import select_packet

spec=importlib.util.spec_from_file_location('independent_auditor',Path(__file__).with_name('audit-semantic-teacher-v2.py'))
auditor=importlib.util.module_from_spec(spec);spec.loader.exec_module(auditor)


def packet(depth=7,bound='',score=400):
    return [f'info depth {depth} seldepth {depth+2} multipv {rank} score cp {score-rank} '
            f'{bound}nodes 32000 tbhits 0 pv {move}'
            for rank,move in enumerate(sorted(m.uci() for m in chess.Board().legal_moves),1)]


class AuditTests(unittest.TestCase):
    def compare(self,lines,all_legal=True):
        board=chess.Board()
        rows,provenance,error=select_packet(lines,board,all_legal=all_legal,min_depth=6)
        independent=auditor.independent_packets(lines,board,all_legal,6)
        self.assertEqual(independent['rejected'],error is not None)
        self.assertEqual(independent['trace'],provenance['emission_trace'])
        self.assertEqual(independent['malformed'],provenance['malformed_scored_line_offsets'])
        self.assertEqual(independent['bestmoves'],provenance['final_bestmove_emissions'])
        if rows is not None:
            depth,ordinal,selected,meta=independent['selected']
            self.assertEqual(depth,provenance['selected_common_depth'])
            self.assertEqual(ordinal,provenance['selected_emission_ordinal'])
            self.assertEqual([r['pv'] for r in selected],[list(r.pv) for r in rows])
            self.assertEqual([r['score'] for r in selected],[r.score for r in rows])
        return independent

    def test_depth_ordinal_bound_mixed_prefix_and_ties(self):
        a=packet(8);b=packet(7);c=packet(8,score=500)
        result=self.compare(a+b+c+packet(9,'upperbound ')+['bestmove h2h4'])
        self.assertEqual(result['selected'][1],3)
        mixed=packet(10);mixed[-1]=mixed[-1].replace('depth 10','depth 9')
        self.compare(a+packet(9)[:3]+mixed+['bestmove h2h4'])
        ties=[line.replace(f'cp {400-rank}','cp 300') for rank,line in enumerate(a,1)]
        self.compare(ties+['bestmove h2h4'])
        self.assertTrue(self.compare(packet(5)+['bestmove h2h4'])['rejected'])

    def test_decisive_and_protocol_corruption_reject_whole_query(self):
        a=packet()
        tail='info depth 9 seldepth 10 multipv 1 score cp 3 nodes 33000 tbhits 0 pv a2a3'
        for decisive in (tail.replace('cp 3','mate 3'),tail.replace('cp 3','cp 20000 upperbound'),
                         tail.replace('tbhits 0','tbhits 1')):
            self.assertTrue(self.compare(a+[decisive,'bestmove h2h4'])['rejected'])
        corruptions=[a+[a[-1]],a[:2]+[a[1]]+a[2:],a[:2]+a[3:],
                     a[:-1]+[a[-1].replace('nodes 32000','nodes 32001')],
                     a[:-1]+[a[-1].replace('cp 380','cp 900')],
                     a[:-1]+[a[-1]+' a1a8'],a[1:]]
        for bad in corruptions:self.assertTrue(self.compare(bad+['bestmove h2h4'])['rejected'])

    def test_all_mandatory_fields_and_final_choice_negatives(self):
        a=packet();words=a[0].split()
        for key in ('depth','seldepth','multipv','score','nodes','tbhits','pv'):
            at=words.index(key);width=3 if key=='score' else 2
            broken=' '.join(words[:at]+words[at+width:])
            if key=='score':
                self.assertIsNone(auditor.independent_line(broken))  # unscored info is not a candidate
            else:
                with self.assertRaises(auditor.PacketSyntax):auditor.independent_line(broken)
            self.compare(a+[broken,'bestmove h2h4'])
        self.assertTrue(self.compare(a+['bestmove a1a8'])['rejected'])
        self.assertTrue(self.compare(['bestmove h2h4']+a)['rejected'])
        self.assertTrue(self.compare(a+['bestmove h2h4','bestmove a2a3'])['rejected'])

    def test_exit_and_wall_admission_do_not_accept_numeric_aliases(self):
        for value in ([False,False],[0,True],[0,None]):
            with self.assertRaises(ValueError):auditor.validate_report_metadata(dict(worker_exitcodes=value,wall_seconds=1))
        for value in (True,float('nan'),float('inf'),-1):
            with self.assertRaises(ValueError):auditor.validate_report_metadata(dict(worker_exitcodes=[0,0],wall_seconds=value))


if __name__=='__main__':unittest.main(verbosity=2)
