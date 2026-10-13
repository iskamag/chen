"""Protocol tests only: fabricated UCI fixture, no real Stockfish searches."""
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from neural.semantic_data import ARCHIVE_SHA256, PGN_SHA256, atomic_json, sha256_file
from neural.semantic_study import (SOURCES, LabelBudget, make_declaration, produce_labels,
                                  query_plan, read_admission, validate_declaration)
from neural.tests.test_semantic_data import position

FAKE = '''import sys, chess, time
board=chess.Board(); width=1
for raw in sys.stdin:
 line=raw.strip()
 if line=='uci': print('id name Stockfish 19 fixture'); print('uciok',flush=True)
 elif line=='isready': print('readyok',flush=True)
 elif line.startswith('setoption name MultiPV'): width=int(line.split()[-1])
 elif line.startswith('position startpos'):
  board=chess.Board()
  if ' moves ' in line:
   for m in line.split(' moves ')[1].split(): board.push_uci(m)
 elif line.startswith('go '):
  if MODE=='hang': time.sleep(10)
  else:
   moves=sorted(m.uci() for m in board.legal_moves)
   for rank,m in enumerate(moves[:width],1):
    print(f'info depth 7 seldepth 9 multipv {rank} score cp {100-rank} nodes 16 pv {m}')
   print('bestmove '+moves[0],flush=True)
 elif line=='quit': break
'''


def fixture(directory, mode='ok'):
    root=Path(directory); repo=root/'repo'; repo.mkdir()
    for name in SOURCES:
        file=repo/name; file.parent.mkdir(parents=True,exist_ok=True); file.write_text('fixture source only\n')
    binary=root/'fake-sf'
    binary.write_text('#!'+sys.executable+'\nMODE='+repr(mode)+'\n'+FAKE); binary.chmod(0o755)
    observed=asdict(position())
    sources={name:[dict(game_id=name, positions=[dict(observed,game_id=name)],action_root_index=0)]
             for name in ('train','validation','test')}
    admission=dict(kind='fresh_licensed_real_game_family_admission',teacher_labels_generated=False,
                   cross_split_overlap=0,spent_input_overlap=0,pgn_sha256=PGN_SHA256,
                   source=dict(archive_sha256=ARCHIVE_SHA256),families={name:1 for name in sources},
                   source_records=sources,source_sha256=hashlib.sha256(json.dumps(sources,sort_keys=True).encode()).hexdigest(),
                   spent_reports=[])
    admission_path=root/'admission.json'; atomic_json(admission_path,admission)
    return repo,binary,admission_path


class ProtocolTests(unittest.TestCase):
    def setUp(self):
        # Isolated fixture-only hash admission. Real producer has no fallback.
        self.binary_patch = None

    def pin_fixture(self, binary):
        self.binary_patch = patch('neural.semantic_study.STOCKFISH_SHA256', sha256_file(binary))
        self.binary_patch.start(); self.addCleanup(self.binary_patch.stop)

    def test_cold_validation_pins_sources_games_binary_and_budget(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,binary,admission=fixture(directory)
            with self.assertRaisesRegex(ValueError,'official'):
                make_declaration(admission,binary,repo=repo)
            self.pin_fixture(binary)
            declaration=make_declaration(admission,binary,repo=repo)
            self.assertEqual(declaration['workload']['queries'],6)
            self.assertEqual(declaration['workload']['planned_action_rows'],60)
            self.assertEqual(declaration['workload']['requested_nodes'],3*(32000+256000))
            self.assertFalse(declaration['model_admission']['training_authorized'])
            seal=Path(directory)/'decl.json'; atomic_json(seal,declaration)
            with self.assertRaisesRegex(ValueError,'sealed'):
                validate_declaration(seal,admission,binary,repo=repo)
            declaration['status']='sealed_for_bounded_labels'; atomic_json(seal,declaration)
            validate_declaration(seal,admission,binary,repo=repo)
            original=(repo/SOURCES[0]).read_text(); (repo/SOURCES[0]).write_text(original+'changed')
            with self.assertRaisesRegex(ValueError,'differs'): validate_declaration(seal,admission,binary,repo=repo)
            (repo/SOURCES[0]).write_text(original)
            altered=json.loads(admission.read_text()); altered['source_records']['train'][0]['positions'][0]['fen']='bad'
            atomic_json(admission,altered)
            with self.assertRaisesRegex(ValueError,'fingerprint'): validate_declaration(seal,admission,binary,repo=repo)
        with self.assertRaises(ValueError): LabelBudget(workers=3).validate()
        with self.assertRaises(ValueError): LabelBudget(wall_seconds=True).validate()
        with self.assertRaises(ValueError): LabelBudget(standalone_nodes=32001).validate()

    def test_gzip_admission_and_whole_family_complete_or_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,binary,admission=fixture(directory)
            self.pin_fixture(binary)
            compressed=admission.with_suffix('.json.gz')
            compressed.write_bytes(gzip.compress(admission.read_bytes(),mtime=0))
            self.assertEqual(read_admission(admission),read_admission(compressed))
            budget=LabelBudget(standalone_nodes=16,all_action_nodes=16,wall_seconds=10,request_seconds=2)
            declaration=make_declaration(compressed,binary,repo=repo,budget=budget)
            declaration['status']='sealed_for_bounded_labels'
            seal=Path(directory)/'decl.json'; atomic_json(seal,declaration)
            out=Path(directory)/'labels'
            result=produce_labels(seal,compressed,binary,out,repo=repo)
            self.assertEqual(result['status'],'complete')
            self.assertEqual(result['completed_queries'],6)
            self.assertEqual(result['worker_exitcodes'],[0,0])
            self.assertEqual(result['whole_family_accepted'],{'train':['train'],'validation':['validation'],'test':['test']})
            rows=[json.loads(p.read_text()) for p in out.glob('*.json') if len(p.stem)==64]
            self.assertEqual(len(rows),6)
            self.assertTrue(all(row['labels']['score_frame']['reference_material']==58 for row in rows))
            with self.assertRaisesRegex(ValueError,'empty'): produce_labels(seal,compressed,binary,out,repo=repo)

    def test_whole_job_deadline_closes_partial_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,binary,admission=fixture(directory,'hang')
            self.pin_fixture(binary)
            budget=LabelBudget(standalone_nodes=16,all_action_nodes=16,wall_seconds=1,request_seconds=2)
            declaration=make_declaration(admission,binary,repo=repo,budget=budget)
            declaration['status']='sealed_for_bounded_labels'
            seal=Path(directory)/'decl.json'; atomic_json(seal,declaration)
            result=produce_labels(seal,admission,binary,Path(directory)/'labels',repo=repo)
            self.assertEqual(result['status'],'partial_closed_wall_ceiling')
            self.assertEqual(result['completed_queries'],0)
            self.assertTrue(all(code is not None for code in result['worker_exitcodes']))
            self.assertTrue(all(not rows for rows in result['whole_family_accepted'].values()))
            self.assertLess(result['wall_seconds'],8)


if __name__=='__main__': unittest.main()
