"""Protocol tests only: fabricated UCI fixture, no real Stockfish searches."""
from dataclasses import asdict
import gzip
import ctypes
import hashlib
import json
from pathlib import Path
import os
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from neural.semantic_data import ARCHIVE_SHA256, PGN_SHA256, atomic_json, sha256_file
from neural.semantic_study import (SOURCES, LabelBudget, _OwnedSubreaper, make_declaration, produce_labels,
                                  query_plan, read_admission, validate_declaration)
from neural.tests.test_semantic_data import position

FAKE = '''import sys, chess, time, os, signal
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
  elif MODE=='orphan': os.kill(os.getppid(),signal.SIGKILL); time.sleep(10)
  else:
   moves=sorted(m.uci() for m in board.legal_moves)
   for rank,m in enumerate(moves[:width],1):
    print(f'info depth 7 seldepth 9 multipv {rank} score cp {100-rank} tbhits 0 nodes 16 pv {m}')
   print('bestmove '+moves[0],flush=True)
 elif line=='quit':
  if MODE=='close_error': sys.exit(1)
  break
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
            budget=LabelBudget(standalone_nodes=16,all_action_nodes=16,wall_seconds=20,request_seconds=2)
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
            for file in out.glob('worker-*-setup.json'):
                metadata=json.loads(file.read_text()); pid=metadata['engine_pid']
                self.assertEqual(metadata['blocked_cancellation_signals'],[])
                self.assertFalse(Path('/proc',str(pid)).exists(), 'normal engine child was not reaped')
            with self.assertRaisesRegex(ValueError,'empty'): produce_labels(seal,compressed,binary,out,repo=repo)

    def test_whole_job_deadline_closes_partial_attempt(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,binary,admission=fixture(directory,'hang')
            self.pin_fixture(binary)
            budget=LabelBudget(standalone_nodes=16,all_action_nodes=16,wall_seconds=15,request_seconds=20)
            declaration=make_declaration(admission,binary,repo=repo,budget=budget)
            declaration['status']='sealed_for_bounded_labels'
            seal=Path(directory)/'decl.json'; atomic_json(seal,declaration)
            result=produce_labels(seal,admission,binary,Path(directory)/'labels',repo=repo)
            self.assertEqual(result['status'],'partial_closed_wall_ceiling')
            self.assertEqual(result['completed_queries'],0)
            self.assertTrue(all(code is not None for code in result['worker_exitcodes']))
            self.assertTrue(all(not rows for rows in result['whole_family_accepted'].values()))
            self.assertLess(result['wall_seconds'],15)
            self.assertEqual(result['work_budget_seconds'],5)
            setup=list((Path(directory)/'labels').glob('worker-*-setup.json'))
            self.assertEqual(len(setup),2, 'deadline fixture must reach both actual engine children')
            for file in setup:
                metadata=json.loads(file.read_text())
                self.assertFalse(Path('/proc',str(metadata['engine_pid'])).exists(), 'timed-out engine child was not reaped')
                self.assertFalse(Path('/proc',str(metadata['worker_pid'])).exists(), 'worker was not reaped')

    def test_crashed_worker_orphan_reaped_and_subreaper_restored(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,binary,admission=fixture(directory,'orphan')
            self.pin_fixture(binary)
            before=ctypes.c_int(); libc=ctypes.CDLL(None)
            self.assertEqual(libc.prctl(37,ctypes.byref(before),0,0,0),0)
            declaration=make_declaration(admission,binary,repo=repo,
                budget=LabelBudget(standalone_nodes=16,all_action_nodes=16,wall_seconds=20))
            declaration['status']='sealed_for_bounded_labels'
            seal=Path(directory)/'decl.json'; atomic_json(seal,declaration)
            out=Path(directory)/'labels'
            result=produce_labels(seal,admission,binary,out,repo=repo)
            self.assertEqual(result['status'],'partial_closed_worker_failure')
            self.assertEqual(result['worker_exitcodes'],[-9,-9])
            self.assertEqual(len(result['adopted_children_reaped']),2)
            pids={row['pid'] for row in result['adopted_children_reaped']}
            for file in out.glob('worker-*-setup.json'):
                child=json.loads(file.read_text())['engine_pid']
                self.assertIn(child,pids)
                self.assertFalse(Path('/proc',str(child)).exists(),'orphan child remained running or zombie')
            after=ctypes.c_int()
            self.assertEqual(libc.prctl(37,ctypes.byref(after),0,0,0),0)
            self.assertEqual(before.value,after.value)

    def test_parent_sigterm_and_repeat_signals_retire_workers_and_engines(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,binary,admission=fixture(directory,'hang'); self.pin_fixture(binary)
            declaration=make_declaration(admission,binary,repo=repo,
                budget=LabelBudget(standalone_nodes=16,all_action_nodes=16,wall_seconds=30))
            declaration['status']='sealed_for_bounded_labels'
            seal=Path(directory)/'decl.json'; atomic_json(seal,declaration)
            out=Path(directory)/'labels'
            script=("from neural import semantic_study as s; "
                    f"s.STOCKFISH_SHA256={sha256_file(binary)!r}; "
                    f"s.produce_labels({str(seal)!r},{str(admission)!r},{str(binary)!r},{str(out)!r},repo={str(repo)!r})")
            supervisor_reaper=_OwnedSubreaper()
            self.addCleanup(supervisor_reaper.close)
            child=subprocess.Popen([sys.executable,'-c',script],stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                   start_new_session=True)
            setup=[]
            try:
                deadline=time.monotonic()+10
                while len(list(out.glob('worker-*-setup.json'))) != 2 and time.monotonic()<deadline:
                    if child.poll() is not None: self.fail('parent fixture exited before engine readiness')
                    time.sleep(.02)
                setup=[json.loads(p.read_text()) for p in out.glob('worker-*-setup.json')]
                self.assertEqual(len(setup),2)
                time.sleep(.1)  # both fake engines enter their blocked query
                child.send_signal(signal.SIGTERM)
                time.sleep(.1)
                child.send_signal(signal.SIGTERM); child.send_signal(signal.SIGINT)
                transcript,_=child.communicate(timeout=10)
                self.assertNotEqual(child.returncode,0)
                self.assertIn(b'TeacherCancelled',transcript)
                result=json.loads((out/'report.json').read_text())
                self.assertEqual(result['status'],'partial_closed_interruption')
                self.assertEqual(result['adopted_children_reaped'],[], 'graceful workers must reap their own engine children')
                for row in setup:
                    self.assertEqual(row['blocked_cancellation_signals'],[])
                    self.assertFalse(Path('/proc',str(row['worker_pid'])).exists())
                    self.assertFalse(Path('/proc',str(row['engine_pid'])).exists())
                errors=[json.loads(p.read_text()) for p in out.glob('worker-*-error.json')]
                self.assertEqual(len(errors),2)
                self.assertTrue(all(row['type']=='TeacherCancelled' for row in errors), 'TERM handler must actually execute')
            finally:
                # The regression harness owns retirement too: even a broken
                # producer cannot leave fixture grandchildren running/zombie.
                for row in setup:
                    try: os.killpg(row['worker_pid'],signal.SIGKILL)
                    except ProcessLookupError: pass
                if child.poll() is None: child.kill(); child.wait(timeout=5)
                try: os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError: pass
                cleanup_deadline=time.monotonic()+5
                for row in setup:
                    supervisor_reaper.reap_group(row['worker_pid'],cleanup_deadline)
                supervisor_reaper.reap_group(child.pid,cleanup_deadline)
                child.stdout.close()

    def test_all_records_plus_failed_close_never_admits_labels(self):
        with tempfile.TemporaryDirectory() as directory:
            repo,binary,admission=fixture(directory,'close_error'); self.pin_fixture(binary)
            declaration=make_declaration(admission,binary,repo=repo,
                budget=LabelBudget(standalone_nodes=16,all_action_nodes=16,wall_seconds=20))
            declaration['status']='sealed_for_bounded_labels'
            seal=Path(directory)/'decl.json'; atomic_json(seal,declaration)
            result=produce_labels(seal,admission,binary,Path(directory)/'labels',repo=repo)
            self.assertEqual(result['completed_queries'],6)
            self.assertEqual(result['worker_exitcodes'],[1,1])
            self.assertEqual(result['status'],'partial_closed_worker_failure')
            self.assertTrue(all(not rows for rows in result['whole_family_accepted'].values()))
            self.assertEqual(sum(map(len,result['mechanically_complete_families_not_admitted_if_attempt_failed'].values())),3)


if __name__=='__main__': unittest.main()
