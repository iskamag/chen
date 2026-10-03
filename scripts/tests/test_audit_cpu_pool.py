"""Saved-game/protocol negative fixtures only; no engines or GPU execution."""
import importlib.util
import json
import math
from pathlib import Path
import random
import tempfile
import unittest

import chess
import chess.pgn

spec = importlib.util.spec_from_file_location('cpu_pool_auditor', Path(__file__).parents[1] / 'audit_cpu_pool.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CPUPoolAudit(unittest.TestCase):
    def fixture(self, directory, capped=False, drawn=False):
        root = Path(directory)
        chen, sf, collector = root/'chen', root/'stockfish', root/'match.py'
        chen.write_text('synthetic CPU binary'); sf.write_text('synthetic opponent binary'); collector.write_text('# synthetic collector')
        start = chess.STARTING_FEN
        mate = '7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'
        draw = '4k3/8/8/8/8/8/8/R3K3 w - - 98 1'
        opening = root/'openings.txt'
        opening.write_text(''.join(f'fixture{i} | {start if capped and i == 0 else draw if drawn and i == 0 else mate}\n' for i in range(8)))
        declaration = dict(kind=module.KIND, source_commit='28c6f86caa57e360798afc38e9f50a33aa1729ed',
            requested_seconds_per_move=1.0, max_plies=512, rounds=1, opening_pairs=8,
            authored_pairs_independent=False, scope='synthetic audit fixture only', opening_path=str(opening),
            opening_sha256=module.sha(opening), engine_sha256=module.sha(chen), stockfish_sha256=module.sha(sf),
            match_script_sha256=module.sha(collector),
            source_and_binary_sha256={str(p):module.sha(p) for p in (chen,sf,collector,opening)},lanes=[])
        cap_moves = self.long_quiet_game() if capped else None
        for lane_id,pvs,elo,cpu in module.LANES:
            lane = root/lane_id; run = lane/'run'; run.mkdir(parents=True)
            common = dict(Backend='alphabeta', Eval='tapered', Level=20, Hash=16,
                UseTT=True, Quiescence=True, MoveOrdering=True, AspirationWindow=0, PVS=pvs)
            config = dict(candidate=dict(command=[str(chen)],options=common,artifacts=dict(engine=str(chen))),
                base=dict(command=[str(sf)],options=dict(Threads=1,Hash=16,NumaPolicy='none',
                    UCI_LimitStrength=True,UCI_Elo=elo,SyzygyProbeLimit=0),artifacts=dict(engine=str(sf))))
            config_path=lane/'config.json'; config_path.write_text(json.dumps(config))
            entry=dict(id=lane_id,run=str(run),pvs=pvs,stockfish_elo=elo,cpu=cpu,
                config_path=str(config_path),config_sha256=module.sha(config_path)); declaration['lanes'].append(entry)
            metadata=dict(config=config,concurrency=1,chess_version=chess.__version__,
                opening_sha256=module.sha(opening),match_script_sha256=module.sha(collector),
                launcher_sha256=dict(base=module.sha(sf),candidate=module.sha(chen)),
                declared_artifact_sha256=dict(base=dict(engine=module.sha(sf)),candidate=dict(engine=module.sha(chen))),
                artifact_scope=dict(base='declared-files',candidate='declared-files'),
                inherited_runtime_environment={},budgets=dict(movetime=1.0,depth=None,nodes=None,
                    rounds=1,max_plies=512,independent_opening_pairs=False,
                    openings=str(opening),config=str(config_path),out=str(run)))
            (run/'metadata.json').write_text(json.dumps(metadata));rows=[];games=[]
            for index in range(16):
                is_start = capped and index//2==0
                is_draw = drawn and index//2==0
                board=chess.Board(start if is_start else draw if is_draw else mate); game=chess.pgn.Game();game.setup(board)
                color=chess.WHITE if index%2==0 else chess.BLACK
                game.headers.update(Opening=f'fixture{index//2}',White='candidate' if color else 'base',Black='base' if color else 'candidate')
                cap = capped and lane_id=='off1800' and index==0
                moves=cap_moves if cap else ['f2f3','e7e5','g2g4','d8h4'] if is_start else ['a1a2'] if is_draw else ['f7g7']
                node=game;trace=[]
                for ply,uci in enumerate(moves):
                    move=chess.Move.from_uci(uci)
                    trace.append(dict(ply=ply,engine='candidate' if board.turn==color else 'base',fen=board.fen(en_passant='fen'),
                        move=uci,wall_ns=1000000000,reported=dict(depth=1,nodes=1,time=1.0)))
                    self.assertIn(move,board.legal_moves);board.push(move);node=node.add_variation(move)
                outcome=board.outcome(claim_draw=True)
                result='*' if cap else outcome.result();reason='ply_cap_inconclusive' if cap else outcome.termination.name
                score=None if cap else .5 if outcome.winner is None else float(outcome.winner==color)
                game.headers.update(Result=result,Termination=reason);games.append(str(game))
                rows.append(dict(opening=f'fixture{index//2}',candidate_color='white' if color else 'black',round=0,
                    failure=None,result=result,candidate_score=score,termination=reason,plies=len(trace),move_trace=trace,final_fen=board.fen(en_passant='fen')))
            (run/'games.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows));(run/'games.pgn').write_text('\n\n'.join(games)+'\n')
            pairs=[tuple(r['candidate_score'] for r in rows[i:i+2]) for i in range(0,16,2)
                if all(r['candidate_score'] is not None for r in rows[i:i+2])]
            scores=[s for pair in pairs for s in pair];mean=sum(scores)/len(scores)
            bins=[sum(round(2*sum(pair))==i for pair in pairs) for i in range(5)]
            summary=dict(completed_pairs=len(pairs),incomplete_pairs=8-len(pairs),technical_failures=0,
                games=len(scores),candidate_score=mean,pentanomial=bins,
                logistic_relative_elo_point_estimate=400*math.log10(mean/(1-mean)),
                paired_bootstrap_score_interval_95_diagnostic=None,independent_opening_pairs_asserted=False)
            (run/'summary.json').write_text(json.dumps(summary))
        path=root/'declaration.json';path.write_text(json.dumps(declaration));return path

    def long_quiet_game(self):
        # Irreversible pawn moves reset the clock. Quiet moves avoid repetition
        # and preserve material, allowing a REAL nonterminal512-ply cap witness.
        for attempt in range(16):
            rng=random.Random(53+attempt);board=chess.Board();moves=[]
            for _ in range(512):
                legal=list(board.legal_moves);rng.shuffle(legal)
                legal.sort(key=lambda m:(board.is_capture(m),
                    int(board.piece_type_at(m.from_square)==chess.PAWN) != (board.halfmove_clock>=80)))
                for move in legal:
                    board.push(move)
                    if board.outcome(claim_draw=True) is None:
                        moves.append(move.uci());break
                    board.pop()
                else:break
            if len(moves)==512:
                self.assertIsNone(board.outcome(claim_draw=True));return moves
        self.fail('bounded cap fixture could not find a legal nonterminal continuation')

    def edit_rows(self,path,lane,edit):
        target=path.parent/lane/'run/games.jsonl';rows=[json.loads(s) for s in target.read_text().splitlines()]
        edit(rows);target.write_text(''.join(json.dumps(r)+'\n' for r in rows))

    def test_complete_pool_keeps_default_off_separate(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory);result=module.audit(path)
            self.assertEqual(result['status'],'COMPLETE')
            self.assertEqual(result['pvs_on_configured_pool_fit']['included_games'],48)
            self.assertAlmostEqual(result['pvs_on_configured_pool_fit']['configured_setting_logistic_point_estimate'],1800.)
            self.assertEqual(result['default_pvs_off_1800_fit']['included_games'],16)
            self.assertIsNone(result['lichess_rating']);self.assertIsNone(result['confidence_interval'])
            self.assertFalse(result['independence_asserted']);self.assertIsNone(result['architecture_winner'])

    def test_real_cap_excludes_whole_pair_and_exposes_finished_partner(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory,capped=True);result=module.audit(path);off=result['lanes'][0]
            self.assertEqual((off['included_games'],off['capped_unscored_games'],off['unpaired_finished_games']),(14,1,1))
            self.assertEqual(off['unpaired_finished_game_indices'],[1])
            self.assertEqual(result['pvs_on_configured_pool_fit']['included_games'],48)
            self.edit_rows(path,'off1800',lambda rows:rows[0].update(candidate_score=.5))
            with self.assertRaisesRegex(ValueError,'result/score'):module.audit(path)

    def test_rule_draw_is_a_real_half_point_not_cap_imputation(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory,drawn=True);result=module.audit(path)
            self.assertEqual(result['lanes'][0]['draws'],2)
            self.assertEqual(result['lanes'][0]['complete_pairs'][0]['scores'],[.5,.5])
            self.assertEqual(result['lanes'][0]['capped_unscored_games'],0)

    def test_missing_tail_requires_explicit_incomplete_state(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory);run=path.parent/'on2000/run'
            rows=(run/'games.jsonl').read_text().splitlines();(run/'games.jsonl').write_text('\n'.join(rows[:3])+'\n')
            parsed=[]
            with (run/'games.pgn').open() as stream:
                while (game:=chess.pgn.read_game(stream)) is not None:parsed.append(str(game))
            (run/'games.pgn').write_text('\n\n'.join(parsed[:3])+'\n');(run/'summary.json').unlink()
            with self.assertRaisesRegex(ValueError,'exact16'):module.audit(path)
            result=module.audit(path,['on2000']);lane=result['lanes'][3]
            self.assertEqual(result['status'],'INCOMPLETE');self.assertEqual(lane['missing_games'],13)
            self.assertEqual(lane['included_games'],2);self.assertEqual(lane['unpaired_finished_games'],1)
            self.assertEqual(result['pvs_on_configured_pool_fit']['included_games'],34)

    def test_failure_prefix_is_unscored_and_never_continued(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory);run=path.parent/'on2000/run'
            board=chess.Board('7k/5Q2/6K1/8/8/8/8/8 w - - 0 1');game=chess.pgn.Game();game.setup(board)
            game.headers.update(Opening='fixture0',White='candidate',Black='base',Result='*',Termination='technical_failure')
            row=dict(opening='fixture0',candidate_color='white',round=0,failure='RuntimeError: synthetic engine stop',
                result='*',candidate_score=None,termination='technical_failure',plies=0,move_trace=[],final_fen=board.fen(en_passant='fen'))
            (run/'games.jsonl').write_text(json.dumps(row)+'\n');(run/'games.pgn').write_text(str(game)+'\n');(run/'summary.json').unlink()
            with self.assertRaises(ValueError):module.audit(path)
            result=module.audit(path,['on2000']);self.assertEqual(result['lanes'][3]['technical_failures'],1)
            self.assertEqual(result['lanes'][3]['included_games'],0)
            self.edit_rows(path,'on2000',lambda rows:rows[0].update(candidate_score=0.))
            with self.assertRaisesRegex(ValueError,'result/score'):module.audit(path,['on2000'])

    def test_explicit_start_failure_without_game_files_has_no_points(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory);run=path.parent/'on2000/run'
            for name in ('games.pgn','games.jsonl','summary.json'):(run/name).unlink()
            with self.assertRaisesRegex(ValueError,'missing saved game'):module.audit(path)
            lane=module.audit(path,['on2000'])['lanes'][3]
            self.assertEqual((lane['observed_games'],lane['missing_games'],lane['included_games']),(0,16,0))
            self.assertIsNone(lane['artifact_sha256']['games.pgn'])

    def test_changed_artifact_and_changed_PVS_option_reject(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory);(path.parent/'chen').write_text('changed binary')
            with self.assertRaisesRegex(ValueError,'hash changed'):module.audit(path)
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory);d=module.read_json(path);entry=d['lanes'][1];config=module.read_json(entry['config_path'])
            config['candidate']['options']['PVS']=False;Path(entry['config_path']).write_text(json.dumps(config));entry['config_sha256']=module.sha(entry['config_path'])
            path.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'candidate policy'):module.audit(path)

    def test_valid_commit_metadata_never_bypasses_frozen_source_pins(self):
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory);d=module.read_json(path);d['source_commit']='a'*40;path.write_text(json.dumps(d))
            self.assertEqual(module.audit(path)['source_commit'],'a'*40)
            (path.parent/'match.py').write_text('# altered source despite valid commit metadata')
            with self.assertRaisesRegex(ValueError,'source/artifact hash changed'):module.audit(path)
        with tempfile.TemporaryDirectory() as directory:
            path=self.fixture(directory);d=module.read_json(path);d['source_commit']='28c6f86-not-a-full-sha';path.write_text(json.dumps(d))
            with self.assertRaisesRegex(ValueError,'declaration'):module.audit(path)

    def test_wrong_trace_timing_and_pair_color_reject(self):
        for edit,message in ((lambda rows:rows[0]['move_trace'][0].update(fen=chess.STARTING_FEN),'trace move'),
                             (lambda rows:rows[0]['move_trace'][0].update(wall_ns=-1),'wall timing'),
                             (lambda rows:rows[0]['move_trace'][0]['reported'].update(time=float('nan')),'UCI telemetry'),
                             (lambda rows:rows[1].update(candidate_color='white'),'pair order')):
            with self.subTest(message=message),tempfile.TemporaryDirectory() as directory:
                path=self.fixture(directory);self.edit_rows(path,'on1800',edit)
                with self.assertRaisesRegex(ValueError,message):module.audit(path)

    def test_false_summary_and_false_interval_reject(self):
        for edit,message in ((lambda s:s.update(candidate_score=.75),'points/pentanomial'),
                             (lambda s:s.update(paired_bootstrap_score_interval_95_diagnostic=[.1,.9]),'independence/interval')):
            with tempfile.TemporaryDirectory() as directory:
                path=self.fixture(directory);target=path.parent/'off1800/run/summary.json';s=module.read_json(target);edit(s);target.write_text(json.dumps(s))
                with self.assertRaisesRegex(ValueError,message):module.audit(path)

    def test_logistic_fit_equation_and_no_fabricated_boundary_estimates(self):
        groups=[dict(setting=setting,games=16,points=16/(1+10**((setting-1875)/400))) for setting in (1600,1800,2000)]
        result=module.configured_fit(groups);self.assertAlmostEqual(result['configured_setting_logistic_point_estimate'],1875.,places=8)
        self.assertLess(abs(result['score_equation_residual']),1e-10)
        single=module.configured_fit([dict(setting=1800,games=16,points=14)])
        self.assertAlmostEqual(single['configured_setting_logistic_point_estimate'],1800+400*math.log10(7.))
        for points,direction in ((0.,'negative_infinity'),(16.,'positive_infinity')):
            boundary=module.configured_fit([dict(setting=1800,games=16,points=points)])
            self.assertIsNone(boundary['configured_setting_logistic_point_estimate']);self.assertEqual(boundary['direction'],direction)
        self.assertEqual(module.configured_fit([])['status'],'no_complete_paired_games')


if __name__=='__main__':unittest.main()
