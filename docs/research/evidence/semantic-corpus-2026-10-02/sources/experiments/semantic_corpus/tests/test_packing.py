"""CPU synthetic serialization/corruption/oracle fixtures; no real teacher data."""
import copy
import hashlib
import json
import math
from pathlib import Path
import struct
import sys
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import chess
from experiments.semantic_corpus import packing as p


def fixture_provenance(*,split='train',fen=chess.STARTING_FEN,moves=(),game_tag='fixture'):
    board=chess.Board(fen);history=[]
    for uci in moves:
        history.append(p.repetition_hash(board));p._push(board,chess.Move.from_uci(uci))
    gid=hashlib.sha256((game_tag+fen+' '.join(moves)).encode()).hexdigest()
    position=dict(game_id=gid,ply=len(moves),fen=board.fen(en_passant='fen'),moves=list(moves),previous_hashes=history)
    family=dict(game_id=gid,game_sha256=hashlib.sha256(b'fixture source').hexdigest(),
                moves_sha256=hashlib.sha256(' '.join(moves).encode()).hexdigest(),source_offset=0,source_end_offset=1,
                ratings=[2000,2000],time_control='600+5',positions=[position],action_root_index=0,all_moves=list(moves))
    records=[]
    legal=list(board.legal_moves)
    # Intentionally MultiPV order differs from canonical order.
    legal.sort(key=lambda m:p.canonical_action(m,board.turn),reverse=True)
    for all_legal in (False,True):
        query=dict(split=split,position=position,all_legal=all_legal,nodes=16)
        query['id']=hashlib.sha256(json.dumps(query,sort_keys=True).encode()).hexdigest()
        selected=legal if all_legal else legal[:1];rows=[];material=p._material(board)
        for i,move in enumerate(selected):
            cp=123-i
            rows.append(dict(uci=move.uci(),canonical_action=p.canonical_action(move,board.turn),root_score_cp=cp,
                             reported_depth=7,seldepth=9,nodes=16,pv=[move.uci()],
                             converted_score=p.source_score_record(cp,material,negate=all_legal)))
        packet=dict(protocol_version=2,selected_emission_ordinal=1,selected_common_depth=7,query_rejection=None)
        labels=dict(target_kind='stockfish_parent_root_normalized_action_cp' if all_legal else 'stockfish_standalone_root_normalized_cp',normalization_root_material=material,rows=rows)
        records.append(dict(query=query,labels=labels,rejection=None,transcript=['fabricated fixture'],packet_provenance=packet))
    pins={k:'0'*64 for k in p.SOURCE_KEYS}
    return p.provenance_from_records(split,[family],records,pins,origin_fen=fen)


def published(tmp,**kwargs):
    path=Path(tmp)/'pack';pin=p._publish(fixture_provenance(**kwargs),path)
    return path,pin,p.read_pack(path,expected_manifest_sha256=pin,expected_source_sha256=p.source_sha256())


def rehash(path,member):
    manifest=json.loads((path/'manifest.json').read_text());raw=(path/member).read_bytes()
    stem={'payload.bin':'payload','provenance.json':'provenance','schedule.json':'schedule'}[member]
    manifest[stem+'_bytes']=len(raw);manifest[stem+'_sha256']=hashlib.sha256(raw).hexdigest()
    (path/'manifest.json').write_bytes(p._dump(manifest))


class PackingTests(unittest.TestCase):
    def test_roundtrip_alignment_complete_action_rank_and_ownership(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,pin,pack=published(tmp)
            self.assertIsInstance(pack.payload,bytes)
            self.assertEqual(pack.manifest['counts']['roots'],1)
            self.assertEqual(pack.values('edges.action'),tuple(sorted(pack.values('edges.action'))))
            self.assertEqual(pack.values('edges.rank'),tuple(range(20,0,-1)))
            self.assertTrue(all(pack.values('edges.ordinary')))
            end=0
            for spec in sorted(pack.segments.values(),key=lambda x:x['offset']):
                self.assertEqual(spec['offset']%64,0)
                self.assertFalse(any(pack.payload[end:spec['offset']]))
                end=spec['offset']+spec['nbytes']
            with self.assertRaises(TypeError):pack.manifest['counts']['roots']=2
            self.assertTrue(pack.view('states.boards').readonly)
            with self.assertRaises(p.CorpusError):p.read_pack(path,expected_split='test')
            with self.assertRaises(p.CorpusError):p.read_pack(path,expected_manifest_sha256='1'*64)

    def test_terminal_children_fifty_repetition_and_insufficient_masks(self):
        fixtures=[dict(fen='7k/8/8/8/8/8/8/KR6 w - - 99 1'),
                  dict(moves=('g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1')),
                  dict(fen='7k/8/8/3b4/8/2N5/8/K7 w - - 0 1')]
        for kwargs in fixtures:
            with self.subTest(kwargs=kwargs),tempfile.TemporaryDirectory() as tmp:
                path,_,pack=published(tmp,**kwargs)
                terminals=pack.values('states.terminal');children=pack.values('edges.child_state')
                self.assertTrue(any(terminals[i] for i in children))
                self.assertEqual(pack.values('edges.ordinary'),tuple(1-terminals[i] for i in children))
                self.assertTrue(all(w>0 for w in pack.values('edges.halfwidth')))
        # Native insufficient policy differs from broad prospective/draw claims.
        self.assertTrue(p.native_insufficient(chess.Board('7k/8/8/8/8/8/8/K7 w - - 0 1')))
        self.assertFalse(p.native_insufficient(chess.Board('7k/8/8/8/8/8/8/KNN5 w - - 0 1')))
        board=chess.Board();self.assertFalse(p.terminal(board,[p.repetition_hash(board)]*2))
        board.halfmove_clock=8;self.assertTrue(p.terminal(board,[p.repetition_hash(board)]*2))

    def test_black_raw_ep_full_unsigned_clock_and_special_moves(self):
        fens=['rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 4294967295',
              'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
              '7k/P7/8/8/8/8/8/K7 w - - 0 1',
              '7k/8/8/3pP3/8/8/8/K7 w - d6 0 1']
        for fen in fens:
            with self.subTest(fen=fen),tempfile.TemporaryDirectory() as tmp:
                _,_,pack=published(tmp,fen=fen)
                fields=pack.values('states.fields');self.assertEqual(fields[2],-1 if chess.Board(fen).ep_square is None else chess.Board(fen).ep_square)
                if fen.endswith('4294967295'):
                    self.assertEqual(fields[4],-1)
                    self.assertTrue(all(fields[5*i+4]==-1 for i in range(pack.manifest['counts']['states'])))
        board=chess.Board('7k/8/8/8/8/8/8/KR6 b - - 65535 4294967295')
        p._push(board,next(iter(board.legal_moves)));self.assertEqual(board.fullmove_number,2**32-1);self.assertEqual(board.halfmove_clock,65535)

    def test_query_sign_material_frame_and_outward_rounding(self):
        for cp in (-19753,-123,0,123,19753):
            for material in (0,17,58,77,78,576):
                for negate in (False,True):
                    mid,width=p.score_interval(cp,material,negate=negate);source=p.source_score_record(cp,material,negate=negate)
                    self.assertLessEqual(mid-width,source['lower_cp']);self.assertGreaterEqual(mid+width,source['upper_cp'])
        for corruption in ('wrong_kind','child_material','negation','width','rank','unknown','duplicate','nodes','PV'):
            prov=fixture_provenance();q=prov['queries'][1]
            if corruption=='wrong_kind':q['target_kind']='stockfish_standalone_root_normalized_cp'
            elif corruption=='child_material':q['query_material']-=1
            elif corruption=='negation':q['rows'][0]['converted_score']['negated']=False
            elif corruption=='width':q['rows'][0]['converted_score']['interval_halfwidth']=0
            elif corruption=='rank':q['rows'][0]['rank']=True
            elif corruption=='unknown':q['rows'][0]['action']=0
            elif corruption=='duplicate':q['rows'][-1]=copy.deepcopy(q['rows'][0])
            elif corruption=='nodes':q['rows'][-1]['nodes']+=1
            else:q['rows'][0]['pv'].append('a1a8')
            with self.subTest(corruption=corruption),self.assertRaises(p.CorpusError):p._assemble(prov,p.Limits())

    def test_corruption_recomputed_hashes_cannot_hide_semantic_failures(self):
        for raw in (b'{"x":1e309}',b'{"x":-1e309}',b'{"x":NaN}'):
            with self.assertRaises(p.CorpusError):p._json(raw)
        for table in ('states.hash','states.fields','history.keys','edges.action','edges.midpoint','edges.ordinary'):
            with self.subTest(table=table),tempfile.TemporaryDirectory() as tmp:
                _,_,pack=published(tmp,moves=('e2e4',));path=Path(tmp)/'pack'
                blob=bytearray(pack.payload);spec=pack.segments[table];blob[spec['offset']]^=1
                (path/'payload.bin').write_bytes(blob);rehash(path,'payload.bin')
                with self.assertRaises(p.CorpusError):p.read_pack(path)
        for change in ('schedule_version_bool','schedule_replacement_int','origin_bool','padding','dtype','duplicate_json'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                path,_,pack=published(tmp)
                if change.startswith('schedule'):
                    schedule=json.loads((path/'schedule.json').read_text());schedule['version' if 'version' in change else 'replacement']=True if 'version' in change else 0
                    (path/'schedule.json').write_bytes(p._dump(schedule));rehash(path,'schedule.json')
                elif change=='origin_bool':
                    m=json.loads((path/'manifest.json').read_text());m['origins'][0]['index']=False;(path/'manifest.json').write_bytes(p._dump(m))
                elif change=='padding':
                    blob=bytearray(pack.payload);last=pack.segments['edges.ordinary'];blob[last['offset']+last['nbytes']]=1
                    (path/'payload.bin').write_bytes(blob);rehash(path,'payload.bin')
                elif change=='dtype':
                    m=json.loads((path/'manifest.json').read_text());m['tables']['states.hash']['dtype']='float32';(path/'manifest.json').write_bytes(p._dump(m))
                else:
                    raw=(path/'manifest.json').read_text().replace('"schema_version": 1','"schema_version": 1, "schema_version": 1');(path/'manifest.json').write_text(raw)
                with self.assertRaises(p.CorpusError):p.read_pack(path)

    def test_caps_atomic_no_replace_and_train_test_boundary(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,pin,_=published(tmp)
            before=(path/'payload.bin').read_bytes()
            with self.assertRaises(p.CorpusError):p._publish(fixture_provenance(),path)
            self.assertEqual(before,(path/'payload.bin').read_bytes())
            with self.assertRaises(p.CorpusError):p._publish(fixture_provenance(),Path(tmp)/'over',p.Limits(payload_bytes=64))
            self.assertFalse((Path(tmp)/'over').exists())
            with self.assertRaises(p.CorpusError):p._publish(fixture_provenance(),Path(tmp)/'over',p.Limits(metadata_bytes=64))
            with self.assertRaises(p.CorpusError):p.read_pack(path,max_bytes=64)
            with self.assertRaises(p.CorpusError):p.pack_audited_split('never-read','never-read',Path(tmp)/'not-packed',split='train',seal_path='never-read',expected_seal_sha256=None)
        with tempfile.TemporaryDirectory() as tmp:
            path,_,_=published(tmp,split='test')
            with self.assertRaises(p.CorpusError):p.read_pack(path,purpose='train')
            with self.assertRaises(p.CorpusError):p.read_pack(path,purpose='test',expected_split='test')

    def test_exact_K_schedule_is_structural_and_complete(self):
        data={'roots.legal_count':[20,3,20,3,1],'observations.state':list(range(7))}
        schedule=p.structural_schedule(data,root_batch_size=2,observation_batch_size=3)
        roots=[i for b in schedule['root_batches'] for i in b['roots']]
        self.assertEqual(sorted(roots),list(range(5)))
        for b in schedule['root_batches']:self.assertTrue(all(data['roots.legal_count'][i]==b['legal_count'] for i in b['roots']))
        self.assertEqual(schedule,p.structural_schedule(data,root_batch_size=2,observation_batch_size=3))

    def test_exact_history_identity_and_no_torch_import(self):
        board=chess.Board();same=board.copy();same.fullmove_number+=1
        self.assertNotEqual(p._state_identity(board,[]),p._state_identity(same,[]))
        self.assertNotEqual(p._state_identity(board,[]),p._state_identity(board,[p.repetition_hash(board)]))
        result=subprocess.run([__import__('sys').executable,'-c',"import sys;import experiments.semantic_corpus.packing;assert 'torch' not in sys.modules"],check=True,capture_output=True)
        prov=fixture_provenance(moves=('e2e4',));prov['families'][0]['source']['positions'][0]['previous_hashes'][0]^=1
        with self.assertRaises(p.CorpusError):p._assemble(prov,p.Limits())
        prov=fixture_provenance(moves=('e2e4',));prov['families'][0]['source']['positions'][0]['fen']=prov['families'][0]['source']['positions'][0]['fen'].replace(' e3 ',' - ')
        with self.assertRaises(p.CorpusError):p._assemble(prov,p.Limits())

    def test_float32_bytes_match_frozen_CPU_score_converter(self):
        try:import torch
        except ImportError:self.skipTest('optional CPU Torch oracle unavailable')
        import importlib.util
        source=Path('docs/research/evidence/semantic-packet-v2-preparation-2026-10-01/neural/score_frames.py')
        spec=importlib.util.spec_from_file_location('frozen_score_frame_oracle',source);module=importlib.util.module_from_spec(spec)
        old_bytecode=sys.dont_write_bytecode
        try:
            sys.dont_write_bytecode=True
            spec.loader.exec_module(module)
        finally:sys.dont_write_bytecode=old_bytecode
        self.assertEqual(p.sha256(source),'efc0e8da2ad65b41523fdcf0fd514fa70e3ddbff066854a88c2ad64726a02f57')
        for fen in (chess.STARTING_FEN,'rnbqkbnr/ppp1pppp/8/3P4/8/8/PPPP1PPP/RNBQKBNR b KQkq - 0 2'):
            board=chess.Board(fen);features=torch.zeros((1,18,8,8),dtype=torch.float32,device='cpu')
            for sq,piece in board.piece_map().items():
                canonical=sq^(0 if board.turn else 56);plane=piece.piece_type-1+(0 if piece.color==board.turn else 6)
                features[0,plane,canonical//8,canonical%8]=1
            for cp in (-19753,-123,0,123,19753):
                for negate in (False,True):
                    oracle=module.sf19_reference_batch(torch.tensor([cp],dtype=torch.float32,device='cpu'),features,score_kind='ordinary_unbounded_cp_no_tablebase',negate=negate)
                    expected=p.score_interval(cp,p._material(board),negate=negate)
                    actual=(oracle['value_cp'].item(),oracle['interval_halfwidth'].item())
                    self.assertEqual(struct.pack('<2f',*actual),struct.pack('<2f',*expected))

    def test_native_Rust_snapshot_oracle_all_fixture_states(self):
        binary=Path('target/release/examples/corpus_oracle')
        if not binary.exists():self.skipTest('root-built CPU oracle unavailable')
        for kwargs in (dict(),dict(moves=('e2e4',)),dict(fen='7k/8/8/8/8/8/8/KR6 w - - 99 1'),dict(fen='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 4294967295')):
            with tempfile.TemporaryDirectory() as tmp:
                _,_,pack=published(tmp,**kwargs)
                _,_,_,meta=p._assemble(fixture_provenance(**kwargs),p.Limits());offsets=pack.values('history.offsets');hist=pack.values('history.keys');lines=[]
                for si,state in enumerate(meta['states']):lines.append(state['fen']+'\t'+','.join(f'{h&((1<<64)-1):x}' for h in hist[offsets[si]:offsets[si+1]]))
                result=subprocess.run([str(binary)],input='\n'.join(lines)+'\n',text=True,capture_output=True,timeout=10,check=True)
                snapshots=[json.loads(s) for s in result.stdout.splitlines()]
                self.assertEqual(len(snapshots),pack.manifest['counts']['states'])
                for si,snapshot in enumerate(snapshots):
                    state=snapshot['state'];self.assertEqual(snapshot['policy'],p.TERMINAL_POLICY)
                    self.assertEqual(state['boards'],[x&((1<<64)-1) for x in pack.values('states.boards')[si*12:(si+1)*12]])
                    packedfields=pack.values('states.fields')[si*5:(si+1)*5]
                    self.assertEqual(state['fields'],[packedfields[0],packedfields[1],packedfields[2],packedfields[3],packedfields[4]&((1<<32)-1)])
                    if si==pack.values('roots.state')[0]:self.assertEqual([r['action'] for r in state['legal']],list(pack.values('edges.action')))
                    self.assertEqual(state['hash'],pack.values('states.hash')[si]&((1<<64)-1))
                    self.assertEqual(state['terminal'],bool(pack.values('states.terminal')[si]))

if __name__=='__main__':unittest.main()
