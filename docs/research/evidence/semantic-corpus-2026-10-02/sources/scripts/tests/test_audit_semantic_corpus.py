"""Fabricated mechanical witnesses only; no corpus pack or teacher process."""
import importlib.util
import hashlib
import json
from pathlib import Path
import struct
import tempfile
import unittest

import chess

spec=importlib.util.spec_from_file_location('corpus_auditor',Path(__file__).resolve().parents[1]/'audit_semantic_corpus.py')
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


class WitnessTests(unittest.TestCase):
    def test_float32_outward_sign_and_source_material(self):
        for material in (0,17,18,27,58,78,576):
            for cp in (0,1,321,-321,19753,-19753):
                positive,width=audit.reference_interval(cp,material,False)
                negative,negative_width=audit.reference_interval(cp,material,True)
                self.assertEqual(negative,-positive);self.assertEqual(width,negative_width)
                self.assertEqual(positive,audit.f32(positive))
                self.assertEqual(width,audit.f32(width))
        board=chess.Board('4k3/pppppppp/8/8/8/8/4q3/R3R1K1 w - - 0 1')
        child=board.copy();child.push_uci('e1e2')
        parent_frame=audit.reference_interval(321,audit.query_material(board),True)
        wrong_child_frame=audit.reference_interval(321,audit.query_material(child),True)
        self.assertNotEqual(parent_frame[0],wrong_child_frame[0])

    def test_numeric_alias_and_decisive_region_rejected(self):
        for arguments in ((True,58,False),(1,True,False),(1,58,1),(19754,58,False),(-19754,58,False)):
            with self.assertRaises(ValueError):audit.reference_interval(*arguments)

    def test_absolute_high_bit_raw_ep_and_canonical_black_orientation(self):
        state=audit.exact_state(chess.Board())
        self.assertTrue(state['boards'][9]&(1<<63))
        self.assertEqual(state['fields'],[0,15,-1,0,1])
        board=chess.Board('4k3/pppppppp/8/8/8/8/4q3/R3R1K1 w - - 0 1')
        move=chess.Move.from_uci('e1e2');mirrored=board.mirror()
        counterpart=chess.Move(move.from_square^56,move.to_square^56)
        self.assertEqual(audit.canonical_action(board,move),audit.canonical_action(mirrored,counterpart))
        raw=chess.Board('4k3/8/8/3p4/8/8/8/4K3 w - d6 0 3');clean=raw.copy();clean.ep_square=None
        self.assertFalse(raw.has_legal_en_passant())
        self.assertEqual(audit.exact_state(raw)['hash'],audit.exact_state(clean)['hash'])
        self.assertNotEqual(audit.exact_state(raw)['fields'],audit.exact_state(clean)['fields'])

    def test_same_exact_state_separate_retained_history(self):
        board,previous=audit.source_position(['g1f3','g8f6','f3g1','f6g8'])
        refreshed=chess.Board(board.fen(en_passant='fen'))
        self.assertEqual(audit.exact_state(board),audit.exact_state(refreshed))
        self.assertEqual(len(previous),4)
        self.assertEqual(previous[0],audit.exact_state(chess.Board())['hash'])
        self.assertNotEqual(previous,[])
        with self.assertRaises(ValueError):audit.source_position(['e2e5'])

    def fixture(self,tmp,**options):
        from experiments.semantic_corpus import packing
        from experiments.semantic_corpus.tests.test_packing import fixture_provenance
        path=Path(tmp)/'pack'
        pin=packing._publish(fixture_provenance(**options),path)
        return path,pin,packing.source_sha256()

    def test_independent_reader_and_all_semantic_bytes(self):
        cases=[{},dict(moves=('e2e4','e7e5')),
               dict(moves=('g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1')),
               dict(fen='7k/8/8/8/8/8/8/KR6 w - - 99 1'),
               dict(fen='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 4294967295'),
               dict(fen='r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'),
               dict(fen='7k/P7/8/8/8/8/8/K7 w - - 0 1'),
               dict(fen='7k/8/8/3pP3/8/8/8/K7 w - d6 0 1')]
        for case in cases:
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                path,pin,source=self.fixture(tmp,**case)
                m,p,v,b=audit.read_binary_pack(path,pin,source)
                states,roots=audit.verify_relations(m,p,v,b)
                self.assertEqual(len(states),m['counts']['states'])
                self.assertEqual(len(roots[0]['edges']),m['counts']['edges'])

    def test_rehashed_state_frame_mask_history_action_corruption(self):
        for table in ('states.boards','states.fields','states.hash','history.keys','edges.action','edges.midpoint','edges.ordinary'):
            with self.subTest(table=table),tempfile.TemporaryDirectory() as tmp:
                path,pin,source=self.fixture(tmp,moves=('e2e4',))
                m,p,v,b=audit.read_binary_pack(path,pin,source)
                data=bytearray(b);data[m['tables'][table]['offset']]^=1
                # The independent semantic pass must fail even if wire hash is updated.
                (path/'payload.bin').write_bytes(data);m['payload_sha256']=hashlib.sha256(data).hexdigest()
                (path/'manifest.json').write_bytes(audit.dump(m));pin=audit.digest(path/'manifest.json')
                m,p,v,b=audit.read_binary_pack(path,pin,source)
                with self.assertRaises(ValueError):audit.verify_relations(m,p,v,b)

    def test_strict_wire_padding_types_and_duplicate_keys(self):
        for change in ('padding','shape_bool','schedule_bool','origin_bool','duplicate'):
            with self.subTest(change=change),tempfile.TemporaryDirectory() as tmp:
                path,pin,source=self.fixture(tmp)
                m=json.loads((path/'manifest.json').read_text())
                if change=='padding':
                    data=bytearray((path/'payload.bin').read_bytes());spec=m['tables']['edges.ordinary'];data[spec['offset']+spec['nbytes']]=1
                    (path/'payload.bin').write_bytes(data);m['payload_sha256']=hashlib.sha256(data).hexdigest()
                elif change=='shape_bool':m['tables']['roots.state']['shape'][0]=True
                elif change=='schedule_bool':
                    schedule=json.loads((path/'schedule.json').read_text());schedule['replacement']=0
                    raw=audit.dump(schedule);(path/'schedule.json').write_bytes(raw);m['schedule_sha256']=hashlib.sha256(raw).hexdigest();m['schedule_bytes']=len(raw)
                elif change=='origin_bool':m['origins'][0]['index']=False
                raw=audit.dump(m)
                if change=='duplicate':raw=raw.replace(b'"schema_version": 1',b'"schema_version": 1, "schema_version": 1')
                (path/'manifest.json').write_bytes(raw);pin=audit.digest(path/'manifest.json')
                with self.assertRaises(ValueError):audit.read_binary_pack(path,pin,source)

    def test_actual_auditor_rejects_fixture_kind_before_archive_reads(self):
        with tempfile.TemporaryDirectory() as tmp:
            path,pin,source=self.fixture(tmp)
            m,p,v,b=audit.read_binary_pack(path,pin,source)
            seal=Path(tmp)/'seal.json'
            from experiments.semantic_corpus import packing
            from dataclasses import asdict
            seal.write_bytes(audit.dump(dict(kind='cpu_semantic_serialization_seal',version=1,status='sealed_no_training',split='train',source_sha256=source,pins=p['pins'],limits=asdict(packing.Limits()))))
            # No actual query record may be opened through a protocol-only fixture.
            with self.assertRaisesRegex(ValueError,'fixture cannot masquerade'):
                audit.verify_archived_sources(m,p,'never-read','never-read',seal,audit.digest(seal),packing.__file__)

    def test_overflow_json_float_in_unused_packet_diagnostics(self):
        with self.assertRaisesRegex(ValueError,'overflowing JSON float'):
            audit.strict_json(b'{"unused_diagnostic": 1e309}')
        with tempfile.TemporaryDirectory() as tmp:
            path,pin,source=self.fixture(tmp)
            prov=json.loads((path/'provenance.json').read_text())
            prov['queries'][0]['packet_provenance']['unused_diagnostic']=0.0
            raw=audit.dump(prov).replace(b'"unused_diagnostic": 0.0',b'"unused_diagnostic": 1e309')
            (path/'provenance.json').write_bytes(raw)
            m=json.loads((path/'manifest.json').read_text());m['provenance_sha256']=hashlib.sha256(raw).hexdigest();m['provenance_bytes']=len(raw)
            (path/'manifest.json').write_bytes(audit.dump(m));pin=audit.digest(path/'manifest.json')
            with self.assertRaisesRegex(ValueError,'overflowing JSON float'):
                audit.read_binary_pack(path,pin,source)

    def test_native_cpu_oracle_all_fixture_states_and_full_roots(self):
        binary=Path('target/release/examples/corpus_oracle');build=Path('results/semantic-corpus/rust-oracle-build.json')
        if not binary.exists() or not build.exists():self.skipTest('root-built CPU oracle unavailable')
        for case in (dict(moves=('e2e4',)),dict(fen='7k/8/8/8/8/8/8/KR6 w - - 99 1'),
                     dict(moves=('g1f3','g8f6','f3g1','f6g8','g1f3','g8f6','f3g1')),
                     dict(fen='rnbqkbnr/pppppppp/8/8/4P3/8/PPPP1PPP/RNBQKBNR b KQkq e3 0 4294967295')):
            with self.subTest(case=case),tempfile.TemporaryDirectory() as tmp:
                path,pin,source=self.fixture(tmp,**case)
                m,p,v,b=audit.read_binary_pack(path,pin,source);states,roots=audit.verify_relations(m,p,v,b)
                report=audit.verify_native_oracle(states,roots,binary,build,audit.digest(build),Path.cwd())
                self.assertEqual(report['state_snapshots_verified'],len(states))
                self.assertEqual(report['legal_child_snapshots_verified'],m['counts']['edges'])


if __name__=='__main__':unittest.main(verbosity=2)
