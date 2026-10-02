"""Archive closed attempt and score-free planned inputs; never starts an engine.

Finalization requires independent packet/frame audit artifacts supplied separately.
"""
from pathlib import Path
from collections import Counter
import argparse,gzip,hashlib,io,json,shutil,sys,tarfile,tempfile
ROOT=Path('/home/iskam/src/vibe/chen')
sys.path.insert(0,str(ROOT))
import chess
from scripts.audit_semantic_admission import feature_key,repetition_hash

sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
def encoded(data):return (json.dumps(data,indent=2,allow_nan=False)+'\n').encode()
def gzjson(path,data):path.write_bytes(gzip.compress(encoded(data),mtime=0))

def prepare():
 stage=ROOT/'results/neural-packet-v2';run=ROOT/'results/semantic-teacher-packet-v2-2026-10-01'
 target=ROOT/'docs/research/evidence/semantic-teacher-packet-v2-2026-10-01'
 assert not target.exists()
 work=Path(tempfile.mkdtemp(prefix='packet-v2-actual-archive-',dir=ROOT/'results'))
 report=json.loads((run/'report.json').read_text());decl=json.loads((run/'declaration.json').read_text())
 assert report['status']=='complete' and report['completed_queries']==report['requested_queries']==3456
 assert len(report['worker_exitcodes'])==2 and all(type(v) is int and v==0 for v in report['worker_exitcodes'])
 assert report['adopted_children_reaped']==[]
 admissionpath=stage/'neural/evidence/packet-v2-admission-2026-10-01.json.gz'
 assert sha(admissionpath)==report['admission_sha256']==decl['admission_sha256']
 assert sha(run/'declaration.json')==report['declaration_sha256']==sha(stage/'neural/evidence/packet-v2-label-sealed-2026-10-01.json')
 admission=json.loads(gzip.decompress(admissionpath.read_bytes()))
 mapping=[]
 def copy(original,name):
  dest=work/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(original,dest)
  mapping.append(dict(original=str(original),archived=name,sha256=sha(dest)))
 # Replay each actual prefix, without using producer encoders/label selectors.
 inputs=[];queries=[];keys={};by_split={};game_ids=set()
 for split in ('train','validation','test'):
  splitkeys=set();observations=children=0
  for family in admission['source_records'][split]:
   assert family['game_id'] not in game_ids;game_ids.add(family['game_id'])
   boards=[]
   for row in family['positions']:
    board=chess.Board();history=[]
    assert row['moves']==family['all_moves'][:row['ply']]
    for uci in row['moves']:
     history.append(repetition_hash(board));move=chess.Move.from_uci(uci);assert move in board.legal_moves;board.push(move)
    assert history==row['previous_hashes'] and board.fen(en_passant='fen')==row['fen']
    boards.append(board);key=feature_key(board);assert key not in keys
    keys[key]=row['fen'];splitkeys.add(key);observations+=1
    inputs.append(dict(split=split,game_id=family['game_id'],kind='standalone_observation',ply=row['ply'],
                       original_fen=row['fen'],canonical_feature_identity=key))
    query=dict(split=split,position=row,all_legal=False,nodes=decl['limits']['standalone_nodes'])
    query['id']=hashlib.sha256(json.dumps(query,sort_keys=True).encode()).hexdigest();queries.append(query)
   parent=boards[family['action_root_index']];parent_row=family['positions'][family['action_root_index']]
   query=dict(split=split,position=parent_row,all_legal=True,nodes=decl['limits']['all_action_nodes'])
   query['id']=hashlib.sha256(json.dumps(query,sort_keys=True).encode()).hexdigest();queries.append(query)
   for move in sorted(parent.legal_moves,key=lambda m:m.uci()):
    child=parent.copy(stack=True);child.push(move);key=feature_key(child);assert key not in keys
    rawfen=child.fen(en_passant='fen');keys[key]=rawfen;splitkeys.add(key);children+=1
    flip=0 if parent.turn else 56;promo={None:0,chess.KNIGHT:1,chess.BISHOP:2,chess.ROOK:3,chess.QUEEN:4}[move.promotion]
    action=(((move.from_square^flip)*64+(move.to_square^flip))*5+promo)
    inputs.append(dict(split=split,game_id=family['game_id'],kind='all_legal_root_child',ply=parent_row['ply']+1,
                       parent_original_fen=parent_row['fen'],action_uci=move.uci(),canonical_action_parent_frame=action,
                       original_fen=rawfen,canonical_feature_identity=key))
  assert len(splitkeys)==admission['canonical_counts'][split]
  assert children==admission['mover_counts'][split]['all_legal_child_rows']
  by_split[split]=dict(families=len(admission['source_records'][split]),observations=observations,
                       legal_children=children,canonical_inputs=len(splitkeys))
 assert len(inputs)==len(keys)==17084 and len(queries)==3456
 assert hashlib.sha256(json.dumps(queries,sort_keys=True).encode()).hexdigest()==decl['query_plan_sha256']
 expected={q['id']:q for q in queries};assert len(expected)==3456
 fingerprints={row['query_id']:row['sha256'] for row in report['source_records']};assert len(fingerprints)==3456 and set(fingerprints)==set(expected)
 for queryid,query in expected.items():
  path=run/(queryid+'.json');assert sha(path)==fingerprints[queryid]
  record=json.loads(path.read_bytes());assert record['query']==query and record['declaration_sha256']==report['declaration_sha256']
 rawfiles=sorted(run.iterdir());assert all(p.is_file() and p.suffix=='.json' for p in rawfiles)
 assert {p.name for p in rawfiles}=={q+'.json' for q in expected}|{'report.json','declaration.json','worker-0-setup.json','worker-1-setup.json'}
 members=[]
 with (work/'raw-records.tar.gz').open('wb') as stream:
  with gzip.GzipFile(fileobj=stream,mode='wb',mtime=0,filename='') as compressed:
   with tarfile.open(fileobj=compressed,mode='w') as tar:
    for path in rawfiles:
     raw=path.read_bytes();info=tarfile.TarInfo(path.name);info.size=len(raw);info.mtime=0;info.mode=0o644
     tar.addfile(info,io.BytesIO(raw));members.append(dict(path=path.name,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
 (work/'raw-member-sha256.json').write_bytes(encoded(members))
 for name in ('report.json','declaration.json','worker-0-setup.json','worker-1-setup.json'):copy(run/name,name)
 copy(admissionpath,'input-admission.json.gz')
 copy(stage/'runtime-manifest.json','runtime-manifest.json')
 for name in ('label-structural-summary.json','label-run.stdout.log','label-run.stderr.log'):copy(stage/name,name)
 copy(Path(__file__),'archive-teacher-evidence.py')
 copy(ROOT/'scripts/audit_semantic_admission.py','independent-input-schema.py')
 gzjson(work/'full-query-plan.json.gz',queries)
 gzjson(work/'original-input-fens.json.gz',dict(kind='all_planned_original_input_FEN_ledger',strength_evidence=False,
            score_metrics_included=False,inputs=inputs,counts=by_split,independent_reconstruction='squarewise_feature_identity_and_full_legal_history_replay'))
 spent=dict(kind='licensed_semantic_teacher_packet_v2_attempt_spent_inputs',
            reason='all fixed requested model inputs spent including rejected families; no retries or rescue',
            canonical_position_fens=[keys[key] for key in sorted(keys)],canonical_input_count=len(keys),
            teacher_attempt=str(run.relative_to(ROOT)),sealed_declaration_sha256=report['declaration_sha256'],
            original_admission_sha256=sha(admissionpath),strength_evidence=False,score_metrics_included=False)
 (work/'spent-inputs.json').write_bytes(encoded(spent))
 (work/'planned-input-structural-summary.json').write_bytes(encoded(dict(kind='independent_all_planned_input_reconstruction',
           input_count=17084,query_count=3456,counts=by_split,
           all_rejected_families_included=True,source_score_values_used=False,heldout_quality_metrics_read=False,
           original_prefixes_raw_EP_and_repetition_hashes_verified=True,
           independent_schema_source_sha256=sha(ROOT/'scripts/audit_semantic_admission.py'),
           original_admission_sha256=sha(admissionpath),query_plan_sha256=decl['query_plan_sha256'])))
 (work/'archive-map.json').write_bytes(encoded(mapping))
 print(str(work))
 return work

if __name__=='__main__':prepare()
