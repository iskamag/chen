"""Fresh restricted-exchange study admission and CPU reference packing.

No model/teacher/GPU during family admission. TEST labels are deliberately not
implemented here. Only complete current graphs enter a model; traces are labels.
"""
import argparse
import ast
from collections import Counter
from dataclasses import asdict, fields
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import random
import sys
import time

import chess

from .exchange_targets import Limits, MissingTarget, complete_initial_actions, exchange_target

SCHEMA = 'chen_exchange_learning_admission_v1'
SPLITS = ('train', 'dev', 'test')

def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()

def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1048576),b''):h.update(block)
    return h.hexdigest()

def read_json(path):
    path=Path(path)
    with (gzip.open(path,'rt') if path.suffix=='.gz' else path.open()) as stream:
        return json.load(stream)

def read_rows(path):
    with gzip.open(path,'rt') as stream:
        for line in stream:
            if line.strip():yield json.loads(line)

def write_rows(path,rows):
    with Path(path).open('xb') as raw:
        with gzip.GzipFile(fileobj=raw,mode='wb',mtime=0) as zipped:
            for row in rows:zipped.write((json.dumps(row,separators=(',',':'))+'\n').encode())

def generator(main):
    scripts=Path(main)/'scripts'
    # Explicit source dependency, no native/runtime import or GPU module loading.
    sys.path.insert(0,str(scripts))
    spec=importlib.util.spec_from_file_location('chen_exchange_reachable_generator',scripts/'generate_synthetic.py')
    module=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=module
    spec.loader.exec_module(module)
    return module

class Ledger:
    """Known-spent input identities from explicitly pinned sources, not all history."""
    def __init__(self,feature_key):
        self.feature_key=feature_key
        self.keys=set();self.families=set();self.moves=set();self.bindings=[]

    def board(self,board,children=False):
        self.keys.add(self.feature_key(board))
        if children:
            for move in list(board.legal_moves):
                board.push(move);self.keys.add(self.feature_key(board));board.pop()

    def walk(self,value):
        if isinstance(value,dict):
            for name,item in value.items():
                if name in ('family_id','game_id') and isinstance(item,str):self.families.add(item)
                elif name in ('moves_sha256',) and isinstance(item,str):self.moves.add(item)
                elif name in ('all_moves','moves') and isinstance(item,list) and all(isinstance(m,str) for m in item):
                    self.moves.add(hashlib.sha256(' '.join(item).encode()).hexdigest())
                elif name in ('fen','root_fen','child_fen') and isinstance(item,str):self.board(chess.Board(item))
                elif name=='feature_key' and isinstance(item,str):self.keys.add(item)
                elif name=='canonical_position_fens':
                    for fen in item:self.board(chess.Board(fen))
                elif isinstance(item,(dict,list)):self.walk(item)
        elif isinstance(value,list):
            for item in value:self.walk(item)

    def add(self,path,kind):
        path=Path(path).resolve()
        before=(len(self.keys),len(self.families),len(self.moves))
        if kind=='key_snapshot':
            data=read_json(path)
            if not isinstance(data,list) or any(not isinstance(k,str) or len(k)!=64 for k in data):
                raise ValueError('explicit feature key snapshot required')
            self.keys.update(data)
        elif kind=='json':self.walk(read_json(path))
        elif kind=='jsonl':
            for row in read_rows(path):self.walk(row)
        elif kind=='fixture_source':self.fixture_source(path)
        else:raise ValueError('unknown ledger input kind')
        self.bindings.append(dict(path=str(path),sha256=sha(path),kind=kind,
            added_keys=len(self.keys)-before[0],added_families=len(self.families)-before[1],
            added_move_identities=len(self.moves)-before[2]))

    def fixture_source(self,path):
        # Read source literals rather than importing modules with device setup.
        # Complete one-ply support is an explicit conservative fixture closure.
        tree=ast.parse(path.read_text())
        self.board(chess.Board(),True)
        for node in ast.walk(tree):
            if isinstance(node,ast.Constant) and isinstance(node.value,str) and node.value.count('/')==7:
                try:board=chess.Board(node.value)
                except ValueError:continue
                for b in (board,board.mirror()):self.board(b,True)
        if path.name=='test_native_topology.py':
            rng,board=random.Random(314159),chess.Board()
            for _ in range(24):
                self.board(board,True)
                moves=list(board.legal_moves)
                if not moves:break
                board.push(rng.choice(moves))
        # Explicit source-defined history fixtures; game-history metadata is not
        # part of the legacy feature key, so both board and full source are bound.
        if path.name in ('test_native_topology.py','test_native_tactical.py'):
            board=chess.Board()
            for move in ('g1f3','g8f6','f3g1','f6g8')*2:
                board.push_uci(move);self.board(board,True)
            board=chess.Board();board.push_uci('e2e4');self.board(board,True)

    def report(self):
        return dict(scope='explicit_known_spent_sources_only',sources=self.bindings,
            feature_count=len(self.keys),family_count=len(self.families),moves_count=len(self.moves),
            key_set_sha256=digest(sorted(self.keys)),family_set_sha256=digest(sorted(self.families)),
            move_set_sha256=digest(sorted(self.moves)))


def known_ledger(main,neural,feature_key):
    main,neural=Path(main),Path(neural)
    evidence=main/'docs/research/evidence'
    ledger=Ledger(feature_key)
    ledger.add(evidence/'synthetic-mechanics-2026-10-03/exclusions.json.gz','key_snapshot')
    # Snapshot already protects roots/descendants of legacy neural admissions;
    # read input records too to retain family and source-game move identities.
    declaration=read_json(evidence/'synthetic-mechanics-2026-10-03/declaration.json')
    legacy=declaration.get('exclusion_sources') or declaration.get('spent_reports') or declaration.get('exclusions')
    if isinstance(legacy,list):
        for row in legacy:
            path=row.get('path') if isinstance(row,dict) else row
            if path:
                path=Path(path)
                if not path.is_absolute():path=neural/path
                ledger.add(path,'json')
    # Explicit legacy admissions (family identities absent from key snapshots).
    for name in ('semantic-data-admission-2026-10-01.json.gz',
                 'semantic-data-unscored-admission-revision1-2026-10-01.json.gz',
                 'search-teacher-gfx1030-2026-09-30.json','value-study-gfx1030-2026-10-01.json',
                 'value-study-pilot-2026-10-01.json','strategic-study-spent-mechanics-2026-10-01.json',
                 'strategic-study-pilot-lineages-2026-10-01.json','strategic-study-gfx1030-2026-10-01.json',
                 'strategic-study-actionblind-roundoff-2026-10-01.json'):
        path=neural/'neural/evidence'/name
        if str(path.resolve()) not in {row['path'] for row in ledger.bindings}:ledger.add(path,'json')
    for path in (evidence/'semantic-teacher-packet-v2-2026-10-01/spent-inputs.json',
                 evidence/'semantic-teacher-packet-v2-2026-10-01/input-admission.json.gz',
                 main/'results/neural-packet-v2/neural/evidence/semantic-teacher-closed-attempt-spent-inputs.json'):
        ledger.add(path,'json')
    for name in ('synthetic-mechanics-2026-10-03','synthetic-mechanics-balanced-2026-10-03'):
        ledger.add(evidence/name/'families.jsonl.gz','jsonl')
    ledger.add(neural/'neural/evidence/exchange-target-pilot-2026-10-03/targets.jsonl.gz','jsonl')
    for path in (main/'gpu/test_native_tactical.py',main/'gpu/test_native_topology.py',
                 neural/'neural/tests/test_exchange_targets.py',neural/'neural/tests/test_tactical_inputs.py'):
        ledger.add(path,'fixture_source')
    return ledger


def family_input_keys(row):
    return {key for root in row['roots'] for key in
        (root['feature_key'],*(a['child_feature_key'] for a in root['actions']))}


def admit_family(row,ledger,occupied):
    if not row['roots']:return 'empty_roots',set()
    keys=family_input_keys(row)
    move_id=hashlib.sha256(' '.join(row['moves']).encode()).hexdigest()
    if row['family_id'] in ledger.families or move_id in ledger.moves:return 'spent_family',keys
    if keys&ledger.keys:return 'spent_feature',keys
    if row['family_id'] in occupied['families'] or move_id in occupied['moves']:return 'duplicate_family',keys
    if keys&occupied['keys']:return 'cross_family_feature',keys
    occupied['families'].add(row['family_id']);occupied['moves'].add(move_id);occupied['keys'].update(keys)
    return None,keys


def prepare(main,neural,output,counts=(256,64,64),seed=20261009,wall_seconds=600):
    output=Path(output);output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();g=generator(main)
    ledger=known_ledger(main,neural,g.feature_key)
    ledger_report=ledger.report()
    if len(counts)!=3 or any(type(n)is not int or n<1 for n in counts):raise ValueError('positive whole-family counts required')
    config=g.Config(seed=seed,families=sum(counts),max_plies=160,roots_per_family=6,
                    min_ply=24,min_separation=12,max_attempts=sum(counts)*40)
    config.validate()
    declaration=dict(schema=SCHEMA,counts=dict(zip(SPLITS,counts)),generator_config=asdict(config),
        labels_generated=False,test_gate='CLOSED',training_performed=False,
        input_identity='legacy mover planes/rights/raw_EP/clipped_clock plus full replay context',
        collision_scope='all planned parent and one-ply complete legal child inputs across whole families',
        deeper_exchange_trace_states='supervision only; not supplied to parent or exact-one-child model',
        source_sha256={str(p):sha(p) for p in (Path(__file__),Path(main)/'scripts/generate_synthetic.py',
            Path(main)/'scripts/audit_semantic_admission.py')},known_spent=ledger_report,wall_seconds=wall_seconds)
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    occupied=dict(keys=set(),families=set(),moves=set());rows={s:[] for s in SPLITS};rejected=Counter()
    # Split schedule fixed BEFORE labels. Candidate rejection depends on inputs only.
    schedule=[s for i in range(max(counts)) for s,n in zip(SPLITS,counts) if i<n]
    for attempt in range(config.max_attempts):
        if time.monotonic()-started>=wall_seconds:raise TimeoutError('fresh input admission bounded wall exhausted')
        row=g.family(config,attempt)
        reason,keys=admit_family(row,ledger,occupied)
        if reason:rejected[reason]+=1;continue
        split=schedule[sum(len(v) for v in rows.values())]
        row['split']=split;rows[split].append(row)
        if sum(len(v) for v in rows.values())==sum(counts):break
    if any(len(rows[s])!=n for s,n in zip(SPLITS,counts)):raise RuntimeError('not enough fresh input families; no label admission')
    paths={}
    for split in SPLITS:
        path=output/f'inputs-{split}.jsonl.gz';write_rows(path,rows[split]);paths[split]=dict(path=path.name,sha256=sha(path),
            families=len(rows[split]),roots=sum(len(r['roots']) for r in rows[split]),
            canonical_inputs=len(set().union(*(family_input_keys(r) for r in rows[split]))))
    admission=dict(schema=SCHEMA,admitted=True,test_gate='CLOSED',labels_generated=False,
        declaration_sha256=sha(output/'declaration.json'),splits=paths,rejections=dict(rejected),
        attempts=attempt+1,wall_seconds=time.monotonic()-started,known_spent=ledger_report,
        family_ids={s:[r['family_id'] for r in rows[s]] for s in SPLITS})
    (output/'admission.json').write_text(json.dumps(admission,indent=2)+'\n')
    return admission


def label_split(directory,split,limits):
    # Check before opening any TEST input, including metadata-rich root records.
    if split not in ('train','dev'):raise ValueError('TEST gate CLOSED: labels forbidden')
    directory=Path(directory);admission=read_json(directory/'admission.json')
    if not admission['admitted'] or admission['test_gate']!='CLOSED':raise ValueError('closed admitted study required')
    source=directory/admission['splits'][split]['path']
    if sha(source)!=admission['splits'][split]['sha256']:raise ValueError('input admission pin changed')
    started=time.monotonic();deadline=started+limits.wall_seconds;nodes=[0];counts=Counter();hist=Counter();result=[]
    def charge():
        if nodes[0]>=limits.global_nodes:raise MissingTarget('global_node_cap')
        nodes[0]+=1
    for family in read_rows(source):
        if family['split']!=split:raise ValueError('whole-family split mismatch')
        roots=[]
        for root in family['roots']:
            board=chess.Board()
            for uci in family['moves'][:root['ply']]:board.push_uci(uci)
            if board.fen(en_passant='fen')!=root['fen']:raise ValueError('root replay identity mismatch')
            targets=[]
            for action in complete_initial_actions(board):
                target=exchange_target(board,action,limits=limits,global_charge=charge,deadline=deadline)
                targets.append(target);counts['actions']+=1
                if target['exact']:
                    counts['closed']+=1;counts['nonzero_correction']+=target['correction_vs_immediate_cp']!=0
                    hist[str(target['dependency_rounds'])]+=1
                else:counts[target['missing_reason']]+=1
            roots.append(dict(ply=root['ply'],feature_key=root['feature_key'],targets=targets))
        result.append(dict(family_id=family['family_id'],split=split,roots=roots))
    path=directory/f'targets-{split}.jsonl.gz';write_rows(path,result)
    report=dict(split=split,input_sha256=sha(source),admission_sha256=sha(directory/'admission.json'),
        target_sha256=sha(path),target_kind='restricted_exchange_material_delta_not_strategic_CP',
        limits=asdict(limits),counts=dict(counts),dependency_rounds=dict(hist),nodes=nodes[0],wall_seconds=time.monotonic()-started)
    (directory/f'labels-{split}.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def pack_split(directory,split):
    if split not in ('train','dev'):raise ValueError('TEST gate CLOSED: packing forbidden')
    import torch
    from .tactical_inputs import ReplaySpec,compact_from_replay,to_tactical_batch
    directory=Path(directory);admission=read_json(directory/'admission.json');labels=read_json(directory/f'labels-{split}.json')
    source=directory/admission['splits'][split]['path'];target=directory/f'targets-{split}.jsonl.gz'
    if sha(source)!=labels['input_sha256'] or sha(target)!=labels['target_sha256'] or sha(directory/'admission.json')!=labels['admission_sha256']:
        raise ValueError('label/input admission changed')
    packed=[]
    sources=iter(read_rows(source))
    for label_family in read_rows(target):
        family=next(sources)
        if family['family_id']!=label_family['family_id'] or family['split']!=split:raise ValueError('family binding mismatch')
        source_roots={r['ply']:r for r in family['roots']}
        for root in label_family['roots']:
            state=compact_from_replay(ReplaySpec(moves=tuple(family['moves'][:root['ply']])),ply_from_search_root=root['ply'])
            batch=to_tactical_batch((state,));k=len(state.actions)
            support={a.id:i for i,a in enumerate(state.actions)}
            correction=torch.zeros(k);mask=torch.zeros(k,dtype=torch.bool);rounds=torch.zeros(k,dtype=torch.int64)
            for target_row in root['targets']:
                if not target_row['exact']:continue
                slot=support[target_row['initial_action_id']]
                correction[slot]=target_row['correction_vs_immediate_cp'];mask[slot]=True;rounds[slot]=target_row['dependency_rounds']
            if not bool(mask.any()):continue
            observation={f.name:getattr(batch,f.name) for f in fields(batch)}
            # Supervision is a sibling object, NEVER an observation tensor field.
            packed.append(dict(observation=observation,correction=correction,mask=mask,rounds=rounds,
                family_id=family['family_id'],feature_key=root['feature_key'],motifs=source_roots[root['ply']]['motifs']))
    if next(sources,None) is not None:raise ValueError('label family count mismatch')
    path=directory/f'pack-{split}.pt'
    with path.open('xb') as stream:torch.save(dict(schema=SCHEMA,split=split,records=packed,
        input_sha256=sha(source),labels_sha256=sha(target),admission_sha256=sha(directory/'admission.json')),stream)
    return dict(path=path.name,sha256=sha(path),roots=len(packed),bytes=path.stat().st_size)


def main():
    parser=argparse.ArgumentParser();sub=parser.add_subparsers(dest='command',required=True)
    prep=sub.add_parser('prepare');prep.add_argument('--main',required=True);prep.add_argument('--neural',required=True);prep.add_argument('--output',required=True)
    prep.add_argument('--counts',nargs=3,type=int,default=(256,64,64));prep.add_argument('--seed',type=int,default=20261009)
    lab=sub.add_parser('label');lab.add_argument('--directory',required=True);lab.add_argument('--split',choices=('train','dev'),required=True)
    pack=sub.add_parser('pack');pack.add_argument('--directory',required=True);pack.add_argument('--split',choices=('train','dev'),required=True)
    args=parser.parse_args()
    if args.command=='prepare':result=prepare(args.main,args.neural,args.output,tuple(args.counts),args.seed)
    elif args.command=='label':result=label_split(args.directory,args.split,Limits(max_plies=16,nodes_per_action=4096,global_nodes=500000,wall_seconds=180))
    else:result=pack_split(args.directory,args.split)
    print(json.dumps(result,indent=2))

if __name__=='__main__':main()
