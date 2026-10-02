#!/usr/bin/env python3
"""Independent CPU corpus admission: bytes, source witnesses and rules only.

No packer/resident imports, Torch, teacher search, model or quality metrics.
The actual-data CLI requires a separately pinned serialization seal.
"""
import argparse
from decimal import Decimal, localcontext
import gzip
import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import tarfile
import time

COEFFICIENTS=(-142.72052667,372.35176398,-340.71073572,415.23490212)
PHASE=(1,3,3,5,9,0)
INTERVAL_ALGORITHM='sf19_reference_batch_nextafter_f32_total_width_v1'
MASK64=(1<<64)-1
CAP=64<<20
TERMINAL_POLICY='chen_current_terminal_v1_mate_stalemate_then_halfmove100_insufficient_threefold_window'
FORMATS={'int64':'q','int32':'i','int16':'h','float32':'f','uint8':'B'}
TABLES=[
 ('states.boards','int64','states',12),('states.fields','int32','states',5),
 ('states.hash','int64','states',1),('states.terminal','uint8','states',1),('states.family','int32','states',1),
 ('states.origin_context','int32','states',1),('states.origin_epoch','int64','states',1),('states.origin_node','int64','states',1),
 ('history.offsets','int64','state_offsets',1),('history.keys','int64','history',1),
 ('observations.state','int32','observations',1),('observations.query','int32','observations',1),
 ('observations.reported_cp','int32','observations',1),('observations.query_material','int32','observations',1),
 ('observations.midpoint','float32','observations',1),('observations.halfwidth','float32','observations',1),('observations.ordinary','uint8','observations',1),
 ('roots.state','int32','roots',1),('roots.query','int32','roots',1),('roots.family','int32','roots',1),
 ('roots.legal_count','int32','roots',1),('roots.edge_offsets','int64','root_offsets',1),
 ('edges.root','int32','edges',1),('edges.child_state','int32','edges',1),('edges.action','int16','edges',1),
 ('edges.reported_cp','int32','edges',1),('edges.midpoint','float32','edges',1),('edges.halfwidth','float32','edges',1),
 ('edges.rank','int16','edges',1),('edges.ordinary','uint8','edges',1)]


def require(condition,message):
    if not condition:raise ValueError(message)


def digest(path):
    value=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(1<<20),b''):value.update(block)
    return value.hexdigest()


def f32(value):
    return struct.unpack('<f',struct.pack('<f',value))[0]


def reference_interval(reported_cp,material,negated):
    """Exact declared binary32 algorithm plus independent Decimal witness."""
    require(type(reported_cp) is int and abs(reported_cp)<19754 and
            type(material) is int and 0<=material<=576 and type(negated) is bool,
            'ordinary integer CP/root material/sign required')
    def polynomial(m):
        a,b,c,d=COEFFICIENTS
        return ((a*m+b)*m+c)*m+d
    m=min(max(material,17),78)/58.0
    factor=polynomial(m)/polynomial(1.0)
    exact64=reported_cp*factor*(-1 if negated else 1)
    midpoint=f32(exact64)
    half=.5*factor
    guard=1e-9*(abs(exact64)+half+1)
    rounded=f32(half+abs(midpoint-exact64)+guard)
    require(math.isfinite(rounded) and rounded>=0,'interval cannot be represented')
    bits=struct.unpack('<I',struct.pack('<f',rounded))[0]
    width=struct.unpack('<f',struct.pack('<I',bits+1))[0]
    with localcontext() as context:
        context.prec=50
        a,b,c,d=map(lambda x:Decimal(str(x)),COEFFICIENTS)
        dm=Decimal(min(max(material,17),78))/Decimal(58)
        denominator=((a+b)+c)+d
        ratio=(((a*dm+b)*dm+c)*dm+d)/denominator
        center=Decimal(reported_cp)*ratio*(-1 if negated else 1)
        radius=ratio/2
        require(Decimal.from_float(midpoint)-Decimal.from_float(width)<=center-radius and
                Decimal.from_float(midpoint)+Decimal.from_float(width)>=center+radius,
                'packed interval narrows theoretical reporting interval')
    return midpoint,width


def canonical_action(board,move):
    import chess
    require(move in board.legal_moves,'action is not legal from source state')
    flip=0 if board.turn else 56
    promotion={None:0,chess.KNIGHT:1,chess.BISHOP:2,chess.ROOK:3,chess.QUEEN:4}[move.promotion]
    return (((move.from_square^flip)*64+(move.to_square^flip))*5)+promotion


def mix_key(offset):
    value=(0x4348454E00000000+offset+0x9E3779B97F4A7C15)&MASK64
    value=((value^(value>>30))*0xBF58476D1CE4E5B9)&MASK64
    value=((value^(value>>27))*0x94D049BB133111EB)&MASK64
    return value^(value>>31)


def exact_state(board):
    """Square-wise absolute-color boards and legal-EP repetition hash."""
    import chess
    pieces=[0]*12
    for square,piece in board.piece_map().items():
        pieces[piece.piece_type-1+(0 if piece.color else 6)]|=1<<square
    rights=sum(1<<i for i,(color,kingside) in enumerate(
        ((chess.WHITE,True),(chess.WHITE,False),(chess.BLACK,True),(chess.BLACK,False)))
        if (board.has_kingside_castling_rights(color) if kingside else board.has_queenside_castling_rights(color)))
    value=mix_key(768+rights)^(mix_key(792) if not board.turn else 0)
    if board.has_legal_en_passant():value^=mix_key(784+chess.square_file(board.ep_square))
    for plane,bits in enumerate(pieces):
        while bits:
            bit=bits&-bits;value^=mix_key(plane*64+bit.bit_length()-1);bits^=bit
    fields=[0 if board.turn else 1,rights,-1 if board.ep_square is None else board.ep_square,
            board.halfmove_clock,board.fullmove_number]
    require(0<=fields[3]<=65535 and 1<=fields[4]<(1<<32),'source clocks outside native packet bounds')
    return dict(boards=pieces,fields=fields,hash=value)


def source_position(moves):
    import chess
    board=chess.Board();previous=[]
    for uci in moves:
        previous.append(exact_state(board)['hash'])
        move=chess.Move.from_uci(uci)
        require(move in board.legal_moves,'illegal full game-prefix source move')
        board.push(move)
    return board,previous


def query_material(board):
    import chess
    return sum(len(board.pieces(piece,color))*PHASE[piece-1]
               for color in (chess.WHITE,chess.BLACK) for piece in range(1,7))


def strict_json(raw):
    def pairs(items):
        result={}
        for key,value in items:
            require(key not in result,'duplicate JSON key');result[key]=value
        return result
    def finite_float(text):
        value=float(text);require(math.isfinite(value),'overflowing JSON float');return value
    return json.loads(raw,object_pairs_hook=pairs,parse_float=finite_float,
                      parse_constant=lambda _:require(False,'nonfinite JSON constant'))


def dump(value):
    return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()


def typed_equal(left,right,message):
    require(dump(left)==dump(right),message)


def bounded_read(path,cap=CAP):
    path=Path(path)
    require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'invalid/oversize file')
    raw=path.read_bytes();require(len(raw)<=cap,'file grew beyond bound');return raw


def read_json(path,gz=False):
    raw=bounded_read(path)
    if gz:
        import io
        with gzip.GzipFile(fileobj=io.BytesIO(raw)) as stream:raw=stream.read(CAP+1)
        require(len(raw)<=CAP,'decompressed metadata cap')
    return strict_json(raw)


def keys(value,expected,label):
    require(type(value) is dict and set(value)==set(expected),label+' keys')


def sha(value):
    return type(value) is str and len(value)==64 and all(x in '0123456789abcdef' for x in value)


def read_binary_pack(path,manifest_sha,source_sha):
    """Independent canonical layout, zero-padding, dtype and hash admission."""
    root=Path(path)
    require(root.is_dir() and not root.is_symlink(),'pack path')
    require({p.name for p in root.iterdir()}=={'manifest.json','payload.bin','provenance.json','schedule.json'},'publication inventory')
    require(sha(manifest_sha) and digest(root/'manifest.json')==manifest_sha,'pack manifest pin')
    m=read_json(root/'manifest.json')
    keys(m,('kind','schema_version','split','source_kind','source_sha256','pins','byte_order','alignment','counts','payload_bytes','payload_sha256','provenance_bytes','provenance_sha256','schedule_bytes','schedule_sha256','tables','terminal_policy','frame_kind','interval_algorithm','origins'),'manifest')
    require(type(m['schema_version']) is int and m['schema_version']==1 and m['kind']=='immutable_cpu_semantic_corpus','wire schema')
    require(m['split'] in ('train','validation','test') and m['source_kind'] in ('audited_packet_v2','fixture_protocol_only'),'split/kind')
    require(sha(source_sha) and m['source_sha256']==source_sha,'packer source binding')
    require(m['byte_order']=='little' and type(m['alignment']) is int and m['alignment']==64,'byteorder/alignment')
    require(m['terminal_policy']==TERMINAL_POLICY and m['frame_kind']=='sf19_reference_material58_approx_cp' and m['interval_algorithm']==INTERVAL_ALGORITHM,'semantic policy')
    expected_origins=[dict(index=0,kind='CPU_reconstructed_exact_history',native_context_uuid=None,native_epoch=-1,native_node=-1,resolver='source_family_query_history_identity_no_native_arena_claim')]
    typed_equal(m['origins'],expected_origins,'native-origin claim')
    countnames=('states','state_offsets','history','observations','roots','root_offsets','edges','families','queries')
    keys(m['counts'],countnames,'counts');counts=m['counts']
    require(all(type(x) is int and 0<=x<2**31 for x in counts.values()),'count types')
    require(counts['state_offsets']==counts['states']+1 and counts['root_offsets']==counts['roots']+1,'CSR geometry')
    require(0<counts['states']<=65536 and 0<counts['families']<=4096 and 0<counts['roots']<=4096 and counts['observations']<=32768 and counts['edges']<=262144 and counts['history']<=4*1024**2,'schema row caps')
    keys(m['tables'],[t[0] for t in TABLES],'tables')
    specs={};offset=0
    for name,dtype,group,width in TABLES:
        offset=(offset+63)&~63;nbytes=counts[group]*width*struct.calcsize('<'+FORMATS[dtype])
        shape=[counts[group]]+([width] if width>1 else [])
        specs[name]=dict(dtype=dtype,shape=shape,offset=offset,nbytes=nbytes);offset+=nbytes
    size=(offset+63)&~63
    typed_equal(m['tables'],specs,'canonical segment specification')
    require(type(m['payload_bytes']) is int and m['payload_bytes']==size<=CAP,'payload geometry/cap')
    rawfiles={}
    for stem,name in (('payload','payload.bin'),('provenance','provenance.json'),('schedule','schedule.json')):
        raw=bounded_read(root/name)
        require(type(m[stem+'_bytes']) is int and len(raw)==m[stem+'_bytes'] and sha(m[stem+'_sha256']) and hashlib.sha256(raw).hexdigest()==m[stem+'_sha256'],'member bytes/hash')
        rawfiles[stem]=raw
    blob=rawfiles['payload'];values={};end=0
    for name,_,_,_ in TABLES:
        spec=specs[name];require(not any(blob[end:spec['offset']]),'nonzero alignment padding')
        end=spec['offset']+spec['nbytes']
        values[name]=tuple(x[0] for x in struct.iter_unpack('<'+FORMATS[spec['dtype']],blob[spec['offset']:end]))
    require(not any(blob[end:]),'nonzero tail padding')
    prov=strict_json(rawfiles['provenance']);schedule=strict_json(rawfiles['schedule'])
    keys(prov,('version','split','source_kind','pins','families','queries'),'provenance')
    require(type(prov['version']) is int and prov['version']==1,'provenance version type')
    require(prov['split']==m['split'] and prov['source_kind']==m['source_kind'],'provenance split/kind')
    typed_equal(prov['pins'],m['pins'],'provenance pins')
    verify_schedule(schedule,values)
    return m,prov,values,blob


def verify_schedule(schedule,values):
    seed=20261008
    orderkey=lambda kind,i:hashlib.sha256(f'{seed}:{kind}:{i}'.encode()).hexdigest()
    buckets={}
    for i,k in enumerate(values['roots.legal_count']):buckets.setdefault(k,[]).append(i)
    batches=[]
    for k in sorted(buckets):
        require(1<=k<=256,'complete root legal count')
        order=sorted(buckets[k],key=lambda i:orderkey('root',i))
        batches.extend(dict(legal_count=k,roots=order[j:j+4]) for j in range(0,len(order),4))
    obs=sorted(range(len(values['observations.state'])),key=lambda i:orderkey('observation',i))
    expected=dict(kind='unsealed_structural_exact_K_schedule_not_learning_config',version=1,seed=seed,algorithm='sha256_seed_domain_row_sort_v1',epochs=1,replacement=False,partial_tail='keep_complete_roots',root_batch_size=4,observation_batch_size=32,root_batches=batches,observation_batches=[obs[j:j+32] for j in range(0,len(obs),32)])
    typed_equal(schedule,expected,'structural schedule/complete coverage')


def source_score(cp,material,negated):
    reference_interval(cp,material,negated)
    def a(m):
        x=min(max(m,17),78)/58.;v,w,y,z=COEFFICIENTS
        return ((v*x+w)*x+y)*x+z
    factor=a(material)/a(58);value=cp*factor*(-1 if negated else 1)
    width=.5*factor;guard=1e-9*(abs(value)+width+1)
    return dict(frame_kind='sf19_reference_material58_approx_cp',reported_uci_cp=cp,normalization_root_material=material,normalization_a=a(material),reference_material=58,factor=factor,negated=negated,value_cp=value,quantization_halfwidth=width,interval_halfwidth=width+guard,lower_cp=value-width-guard,upper_cp=value+width+guard,source_score_kind='ordinary_unbounded_cp_no_tablebase')


def push_exact(board,move):
    board.push(move);board.halfmove_clock=min(board.halfmove_clock,65535);board.fullmove_number=min(board.fullmove_number,(1<<32)-1)


def native_terminal(board,history):
    """Independent declared terminal algebra; native sweep verifies it again."""
    pieces=exact_state(board)['boards'];minors=pieces[1]|pieces[2]|pieces[7]|pieces[8]
    insufficient=not any(pieces[i] for i in (0,3,4,6,9,10))
    if insufficient:
        insufficient=(minors.bit_count()<=1 or (not (pieces[1]|pieces[7]) and
                      (not(minors&0xaa55aa55aa55aa55) or not(minors&0x55aa55aa55aa55aa))))
    repetitions=1+sum(x==exact_state(board)['hash'] for x in (history[-board.halfmove_clock:] if board.halfmove_clock else []))
    return not any(board.legal_moves) or board.halfmove_clock>=100 or insufficient or repetitions>=3


def verify_relations(manifest,prov,values,blob):
    """Reconstruct source prefixes, joins and every wire element independently."""
    import chess
    expected={name:[] for name,_,_,_ in TABLES};expected['history.offsets']=[0];expected['roots.edge_offsets']=[0]
    states=[];roots=[];identities={};used=set()
    require(len(prov['families'])==manifest['counts']['families'] and len(prov['queries'])==manifest['counts']['queries'],'provenance row counts')
    qmap={};position_queries={}
    for qi,q in enumerate(prov['queries']):
        query=q['query'];qid=query['id'];base={k:v for k,v in query.items() if k!='id'}
        require(sha(qid) and hashlib.sha256(json.dumps(base,sort_keys=True).encode()).hexdigest()==qid and qid not in qmap,'query ID/bijection')
        require(query['split']==manifest['split'] and type(query['all_legal']) is bool,'query split/type')
        qmap[qid]=(qi,q)
        position_queries.setdefault((dump(query['position']),query['all_legal']),[]).append((qi,q))
    def add_state(board,history,fi):
        state=exact_state(board);identity=(*state['boards'],*state['fields'],tuple(history))
        if identity in identities:
            si=identities[identity];require(states[si]['family']==fi,'cross-family full-state collision');return si
        si=len(states);identities[identity]=si
        def signed(v,bits):return v-(1<<bits) if v>=1<<(bits-1) else v
        expected['states.boards'].extend(signed(x,64) for x in state['boards'])
        expected['states.fields'].extend(signed(x,32) for x in state['fields'])
        for name,value in (('hash',signed(state['hash'],64)),('terminal',int(native_terminal(board,history))),('family',fi),('origin_context',0),('origin_epoch',-1),('origin_node',-1)):expected['states.'+name].append(value)
        expected['history.keys'].extend(signed(x,64) for x in history);expected['history.offsets'].append(len(expected['history.keys']))
        # Authoritative history is the explicit retained hash sequence; the
        # oracle needs the diagram, not thousands of duplicated move stacks.
        states.append(dict(board=board.copy(stack=False),history=list(history),family=fi));return si
    def query_for(position,all_legal,board):
        matches=position_queries.get((dump(position),all_legal),[])
        require(len(matches)==1,'position query missing/duplicate');qi,q=matches[0];used.add(q['query']['id'])
        require(q['target_kind']==('stockfish_parent_root_normalized_action_cp' if all_legal else 'stockfish_standalone_root_normalized_cp'),'source target kind')
        require(type(q['query_material']) is int and q['query_material']==query_material(board),'actual query material')
        legal={canonical_action(board,m):m for m in board.legal_moves};rows=q['rows']
        require(len(rows)==(len(legal) if all_legal else 1),'whole support geometry')
        seen=set();previous=None
        for rank,row in enumerate(rows,1):
            require(type(row['rank']) is int and row['rank']==rank and type(row['action']) is int and row['action'] in legal and row['action'] not in seen and legal[row['action']].uci()==row['uci'],'rank/action bijection')
            seen.add(row['action']);cp=row['reported_cp']
            reference_interval(cp,q['query_material'],all_legal)
            require(previous is None or cp<=previous,'source rank monotonicity');previous=cp
            typed_equal(row['converted_score'],source_score(cp,q['query_material'],all_legal),'full source frame')
            require(type(row['pv']) is list and row['pv'] and row['pv'][0]==row['uci'],'source full PV identity')
            copy=board.copy(stack=True)
            for uci in row['pv']:
                move=chess.Move.from_uci(uci);require(move in copy.legal_moves,'source full PV legality');push_exact(copy,move)
            require(type(row['depth']) is int and row['depth']==q['packet_provenance']['selected_common_depth']>=6 and type(row['nodes']) is int and row['nodes']>0,'source selected telemetry')
        require(len({r['nodes'] for r in rows})==1 and (not all_legal or seen==set(legal)),'whole common packet')
        return qi,q,legal
    for fi,wrapped in enumerate(prov['families']):
        family=wrapped['source'];positions=family['positions'];boards=[];ids=[]
        require(hashlib.sha256(' '.join(family['all_moves']).encode()).hexdigest()==family['moves_sha256'],'source full mainline hash')
        for position in positions:
            board=chess.Board(wrapped['origin_fen']);history=[]
            require(board.is_valid() and position['moves']==family['all_moves'][:position['ply']] and len(position['moves'])==position['ply'] and position['game_id']==family['game_id'],'source prefix lineage')
            for uci in position['moves']:
                history.append(exact_state(board)['hash']);move=chess.Move.from_uci(uci)
                require(move in board.legal_moves,'source prefix illegal');push_exact(board,move)
            require(board.fen(en_passant='fen')==position['fen'] and history==position['previous_hashes'] and not native_terminal(board,history),'raw EP/history/nonterminal source')
            si=add_state(board,history,fi);boards.append((board,history));ids.append(si)
            qi,q,_=query_for(position,False,board);row=q['rows'][0];mid,width=reference_interval(row['reported_cp'],q['query_material'],False)
            for suffix,value in zip(('state','query','reported_cp','query_material','midpoint','halfwidth','ordinary'),(si,qi,row['reported_cp'],q['query_material'],mid,width,1)):expected['observations.'+suffix].append(value)
        index=family['action_root_index'];board,history=boards[index];qi,q,legal=query_for(positions[index],True,board);ri=len(roots)
        for suffix,value in zip(('state','query','family','legal_count'),(ids[index],qi,fi,len(legal))):expected['roots.'+suffix].append(value)
        edges=[]
        for row in sorted(q['rows'],key=lambda r:r['action']):
            child=board.copy(stack=True);push_exact(child,legal[row['action']]);childhistory=[*history,exact_state(board)['hash']];ci=add_state(child,childhistory,fi)
            mid,width=reference_interval(row['reported_cp'],q['query_material'],True)
            for suffix,value in zip(('root','child_state','action','reported_cp','midpoint','halfwidth','rank','ordinary'),(ri,ci,row['action'],row['reported_cp'],mid,width,row['rank'],int(not native_terminal(child,childhistory)))):expected['edges.'+suffix].append(value)
            edges.append((row['action'],row['uci'],ci))
        expected['roots.edge_offsets'].append(len(expected['edges.root']));roots.append(dict(state=ids[index],edges=edges))
    require(used==set(qmap),'unjoined source query')
    for name,dtype,_,_ in TABLES:
        spec=manifest['tables'][name];fmt='<'+FORMATS[dtype]
        raw=b''.join(struct.pack(fmt,v) for v in expected[name])
        require(len(raw)==spec['nbytes'] and blob[spec['offset']:spec['offset']+spec['nbytes']]==raw,'semantic byte mismatch: '+name)
    require(len(states)==manifest['counts']['states'] and len(roots)==manifest['counts']['roots'],'full state/root counts')
    return states,roots


def verify_archived_sources(manifest,prov,archive,prep,seal_path,seal_sha,packer_source):
    """Compare actual provenance to externally pinned immutable raw records.

    Only this split's accepted records are JSON-parsed. Other archive members
    are streamed for inventory/hash checking, never treated as learning inputs.
    """
    archive=Path(archive);prep=Path(prep)
    require(prov['source_kind']==manifest['source_kind']=='audited_packet_v2','fixture cannot masquerade as actual corpus')
    require(sha(seal_sha) and digest(seal_path)==seal_sha,'external serialization seal pin')
    seal=read_json(seal_path)
    keys(seal,('kind','version','status','split','source_sha256','pins','limits'),'serialization seal')
    require(seal['kind']=='cpu_semantic_serialization_seal' and type(seal['version']) is int and seal['version']==1 and seal['status']=='sealed_no_training' and seal['split']==manifest['split'],'serialization authority/split')
    require(seal['source_sha256']==manifest['source_sha256']==digest(packer_source),'exact serialization source')
    keys(seal['limits'],('payload_bytes','metadata_bytes','states','roots','observations','edges','history','families','record_bytes'),'sealed limits')
    require(all(type(v) is int and v>0 for v in seal['limits'].values()),'sealed limit types')
    require(manifest['payload_bytes']<=seal['limits']['payload_bytes'] and manifest['provenance_bytes']<=seal['limits']['metadata_bytes'] and manifest['schedule_bytes']<=seal['limits']['metadata_bytes'],'sealed byte limits')
    for name in ('states','roots','observations','edges','history','families'):
        require(manifest['counts'][name]<=seal['limits'][name],'sealed row limit')
    auditname='independent-audit/packet-v2-independent-mechanical-audit-2026-10-01.json'
    pins=dict(teacher_archive_manifest_sha256=digest(archive/'manifest.json'),teacher_audit_sha256=digest(archive/auditname),teacher_declaration_sha256=digest(archive/'declaration.json'),input_admission_sha256=digest(archive/'input-admission.json.gz'))
    typed_equal(seal['pins'],pins,'serialization source pins');typed_equal(manifest['pins'],pins,'corpus source pins')
    archived=read_json(archive/'manifest.json')
    for name,h in archived['files_sha256'].items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts and sha(h) and digest(archive/name)==h,'archived evidence pin')
    require(digest(prep/'manifest.json')==archived['preparation_manifest_sha256'],'preparation manifest pin')
    preparation=read_json(prep/'manifest.json')
    for name,h in preparation['files_sha256'].items():
        require(type(name) is str and not Path(name).is_absolute() and '..' not in Path(name).parts and sha(h) and digest(prep/name)==h,'preparation artifact closure')
    audit=read_json(archive/auditname);declaration=read_json(archive/'declaration.json');report=read_json(archive/'report.json')
    require(audit['status']=='pass' and audit['record_count']==3456 and audit['every_record_hash_query_command_packet_frame_and_history_checked'] is True and audit['independent_grammar_emission_selection_and_trace_checked'] is True,'upstream mechanical audit')
    require(audit['report_sha256']==digest(archive/'report.json') and audit['admission_sha256']==pins['input_admission_sha256'] and audit['declaration_sha256']==pins['teacher_declaration_sha256'],'audit hash closure')
    typed_equal(audit['frozen_source_sha256'],declaration['source_sha256'],'frozen source map')
    require(len(declaration['source_sha256'])==16,'frozen source count')
    for name,h in declaration['source_sha256'].items():
        require(name.startswith('neural/') and '..' not in Path(name).parts and digest(prep/name)==h,'frozen preparation source')
    require(digest(prep/'neural/evidence/packet-v2-freeze-2026-10-01.json')==audit['preparation_freeze_sha256'],'preparation freeze pin')
    require(report['status']=='complete' and type(report['worker_exitcodes']) is list and len(report['worker_exitcodes'])==2 and all(type(v) is int and v==0 for v in report['worker_exitcodes']),'faulted source attempt')
    require(report['admission_sha256']==pins['input_admission_sha256'] and report['declaration_sha256']==pins['teacher_declaration_sha256'],'attempt source pins')
    frame=declaration['score_frame']
    typed_equal(frame['normalization_coefficients'],list(COEFFICIENTS),'normalization coefficients')
    typed_equal(frame['material_weights'],list(PHASE),'normalization material weights')
    typed_equal(frame['material_clamp'],[17,78],'normalization clamp')
    require(frame['kind']=='sf19_reference_material58_approx_cp' and type(frame['numeric_guard_relative']) is float and frame['numeric_guard_relative']==1e-9 and frame['batch_halfwidth']=='float32_outward_including_midpoint_cast_error','frame provenance')
    admission=read_json(archive/'input-admission.json.gz',gz=True)
    accepted=report['whole_family_accepted'][manifest['split']]
    require(type(accepted) is list and all(sha(x) for x in accepted) and len(accepted)==len(set(accepted)),'whole-family IDs')
    families=[f for f in admission['source_records'][manifest['split']] if f['game_id'] in set(accepted)]
    require(len(families)==len(accepted)>0,'whole-family admission coverage')
    import chess
    typed_equal(prov['families'],[dict(source=f,origin_fen=chess.STARTING_FEN) for f in families],'source families/lineages')
    plan=read_json(archive/'full-query-plan.json.gz',gz=True)
    require(len(plan)==3456 and hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()==declaration['query_plan_sha256'],'cold query plan')
    wanted=[q for q in plan if q['split']==manifest['split'] and q['position']['game_id'] in set(accepted)]
    typed_equal([q['query'] for q in prov['queries']],wanted,'exact accepted query order/coverage')
    byid={q['query']['id']:q for q in prov['queries']}
    entries=read_json(archive/'raw-member-sha256.json');membermap={}
    for row in entries:
        keys(row,('path','bytes','sha256'),'raw member')
        require(type(row['path']) is str and row['path'] not in membermap and type(row['bytes']) is int and 0<row['bytes']<=8<<20 and sha(row['sha256']),'raw member geometry')
        membermap[row['path']]=row
    require(len(membermap)==3460 and set(membermap)=={q['id']+'.json' for q in plan}|{'report.json','declaration.json','worker-0-setup.json','worker-1-setup.json'},'raw archive inventory')
    typed_equal({x['query_id']:x['sha256'] for x in report['source_records']},{q['id']:membermap[q['id']+'.json']['sha256'] for q in plan},'raw report fingerprints')
    seen=set();checked=0
    with tarfile.open(archive/'raw-records.tar.gz','r:gz') as tar:
        for info in tar:
            require(info.isfile() and info.name in membermap and info.name not in seen and info.size==membermap[info.name]['bytes'],'tar member identity')
            seen.add(info.name);stream=tar.extractfile(info);hasher=hashlib.sha256();raw=None
            if info.name[:-5] in byid:
                raw=stream.read((8<<20)+1);require(len(raw)==info.size,'target record size');hasher.update(raw)
            else:
                for block in iter(lambda:stream.read(1<<20),b''):hasher.update(block)
            require(hasher.hexdigest()==membermap[info.name]['sha256'],'raw archive member SHA')
            if raw is None:continue
            record=strict_json(raw);q=byid[info.name[:-5]];labels=record['labels']
            require(record['rejection'] is None and type(labels) is dict,'rejected raw query')
            typed_equal(record['query'],q['query'],'raw query identity')
            require(q['source_record_sha256']==hasher.hexdigest() and q['transcript_sha256']==hashlib.sha256(dump(record['transcript'])).hexdigest(),'raw record/transcript pin')
            typed_equal(q['packet_provenance'],record['packet_provenance'],'raw packet selection')
            require(q['target_kind']==labels['target_kind'] and type(q['query_material']) is int and q['query_material']==labels['normalization_root_material'],'raw frame/kind')
            expectedrows=[dict(uci=r['uci'],action=r['canonical_action'],reported_cp=r['root_score_cp'],rank=i+1,depth=r['reported_depth'],seldepth=r['seldepth'],nodes=r['nodes'],pv=r['pv'],converted_score=r['converted_score']) for i,r in enumerate(labels['rows'])]
            typed_equal(q['rows'],expectedrows,'raw PV/action/score provenance');checked+=1
    require(seen==set(membermap) and checked==len(wanted),'actual raw query coverage')
    return dict(archive_manifest_sha256=pins['teacher_archive_manifest_sha256'],serialization_seal_sha256=seal_sha,query_records_verified=checked,source_modules_verified=16,raw_archive_member_hashes_verified=len(seen))


def verify_native_oracle(states,roots,binary,build_manifest,build_sha,workspace):
    """One bounded no-score CPU oracle process verifies states and full support."""
    import tempfile
    require(sha(build_sha) and digest(build_manifest)==build_sha,'oracle build manifest pin')
    build=read_json(build_manifest);workspace=Path(workspace).resolve();binary=Path(binary).resolve()
    require(build['kind']=='CPU_corpus_oracle_source_build_binding' and build['GPU_runtime_calls']==0 and digest(binary)==build['binary_sha256'],'CPU oracle executable provenance')
    closure=build['compiled_crate_source_sha256'];require(len(closure)==18,'oracle complete source closure')
    for name,h in closure.items():
        path=(workspace/name).resolve()
        require(path.is_relative_to(workspace) and sha(h) and digest(path)==h,'oracle build source binding')
    work=[(s,None) for s in states]+[(states[r['state']],r) for r in roots]
    lines=[]
    for state,root in work:
        line=state['board'].fen(en_passant='fen')+'\t'+','.join(f'{x:x}' for x in state['history'])
        lines.append(line+('\tchildren' if root else '')+'\n')
    def compare(snapshot,state):
        actual=exact_state(state['board'])
        for field in ('boards','fields','hash'):typed_equal(snapshot[field],actual[field],'native exact '+field)
        require(type(snapshot['terminal']) is bool and snapshot['terminal']==native_terminal(state['board'],state['history']),'native terminal/history parity')
        legal=sorted((canonical_action(state['board'],m),m.uci()) for m in state['board'].legal_moves)
        typed_equal(snapshot['legal'],[dict(action=a,uci=u) for a,u in legal],'native complete action support')
    with tempfile.TemporaryDirectory(prefix='chen-corpus-native-audit-') as tmp:
        output=Path(tmp)/'oracle.jsonl'
        with output.open('wb') as stream:
            proc=subprocess.run([str(binary)],input=''.join(lines).encode(),stdout=stream,stderr=subprocess.PIPE,timeout=120)
        require(proc.returncode==0 and output.stat().st_size<=256<<20,'bounded native oracle failure/output cap')
        count=0
        with output.open('rb') as stream:
            for index,line in enumerate(stream):
                require(index<len(work) and len(line)<=2<<20,'native output row cap')
                result=strict_json(line);state,root=work[index]
                require(result['row']==index and result['policy']==TERMINAL_POLICY,'native row/policy')
                compare(result['state'],state)
                if root:
                    require(len(result['children'])==len(root['edges']),'native full child expansion')
                    for child,(action,uci,si) in zip(result['children'],root['edges']):
                        require(child['action']==action and child['uci']==uci,'native root/action child join');compare(child['state'],states[si])
                count+=1
        require(count==len(work),'native snapshot count')
    return dict(binary_sha256=digest(binary),build_manifest_sha256=build_sha,compiled_source_files_verified=len(closure),state_snapshots_verified=len(states),full_root_expansions_verified=len(roots),legal_child_snapshots_verified=sum(len(r['edges']) for r in roots))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('pack','manifest-sha256','packer-source','archive','preparation-archive','seal','seal-sha256','oracle','oracle-build','oracle-build-sha256','output'):
        parser.add_argument('--'+name,required=True)
    parser.add_argument('--workspace',default=str(Path(__file__).resolve().parents[1]))
    args=parser.parse_args();started=time.monotonic()
    m,prov,values,blob=read_binary_pack(args.pack,args.manifest_sha256,digest(args.packer_source))
    source=verify_archived_sources(m,prov,args.archive,args.preparation_archive,args.seal,args.seal_sha256,args.packer_source)
    states,roots=verify_relations(m,prov,values,blob)
    native=verify_native_oracle(states,roots,args.oracle,args.oracle_build,args.oracle_build_sha256,args.workspace)
    result=dict(kind='independent_CPU_semantic_corpus_mechanical_audit',status='pass',split=m['split'],counts=m['counts'],payload_bytes=m['payload_bytes'],manifest_sha256=args.manifest_sha256,payload_sha256=m['payload_sha256'],packer_source_sha256=m['source_sha256'],auditor_source_sha256=digest(__file__),source=source,native_oracle=native,all_tables_byte_exact_against_independent_source_reconstruction=True,all_full_PVs_and_common_frame_intervals_verified=True,complete_legal_root_support_and_terminal_masks_verified=True,quality_error_ranking_metrics_computed=False,model_training=False,GPU_runtime_calls=0,elapsed_seconds=time.monotonic()-started)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    with output.open('xb') as stream:stream.write(dump(result))
    print(json.dumps({k:result[k] for k in ('status','split','counts','elapsed_seconds')},sort_keys=True))


if __name__=='__main__':main()
