"""CPU-only immutable semantic corpus schema1. No Torch, engine or GPU imports.

Actual audited packing requires an externally pinned CPU serialization seal.
Reader validates relational semantics by independent state/action reconstruction.
"""
from __future__ import annotations
from collections import OrderedDict
from dataclasses import dataclass, asdict
import ctypes
import gzip
import hashlib
import io
import json
import math
import os
from pathlib import Path
import shutil
import struct
import tarfile
import tempfile
from types import MappingProxyType

SCHEMA_VERSION=1
ALIGNMENT=64
TERMINAL_POLICY='chen_current_terminal_v1_mate_stalemate_then_halfmove100_insufficient_threefold_window'
INTERVAL_ALGORITHM='sf19_reference_batch_nextafter_f32_total_width_v1'
FRAME_KIND='sf19_reference_material58_approx_cp'
AS=(-142.72052667,372.35176398,-340.71073572,415.23490212)
PHASE_WEIGHTS=(1,3,3,5,9,0)
SPLITS=('train','validation','test')
FORMATS={'int64':'q','int32':'i','int16':'h','float32':'f','uint8':'B'}
SOURCE_KEYS={'teacher_archive_manifest_sha256','teacher_audit_sha256','teacher_declaration_sha256','input_admission_sha256'}
TABLES=OrderedDict((name,(dtype,group,tail)) for name,dtype,group,tail in (
 ('states.boards','int64','states',(12,)),('states.fields','int32','states',(5,)),
 ('states.hash','int64','states',()),('states.terminal','uint8','states',()),('states.family','int32','states',()),
 ('states.origin_context','int32','states',()),('states.origin_epoch','int64','states',()),('states.origin_node','int64','states',()),
 ('history.offsets','int64','state_offsets',()),('history.keys','int64','history',()),
 ('observations.state','int32','observations',()),('observations.query','int32','observations',()),
 ('observations.reported_cp','int32','observations',()),('observations.query_material','int32','observations',()),
 ('observations.midpoint','float32','observations',()),('observations.halfwidth','float32','observations',()),('observations.ordinary','uint8','observations',()),
 ('roots.state','int32','roots',()),('roots.query','int32','roots',()),('roots.family','int32','roots',()),
 ('roots.legal_count','int32','roots',()),('roots.edge_offsets','int64','root_offsets',()),
 ('edges.root','int32','edges',()),('edges.child_state','int32','edges',()),('edges.action','int16','edges',()),
 ('edges.reported_cp','int32','edges',()),('edges.midpoint','float32','edges',()),('edges.halfwidth','float32','edges',()),
 ('edges.rank','int16','edges',()),('edges.ordinary','uint8','edges',())))

class CorpusError(ValueError):pass

@dataclass(frozen=True)
class Limits:
    payload_bytes:int=64*1024**2
    metadata_bytes:int=64*1024**2
    states:int=65536
    roots:int=4096
    observations:int=32768
    edges:int=262144
    history:int=4*1024**2
    families:int=4096
    record_bytes:int=8*1024**2
    def validate(self):
        if any(type(x) is not int or x<1 for x in asdict(self).values()):raise CorpusError('positive integer limits required')
        if max(self.states,self.roots,self.observations,self.edges,self.history,self.families)>=2**31:raise CorpusError('signed-int32 row capacity exceeded')


def _require(ok,message):
    if not ok:raise CorpusError(message)
def _keys(value,keys,name):_require(type(value) is dict and set(value)==set(keys),name+' keys differ')
def _int(x,lo,hi):return type(x) is int and lo<=x<=hi
def _sha_value(x):return type(x) is str and len(x)==64 and all(c in '0123456789abcdef' for c in x)
def sha256(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(1048576),b''):h.update(b)
    return h.hexdigest()
def source_sha256():return sha256(__file__)
def _dump(value):return (json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
def _json(raw):
    def pairs(items):
        d={}
        for key,value in items:
            if key in d:raise CorpusError('duplicate JSON key')
            d[key]=value
        return d
    def finite_float(token):
        value=float(token)
        _require(math.isfinite(value),'nonfinite JSON exponent')
        return value
    try:return json.loads(raw,object_pairs_hook=pairs,parse_float=finite_float,parse_constant=lambda x:(_ for _ in ()).throw(CorpusError('nonfinite JSON')))
    except (ValueError,UnicodeError) as e:raise CorpusError('invalid strict JSON: '+str(e)) from e

def _read(path,cap):
    path=Path(path);_require(path.is_file() and not path.is_symlink() and path.stat().st_size<=cap,'file exceeds cap or invalid owner path')
    with path.open('rb') as f:
        b=f.read(cap+1)
    _require(len(b)<=cap,'file grew beyond cap');return b

def _gunzip(raw,cap):
    with gzip.GzipFile(fileobj=io.BytesIO(raw)) as f:b=f.read(cap+1)
    _require(len(b)<=cap,'decompressed metadata cap exceeded');return b

def _signed(value,bits):return value if value<1<<(bits-1) else value-(1<<bits)
def _f32(value):return struct.unpack('<f',struct.pack('<f',value))[0]
def _a(material):
    m=min(max(material,17),78)/58.0
    return ((AS[0]*m+AS[1])*m+AS[2])*m+AS[3]

def source_score_record(cp,material,*,negate):
    _require(_int(cp,-19753,19753) and _int(material,0,576) and type(negate) is bool,'typed ordinary source score')
    factor=_a(material)/_a(58);value=cp*factor*(-1 if negate else 1);width=.5*factor;guard=1e-9*(abs(value)+width+1)
    return dict(frame_kind=FRAME_KIND,reported_uci_cp=cp,normalization_root_material=material,normalization_a=_a(material),
                reference_material=58,factor=factor,negated=negate,value_cp=value,quantization_halfwidth=width,
                interval_halfwidth=width+guard,lower_cp=value-width-guard,upper_cp=value+width+guard,
                source_score_kind='ordinary_unbounded_cp_no_tablebase')


def score_interval(cp,material,*,negate):
    _require(_int(cp,-19753,19753) and _int(material,0,576) and type(negate) is bool,'ordinary CP/query material/sign required')
    factor=_a(material)/_a(58);exact=cp*factor*(-1 if negate else 1)
    guard=1e-9*(abs(exact)+.5*factor+1)
    midpoint=_f32(exact);total=.5*factor+abs(midpoint-exact)+guard
    rounded=_f32(total);bits=struct.unpack('<I',struct.pack('<f',rounded))[0]
    _require(bits<0x7f7fffff,'interval float32 overflow')
    width=struct.unpack('<f',struct.pack('<I',bits+1))[0]
    _require(midpoint-width<=exact-.5*factor-guard and midpoint+width>=exact+.5*factor+guard,'interval narrowing')
    return midpoint,width


def repetition_hash(board):
    import chess
    def key(offset):
        mask=(1<<64)-1;v=(0x4348454E00000000+offset+0x9E3779B97F4A7C15)&mask
        v=((v^(v>>30))*0xBF58476D1CE4E5B9)&mask;v=((v^(v>>27))*0x94D049BB133111EB)&mask
        return v^(v>>31)
    rights=_rights(board);value=key(768+rights)^(key(792) if not board.turn else 0)
    if board.has_legal_en_passant():value^=key(784+chess.square_file(board.ep_square))
    for sq,piece in board.piece_map().items():value^=key((piece.piece_type-1+(0 if piece.color else 6))*64+sq)
    return value

def _rights(board):
    import chess
    return sum(1<<i for i,(c,k) in enumerate(((chess.WHITE,1),(chess.WHITE,0),(chess.BLACK,1),(chess.BLACK,0)))
               if (board.has_kingside_castling_rights(c) if k else board.has_queenside_castling_rights(c)))

def native_insufficient(board):
    boards=_boards(board)
    if any(boards[i] for i in (0,3,4,6,9,10)):return False
    knights=boards[1]|boards[7];bishops=boards[2]|boards[8]
    if (knights|bishops).bit_count()<=1:return True
    if knights:return False
    return not(bishops&0xaa55aa55aa55aa55) or not(bishops&0x55aa55aa55aa55aa)

def terminal(board,history):
    key=repetition_hash(board)
    return (not any(board.legal_moves) or board.halfmove_clock>=100 or native_insufficient(board) or
            1+sum(h==key for h in history[-board.halfmove_clock:] if board.halfmove_clock)>2)

def _boards(board):
    import chess
    return [board.pieces_mask(kind,color) for color in (chess.WHITE,chess.BLACK) for kind in range(1,7)]

def _material(board):return sum(v.bit_count()*w for v,w in zip(_boards(board),PHASE_WEIGHTS*2))
def canonical_action(move,turn):
    import chess
    flip=0 if turn else 56;promo={None:0,chess.KNIGHT:1,chess.BISHOP:2,chess.ROOK:3,chess.QUEEN:4}[move.promotion]
    return ((move.from_square^flip)*64+(move.to_square^flip))*5+promo

def _fields(board):
    _require(0<=board.halfmove_clock<=65535 and 1<=board.fullmove_number<2**32,'exact clock range')
    return [int(not board.turn),_rights(board),-1 if board.ep_square is None else board.ep_square,board.halfmove_clock,board.fullmove_number]

def _push(board,move):
    board.push(move);board.halfmove_clock=min(board.halfmove_clock,65535);board.fullmove_number=min(board.fullmove_number,2**32-1)

def _history_digest(history):return hashlib.sha256(b''.join(struct.pack('<Q',h) for h in history)).hexdigest()
def _state_identity(board,history):return hashlib.sha256(struct.pack('<12QIIiII',*_boards(board),*_fields(board))+bytes.fromhex(_history_digest(history))).hexdigest()


def _position(origin,row):
    import chess
    _keys(row,('game_id','ply','fen','moves','previous_hashes'),'position')
    _require(_sha_value(row['game_id']) and _int(row['ply'],0,512) and type(row['moves']) is list and len(row['moves'])==row['ply'],'position prefix geometry')
    _require(type(row['fen']) is str and len(row['fen'])<=256 and type(row['previous_hashes']) is list and len(row['previous_hashes'])==row['ply'],'position history geometry')
    board=chess.Board(origin);history=[]
    _require(board.is_valid(),'invalid origin position')
    for uci in row['moves']:
        _require(type(uci) is str and len(uci)<=5,'move schema');history.append(repetition_hash(board))
        try:move=chess.Move.from_uci(uci)
        except ValueError as e:raise CorpusError('malformed prefix move') from e
        _require(move in board.legal_moves,'illegal prefix move');_push(board,move)
    _require(row['fen']==board.fen(en_passant='fen') and row['previous_hashes']==history,'prefix/rawEP/history identity differs')
    return board,history


def _layout(counts):
    result={};offset=0
    for name,(dtype,group,tail) in TABLES.items():
        shape=[counts[group],*tail];n=math.prod(shape)*struct.calcsize('<'+FORMATS[dtype]);offset=(offset+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT
        result[name]=dict(dtype=dtype,shape=shape,offset=offset,nbytes=n);offset+=n
    return result,(offset+ALIGNMENT-1)//ALIGNMENT*ALIGNMENT

def _assemble(provenance,limits):
    """Rebuild all tables from full histories and typed query witnesses."""
    import chess
    limits.validate();_keys(provenance,('version','split','source_kind','pins','families','queries'),'provenance')
    _require(provenance['version']==1 and type(provenance['version']) is int and provenance['split'] in SPLITS,'provenance schema/split')
    _require(provenance['source_kind'] in ('audited_packet_v2','fixture_protocol_only'),'source kind')
    _keys(provenance['pins'],SOURCE_KEYS,'cold source pins');_require(all(_sha_value(x) for x in provenance['pins'].values()),'source pin grammar')
    families=provenance['families'];queries=provenance['queries'];_require(type(families) is list and 0<len(families)<=limits.families and type(queries) is list,'family/query tables')
    _require(len(queries)<=limits.observations+limits.roots,'query count cap')
    byquery={};query_index={}
    for index,q in enumerate(queries):
        _keys(q,('query','source_record_sha256','transcript_sha256','packet_provenance','target_kind','query_material','rows'),'query provenance')
        _require(_sha_value(q['source_record_sha256']) and _sha_value(q['transcript_sha256']),'query source hash')
        query=q['query'];_keys(query,('split','position','all_legal','nodes','id'),'query')
        _keys(query['position'],('game_id','ply','fen','moves','previous_hashes'),'query position')
        _require(_int(query['position']['ply'],0,512) and type(q['rows']) is list and 0<len(q['rows'])<=256,'query preflight geometry')
        _require(query['split']==provenance['split'] and type(query['all_legal']) is bool and _int(query['nodes'],1,256000),'query split/budget/kind')
        payload={k:v for k,v in query.items() if k!='id'}
        _require(_sha_value(query['id']) and hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()==query['id'],'query ID differs')
        _require(query['id'] not in byquery,'duplicate query ID');byquery[query['id']]=q;query_index[query['id']]=index
    _require(all(type(f) is dict and type(f.get('source')) is dict and type(f['source'].get('positions')) is list and all(type(p) is dict and _int(p.get('ply'),0,512) for p in f['source']['positions']) for f in families),'family preflight geometry')
    observations=sum(len(f['source']['positions']) for f in families)
    edges=sum(len(q['rows']) for q in queries if q['query']['all_legal'])
    hist=sum(p['ply'] for f in families for p in f['source']['positions'])+sum(len(q['rows'])*(q['query']['position']['ply']+1) for q in queries if q['query']['all_legal'])
    _require(observations<=limits.observations and len(families)<=limits.roots and edges<=limits.edges and observations+edges<=limits.states and hist<=limits.history,'preflight row/history caps before allocation')
    upper=dict(states=observations+edges,state_offsets=observations+edges+1,history=hist,observations=observations,roots=len(families),root_offsets=len(families)+1,edges=edges)
    _require(_layout(upper)[1]<=limits.payload_bytes,'preflight payload cap before allocation')
    data={k:[] for k in TABLES};data['history.offsets']=[0];data['roots.edge_offsets']=[0]
    state_meta=[];state_map={};family_ids=set();used_queries=set();root_meta=[]
    def add_state(board,history,fi):
        identity=_state_identity(board,history)
        if identity in state_map:
            si=state_map[identity];_require(data['states.family'][si]==fi,'cross-family exact state collision');return si
        si=len(state_meta);_require(si<limits.states and len(data['history.keys'])+len(history)<=limits.history,'state/history capacity')
        state_map[identity]=si;h=repetition_hash(board);fields=_fields(board)
        data['states.boards'].extend(_signed(x,64) for x in _boards(board));data['states.fields'].extend(_signed(x,32) if x>=0 else x for x in fields)
        for name,value in (('hash',_signed(h,64)),('terminal',int(terminal(board,history))),('family',fi),('origin_context',0),('origin_epoch',-1),('origin_node',-1)):data['states.'+name].append(value)
        data['history.keys'].extend(_signed(x,64) for x in history);data['history.offsets'].append(len(data['history.keys']))
        state_meta.append(dict(identity=identity,fen=board.fen(en_passant='fen'),history_sha256=_history_digest(history),family=fi,origin_context=0))
        return si
    def find_query(row,all_legal):
        matches=[q for q in queries if q['query']['position']==row and q['query']['all_legal'] is all_legal]
        _require(len(matches)==1,'missing/duplicate typed query for position');q=matches[0];qid=q['query']['id'];used_queries.add(qid);return q,query_index[qid]
    def check_labels(q,board,all_legal):
        _require(q['target_kind']==('stockfish_parent_root_normalized_action_cp' if all_legal else 'stockfish_standalone_root_normalized_cp'),'teacher perspective/query kind')
        _require(type(q['query_material']) is int and q['query_material']==_material(board),'wrong search-root normalization material')
        packet=q['packet_provenance'];_require(type(packet) is dict and packet.get('protocol_version')==2 and type(packet.get('protocol_version')) is int and packet.get('query_rejection') is None,'ordinary packet provenance')
        _require(_int(packet.get('selected_emission_ordinal'),1,1000000) and _int(packet.get('selected_common_depth'),6,512),'packet selection depth/ordinal')
        legal={canonical_action(m,board.turn):m for m in board.legal_moves};rows=q['rows'];_require(type(rows) is list and len(rows)==(len(legal) if all_legal else 1) and 0<len(rows)<=256,'complete query action geometry')
        seen=set();last_cp=None
        for rank,row in enumerate(rows,1):
            _keys(row,('uci','action','reported_cp','rank','depth','seldepth','nodes','pv','converted_score'),'teacher row')
            _require(_int(row['action'],0,20479) and row['action'] in legal and legal[row['action']].uci()==row['uci'] and row['action'] not in seen,'unknown/duplicate action join')
            _require(type(row['pv']) is list and 0<len(row['pv'])<=512 and row['pv'][0]==row['uci'],'source PV/action identity')
            pvboard=board.copy(stack=True)
            for uci in row['pv']:
                _require(type(uci) is str and len(uci)<=5,'source PV move schema')
                try:move=chess.Move.from_uci(uci)
                except ValueError as e:raise CorpusError('malformed source PV') from e
                _require(move in pvboard.legal_moves,'illegal source PV');_push(pvboard,move)
            witness=source_score_record(row['reported_cp'],q['query_material'],negate=all_legal)
            _require(json.dumps(row['converted_score'],sort_keys=True,allow_nan=False)==json.dumps(witness,sort_keys=True,allow_nan=False),'source converted frame/sign/interval differs')
            seen.add(row['action']);_require(type(row['rank']) is int and row['rank']==rank,'rank order differs')
            _require(_int(row['reported_cp'],-19753,19753) and (last_cp is None or last_cp>=row['reported_cp']),'nonordinary/nonmonotonic score ranks');last_cp=row['reported_cp']
            _require(row['depth']==packet['selected_common_depth'] and type(row['depth']) is int and _int(row['seldepth'],0,512) and _int(row['nodes'],1,2**63-1),'row source telemetry')
        _require(len({r['nodes'] for r in rows})==1,'mixed emission node stamp')
        _require(not all_legal or seen==set(legal),'incomplete all-legal support');return rows,legal
    for fi,wrapped in enumerate(families):
        _keys(wrapped,('source','origin_fen'),'family provenance');family=wrapped['source']
        _keys(family,('game_id','game_sha256','moves_sha256','source_offset','source_end_offset','ratings','time_control','positions','action_root_index','all_moves'),'family source')
        _require(_sha_value(family['game_id']) and family['game_id'] not in family_ids and _sha_value(family['game_sha256']) and _sha_value(family['moves_sha256']),'family identity');family_ids.add(family['game_id'])
        _require(_int(family['source_offset'],0,2**63-1) and _int(family['source_end_offset'],family['source_offset']+1,2**63-1),'PGN byte span')
        _require(type(family['all_moves']) is list and len(family['all_moves'])<=512 and all(type(m) is str and len(m)<=5 for m in family['all_moves']),'family full-prefix geometry')
        _require(hashlib.sha256(' '.join(family['all_moves']).encode()).hexdigest()==family['moves_sha256'],'mainline hash')
        _require(type(family['ratings']) is list and len(family['ratings'])==2 and all(_int(x,0,5000) for x in family['ratings']) and type(family['time_control']) is str and len(family['time_control'])<=64,'family metadata types')
        positions=family['positions'];_require(type(positions) is list and 0<len(positions)<=8 and _int(family['action_root_index'],0,len(positions)-1),'family positions/root index')
        if provenance['source_kind']=='audited_packet_v2':_require(len(positions)==8 and wrapped['origin_fen']==chess.STARTING_FEN,'audited family source geometry')
        boards=[];state_ids=[]
        for row in positions:
            _require(row['game_id']==family['game_id'] and row['moves']==family['all_moves'][:row['ply']],'family lineage')
            board,history=_position(wrapped['origin_fen'],row);_require(not terminal(board,history),'terminal standalone source')
            si=add_state(board,history,fi);boards.append((board,history));state_ids.append(si)
            q,qi=find_query(row,False);labelrows,_=check_labels(q,board,False);mid,width=score_interval(labelrows[0]['reported_cp'],q['query_material'],negate=False)
            _require(len(data['observations.state'])<limits.observations,'observation capacity')
            values=(si,qi,labelrows[0]['reported_cp'],q['query_material'],mid,width,1)
            for suffix,value in zip(('state','query','reported_cp','query_material','midpoint','halfwidth','ordinary'),values):data['observations.'+suffix].append(value)
        board,history=boards[family['action_root_index']];si=state_ids[family['action_root_index']];q,qi=find_query(positions[family['action_root_index']],True);rows,legal=check_labels(q,board,True)
        ri=len(root_meta);_require(ri<limits.roots and len(data['edges.root'])+len(rows)<=limits.edges,'root/edge capacity')
        for suffix,value in zip(('state','query','family','legal_count'),(si,qi,fi,len(rows))):data['roots.'+suffix].append(value)
        root_meta.append(dict(family=fi,query=qi,state=si,source_query_id=q['query']['id']))
        for row in sorted(rows,key=lambda r:r['action']):
            child=board.copy(stack=True);_push(child,legal[row['action']]);child_history=[*history,repetition_hash(board)];ci=add_state(child,child_history,fi)
            mid,width=score_interval(row['reported_cp'],q['query_material'],negate=True)
            values=(ri,ci,row['action'],row['reported_cp'],mid,width,row['rank'],1-data['states.terminal'][ci])
            for suffix,value in zip(('root','child_state','action','reported_cp','midpoint','halfwidth','rank','ordinary'),values):data['edges.'+suffix].append(value)
        data['roots.edge_offsets'].append(len(data['edges.root']))
    _require(used_queries==set(byquery),'unexpected unjoined query provenance')
    counts=dict(states=len(state_meta),state_offsets=len(state_meta)+1,history=len(data['history.keys']),observations=len(data['observations.state']),roots=len(root_meta),root_offsets=len(root_meta)+1,edges=len(data['edges.root']),families=len(families),queries=len(queries))
    specs,nbytes=_layout(counts);_require(nbytes<=limits.payload_bytes,'payload byte cap before allocation')
    blob=bytearray(nbytes)
    for name,values in data.items():
        spec=specs[name];_require(len(values)*struct.calcsize('<'+FORMATS[spec['dtype']])==spec['nbytes'],'internal table shape')
        fmt='<'+FORMATS[spec['dtype']];size=struct.calcsize(fmt)
        for index,value in enumerate(values):struct.pack_into(fmt,blob,spec['offset']+index*size,value)
    return bytes(blob),specs,counts,dict(states=state_meta,roots=root_meta)


def _dump_limited(value,cap):
    out=io.BytesIO();total=0
    for text in json.JSONEncoder(sort_keys=True,indent=2,allow_nan=False).iterencode(value):
        _require(len(text)<=cap,'metadata string cap');chunk=text.encode();total+=len(chunk);_require(total+1<=cap,'metadata byte cap');out.write(chunk)
    out.write(b'\n');return out.getvalue()

def _freeze(value):
    if isinstance(value,dict):return MappingProxyType({k:_freeze(v) for k,v in value.items()})
    if isinstance(value,list):return tuple(_freeze(v) for v in value)
    return value

@dataclass(frozen=True)
class CorpusPack:
    payload:bytes
    manifest:object
    provenance:object
    schedule:object
    @property
    def segments(self):return self.manifest['tables']
    def view(self,name):
        spec=self.segments[name];return memoryview(self.payload)[spec['offset']:spec['offset']+spec['nbytes']]
    def values(self,name):return tuple(v[0] for v in struct.iter_unpack('<'+FORMATS[self.segments[name]['dtype']],self.view(name)))


def structural_schedule(data,*,seed=20261008,root_batch_size=4,observation_batch_size=32):
    """Exact-K structural one-epoch schedule; explicitly not a learning config."""
    _require(type(seed) is int and _int(root_batch_size,1,4096) and _int(observation_batch_size,1,4096),'schedule bounds')
    key=lambda domain,index:hashlib.sha256(f'{seed}:{domain}:{index}'.encode()).hexdigest()
    buckets={}
    for ri,k in enumerate(data['roots.legal_count']):buckets.setdefault(k,[]).append(ri)
    batches=[]
    for k in sorted(buckets):
        order=sorted(buckets[k],key=lambda i:key('root',i))
        for start in range(0,len(order),root_batch_size):batches.append(dict(legal_count=k,roots=order[start:start+root_batch_size]))
    order=sorted(range(len(data['observations.state'])),key=lambda i:key('observation',i))
    return dict(kind='unsealed_structural_exact_K_schedule_not_learning_config',version=1,seed=seed,
                algorithm='sha256_seed_domain_row_sort_v1',epochs=1,replacement=False,partial_tail='keep_complete_roots',
                root_batch_size=root_batch_size,observation_batch_size=observation_batch_size,
                root_batches=batches,observation_batches=[order[i:i+observation_batch_size] for i in range(0,len(order),observation_batch_size)])

def _schedule_from_blob(blob,specs,seed=20261008):
    data={}
    for name in ('roots.legal_count','observations.state'):
        spec=specs[name];view=memoryview(blob)[spec['offset']:spec['offset']+spec['nbytes']]
        data[name]=[v[0] for v in struct.iter_unpack('<i',view)]
    return structural_schedule(data,seed=seed)

def _manifest(prov,blob,specs,counts,provenance_bytes,schedule_bytes):
    return dict(kind='immutable_cpu_semantic_corpus',schema_version=1,split=prov['split'],source_kind=prov['source_kind'],
                source_sha256=source_sha256(),pins=prov['pins'],byte_order='little',alignment=64,
                counts=counts,payload_bytes=len(blob),payload_sha256=hashlib.sha256(blob).hexdigest(),
                provenance_bytes=len(provenance_bytes),provenance_sha256=hashlib.sha256(provenance_bytes).hexdigest(),
                schedule_bytes=len(schedule_bytes),schedule_sha256=hashlib.sha256(schedule_bytes).hexdigest(),
                tables=specs,terminal_policy=TERMINAL_POLICY,frame_kind=FRAME_KIND,interval_algorithm=INTERVAL_ALGORITHM,
                origins=[dict(index=0,kind='CPU_reconstructed_exact_history',native_context_uuid=None,native_epoch=-1,native_node=-1,
                              resolver='source_family_query_history_identity_no_native_arena_claim')])


def _rename_no_replace(source,destination):
    libc=ctypes.CDLL(None,use_errno=True)
    _require(hasattr(libc,'renameat2'),'atomic Linux renameat2 required; no unsafe publication fallback')
    call=libc.renameat2;call.argtypes=[ctypes.c_int,ctypes.c_char_p,ctypes.c_int,ctypes.c_char_p,ctypes.c_uint];call.restype=ctypes.c_int
    if call(-100,os.fsencode(source),-100,os.fsencode(destination),1):
        error=ctypes.get_errno();raise OSError(error,os.strerror(error),str(destination))


def _publish(prov,destination,limits=Limits()):
    """Private builder used by fixtures; production entry is seal-gated below."""
    limits.validate();destination=Path(destination);_require(not destination.exists(),'publication destination already exists')
    # Cap serialized source metadata before allocating a blob or large row tables.
    metadata=_dump_limited(prov,limits.metadata_bytes)
    blob,specs,counts,_=_assemble(prov,limits)
    schedule=_schedule_from_blob(blob,specs);schedule_bytes=_dump_limited(schedule,limits.metadata_bytes)
    manifest=_manifest(prov,blob,specs,counts,metadata,schedule_bytes);manifest_bytes=_dump_limited(manifest,limits.metadata_bytes)
    destination.parent.mkdir(parents=True,exist_ok=True);temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'-',dir=destination.parent))
    try:
        for name,raw in (('payload.bin',blob),('provenance.json',metadata),('schedule.json',schedule_bytes),('manifest.json',manifest_bytes)):
            with (temporary/name).open('xb') as f:f.write(raw);f.flush();os.fsync(f.fileno())
        # Validate cold bytes before atomic no-replace publication.
        read_pack(temporary,max_bytes=limits.payload_bytes,max_metadata_bytes=limits.metadata_bytes,
                  expected_manifest_sha256=hashlib.sha256(manifest_bytes).hexdigest(),expected_source_sha256=source_sha256(),expected_split=prov['split'])
        _rename_no_replace(temporary,destination)
        fd=os.open(destination.parent,os.O_RDONLY|os.O_DIRECTORY)
        try:os.fsync(fd)
        finally:os.close(fd)
    finally:
        if temporary.exists():shutil.rmtree(temporary)
    return hashlib.sha256(manifest_bytes).hexdigest()


def read_pack(path,*,max_bytes=64*1024**2,max_metadata_bytes=64*1024**2,
              expected_manifest_sha256=None,expected_source_sha256=None,purpose='audit',expected_split=None):
    """Open exactly one split. Audit is mechanical; train is train-only.

    Evaluation checkpoint/selection gates belong to resident.load_corpus. CPU
    reader rejects evaluation purposes without that outer capability rather than
    accepting a boolean 'test allowed'. No all-splits dictionary is returned.
    """
    _require(_int(max_bytes,1,2**63-1) and _int(max_metadata_bytes,1,2**63-1),'reader byte limits')
    _require(purpose in ('audit','train'),'reader purpose requires audit or train; evaluation gate owned by resident loader')
    root=Path(path);_require(root.is_dir() and not root.is_symlink(),'pack path')
    _require({p.name for p in root.iterdir()}=={'payload.bin','manifest.json','provenance.json','schedule.json'},'unknown/missing publication files')
    manifest_bytes=_read(root/'manifest.json',max_metadata_bytes)
    if expected_manifest_sha256 is not None:_require(_sha_value(expected_manifest_sha256) and hashlib.sha256(manifest_bytes).hexdigest()==expected_manifest_sha256,'manifest pin differs')
    m=_json(manifest_bytes)
    _keys(m,('kind','schema_version','split','source_kind','source_sha256','pins','byte_order','alignment','counts','payload_bytes','payload_sha256','provenance_bytes','provenance_sha256','schedule_bytes','schedule_sha256','tables','terminal_policy','frame_kind','interval_algorithm','origins'),'manifest')
    _require(m['kind']=='immutable_cpu_semantic_corpus' and type(m['schema_version']) is int and m['schema_version']==1 and m['split'] in SPLITS,'manifest schema/split')
    if expected_split is not None:_require(expected_split in SPLITS and m['split']==expected_split,'split differs')
    _require(purpose!='train' or m['split']=='train','training cannot open heldout split')
    _require(_sha_value(m['source_sha256']) and m['source_sha256']==source_sha256(),'packer/reader current source hash differs')
    if expected_source_sha256 is not None:_require(_sha_value(expected_source_sha256) and m['source_sha256']==expected_source_sha256,'source pin differs')
    _require(m['byte_order']=='little' and type(m['alignment']) is int and m['alignment']==64 and m['terminal_policy']==TERMINAL_POLICY and m['frame_kind']==FRAME_KIND and m['interval_algorithm']==INTERVAL_ALGORITHM,'wire/math/terminal contract differs')
    _require(_int(m['payload_bytes'],0,max_bytes) and _int(m['provenance_bytes'],1,max_metadata_bytes) and _int(m['schedule_bytes'],1,max_metadata_bytes),'manifest byte caps')
    _require(all(_sha_value(m[k]) for k in ('payload_sha256','provenance_sha256','schedule_sha256')),'payload hash grammar')
    limits=Limits(payload_bytes=max_bytes,metadata_bytes=max_metadata_bytes)
    _keys(m['counts'],('states','state_offsets','history','observations','roots','root_offsets','edges','families','queries'),'row counts')
    _require(all(_int(v,0,2**31-1) for v in m['counts'].values()),'row count type/range')
    specs,size=_layout(m['counts']);_require(m['tables']==specs and m['payload_bytes']==size,'aligned table shape/dtype/span differs')
    for name,spec in m['tables'].items():
        _keys(spec,('dtype','shape','offset','nbytes'),'table spec')
        _require(type(spec['shape']) is list and all(type(x) is int for x in spec['shape']) and type(spec['offset']) is int and type(spec['nbytes']) is int,'table spec types')
    blob=_read(root/'payload.bin',max_bytes);metadata=_read(root/'provenance.json',max_metadata_bytes);schedule_bytes=_read(root/'schedule.json',max_metadata_bytes)
    for raw,length,h in ((blob,m['payload_bytes'],m['payload_sha256']),(metadata,m['provenance_bytes'],m['provenance_sha256']),(schedule_bytes,m['schedule_bytes'],m['schedule_sha256'])):
        _require(len(raw)==length and hashlib.sha256(raw).hexdigest()==h,'payload/metadata bytes differ')
    provenance=_json(metadata);rebuilt,expected_specs,counts,_=_assemble(provenance,limits)
    _require(rebuilt==blob and expected_specs==m['tables'] and counts==m['counts'],'serialized semantic tables/padding differ')
    _require(provenance['split']==m['split'] and provenance['source_kind']==m['source_kind'] and provenance['pins']==m['pins'],'provenance cold binding differs')
    _require(_dump(m['origins'])==_dump(_manifest(provenance,blob,specs,counts,metadata,schedule_bytes)['origins']),'CPU origin resolver differs')
    schedule=_json(schedule_bytes);_require(_dump(schedule)==_dump(_schedule_from_blob(blob,specs)),'structural exact-K schedule differs')
    return CorpusPack(blob,_freeze(m),_freeze(provenance),_freeze(schedule))


def provenance_from_records(split,families,records,pins,*,source_kind='fixture_protocol_only',origin_fen=None):
    """Convert explicit source witnesses; no archive is read by this helper."""
    import chess
    _require(split in SPLITS and type(families) is list and type(records) is list,'explicit single-split source lists')
    queries=[]
    for record in records:
        _require(type(record) is dict and record.get('rejection') is None and type(record.get('labels')) is dict,'rejected source query cannot enter corpus')
        labels=record['labels'];rows=labels['rows']
        queries.append(dict(query=record['query'],source_record_sha256=record.get('source_record_sha256',hashlib.sha256(_dump(record)).hexdigest()),
                            transcript_sha256=hashlib.sha256(_dump(record.get('transcript',[]))).hexdigest(),
                            packet_provenance=record['packet_provenance'],target_kind=labels['target_kind'],query_material=labels['normalization_root_material'],
                            rows=[dict(uci=row['uci'],action=row['canonical_action'],reported_cp=row['root_score_cp'],rank=i+1,
                                       depth=row['reported_depth'],seldepth=row['seldepth'],nodes=row['nodes'],pv=row['pv'],converted_score=row['converted_score']) for i,row in enumerate(rows)]))
    return dict(version=1,split=split,source_kind=source_kind,pins=pins,
                families=[dict(source=family,origin_fen=origin_fen or chess.STARTING_FEN) for family in families],queries=queries)


def pack_audited_split(archive,preparation_archive,destination,*,split,seal_path,expected_seal_sha256):
    """Actual packing only after separately pinned CPU-only serialization seal.

    Each invocation owns one split. There is no training loader or all-splits
    return object, and sealed test files carry no learning authorization.
    """
    _require(split in SPLITS and _sha_value(expected_seal_sha256),'split/seal pin required')
    archive=Path(archive);prep=Path(preparation_archive);sealraw=_read(seal_path,1048576)
    _require(hashlib.sha256(sealraw).hexdigest()==expected_seal_sha256,'serialization seal hash differs');seal=_json(sealraw)
    _keys(seal,('kind','version','status','split','source_sha256','pins','limits'),'serialization seal')
    _require(seal['kind']=='cpu_semantic_serialization_seal' and type(seal['version']) is int and seal['version']==1 and seal['status']=='sealed_no_training' and seal['split']==split and seal['source_sha256']==source_sha256(),'serialization source/split seal differs')
    _keys(seal['limits'],asdict(Limits()),'serialization limits');limits=Limits(**seal['limits']);limits.validate()
    pins=dict(teacher_archive_manifest_sha256=sha256(archive/'manifest.json'),
              teacher_audit_sha256=sha256(archive/'independent-audit/packet-v2-independent-mechanical-audit-2026-10-01.json'),
              teacher_declaration_sha256=sha256(archive/'declaration.json'),input_admission_sha256=sha256(archive/'input-admission.json.gz'))
    _require(seal['pins']==pins,'teacher archive/audit/declaration/admission seal differs')
    manifest=_json(_read(archive/'manifest.json',limits.metadata_bytes))
    for name in ('report.json','declaration.json','input-admission.json.gz','raw-member-sha256.json','raw-records.tar.gz'):
        _require(sha256(archive/name)==manifest['files_sha256'][name],'teacher archive member pin differs')
    audit=_json(_read(archive/'independent-audit/packet-v2-independent-mechanical-audit-2026-10-01.json',limits.metadata_bytes))
    _require(audit.get('status')=='pass' and audit.get('every_record_hash_query_command_packet_frame_and_history_checked') is True and audit.get('independent_grammar_emission_selection_and_trace_checked') is True and audit.get('record_count')==3456,'independent audit is not complete/passing')
    _require(audit['declaration_sha256']==pins['teacher_declaration_sha256'] and audit['admission_sha256']==pins['input_admission_sha256'],'audit frame/history source binding')
    report=_json(_read(archive/'report.json',limits.metadata_bytes));_require(report['status']=='complete' and type(report['worker_exitcodes']) is list and len(report['worker_exitcodes'])==2 and all(type(v) is int and v==0 for v in report['worker_exitcodes']),'faulted teacher attempt')
    _require(audit['report_sha256']==sha256(archive/'report.json') and report['declaration_sha256']==pins['teacher_declaration_sha256'] and report['admission_sha256']==pins['input_admission_sha256'],'audit/report hash binding differs')
    declaration=_json(_read(archive/'declaration.json',limits.metadata_bytes))
    _require(declaration['status']=='sealed_for_bounded_labels' and declaration['admission_sha256']==pins['input_admission_sha256'],'teacher declaration binding differs')
    _require(sha256(prep/'manifest.json')==manifest['preparation_manifest_sha256'] and sha256(prep/'neural/evidence/packet-v2-freeze-2026-10-01.json')==audit['preparation_freeze_sha256'],'preparation source archive binding differs')
    _require(_dump(audit['frozen_source_sha256'])==_dump(declaration['source_sha256']) and len(declaration['source_sha256'])==16,'audited source hash map differs')
    for name,h in declaration['source_sha256'].items():
        _require(type(name) is str and name.startswith('neural/') and '..' not in Path(name).parts and _sha_value(h) and sha256(prep/name)==h,'frozen source bytes differ')
    frame=declaration['score_frame']
    _require(_dump(frame['normalization_coefficients'])==_dump(list(AS)) and frame['kind']==FRAME_KIND and _dump(frame['material_weights'])==_dump(list(PHASE_WEIGHTS)) and _dump(frame['material_clamp'])==_dump([17,78]) and type(frame['numeric_guard_relative']) is float and frame['numeric_guard_relative']==1e-9 and frame['batch_halfwidth']=='float32_outward_including_midpoint_cast_error' and frame['batch_midpoint_dtype']=='float32' and type(frame['material_divisor']) is float and frame['material_divisor']==58.0 and type(frame['reference_a']) is float and frame['reference_a']==_a(58) and type(frame['reference_material']) is int and frame['reference_material']==58 and frame['source_rounding']=='std_round_nearest_half_away_from_zero','teacher frame algorithm binding differs')
    admission=_json(_gunzip(_read(archive/'input-admission.json.gz',limits.metadata_bytes),limits.metadata_bytes))
    accepted_list=report['whole_family_accepted'][split];_require(type(accepted_list) is list and len(set(accepted_list))==len(accepted_list) and all(_sha_value(x) for x in accepted_list),'accepted family schema')
    accepted=set(accepted_list);families=[f for f in admission['source_records'][split] if f['game_id'] in accepted]
    _require(len(families)==len(accepted)>0,'accepted family join differs')
    plan=_json(_gunzip(_read(archive/'full-query-plan.json.gz',limits.metadata_bytes),limits.metadata_bytes))
    _require(type(plan) is list and len(plan)==3456 and hashlib.sha256(json.dumps(plan,sort_keys=True).encode()).hexdigest()==declaration['query_plan_sha256'],'immutable query plan binding differs')
    all_ids=[];order={};wanted={}
    for i,q in enumerate(plan):
        _keys(q,('split','position','all_legal','nodes','id'),'planned query')
        _require(_sha_value(q['id']) and q['id'] not in order,'duplicate planned query ID');order[q['id']]=i;all_ids.append(q['id'])
        if q['split']==split and q['position']['game_id'] in accepted:wanted[q['id']]=q
    entries=_json(_read(archive/'raw-member-sha256.json',limits.metadata_bytes));_require(type(entries) is list and len(entries)==3460,'raw member count')
    membermap={}
    for row in entries:
        _keys(row,('path','bytes','sha256'),'raw member');_require(type(row['path']) is str and row['path'] not in membermap and _int(row['bytes'],1,limits.record_bytes) and _sha_value(row['sha256']),'raw member duplicates/schema')
        membermap[row['path']]=row
    _require(set(membermap)=={qid+'.json' for qid in all_ids}|{'report.json','declaration.json','worker-0-setup.json','worker-1-setup.json'},'unexpected raw member inventory')
    fingerprints=report['source_records'];_require(type(fingerprints) is list and len(fingerprints)==3456,'reported fingerprints geometry')
    seen=set()
    for row in fingerprints:
        _keys(row,('query_id','sha256'),'reported query fingerprint');_require(row['query_id'] not in seen and row['query_id'] in order and row['sha256']==membermap[row['query_id']+'.json']['sha256'],'query report hash binding')
        seen.add(row['query_id'])
    records=[];seen_members=set()
    with tarfile.open(archive/'raw-records.tar.gz') as tar:
        for info in tar:
            _require(info.isfile() and info.name in membermap and info.name not in seen_members and info.size==membermap[info.name]['bytes'] and info.size<=limits.record_bytes,'raw query member bounds/schema')
            seen_members.add(info.name)
            if info.name[:-5] not in wanted:continue # No heldout labels are parsed for a train split.
            raw=tar.extractfile(info).read(limits.record_bytes+1);_require(len(raw)==info.size and hashlib.sha256(raw).hexdigest()==membermap[info.name]['sha256'],'raw query bytes differ')
            record=_json(raw);_require(record['query']==wanted[info.name[:-5]],'target query payload differs')
            record['source_record_sha256']=hashlib.sha256(raw).hexdigest();records.append(record)
    _require(seen_members==set(membermap) and len(records)==len(wanted),'raw target query coverage differs')
    records.sort(key=lambda r:order[r['query']['id']])
    provenance=provenance_from_records(split,families,records,pins,source_kind='audited_packet_v2')
    return _publish(provenance,destination,limits)
