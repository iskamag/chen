#!/usr/bin/env python3
"""Independent finite-policy rules/score certificate checks; no rating claim."""
import argparse,ctypes as c,hashlib,json,threading,time
from pathlib import Path
import chess
import bounded_search_probe as ref
from rules_probe import pack
class Child(c.Structure):
    _fields_=[('bound',ref.Result)]+[(n,c.c_uint32)for n in ('status','canonical_index','stack_pointer','reserved')]+[('original_alpha',c.c_int32),('original_beta',c.c_int32),('action',ref.Move),('padding',c.c_uint32)]
class Snapshot(c.Structure):
    _fields_=[('published',ref.Result)]+[(n,c.c_uint64)for n in ('root_hash','root_epoch','nodes','qnodes','iteration_nodes','elapsed_ticks')]+[(n,c.c_uint32)for n in ('abi_version','struct_size','requested_depth','root_count','completed_children','exact_children','upper_children','paused_children','abort_reason','iteration_complete','terminal','reserved')]+[('fallback',ref.Move),('padding',c.c_uint32)]
class Stats(c.Structure):
    _fields_=[(n,c.c_double)for n in ('h2d_ms','kernel_ms','d2h_ms')]+[(n,c.c_uint64)for n in ('wall_clock_khz','scratch_bytes','history_capacity')]
assert c.sizeof(Child)==336 and c.sizeof(Snapshot)==408

def cases():
    positions=[('start',chess.STARTING_FEN),('promotion','4k3/P7/8/8/8/8/7p/4K3 w - - 0 1'),('ep','4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1'),('castle','r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1'),('mate1','7k/5Q2/6K1/8/8/8/8/8 w - - 0 1'),('check','4k3/8/8/8/8/8/4r3/4K3 w - - 0 1'),('quiet','4k3/8/8/8/8/8/P7/4K3 b - - 0 1')]
    rows=[]
    for name,fen in positions:
        for depth in (1,2):
            case=dict(name=f'{name}-d{depth}',fen=fen,depth=depth,max_qply=8,evaluator=1,quiescence=0,root_ply=0,node_budget=1000000,history=[],alpha=-32000,beta=32000)
            case['expected']=ref.oracle(case,exhaustive=True);assert case['expected']['kind']==0;rows.append(case)
    for name,fen in positions[1:3]+positions[4:]:
        case=dict(name=name+'-q',fen=fen,depth=1,max_qply=8,evaluator=1,quiescence=1,root_ply=0,node_budget=1000000,history=[],alpha=-32000,beta=32000)
        case['expected']=ref.oracle(case,exhaustive=True);assert case['expected']['kind']==0;rows.append(case)
    fen='4k3/8/8/8/8/8/P7/4K3 w - - 8 10';key=pack(chess.Board(fen)).hash
    for name,history in [('root-threefold',[key,0,key,0]),('root-not-threefold',[key,0])]:
        case=dict(name=name,fen=fen,depth=2,max_qply=8,evaluator=0,quiescence=0,root_ply=0,node_budget=1000000,history=history,alpha=-32000,beta=32000)
        case['expected']=ref.oracle(case,exhaustive=True);rows.append(case)
    for name,fen in [('mate','7k/6Q1/6K1/8/8/8/8/8 b - - 0 1'),('fifty','4k3/8/8/8/8/8/P7/4K3 w - - 100 10')]:
        case=dict(name=name,fen=fen,depth=2,max_qply=8,evaluator=1,quiescence=1,root_ply=0,node_budget=1000000,history=[],alpha=-32000,beta=32000)
        case['expected']=ref.oracle(case,exhaustive=True);rows.append(case)
    return rows

def job(case,budget=None):
    return ref.Job(pack(chess.Board(case['fen'])),case['node_budget']if budget is None else budget,0,len(case['history']),1,192,0,case['max_qply'],case['evaluator'],case['quiescence'],0,0,-32000,32000)
def main():
    parser=argparse.ArgumentParser();parser.add_argument('--prepare',action='store_true');parser.add_argument('--cases',required=True);parser.add_argument('--library');parser.add_argument('--output');args=parser.parse_args()
    if args.prepare:
        rows=cases();Path(args.cases).write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n');print(json.dumps({'status':'PASS','exhaustive_roots':len(rows)}));return
    rows=json.loads(Path(args.cases).read_text());lib=c.CDLL(args.library);error=c.create_string_buffer(1024)
    lib.chen_root_pool_abi_version.restype=c.c_uint32;assert lib.chen_root_pool_abi_version()==3
    lib.chen_root_pool_create.argtypes=[c.c_int32,c.c_char_p,c.c_uint64];lib.chen_root_pool_create.restype=c.c_void_p
    lib.chen_root_pool_reset.argtypes=[c.c_void_p,c.POINTER(ref.Job),c.POINTER(c.c_uint64),c.c_uint64,c.c_uint64,c.POINTER(Snapshot),c.c_char_p,c.c_uint64]
    lib.chen_root_pool_begin.argtypes=[c.c_void_p,c.c_uint32,c.POINTER(Snapshot),c.c_char_p,c.c_uint64]
    lib.chen_root_pool_advance.argtypes=[c.c_void_p,c.c_uint64,c.POINTER(Snapshot),c.POINTER(Child),c.c_uint32,c.POINTER(Stats),c.c_char_p,c.c_uint64]
    for n in ('cancel','destroy'):getattr(lib,'chen_root_pool_'+n).argtypes=[c.c_void_p]
    ctx=lib.chen_root_pool_create(0,error,1024);assert ctx,error.value
    hard_deadline=time.monotonic()+25
    report={'status':'INCOMPLETE','rows':[],'controls':[],'library_sha256':hashlib.sha256(Path(args.library).read_bytes()).hexdigest(),'cases_sha256':hashlib.sha256(Path(args.cases).read_bytes()).hexdigest()}
    version=c.c_int();lib.hipRuntimeGetVersion(c.byref(version));providers=sorted({line.split()[-1]for line in Path('/proc/self/maps').read_text().splitlines()if 'libamdhip64.so'in line});assert len(providers)==1
    report['runtime']={'version':version.value,'path':providers[0],'sha256':hashlib.sha256(Path(providers[0]).read_bytes()).hexdigest()}
    def reset(case,budget=None):
        j=job(case,budget);h=(c.c_uint64*len(case['history']))(*case['history']);s=Snapshot();assert lib.chen_root_pool_reset(ctx,c.byref(j),h,len(h),123,c.byref(s),error,1024)==0,error.value
        assert s.root_hash==j.root.hash and s.root_epoch==123 and s.root_count==chess.Board(case['fen']).legal_moves.count();return s
    def begin(depth):
        s=Snapshot();assert lib.chen_root_pool_begin(ctx,depth,c.byref(s),error,1024)==0,error.value;return s
    def advance(ns=50_000_000):
        s=Snapshot();children=(Child*256)();stats=Stats();start=time.monotonic();assert lib.chen_root_pool_advance(ctx,ns,c.byref(s),children,256,c.byref(stats),error,1024)==0,error.value
        assert s.qnodes<=s.nodes<=1000000 and s.abi_version==3 and s.struct_size==408
        return s,children,stats,(time.monotonic()-start)*1000
    def finish(depth):
        s=begin(depth);trajectory=[];prior=[0]*256;windows={}
        for _ in range(150):
            assert time.monotonic()<hard_deadline, "25s bounded probe cap"
            s,ch,stats,ms=advance();trajectory.append({'nodes':s.nodes,'iteration_nodes':s.iteration_nodes,'completed_children':s.completed_children,'exact_children':s.exact_children,'upper_children':s.upper_children,'paused_children':s.paused_children,'kernel_ms':stats.kernel_ms,'wall_ms':ms})
            for i in range(s.root_count):
                cc=ch[i];assert cc.bound.nodes>=prior[i];prior[i]=cc.bound.nodes
                if cc.status:
                    window=(cc.original_alpha,cc.original_beta)
                    if i in windows:assert windows[i]==window
                    windows[i]=window
                if cc.status!=2:assert cc.bound.kind==3 and cc.bound.score==cc.bound.pv_length==cc.bound.completed_depth==0
            if s.iteration_complete or s.abort_reason:return s,ch,stats,trajectory
        raise AssertionError('bounded fixture did not finish in150epochs')
    try:
        for case in rows:
            assert ref.oracle(case,exhaustive=True)==case['expected'];s=reset(case)
            if s.terminal:
                assert s.published.kind==0 and s.published.score==case['expected']['score'];report['rows'].append({'name':case['name'],'terminal':True,'score':s.published.score});continue
            # Depth progression is explicit even for independently prepared d2.
            for depth in range(1,case['depth']+1):s,ch,stats,trajectory=finish(depth)
            assert not s.abort_reason and s.iteration_complete and s.published.kind==0
            observed=ref.packet(s.published);assert observed['qnodes']<=observed['nodes'];assert observed['score']==case['expected']['score'] and observed['completed_depth']==case['depth'],(case['name'],observed,case['expected'])
            optimal=case['expected']['optimal_moves'];assert observed['pv'][0]==optimal[0],(case['name'],observed['pv'],optimal)
            board=chess.Board(case['fen']);canonical=ref.canonical(board);winner=next(i for i,m in enumerate(canonical)if m.uci()==observed['pv'][0]);inc=s.published.score
            for i in range(s.root_count):
                cc=ch[i];assert cc.status==2 and cc.action.uci()==canonical[i].uci() and cc.canonical_index==i
                childboard=board.copy();childboard.push(canonical[i]);childcase=dict(case,fen=childboard.fen(en_passant='fen'),depth=case['depth']-1,root_ply=1,history=case['history']+[pack(board).hash])
                truth=ref.oracle(childcase,exhaustive=True);assert truth['kind']==0
                if cc.bound.kind==0:assert cc.bound.score==truth['score']
                elif cc.bound.kind==1:assert cc.bound.score<=truth['score']
                else:raise AssertionError('unexpected child upper frame')
                assert -cc.bound.score<=inc and (i>=winner or -cc.bound.score<inc)
                pvboard=childboard.copy()
                for m in ref.packet(cc.bound)['pv']:move=chess.Move.from_uci(m);assert move in pvboard.legal_moves;pvboard.push(move)
            report['rows'].append({'name':case['name'],'published':observed,'trajectory':trajectory,'children':[{'action':ch[i].action.uci(),'kind':ch[i].bound.kind,'score':ch[i].bound.score,'nodes':ch[i].bound.nodes,'window':[ch[i].original_alpha,ch[i].original_beta]}for i in range(s.root_count)]})
        # Resume a paused epoch, require immutable completed earlier publication.
        case=rows[0];reset(case);s,ch,stats,trace=finish(1);prior=bytes(s.published);begin(2);s,ch,stats,ms=advance(1)
        assert not s.iteration_complete and bytes(s.published)==prior and s.paused_children
        sentinel=Snapshot();c.memset(c.byref(sentinel),0x5a,c.sizeof(sentinel));before=bytes(sentinel)
        assert lib.chen_root_pool_begin(ctx,3,c.byref(sentinel),error,1024)==-1 and bytes(sentinel)==before
        for _ in range(50):
            s,ch,stats,ms=advance()
            if s.iteration_complete:break
        assert s.iteration_complete and s.published.completed_depth==2
        report['controls'].append({'name':'pause-resume-retention-and-active-begin-reject','status':'PASS','depth':2})
        # Strict node cap and last completed publication survive failure.
        reset(case,1);begin(1);s,ch,stats,ms=advance();assert s.abort_reason==1 and s.nodes==1 and s.published.kind==3
        report['controls'].append({'name':'strict-global-node-cap','nodes':s.nodes,'status':'PASS'})
        reset(case);s,ch,stats,trace=finish(1);prior=bytes(s.published);begin(2);lib.chen_root_pool_cancel(ctx);s,ch,stats,ms=advance();assert s.abort_reason==7 and bytes(s.published)==prior
        report['controls'].append({'name':'cancel-retains-completed-root','status':'PASS','wall_ms':ms})
        # Reset clears sticky cancellation, root history and all old task packets.
        reset(case);s,ch,stats,trace=finish(1);assert s.published.completed_depth==1 and not s.abort_reason
        report['controls'].append({'name':'reset-after-cancel','status':'PASS'})
        bad=job(case);bad.root.hash^=1;s=Snapshot();assert lib.chen_root_pool_reset(ctx,c.byref(bad),None,0,125,c.byref(s),error,1024)==0;assert s.abort_reason==4
        report['controls'].append({'name':'device-position-hash-reject','status':'PASS'})
        report['admission']={n:getattr(stats,n)for n,_ in stats._fields_};report['status']='PASS'
    finally:
        assert lib.chen_root_pool_destroy(ctx)==0
        if args.output:Path(args.output).write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
    print(json.dumps({'status':report['status'],'roots':len(report['rows']),'controls':len(report['controls'])}))
if __name__=='__main__':main()
