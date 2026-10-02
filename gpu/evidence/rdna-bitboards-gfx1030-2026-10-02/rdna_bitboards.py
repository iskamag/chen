#!/usr/bin/env python3
"""Deterministic independent bit-by-bit oracle and warm HIP runner.

The benchmark binary runs only inside the specified ROCm container. The output
directory must be new; failed experiments are preserved rather than overwritten.
No production engine imports, neural libraries, or rating inference.
"""
import argparse
import hashlib
import json
from pathlib import Path
import random
import statistics
import struct
import subprocess

MASK=(1<<64)-1
MODES=('transpose_loop','transpose_swar','popcount64','popcount_halves',
       'popcount_swar','rook_walk','rook_kogge_stone','rook_wave32')

def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def transpose(x):
    return sum(((x>>s)&1)<<((s%8)*8+s//8) for s in range(64))

def rook(x,s):
    result=0
    for df,dr in ((1,0),(-1,0),(0,1),(0,-1)):
        f,r=s%8+df,s//8+dr
        while 0<=f<8 and 0<=r<8:
            b=1<<(r*8+f); result|=b
            if x&b: break
            f+=df; r+=dr
    return result

def fixture(n):
    rng=random.Random(20261002)
    # Dense, empty, single-bit, alternating files/ranks, sparse, random; every
    # origin square is represented in each sufficiently large fixture.
    constants=(0,MASK,0xaa55aa55aa55aa55,0x0101010101010101,
               0x8080808080808080,0xff000000000000ff)
    rows=[]
    for i in range(n):
        s=i%64; k=(i//64)%10
        if k<len(constants): x=constants[k]
        elif k==6: x=1<<s
        elif k==7: x=sum(1<<v for v in rng.sample(range(64),8))
        else: x=rng.getrandbits(64)
        rows.append((x,s))
    return rows

def expected(rows,rounds):
    out=[[],[],[]]
    for x,s in rows:
        sums=[0,0,0]
        for r in range(rounds):
            sums[0]^=transpose(x); sums[1]^=x.bit_count(); sums[2]^=rook(x,s)
            x=(((x<<9)|(x>>55))^(0x9e3779b97f4a7c15+r))&MASK
            s=(s+13)&63
        for d,v in zip(out,sums): d.append(v)
    return out

def call(argv):
    return subprocess.run(argv,check=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=180)

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--container',default='cpp-rocm')
    ap.add_argument('--binary',default='/tmp/chen-bitboard-microbench')
    ap.add_argument('--output',type=Path,required=True)
    ap.add_argument('--sizes',default='64,1024,16384')
    ap.add_argument('--rounds',type=int,default=64)
    ap.add_argument('--repeats',type=int,default=15)
    args=ap.parse_args()
    sizes=[int(v) for v in args.sizes.split(',')]
    if not sizes or any(n<1 or n>1048576 for n in sizes): ap.error('size outside 1..1048576')
    if not 1<=args.rounds<=4096 or not 1<=args.repeats<=101: ap.error('rounds or repeats outside bound')
    args.output.mkdir(parents=True,exist_ok=False)
    study={'schema':1,'seed':20261002,'scope':'Resident primitive latency with evolving input and XOR reduction; transfers excluded, no engine speed or strength claim',
           'source_sha256':sha('gpu/bitboard_microbench.hip'),'runner_sha256':sha(__file__),
           'binary_sha256':call(['docker','exec',args.container,'sha256sum',args.binary]).stdout.decode().split()[0],
           'build_command':'hipcc -O3 --offload-arch=gfx1030 gpu/bitboard_microbench.hip -o microbench',
           'cases':[]}
    # Round 1 independently checks intermediate outputs; odd 7 checks recurrence
    # before the long timing run, whose final outputs are also fully checked.
    cases=[(max(sizes),1,1),(max(sizes),7,1)]+[(n,args.rounds,args.repeats) for n in sizes]
    for number,(n,rounds,repeats) in enumerate(cases):
        rows=fixture(n); prefix='case-'+str(number)
        ip=args.output/(prefix+'.input.bin'); ip.write_bytes(b''.join(struct.pack('<QII',x,s,0) for x,s in rows))
        remote='/tmp/chen-bitboards-'+args.output.name+'-'+prefix
        call(['docker','cp',str(ip),args.container+':'+remote+'.input.bin'])
        run=call(['docker','exec',args.container,args.binary,remote+'.input.bin',remote,str(n),str(rounds),str(repeats)])
        (args.output/(prefix+'.stdout.json')).write_bytes(run.stdout)
        (args.output/(prefix+'.stderr.txt')).write_bytes(run.stderr)
        result=json.loads(run.stdout); reference=expected(rows,rounds)
        if result['n']!=n or result['rounds']!=rounds or result['warp_size']!=32 or not result['arch'].startswith('gfx1030'):
            raise ValueError('runtime identity or dimensions differ')
        mode_results=[]
        for m,name in enumerate(MODES):
            op=args.output/(prefix+'.'+str(m)+'.bin')
            call(['docker','cp',args.container+':'+remote+'.'+str(m)+'.bin',str(op)])
            raw=op.read_bytes()
            if len(raw)!=n*8: raise ValueError('incorrect result length')
            got=struct.unpack('<'+'Q'*n,raw); family=0 if m<2 else 1 if m<5 else 2
            if list(got)!=reference[family]:
                bad=next(i for i,(a,b) in enumerate(zip(got,reference[family])) if a!=b)
                raise ValueError(f'{name} mismatch row {bad}: {got[bad]:x} != {reference[family][bad]:x}')
            samples=result['samples'][m]['milliseconds']
            mode_results.append({'mode':m,'name':name,'checked_results':n,'output_sha256':sha(op),
                                 'median_ms':statistics.median(samples),'min_ms':min(samples),'max_ms':max(samples),
                                 'median_ns_per_operation':statistics.median(samples)*1e6/(n*rounds)})
        study['cases'].append({'number':number,'n':n,'rounds':rounds,'repeats':repeats,'input_sha256':sha(ip),
                               'runtime':result,'modes':mode_results})
        print(f'{prefix}: {n} inputs x {rounds} rounds, all 8 variants independently verified',flush=True)
    (args.output/'report.json').write_text(json.dumps(study,indent=2,allow_nan=False)+'\n')
    files={str(p.relative_to(args.output)):sha(p) for p in sorted(args.output.iterdir()) if p.is_file()}
    (args.output/'manifest.json').write_text(json.dumps({'sha256':files},indent=2)+'\n')

if __name__=='__main__': main()
