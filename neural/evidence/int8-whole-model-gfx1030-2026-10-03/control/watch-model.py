"""Bounded owned inference process group, never retry or train."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--pin',required=True);a=p.parse_args()
b=Path(a.base).resolve();c=b/'control';h=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
assert h(c/'declaration.json')==a.pin
d=json.loads((c/'declaration.json').read_text());assert not d['training']and d['splits']==['train']
def write(name,row):
    with(c/name).open('x')as f:json.dump(row,f,indent=2,sort_keys=True);f.write('\n');f.flush();os.fsync(f.fileno())
write('watchdog-claim.json',dict(kind='owned_int8_diagnostic_claim_v1',declaration_sha256=a.pin,no_retry=True))
command=['python3','-m','neural.profile_int8_inference','--declaration',str(c/'declaration.json'),'--sha256',a.pin,'--output',str(b/'attempt-1')]
env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join([str(b/'neural-worktree'),d['main_root'],str(b)]),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
assert not env.get('HSA_OVERRIDE_GFX_VERSION')
started=time.monotonic();child=None;signal_seen=[None]
def stop(sig,frame):
    signal_seen[0]=sig
    if child is not None and child.poll()is None:os.killpg(child.pid,signal.SIGTERM)
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
with(c/'stdout.log').open('xb')as out,(c/'stderr.log').open('xb')as err:
    child=subprocess.Popen(command,cwd=b/'neural-worktree',env=env,stdout=out,stderr=err,start_new_session=True)
    try:
        write('watchdog-started.json',dict(status='running',watchdog_pid=os.getpid(),child_pid=child.pid,child_pgid=child.pid,command=command,external_cap_seconds=660,runner_cap_seconds=600,source_sha256=h(__file__),declaration_sha256=a.pin))
        timeout=False
        try:code=child.wait(timeout=660)
        except subprocess.TimeoutExpired:
            timeout=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    finally:
        if child.poll()is None:os.killpg(child.pid,signal.SIGKILL);child.wait(timeout=15)
    write('watchdog-terminal.json',dict(status='terminal',exitcode=code,timed_out=timeout,interrupted_signal=signal_seen[0],elapsed_seconds=time.monotonic()-started,child_pid=child.pid,child_retired=child.poll()is not None,stdout_sha256=h(c/'stdout.log'),stderr_sha256=h(c/'stderr.log'),no_retry=True))
    raise SystemExit(code if code>=0 else 128-code)
