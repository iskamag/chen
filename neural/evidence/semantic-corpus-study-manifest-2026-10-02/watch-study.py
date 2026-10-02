"""One owned process group, fixed external cap, no retry or resume."""
import argparse,hashlib,json,os,signal,subprocess,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--base',required=True);p.add_argument('--suite-pin',required=True);args=p.parse_args()
base=Path(args.base).resolve();control=base/'control';h=lambda path:hashlib.sha256(Path(path).read_bytes()).hexdigest()
assert h(control/'suite-sealed.json')==args.suite_pin
# This independent watchdog allowance includes startup and final retirement.
# The reviewed runner separately rejects completed work past its 7200s ceiling.
cap=7350
command=['python3','-m','neural.run_semantic_corpus_study','run','--declaration',str(control/'suite-sealed.json'),'--declaration-sha256',args.suite_pin,'--output',str(base/'attempt-1')]
env=os.environ.copy();env.update(PYTHONPATH=os.pathsep.join([str(base/'neural-worktree'),str(base/'main'),str(base)]),PYTHONDONTWRITEBYTECODE='1',PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
assert not env.get('HSA_OVERRIDE_GFX_VERSION')
started=time.monotonic();child=None
interrupted=[None]
def stop(sig,frame):
    interrupted[0]=sig
    if child is not None and child.poll() is None:os.killpg(child.pid,signal.SIGTERM)
signal.signal(signal.SIGINT,stop);signal.signal(signal.SIGTERM,stop)
def write(name,value):
    raw=(json.dumps(value,sort_keys=True,indent=2,allow_nan=False)+'\n').encode()
    with (control/name).open('xb')as f:f.write(raw);f.flush();os.fsync(f.fileno())
write('watchdog-claim.json',dict(kind='semantic_external_watchdog_unique_claim_v1',suite_sha256=args.suite_pin,no_retry=True))
with (control/'study.stdout.log').open('xb')as stdout,(control/'study.stderr.log').open('xb')as stderr:
    child=subprocess.Popen(command,cwd=base/'neural-worktree',env=env,stdout=stdout,stderr=stderr,start_new_session=True)
    try:
        write('watchdog-started.json',dict(kind='semantic_external_watchdog_v1',status='running',watchdog_pid=os.getpid(),owned_child_pid=child.pid,owned_child_pgid=child.pid,command=command,external_timeout_seconds=cap,runner_timeout_seconds=7200,phase_completion_timeout_seconds=900,phase_timeout_scope='runner rejects after draining work, not independent preemption',suite_sha256=args.suite_pin,source_sha256=h(__file__),environment={key:env[key]for key in ['PYTHONPATH','PYTHONDONTWRITEBYTECODE','OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},no_retry=True))
        timed_out=False
        try:code=child.wait(timeout=cap)
        except subprocess.TimeoutExpired:
            timed_out=True;os.killpg(child.pid,signal.SIGTERM)
            try:code=child.wait(timeout=30)
            except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);code=child.wait(timeout=15)
    finally:
        # No exception in bookkeeping/waiting may leave the owned child alive.
        if child.poll() is None:
            os.killpg(child.pid,signal.SIGKILL)
            child.wait(timeout=15)
    write('watchdog-terminal.json',dict(kind='semantic_external_watchdog_terminal_v1',status='terminal',exitcode=code,timed_out=timed_out,interrupted_signal=interrupted[0],elapsed_seconds=time.monotonic()-started,child_pid=child.pid,child_retired=child.poll()is not None,stdout_sha256=h(control/'study.stdout.log'),stderr_sha256=h(control/'study.stderr.log'),no_retry=True))
    raise SystemExit(code if code>=0 else 128-code)
