"""Bounded stdlib lifecycle fixtures; no Torch, corpus or training."""
from pathlib import Path
import hashlib,json,os,subprocess,sys,tempfile,time,signal
watch=Path(__file__).with_name('watch-study.py')
def setup(base,body):
    (base/'control').mkdir();(base/'main').mkdir();(base/'neural-worktree/neural').mkdir(parents=True)
    (base/'neural-worktree/neural/run_semantic_corpus_study.py').write_text(body)
    seal=base/'control/suite-sealed.json';seal.write_text('{"kind":"mechanical_lifecycle_fixture_no_training"}')
    return [sys.executable,str(watch),'--base',str(base),'--suite-pin',hashlib.sha256(seal.read_bytes()).hexdigest()]
with tempfile.TemporaryDirectory()as folder:
    base=Path(folder);cmd=setup(base,'print("mechanical exit, no training")\n')
    result=subprocess.run(cmd,capture_output=True,timeout=15);assert result.returncode==0,result.stderr
    terminal=json.loads((base/'control/watchdog-terminal.json').read_text());assert terminal['child_retired'] and terminal['exitcode']==0
    again=subprocess.run(cmd,capture_output=True,timeout=15);assert again.returncode!=0
with tempfile.TemporaryDirectory()as folder:
    base=Path(folder);cmd=setup(base,'import time;time.sleep(60)\n')
    parent=subprocess.Popen(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    for _ in range(100):
        if (base/'control/watchdog-started.json').exists():break
        time.sleep(.02)
    started=json.loads((base/'control/watchdog-started.json').read_text());os.kill(parent.pid,signal.SIGTERM)
    parent.communicate(timeout=15);terminal=json.loads((base/'control/watchdog-terminal.json').read_text())
    assert terminal['child_retired'] and terminal['interrupted_signal']==signal.SIGTERM and terminal['exitcode']==-signal.SIGTERM
    assert not Path('/proc',str(started['owned_child_pid'])).exists()
print(json.dumps({'passed':True,'fixtures':['normal completion and child retirement','exclusive durable claim rejects duplicate before spawn','parent SIGTERM retires owned process group and journals terminal'], 'GPU_calls':0,'training':False}))
