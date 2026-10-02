"""Create a deterministic closed-runtime archive; never invoke GPU execution."""
import gzip,hashlib,json,shutil,tarfile
from pathlib import Path
base=Path(__file__).resolve().parent
out=base.parents[2]/'neural/evidence/semantic-corpus-study-manifest-2026-10-02'
out.mkdir(parents=True,exist_ok=True)
h=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
members={}
for directory,prefix in [(base/'closure',''),(base/'closed-control','control'),(base/'attempt-1','attempt-1')]:
    for path in sorted(directory.rglob('*')):
        if path.is_dir():continue
        assert path.is_file() and not path.is_symlink()
        name='/'.join(x for x in [prefix,path.relative_to(directory).as_posix()] if x)
        if name in members:assert h(path)==h(members[name])
        members[name]=path
rows=[dict(path=name,sha256=h(path),bytes=path.stat().st_size) for name,path in sorted(members.items())]
(out/'runtime-members.json').write_text(json.dumps(rows,indent=2,sort_keys=True)+'\n')
with (out/'runtime-payloads.tar.gz').open('wb') as raw,gzip.GzipFile(fileobj=raw,mode='wb',filename='',mtime=0) as zipped,tarfile.open(fileobj=zipped,mode='w') as tar:
    for name,path in sorted(members.items()):
        info=tar.gettarinfo(str(path),arcname=name)
        info.uid=info.gid=0;info.uname=info.gname='';info.mtime=0
        with path.open('rb') as stream:tar.addfile(info,stream)
for path in base.iterdir():
    if path.is_file() and path.name!='archive-terminal.py':shutil.copy2(path,out/path.name)
shutil.copy2(__file__,out/'archive-terminal.py')
for path in (base/'attempt-1').glob('*.json'):shutil.copy2(path,out/path.name)
for path in (base/'closed-control').glob('watchdog-*.json'):shutil.copy2(path,out/path.name)
for path in (base/'closed-control').glob('study.*.log'):shutil.copy2(path,out/path.name)
README='''# Closed corrected semantic GPU study

This is one fresh, separately reviewed bounded attempt after the earlier
checkpoint-bundle size failure. The earlier failed attempt remains closed.
This run completed all 8 anchor and 32 transition phases (9,216 updates) on
actual gfx1030 ROCm, froze all 40 checkpoint files under a 33KB manifest,
then evaluated the declared validation split. The unchanged joint gate selected
NONE. TEST remained closed; there is no model promotion or Elo claim.

The owned watchdog recorded exit0 after 144.173732 seconds; runner elapsed was
142.158113 seconds and summed training phase time was 107.045102 seconds.
Watchdog PID5099, training PID5105 and PGID5105 were independently observed
absent after completion. Stderr contains only the pinned missing amdgpu.ids
metadata warning. No retry or resume occurred.

`runtime-payloads.tar.gz` preserves exact isolated sources, all three immutable
packs, old native baseline sources/artifacts, path-rebased witness, declarations,
seals, logs and all individual states. `runtime-members.json` pins each regular
member. `manifest.json` pins the review payloads. Baseline core/source is the
historical f7e967 witness closure, before later main SWAR adoption.

`validation-results.json` contains every baseline and all 16 arm x 2 seed metrics
and diagnostics. `independent-terminal-replay.json` independently checks all40
file hashes, finite typed states, architecture keys and state fingerprints,
complete updates, frozen metric bindings and strict all-seed/all-baseline gate.
It reads the already published metrics; it does not recalculate quality from
corpus or run any GPU work. The first cold audit mistakenly required empty
stderr; its failed log is retained, and the final audit explicitly verifies the
exact harmless metadata warning.

Some anchors improve scalar error and some transitions improve root regret;
the per-seed optimal agreement condition still prevents promotion. Conditioned
and transport arms have equal root ranking metrics in this run, while their
latent diagnostics differ. This is an observation, not evidence that typed
routing is generally useless. Future research needs separately declared exact
child inference and transformer controls; current validation is development
evidence and must not become an untouched final evaluation claim.
'''
(out/'README.md').write_text(README)
assert json.loads((out/'independent-terminal-replay.json').read_text())['status']=='pass'
manifest=dict(kind='semantic_closed_corrected_GPU_study_v1',status='complete_selection_NONE',
              no_retry_or_resume=True,validation_opened=True,test_opened=False,
              quality_metrics_scope='declared_validation_only',rating_claim=None,
              runtime_member_count=len(rows),runtime_regular_bytes=sum(row['bytes']for row in rows),
              files_sha256={path.name:h(path)for path in sorted(out.iterdir())if path.is_file()and path.name!='manifest.json'})
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
print(json.dumps(dict(archive=str(out),manifest_sha256=h(out/'manifest.json'),tar_sha256=h(out/'runtime-payloads.tar.gz'),tar_bytes=(out/'runtime-payloads.tar.gz').stat().st_size,members=len(rows))))
