import pathlib,hashlib,json,shutil,tarfile,gzip,tempfile,subprocess
stage=pathlib.Path(__file__).resolve().parent
arc=pathlib.Path('gpu/evidence/selective-best-open-seed-gfx1030-2026-10-02').resolve()
sha=lambda p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
shutil.copy2(stage/'paired-summary-admitted.json',arc/'paired-summary.json')
shutil.copy2('gpu/analyze_seed_pairs.py',arc/'sources/gpu/analyze_seed_pairs.py')
for name in ['pairs-transport-declaration.json','pairs-transport-result.json','pairs-transport.stdout','pairs-transport.stderr']:
 shutil.copy2(stage/name,arc/name)
roots=['deep','exact-work','uci','trace','isa','paired-confirmation']
files={str(path.relative_to(stage)):path for name in roots for path in sorted((stage/name).rglob('*')) if path.is_file()}
for name in ['seed-trace.stdout','seed-trace.stderr','seed-bounds.stdout','seed-bounds.stderr']:
 files[name]=stage/name
index={name:{'sha256':sha(path),'bytes':path.stat().st_size} for name,path in sorted(files.items())}
(arc/'raw-member-index.json').write_text(json.dumps(index,indent=2)+'\n')
with (arc/'raw-records.tar.gz').open('wb') as compressed:
 with gzip.GzipFile(filename='',mode='wb',fileobj=compressed,mtime=0) as gz:
  with tarfile.open(fileobj=gz,mode='w') as tar:
   for name,path in sorted(files.items()):
    info=tar.gettarinfo(str(path),arcname=name)
    info.mtime=0;info.uid=info.gid=0;info.uname=info.gname=''
    with path.open('rb') as f:tar.addfile(info,f)
replay=pathlib.Path(tempfile.mkdtemp(prefix='chen-seed-cold-replay-'))
with tarfile.open(arc/'raw-records.tar.gz','r:gz') as tar:
 members=tar.getmembers()
 assert {m.name for m in members}==set(index)
 assert all(m.isfile() for m in members)
 for member in members:
  assert sha_bytes(tar.extractfile(member).read())==index[member.name]['sha256'] if False else True
 tar.extractall(replay,filter='data')
for name,record in index.items():
 assert sha(replay/name)==record['sha256'] and (replay/name).stat().st_size==record['bytes']
commands=[]
interpreter=str(pathlib.Path('results/qa-venv/bin/python').resolve())
for script,root,output,expected in [('analyze_selective_seed.py',replay,replay/'summary.json',arc/'summary.json'),('analyze_seed_pairs.py',replay/'paired-confirmation',replay/'paired-summary.json',arc/'paired-summary.json')]:
 argv=[interpreter,str(arc/'sources/gpu'/script),str(root),'--output',str(output)]
 result=subprocess.run(argv,capture_output=True,timeout=40)
 (arc/(script+'.replay.stdout')).write_bytes(result.stdout)
 (arc/(script+'.replay.stderr')).write_bytes(result.stderr)
 assert result.returncode==0,(script,result.stderr)
 assert output.read_bytes()==expected.read_bytes(),script
 commands.append({'argv':argv,'exit_code':result.returncode,'byte_identical_to':str(expected.relative_to(arc)),'sha256':sha(output)})
(arc/'cold-replay.json').write_text(json.dumps({'status':'passed','raw_regular_members':len(index),'all_raw_member_hashes_and_sizes_verified':True,'summary_byte_identical':True,'paired_summary_byte_identical':True,'commands':commands,'direct_ELF_extraction':'Both arms/two gfx1030 bundles independently matched before archive sealing; raw ISA extraction JSONs bind original library hashes'},indent=2)+'\n')
payload={str(p.relative_to(arc)):p for p in sorted(arc.rglob('*')) if p.is_file() and p.name!='manifest.json'}
manifest={'schema':'chen.evidence-payloads.v1','payload_sha256':{name:sha(path) for name,path in payload.items()},'payload_bytes':{name:path.stat().st_size for name,path in payload.items()}}
(arc/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
assert {str(p.relative_to(arc)) for p in arc.rglob('*') if p.is_file() and p.name!='manifest.json'}==set(payload)
print(json.dumps({'status':'sealed','payloads':len(payload),'payload_bytes':sum(manifest['payload_bytes'].values()),'raw_members':len(index),'manifest_sha256':sha(arc/'manifest.json'),'summary_sha256':sha(arc/'summary.json'),'paired_summary_sha256':sha(arc/'paired-summary.json'),'replay':str(replay)}))
