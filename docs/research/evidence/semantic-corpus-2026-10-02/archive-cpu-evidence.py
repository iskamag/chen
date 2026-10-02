"""Archive the closed CPU serialization/audit milestone without model metrics."""
from pathlib import Path
import gzip
import hashlib
import io
import json
import shutil
import tarfile

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / 'results/semantic-corpus'
DEST = ROOT / 'docs/research/evidence/semantic-corpus-2026-10-02'
DEST.mkdir(exist_ok=False)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def copy(source, relative):
    target = DEST / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    assert sha(source) == sha(target)


publication = json.loads((BASE / 'publication.json').read_bytes())
summary = json.loads((BASE / 'audit-all-splits.json').read_bytes())
assert summary['status'] == 'pass'
assert summary['aggregate_counts']['states'] == 15501
assert summary['aggregate_counts']['edges'] == 12701
assert summary['quality_error_ranking_metrics_computed'] is False
for split, entry in publication.items():
    folder = Path(entry['path'])
    assert sha(folder / 'manifest.json') == entry['manifest_sha256']
    assert sha(Path(entry['seal_path'])) == entry['seal_sha256']
    report = json.loads((BASE / ('audit-' + split + '.json')).read_bytes())
    assert report['status'] == 'pass' and report['manifest_sha256'] == entry['manifest_sha256']
    assert sha(BASE / ('audit-' + split + '.json')) == summary['splits'][split]['report_sha256']
    copy(folder / 'manifest.json', Path('splits') / split / 'manifest.json')

# Stable names, metadata and gzip header allow byte-comparable replay archives.
with (DEST / 'packs.tar.gz').open('xb') as out:
    with gzip.GzipFile(filename='', mode='wb', fileobj=out, mtime=0, compresslevel=6) as zipped:
        with tarfile.open(fileobj=zipped, mode='w|') as archive:
            for split in sorted(publication):
                for path in sorted(Path(publication[split]['path']).iterdir()):
                    assert path.is_file() and path.name in {'manifest.json', 'payload.bin', 'provenance.json', 'schedule.json'}
                    data = path.read_bytes()
                    info = tarfile.TarInfo(split + '/' + path.name)
                    info.size, info.mtime, info.mode = len(data), 0, 0o644
                    archive.addfile(info, io.BytesIO(data))

names = ['packing-source-freeze.json', 'packing-tests.log', 'rust-oracle-build.json',
         'rust-oracle-special-cases.json', 'produce-packs.py', 'archive-cpu-evidence.py',
         'publication.json', 'production.stdout.log', 'production.stderr.log',
         'audit-all-splits.json']
for split in publication:
    names.extend([split + '-publication.json', 'seals/' + split + '.json',
                  'audit-' + split + '.json', 'audit-' + split + '.stdout.log',
                  'audit-' + split + '.stderr.log'])
for name in names:
    copy(BASE / name, name)
copy(Path('/tmp/chen-corpus-independent-review-tests.log'), 'independent-review/44-cpu-tests.log')
sources = ['experiments/semantic_corpus/__init__.py', 'experiments/semantic_corpus/packing.py',
           'experiments/semantic_corpus/resident.py', 'experiments/semantic_corpus/test_resident.py',
           'experiments/semantic_corpus/tests/test_packing.py', 'scripts/audit_semantic_corpus.py',
           'scripts/tests/test_audit_semantic_corpus.py', 'scripts/check_corpus_oracle.py']
build = json.loads((BASE / 'rust-oracle-build.json').read_bytes())
for name, expected in build['compiled_crate_source_sha256'].items():
    assert sha(ROOT / name) == expected
    sources.append(name)
for name in sorted(set(sources)):
    copy(ROOT / name, Path('sources') / name)
copy(ROOT / 'target/release/examples/corpus_oracle', 'corpus-oracle')
assert sha(DEST / 'corpus-oracle') == build['binary_sha256']
copy(ROOT / 'experiments/semantic_corpus/README.md', 'package-README.md')
manifest = {str(p.relative_to(DEST)): sha(p) for p in sorted(DEST.rglob('*')) if p.is_file()}
with (DEST / 'manifest.json').open('xb') as out:
    out.write((json.dumps({'kind': 'closed_CPU_semantic_corpus_evidence', 'files_sha256': manifest}, sort_keys=True, indent=2) + '\n').encode())
print(json.dumps({'archive': str(DEST), 'files': len(manifest), 'manifest_sha256': sha(DEST / 'manifest.json'), 'packs_tar_sha256': sha(DEST / 'packs.tar.gz')}, sort_keys=True))
