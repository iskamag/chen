"""Root-controlled, CPU-only publication of three independently sealed splits."""
import hashlib
import json
import sys
from pathlib import Path

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
BASE = ROOT / 'results/semantic-corpus'
ARCHIVE = ROOT / 'docs/research/evidence/semantic-teacher-packet-v2-2026-10-01'
PREPARATION = ROOT / 'docs/research/evidence/semantic-packet-v2-preparation-2026-10-01'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


freeze = json.loads((BASE / 'packing-source-freeze.json').read_bytes())
for name, expected in freeze['source_sha256'].items():
    assert sha(ROOT / name) == expected, name
pins = {
    'teacher_archive_manifest_sha256': ('manifest.json', '6404b1fdd3af6470be529ba5305647ee9478e6f5f6081e1a98f6dbb3bc5d0cc9'),
    'teacher_audit_sha256': ('independent-audit/packet-v2-independent-mechanical-audit-2026-10-01.json', '920866ed2a95a14f4495caceda37ed1cbef9639f511e6f13e9ee31b0737d2d5d'),
    'teacher_declaration_sha256': ('declaration.json', 'a6f09aa8f34eaffa097b6602dc301cd00cd0e86f810ed3a357c4ce7daee08bda'),
    'input_admission_sha256': ('input-admission.json.gz', '36ea77f09f05b94043643dfb1132141717bcd4b56157b6ff43bf5917a65c5b6f'),
}
for name, (relative, expected) in pins.items():
    assert sha(ARCHIVE / relative) == expected, name
from experiments.semantic_corpus.packing import pack_audited_split

entries = {}
for split in ('train', 'validation', 'test'):
    seal = dict(kind='cpu_semantic_serialization_seal', version=1,
                status='sealed_no_training', split=split,
                source_sha256=freeze['source_sha256']['experiments/semantic_corpus/packing.py'],
                pins={k: v[1] for k, v in pins.items()}, limits=freeze['limits'])
    seal_path = BASE / 'seals' / (split + '.json')
    with seal_path.open('xb') as out:
        out.write((json.dumps(seal, sort_keys=True, indent=2) + '\n').encode())
    seal_sha = sha(seal_path)
    destination = BASE / 'packet-v2' / split
    manifest_sha = pack_audited_split(ARCHIVE, PREPARATION, destination,
                                     split=split, seal_path=seal_path,
                                     expected_seal_sha256=seal_sha)
    manifest = json.loads((destination / 'manifest.json').read_bytes())
    entries[split] = dict(path=str(destination), manifest_sha256=manifest_sha,
                          seal_path=str(seal_path), seal_sha256=seal_sha,
                          counts=manifest['counts'], payload_bytes=(destination / 'payload.bin').stat().st_size)
    with (BASE / (split + '-publication.json')).open('xb') as out:
        out.write((json.dumps(entries[split], sort_keys=True, indent=2) + '\n').encode())
    print(json.dumps({split: entries[split]}, sort_keys=True), flush=True)
with (BASE / 'publication.json').open('xb') as out:
    out.write((json.dumps(entries, sort_keys=True, indent=2) + '\n').encode())
