"""Unscored fresh admission only. Never constructs StockfishSession or invokes go."""
from pathlib import Path
import gzip
import json
import time
from neural.semantic_data import admit_families, atomic_json, sha256_file, ARCHIVE_SHA256, PGN_SHA256
from neural.semantic_study import make_declaration

root=Path(__file__).resolve().parent
prior=root.parent/'neural-staging/neural/evidence/semantic-data-admission-2026-10-01.json'
old=json.loads(prior.read_text())
spent=[row['path'] for row in old['spent_reports']]+['neural/evidence/semantic-teacher-closed-attempt-spent-inputs.json']
pgn=Path('/home/iskam/src/vibe/chen-neural/results/semantic-data/lichess_db_standard_rated_2013-01.pgn')
binary=Path('/home/iskam/src/vibe/chen/results/match-tools/stockfish/stockfish-linux-x86-64-universal')
archive=pgn.with_suffix(pgn.suffix+'.zst')
assert sha256_file(archive)==ARCHIVE_SHA256 and sha256_file(pgn)==PGN_SHA256
source=dict(old['source'],archive_path=str(archive),pgn_path=str(pgn))
start=time.monotonic()
groups,manifest=admit_families(pgn,seed=20261008,spent_reports=spent)
manifest.update(source=source,teacher_labels_generated=False,packet_protocol_version=2,
                admission_source_sha256=sha256_file(root/'neural/semantic_data.py'),
                preparation_source_sha256=sha256_file(__file__),
                prior_closed_admission_sha256=sha256_file(prior),
                prior_closed_compressed_admission_sha256=sha256_file(prior.with_suffix('.json.gz')),
                prior_closed_sealed_declaration_sha256='05ead77fd024c67ad789e723540ae258604dd15275c7cd7cf64569c0140b7b34',
                reason='independent_new_seed_all_previous_planned_inputs_spent_no_retry_relabel_or_replacement',
                admission_wall_seconds=time.monotonic()-start)
path=root/'neural/evidence/packet-v2-admission-2026-10-01.json.gz'
raw=(json.dumps(manifest,indent=2,allow_nan=False)+'\n').encode()
temporary=path.with_name(path.name+'.tmp');temporary.write_bytes(gzip.compress(raw,mtime=0));temporary.replace(path)
summary={key:value for key,value in manifest.items() if key!='source_records'}
summary['full_provenance']={'path':str(path.relative_to(root)),'sha256':sha256_file(path),'decoded_sha256':__import__('hashlib').sha256(raw).hexdigest()}
atomic_json(root/'neural/evidence/packet-v2-admission-summary-2026-10-01.json',summary)
atomic_json(root/'neural/evidence/packet-v2-label-draft-2026-10-01.json',make_declaration(path,binary,repo=root))
print(json.dumps(summary,indent=2))
