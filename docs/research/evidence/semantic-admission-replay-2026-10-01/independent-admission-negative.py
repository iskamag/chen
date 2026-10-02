import copy
import gzip
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile

repo = Path('/home/iskam/src/vibe/chen')
staging = repo / 'results/neural-staging'
spec = importlib.util.spec_from_file_location('audit', repo / 'scripts/audit_semantic_admission.py')
module = importlib.util.module_from_spec(spec); spec.loader.exec_module(module)
source = staging / 'neural/evidence/semantic-data-admission-2026-10-01.json.gz'
original = json.loads(gzip.decompress(source.read_bytes()))
checks = []
for case in ('history_bit_flip', 'halfmove_clock_change'):
    altered = copy.deepcopy(original)
    row = altered['source_records']['train'][0]['positions'][0]
    if case == 'history_bit_flip':
        row['previous_hashes'][0] ^= 1
    else:
        fen = row['fen'].split(); fen[4] = str(int(fen[4]) + 1); row['fen'] = ' '.join(fen)
    # Recompute the source fingerprint: these controls must fail replay itself,
    # not merely the outer content-integrity check.
    altered['source_sha256'] = hashlib.sha256(json.dumps(altered['source_records'], sort_keys=True).encode()).hexdigest()
    with tempfile.TemporaryDirectory(prefix='chen-admission-negative-') as directory:
        path = Path(directory) / 'admission.json'; path.write_text(json.dumps(altered))
        try:
            module.audit(path, '/home/iskam/src/vibe/chen-neural/results/semantic-data/lichess_db_standard_rated_2013-01.pgn', staging)
        except AssertionError as error:
            import traceback
            frame = traceback.extract_tb(error.__traceback__)[-1]
            checks.append(dict(case=case, rejected=True, replay_check=frame.line, line=frame.lineno))
        else:
            raise RuntimeError('mutated admission passed: ' + case)
report = dict(kind='semantic_admission_replay_negative_controls', source_sha256=module.sha(source),
              auditor_sha256=module.sha(module.__file__), driver_sha256=module.sha(__file__),
              checks=checks, teacher_searches=0, model_training=False)
(staging / 'independent-admission-negative.json').write_text(json.dumps(report, indent=2) + '\n')
print(json.dumps(report, indent=2))
