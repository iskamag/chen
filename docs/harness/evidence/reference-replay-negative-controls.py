#!/usr/bin/env python3
"""Corrupt copies only; confirm the independent archived-game verifier rejects them."""
import importlib.util
import json
from pathlib import Path
import shutil
import tempfile

root = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location('reference_replay', root / 'reference-replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
for kind in ['json_result_disagrees', 'illegal_pgn_move', 'missing_pair_game']:
    with tempfile.TemporaryDirectory(prefix='chen-reference-negative-') as temporary:
        copy = Path(temporary) / 'evidence'
        shutil.copytree(root, copy)
        if kind == 'illegal_pgn_move':
            path = copy / 'reference-games.pgn'
            text = path.read_text()
            assert '4. O-O Nf6' in text
            path.write_text(text.replace('4. O-O Nf6', '4. Ke8 Nf6', 1))
        else:
            path = copy / 'reference-games.jsonl'
            rows = [json.loads(line) for line in path.read_text().splitlines()]
            if kind == 'json_result_disagrees':
                rows[0]['result'] = '1-0'
            else:
                rows.pop()
            path.write_text(''.join(json.dumps(row) + '\n' for row in rows))
        try:
            replay.verify(copy)
        except ValueError as error:
            print(json.dumps({'control': kind, 'rejected': True, 'reason': str(error)}))
        else:
            raise RuntimeError('corrupt evidence accepted: ' + kind)
print(json.dumps({'negative_controls': 3, 'all_rejected': True}))
