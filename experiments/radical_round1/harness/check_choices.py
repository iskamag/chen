#!/usr/bin/env python3
"""Measure actual selected moves on frozen board-local tactical obligations.

This is a challenge corpus, not a rating test. Labels and move selection remain
separate: a classifier without a policy cannot be credited with selected moves.
"""
import argparse
import hashlib
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--threats', type=Path, required=True)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    labels = {r['id']: r for r in map(json.loads, args.oracle.read_text().splitlines())}
    threats = {r['id']: r for r in map(json.loads, args.threats.read_text().splitlines())}
    rows = {}
    for row in map(json.loads, args.answers.read_text().splitlines()):
        assert row['position'] not in rows, 'duplicate position'
        rows[row['position']] = row
    assert rows.keys() == labels.keys() == threats.keys(), 'incomplete inventory'
    mate_total = mate_hit = avoidable = blunders = illegal = excluded = 0
    examples = []
    for rid, row in rows.items():
        label, threat, move = labels[rid], threats[rid], row['bestmove']
        if label['legal'] and move not in label['legal']:
            illegal += 1
        if label['draw_clock'] or label['insufficient_material']:
            excluded += 1
            continue
        if label['mates']:
            mate_total += 1
            mate_hit += move in label['mates']
            if move not in label['mates'] and len(examples) < 5:
                examples.append({'id': rid, 'fen': label['fen'], 'chosen': move, 'mates': label['mates']})
        if threat['safe_from_mate1'] and threat['allows_mate1']:
            avoidable += 1
            blunders += move in threat['allows_mate1']
    report = {'roots': len(rows), 'excluded_root_draw_contexts': excluded,
              'mate1_roots': mate_total, 'mate1_selected': mate_hit,
              'avoidable_opposing_mate1_roots': avoidable,
              'opposing_mate1_blunders': blunders, 'illegal_selected': illegal,
              'miss_examples': examples,
              'scope': 'Actual selected moves on constructed/seeded held-out challenge corpus; no Elo estimate',
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.oracle, args.threats, args.answers, Path(__file__)]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
