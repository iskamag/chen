#!/usr/bin/env python3
"""Audit complete relation classifications and nested mate witnesses independently.

Both input inventories must be frozen before the tested executable sees CERT.
Timing and scalar assertions printed by the executable are not audit evidence.
"""
import argparse
import hashlib
import json
from pathlib import Path

import chess


def unique(values, context):
    assert len(values) == len(set(values)), (context, 'duplicate move')
    return set(values)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--oracle', type=Path, required=True)
    parser.add_argument('--threats', type=Path, required=True)
    parser.add_argument('--answers', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    oracle = {row['id']: row for row in map(json.loads, args.oracle.read_text().splitlines())}
    threats = {row['id']: row for row in map(json.loads, args.threats.read_text().splitlines())}
    assert oracle and oracle.keys() == threats.keys(), 'oracle inventories differ'
    classifications = {}
    witnessed = set()
    positive = 0
    for line in args.answers.read_text().splitlines():
        if line.startswith('{'):
            row = json.loads(line)
            if row.get('kind') != 'classification':
                continue
            rid, variant = row['id'], row['variant']
            assert rid in oracle, ('unexpected root', rid)
            covered = classifications.setdefault(variant, set())
            assert rid not in covered, ('duplicate root', variant, rid)
            covered.add(rid)
            for field in ['legal', 'checking', 'mates', 'stalemates']:
                got = unique(row[field], (rid, variant, field))
                assert got == set(oracle[rid][field]), (rid, variant, field, got, oracle[rid][field])
            assert not row['unknown'], (rid, variant, 'unbounded run returned unknown')
        elif line.startswith('THREAT\t'):
            _, rid, move, witness, complete = line.split('\t')
            assert rid in oracle and complete == 'true', ('incomplete threat', rid, move)
            key = rid, move
            assert key not in witnessed, ('duplicate threat action', key)
            witnessed.add(key)
            expected = threats[rid]
            assert move in expected['legal'], ('illegal root action', key)
            assert (witness != '-') == (move in expected['allows_mate1']), ('wrong threat result', key)
            if witness != '-':
                board = chess.Board(oracle[rid]['fen'])
                board.push_uci(move)
                reply = chess.Move.from_uci(witness)
                assert reply in board.legal_moves, ('illegal witness', key, witness)
                board.push(reply)
                assert board.is_checkmate(), ('witness does not mate', key, witness)
                positive += 1
    assert classifications, 'no classifications'
    for variant, covered in classifications.items():
        assert covered == oracle.keys(), ('missing roots', variant, oracle.keys() - covered)
    all_actions = {(rid, move) for rid, row in threats.items() for move in row['legal']}
    assert witnessed == all_actions, ('missing threat coverage', len(all_actions - witnessed))
    report = {'status': 'PASS', 'roots': len(oracle),
              'classification_variants': {k: len(v) for k, v in classifications.items()},
              'nested_actions': len(witnessed), 'positive_mate_witnesses': positive,
              'python_chess': chess.__version__,
              'sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in [args.oracle, args.threats, args.answers, Path(__file__)]}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
