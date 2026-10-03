"""Fixed TRAIN-family complete-root teacher pilot. Never supplies model inputs.

Plan, source/data/options/binary/net declarations precede search. Runs once in
an empty attempt directory; faults retain partial evidence and admit no pilot.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import gzip
import hashlib
import itertools
import json
import os
from pathlib import Path
import sys
import time

import chess

from .ordinary_packets import POLICY, parse_pv, select_packet
from .tactical_inputs import ReplaySpec, action_id, compact_from_replay
from .teacher_session import GamePosition, StockfishSession, sha256_file

SF_SHA = '0f83d24cc46d2c66c60f16001af5444873bc112b7d028594513426894c12da19'
BASE_NODES = 262144
CALIBRATION_NODES = 4 * BASE_NODES
SOURCES = ('neural/synthetic_teacher.py', 'neural/ordinary_packets.py',
           'neural/teacher_session.py', 'neural/tactical_inputs.py', 'neural/score_frames.py')
SF_SOURCES = ('src/search.cpp', 'src/uci.cpp', 'src/score.cpp', 'src/evaluate.h',
              'src/nnue/network.cpp', 'Copying.txt')


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def save(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    temporary.replace(path)


def make_plan(shard, *, seed=20261003, roots=32, calibration=8):
    """Deterministic family-disjoint mover-balanced sampling, before any labels."""
    if type(seed) is not int or type(roots) is not int or roots < 2 or roots % 2 or not 0 <= calibration <= roots:
        raise ValueError('integer seed, even root count and bounded calibration count required')
    candidates, families = [], set()
    with gzip.open(shard, 'rt') as stream:
        for line in stream:
            family = json.loads(line)
            if (family.get('schema') != 'chen_reachable_mechanics_v1' or family.get('split') != 'train' or
                    family.get('outcome_label') is not None or family['family_id'] in families):
                raise ValueError('unique whole-family TRAIN mechanics shard without outcomes required')
            families.add(family['family_id'])
            for root in family['roots']:
                if root['terminal'] is None and root['actions']:
                    key = digest([seed, family['family_id'], root['ply']])
                    candidates.append((key, family, root))
    candidates.sort(key=lambda row: row[0])
    selected, used = [], set()
    for ordinal in range(roots):
        mover = 'white' if ordinal % 2 == 0 else 'black'
        candidate = next((c for c in candidates if c[1]['family_id'] not in used and c[2]['mover'] == mover), None)
        if candidate is None:
            raise ValueError('not enough distinct TRAIN families for balanced fixed plan')
        _, family, root = candidate
        used.add(family['family_id'])
        moves = tuple(family['moves'][:root['ply']])
        state = compact_from_replay(ReplaySpec(moves=moves), ply_from_search_root=root['ply'])
        ids = [a.id for a in state.actions]
        if (state.fen != root['fen'] or ids != [a['id'] for a in root['actions']] or
                state.source_attacks != tuple(int(x, 16) for x in root['attacks']) or
                state.history_counters != tuple(root['history_counters'])):
            raise ValueError('selected root replay/support/topology/history mismatch')
        row = dict(family_id=family['family_id'], ply=root['ply'], mover=mover, fen=state.fen,
                   moves=list(moves), previous_hashes=list(state.previous_hashes),
                   canonical_action_ids=ids, motifs=root['motifs'], state_context_sha256=state.context_sha256)
        row['root_id'] = digest(row)
        selected.append(row)
    calibration_ids = [r['root_id'] for r in sorted(selected, key=lambda r: digest([seed, 'calibration', r['root_id']]))[:calibration]]
    jobs = [dict(root_id=row['root_id'], budget='base', nodes=BASE_NODES) for row in selected]
    jobs += [dict(root_id=key, budget='calibration_4x', nodes=CALIBRATION_NODES) for key in calibration_ids]
    for job in jobs:
        job['query_id'] = digest(job)
    return dict(schema='chen_synthetic_teacher_plan_v1', split='train', seed=seed,
                selection='hash_order_distinct_families_alternating_mover_no_observation_replacement',
                shard_sha256=sha256_file(shard), roots=selected, calibration_root_ids=calibration_ids,
                jobs=jobs, requested_nodes=sum(j['nodes'] for j in jobs),
                requested_nodes_are_not_exact_actual_work=True, teacher_labels_in_model_inputs=False)


def ordinary_labels(lines, root, *, min_depth=6):
    """All root action scores in PARENT mover frame, from one eligible emission."""
    board = chess.Board()
    for uci in root['moves']:
        board.push_uci(uci)
    if board.fen(en_passant='fen') != root['fen']:
        raise ValueError('root/history mismatch')
    rows, provenance, reason = select_packet(lines, board, all_legal=True, min_depth=min_depth)
    evidence = []
    for offset, line in enumerate(lines):
        try:
            parsed = parse_pv(line)
        except ValueError:
            evidence.append(dict(line_offset=offset, type='malformed_scored_grammar'))
            continue
        if parsed is not None:
            kind = ('mate_distance' if parsed.score_kind == 'mate' else
                    'tablebase_or_reserved_cp' if parsed.tablebase_hits or abs(parsed.score) >= 19754 else
                    'bounded_cp' if parsed.bound else 'ordinary_cp_search_estimate')
            evidence.append(dict(line_offset=offset, type=kind, **asdict(parsed)))
    if reason:
        return None, provenance, reason, evidence
    from .score_frames import PHASE_WEIGHTS, sf19_reference_score, score_frame_manifest
    material = sum(len(board.pieces(kind, color)) * PHASE_WEIGHTS[kind - 1]
                   for color in (True, False) for kind in range(1, 7))
    targets = [dict(canonical_action=action_id(chess.Move.from_uci(r.pv[0]), board.turn),
                    rank=r.multipv, uci=r.pv[0], reported_uci_cp=r.score,
                    converted_score=sf19_reference_score(r.score, material,
                        score_kind='ordinary_unbounded_cp_no_tablebase', negate=False),
                    reported_depth=r.depth, seldepth=r.seldepth, reported_nodes=r.nodes, pv=list(r.pv)) for r in rows]
    targets.sort(key=lambda r: r['canonical_action'])
    if [r['canonical_action'] for r in targets] != root['canonical_action_ids']:
        raise ValueError('selected point packet lost complete support')
    return dict(target_kind='complete_root_action_ordinary_interval', perspective='parent_root_mover',
                root_material=material, score_frame=score_frame_manifest(), rows=targets,
                common_reported_depth=rows[0].depth, selected_reported_nodes=rows[0].nodes,
                exact_minimax_or_uniform_child_horizon=False, terminal_or_wdl_target=False), provenance, None, evidence


def compare_budgets(base, high):
    """Calibration, not label confidence or architecture quality. Canonical ties."""
    if base is None or high is None:
        return dict(comparable=False, reason='one_or_both_ordinary_packets_missing')
    a = {r['canonical_action']: r['reported_uci_cp'] for r in base['rows']}
    b = {r['canonical_action']: r['reported_uci_cp'] for r in high['rows']}
    if set(a) != set(b):
        raise ValueError('calibration support differs')
    choice = lambda scores: min(scores, key=lambda k: (-scores[k], k))
    changes = Counter()
    for x, y in itertools.combinations(sorted(a), 2):
        sa = (a[x] > a[y]) - (a[x] < a[y])
        sb = (b[x] > b[y]) - (b[x] < b[y])
        changes['pairs'] += 1
        changes['ordering_or_tie_changed'] += sa != sb
        changes['strict_reversal'] += sa * sb == -1
        changes['base_tie'] += sa == 0
        changes['high_tie'] += sb == 0
    return dict(comparable=True, base_choice=choice(a), high_choice=choice(b),
                choice_changed=choice(a) != choice(b), **changes)


def prepare(shard, binary, sf_source, output, *, repo='.'):
    """No search: fixed plan and metadata-only embedded-network export."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    plan = make_plan(shard)
    save(output/'plan.json', plan)
    if sha256_file(binary) != SF_SHA:
        raise ValueError('official SF19 binary differs')
    sf_source = Path(sf_source)
    sources = {name: sha256_file(sf_source/name) for name in SF_SOURCES}
    if sources['src/search.cpp'] != POLICY['search_source_sha256']:
        raise ValueError('ordinary packet SF search source differs')
    # Explicitly export/hash the embedded default net; no search/eval command.
    net = (output/'nn-1a298aa575a0.nnue').resolve()
    with StockfishSession(binary, expected_sha256=SF_SHA) as session:
        session.send('export_net ' + str(net))
        session.send('isready')
        session.receive(lambda line: line == 'readyok')
        metadata = dict(identity=session.identity, setup_trace=session.setup_trace,
                        export_trace=session.take_trace())
    net_sha = sha256_file(net)
    if not net_sha.startswith('1a298aa575a0'):
        raise ValueError('exported embedded net differs from declared SF19 default')
    from .score_frames import score_frame_manifest
    declaration = dict(schema='chen_synthetic_teacher_declaration_v1', status='frozen_for_one_bounded_train_pilot',
        plan_sha256=sha256_file(output/'plan.json'), shard_sha256=sha256_file(shard),
        source_sha256={name: sha256_file(Path(repo)/name) for name in SOURCES},
        stockfish=dict(binary_sha256=SF_SHA, source_sha256=sources, release='sf_19',
                       embedded_net_name=net.name, embedded_net_sha256=net_sha,
                       source_license='GPL-3.0-or-later', options='verbatim teacher_session setup trace'),
        protocol=POLICY, score_frame=score_frame_manifest(), cpu_affinity=[4],
        limits=dict(request_seconds=20, cooperative_seconds=600, external_seconds=660, min_depth=6),
        strength_evidence=False, pilot_contains_training=False, heldout_reads=False)
    save(output/'metadata.json', metadata)
    declaration['metadata_sha256'] = sha256_file(output/'metadata.json')
    save(output/'declaration.json', declaration)
    # Export is reproducible from the pinned binary; preserve its hash, not 99MB duplication.
    net.unlink()
    return declaration


def run(output, binary, shard, *, declaration_sha256, repo='.'):
    output = Path(output)
    raw = (output/'declaration.json').read_bytes()
    declaration = json.loads(raw)
    plan = json.loads((output/'plan.json').read_text())
    from .score_frames import score_frame_manifest
    if (hashlib.sha256(raw).hexdigest() != declaration_sha256 or
            declaration['status'] != 'frozen_for_one_bounded_train_pilot' or sha256_file(binary) != SF_SHA or
            sha256_file(output/'plan.json') != declaration['plan_sha256'] or
            sha256_file(output/'metadata.json') != declaration['metadata_sha256'] or
            sha256_file(shard) != declaration['shard_sha256'] or
            declaration['source_sha256'] != {name:sha256_file(Path(repo)/name) for name in SOURCES} or
            declaration['protocol'] != POLICY or declaration['score_frame'] != score_frame_manifest() or
            declaration['limits'] != dict(request_seconds=20,cooperative_seconds=600,external_seconds=660,min_depth=6) or
            declaration['cpu_affinity'] != [4] or
            plan != make_plan(shard) or os.sched_getaffinity(0) != {4}):
        raise ValueError('frozen source/data/options/plan/CPU admission differs')
    attempt = output/'attempt'
    attempt.mkdir(exist_ok=False)  # no resume, retry, overwrite or cherry-picked replacement
    started = time.monotonic()
    deadline = started + declaration['limits']['cooperative_seconds'] - 10
    roots = {r['root_id']: r for r in plan['roots']}
    records, session = [], None
    report = dict(status='running', started_utc=datetime.now(timezone.utc).isoformat(),
                  owner_pid=os.getpid(), declaration_sha256=hashlib.sha256(raw).hexdigest())
    save(attempt/'report.json', report)
    try:
        with StockfishSession(binary, expected_sha256=SF_SHA) as session:
            report['stockfish_pid'] = session.process.pid
            save(attempt/'setup.json', dict(identity=session.identity, trace=session.setup_trace))
            save(attempt/'report.json', report)
            for index, job in enumerate(plan['jobs']):
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError('cooperative pilot deadline')
                session.timeout = min(declaration['limits']['request_seconds'], remaining)
                root = roots[job['root_id']]
                position = GamePosition(root['family_id'], root['ply'], root['fen'],
                                        tuple(root['moves']), tuple(root['previous_hashes']))
                tick = time.monotonic()
                lines = session.analyze(position, nodes=job['nodes'], all_legal=True,
                                        declaration_sha256=report['declaration_sha256'])
                labels, provenance, reason, evidence = ordinary_labels(lines, root)
                record = dict(job=job, root_id=root['root_id'], labels=labels, missing_reason=reason,
                    packet_provenance=provenance, typed_score_evidence=evidence,
                    transcript=session.take_trace(), elapsed_seconds=time.monotonic()-tick,
                    maximum_reported_nodes=max((r.get('nodes',0) for r in evidence),default=0))
                save(attempt/(job['query_id']+'.json'), record)
                records.append(record)
                print(json.dumps(dict(completed=index+1,total=len(plan['jobs']),budget=job['budget'],
                                      accepted=reason is None,reason=reason)),flush=True)
        if time.monotonic() > deadline:
            raise TimeoutError('final pilot deadline')
        by_job = {(r['root_id'],r['job']['budget']):r for r in records}
        calibration = [dict(root_id=key, **compare_budgets(by_job[key,'base']['labels'],
                                   by_job[key,'calibration_4x']['labels'])) for key in plan['calibration_root_ids']]
        report.update(status='complete', queries=len(records),
            accepted=dict(Counter(r['job']['budget'] for r in records if r['labels'] is not None)),
            missing_reasons=dict(Counter(r['missing_reason'] for r in records if r['missing_reason'])),
            ordinary_action_rows=sum(len(r['labels']['rows']) for r in records if r['labels']),
            generation_wall_seconds=time.monotonic()-started,
            requested_nodes=sum(r['job']['nodes'] for r in records),
            sum_maximum_reported_nodes=sum(r['maximum_reported_nodes'] for r in records),
            sum_selected_packet_nodes=sum(r['labels']['selected_reported_nodes'] for r in records if r['labels']),
            calibration=calibration, strength_evidence=False, training=False, teacher_inputs=False)
    except BaseException as error:
        report.update(status='failed_no_pilot_admission', error=dict(type=type(error).__name__,message=str(error)),
            completed_queries=len(records), partial_transcript=[] if session is None else session.take_trace(),
            generation_wall_seconds=time.monotonic()-started)
        save(attempt/'report.json', report)
        raise
    finally:
        report['terminal_utc'] = datetime.now(timezone.utc).isoformat()
        report['stockfish_retired'] = session is None or session.process.poll() is not None
        save(attempt/'report.json', report)
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('mode',choices=('prepare','run'))
    parser.add_argument('--shard',required=True)
    parser.add_argument('--binary',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--sf-source')
    parser.add_argument('--declaration-sha256')
    args = parser.parse_args()
    if args.mode == 'prepare':
        if not args.sf_source:
            parser.error('prepare requires --sf-source')
        result = prepare(args.shard,args.binary,args.sf_source,args.output)
    else:
        if not args.declaration_sha256:
            parser.error('run requires the pre-reviewed declaration SHA256')
        result = run(args.output,args.binary,args.shard,declaration_sha256=args.declaration_sha256)
    print(json.dumps(result,indent=2),flush=True)


if __name__ == '__main__':
    main()
