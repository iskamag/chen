#!/usr/bin/env python3
"""Matched DEV action diagnostic; approximate SF operational regret, never a rating."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import statistics
import time
import chess
import chess.engine


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def position(row):
    expected = chess.Board(row['fen'])
    if not expected.is_valid():
        raise ValueError('invalid FEN')
    board = chess.Board()
    reason = None
    try:
        for move in row.get('history', []):
            board.push_uci(move)
        if board.fen(en_passant='fen') != expected.fen(en_passant='fen'):
            reason = 'history from start does not reproduce complete FEN; used FEN without repetition history'
    except Exception as exc:
        reason = f'history invalid from start ({exc}); used FEN without repetition history'
    if reason:
        board = expected
    return board, reason


def launch(entry):
    engine = chess.engine.SimpleEngine.popen_uci(entry['command'], timeout=30,
        cwd=entry.get('cwd'), env={**os.environ, **entry.get('env', {})})
    try:
        engine.configure(entry.get('options', {}))
    except Exception:
        engine.close()
        raise
    return engine


def score_info(info, turn):
    score = info.get('score')
    relative = score.pov(turn) if score is not None else None
    return {**{k: info.get(k) for k in ('depth', 'seldepth', 'nodes', 'time', 'lowerbound', 'upperbound')},
            'cp': relative.score() if relative is not None else None,
            'mate': relative.mate() if relative is not None else None,
            'pv': [m.uci() for m in info.get('pv', [])]}


def reference(sf, board, nodes, move=None):
    started = time.perf_counter()
    try:
        with launch({'command': [sf], 'options': {'Threads': 1, 'Hash': 16}}) as engine:
            latest = {}; completed = None; consumed = 0
            with engine.analysis(board, chess.engine.Limit(nodes=nodes),
                                 root_moves=[chess.Move.from_uci(move)] if move else None,
                                 game=object(), info=chess.engine.INFO_ALL) as stream:
                for info in stream:
                    consumed = max(consumed, info.get('nodes', 0))
                    if 'score' in info:
                        latest = dict(info)
                        if not info.get('lowerbound') and not info.get('upperbound'):
                            completed = dict(info)
            result = score_info(completed if completed is not None else latest, board.turn)
            result['query_nodes'] = consumed
            result['score_selection'] = 'latest_unbounded_message' if completed is not None else 'last_bounded_message'
            result['last_scored_message'] = score_info(latest, board.turn)
            result.update({'engine_id': engine.id, 'error': None})
    except Exception as exc:
        result = {'error': f'{type(exc).__name__}: {exc}'}
    result['wall_seconds'] = time.perf_counter() - started
    return result


def decision(entry, board, seconds):
    started = time.perf_counter()
    try:
        with launch(entry) as engine:
            ready = time.perf_counter()
            result = engine.play(board, chess.engine.Limit(time=seconds), game=object(), info=chess.engine.INFO_ALL)
            elapsed = time.perf_counter() - ready
            legal = result.move in board.legal_moves
            return {'move': result.move.uci() if result.move else None, 'legal': legal,
                    'decision_wall_seconds': elapsed, 'total_wall_seconds': time.perf_counter()-started,
                    'reported': score_info(result.info, board.turn), 'engine_id': engine.id,
                    'error': None if legal else 'illegal or absent move'}
    except Exception as exc:
        return {'error': f'{type(exc).__name__}: {exc}', 'total_wall_seconds': time.perf_counter()-started}


def ordinary(result):
    return (result and not result.get('error') and result.get('cp') is not None
            and result.get('mate') is None and not result.get('lowerbound') and not result.get('upperbound'))


def summarize(rows):
    gaps = []; losses = {'base': [], 'candidate': []}; mate_cases = []; failures = []; different = 0
    for row in rows:
        if row.get('error'):
            failures.append({'id': row['id'], 'error': row['error']}); continue
        decisions = row['decisions']; scores = row['forced']; ref = row['reference']
        for name, value in decisions.items():
            if value.get('error'): failures.append({'id': row['id'], 'stage': name, 'error': value['error']})
        for name, value in [('reference', ref), *scores.items()]:
            if value.get('error'): failures.append({'id': row['id'], 'stage': name, 'error': value['error']})
        moves = [decisions[n].get('move') for n in ['base', 'candidate']]
        if all(moves): different += moves[0] != moves[1]
        chosen = [scores.get(m) for m in moves]
        if any(v and v.get('mate') is not None for v in [ref, *chosen]):
            mate_cases.append({'id': row['id'], 'reference': ref, 'moves': moves, 'chosen': chosen})
        # Common ordinary-CP support only; keep signed gaps, do not clamp negatives.
        if ordinary(ref) and all(ordinary(v) for v in chosen):
            gaps.append(chosen[1]['cp'] - chosen[0]['cp'])
            for name, value in zip(['base', 'candidate'], chosen): losses[name].append(ref['cp']-value['cp'])
    return {'roots': len(rows), 'different_moves': different, 'ordinary_common_roots': len(gaps),
            'candidate_minus_base_forced_cp_mean': statistics.mean(gaps) if gaps else None,
            'candidate_minus_base_forced_cp_median': statistics.median(gaps) if gaps else None,
            'ordinary_reference_gap': {name: {'mean_signed_cp': statistics.mean(v) if v else None,
                'over_100cp': sum(x>100 for x in v), 'over_300cp': sum(x>300 for x in v),
                'negative_gap_count': sum(x<0 for x in v)} for name,v in losses.items()},
            'mate_case_count': len(mate_cases), 'mate_cases': mate_cases, 'failures': failures,
            'history_fallbacks': sum(bool(r.get('history_fallback')) for r in rows),
            'interpretation': 'Approximate operational regret at separate finite search horizons, not an exact loss certificate or rating. Forced searches can outrank the unrestricted reference. Mate scores are separate. DEV is for development, not closed certification.'}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', required=True); p.add_argument('--dev'); p.add_argument('--positions')
    p.add_argument('--reuse-decisions', help='prior records with identical positions/config; teacher rescore only')
    p.add_argument('--out', required=True); p.add_argument('--sf', required=True)
    p.add_argument('--count', type=int, default=128); p.add_argument('--seed', default='jpeg2-dev-probe-20261015')
    p.add_argument('--seconds', type=float, default=.5); p.add_argument('--nodes', type=int, default=128000)
    args=p.parse_args()
    if bool(args.dev)==bool(args.positions): p.error('exactly one of --dev or --positions required')
    out=Path(args.out); out.mkdir(parents=True, exist_ok=False)
    source=Path(args.dev or args.positions)
    rows=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    if any(row.get('split')!='dev' for row in rows): raise ValueError('input must contain only explicitly DEV records')
    if args.dev:
        rows=sorted(rows, key=lambda r: hashlib.sha256((args.seed+'|'+r['id']).encode()).digest())[:args.count]
    # Deliberately strip all teacher labels before engine work and persist exact sample.
    rows=[{k:r[k] for k in ('id','split','fen','history') if k in r} for r in rows]
    (out/'positions.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in rows))
    config=json.loads(Path(args.config).read_text())
    metadata={'reused_decisions_sha256': sha(args.reuse_decisions) if args.reuse_decisions else None, 'arguments':vars(args), 'config':config, 'source_sha256':sha(source),
        'positions_sha256':sha(out/'positions.jsonl'), 'script_sha256':sha(__file__),
        'config_sha256':sha(args.config), 'sf_sha256':sha(args.sf), 'python_chess':chess.__version__,
        'affinity':sorted(os.sched_getaffinity(0)), 'fresh_process_per_decision_and_teacher_query':True,
        'artifacts':{name:{'binary':sha(entry['command'][0]),'declared':{k:sha(v) for k,v in entry.get('artifacts',{}).items()}} for name,entry in config.items()}}
    if args.reuse_decisions:
        prior_metadata = json.loads((Path(args.reuse_decisions).parent/'metadata.json').read_text())
        for field in ('config_sha256', 'positions_sha256', 'artifacts'):
            if prior_metadata[field] != metadata[field]:
                raise ValueError(f'reused decision provenance mismatch: {field}')
    dump(out/'metadata.json',metadata)
    reused = {r['id']: r for r in (json.loads(line) for line in Path(args.reuse_decisions).read_text().splitlines())} if args.reuse_decisions else {}
    results=[]
    with (out/'records.jsonl').open('w') as log:
        for i,row in enumerate(rows):
            record={'id':row['id'],'fen':row['fen']}
            try:
                board, fallback=position(row); record['history_fallback']=fallback
                record['root_terminal']=str(board.outcome(claim_draw=True))
                if not any(board.legal_moves): raise ValueError('no legal root actions')
                # Alternate execution order to reduce systematic wall-budget order effects.
                order=['base','candidate'] if i%2==0 else ['candidate','base']
                if args.reuse_decisions:
                    prior = reused[row['id']]
                    if prior['fen'] != row['fen']: raise ValueError('reused decision FEN mismatch')
                    record['decisions'] = prior['decisions']
                    record['decisions_reused_from'] = args.reuse_decisions
                else:
                    record['decisions']={name:decision(config[name],board,args.seconds) for name in order}
                record['reference']=reference(args.sf,board,args.nodes)
                moves=sorted({v['move'] for v in record['decisions'].values() if not v.get('error')})
                record['forced']={m:reference(args.sf,board,args.nodes,m) for m in moves}
            except Exception as exc:
                record['error']=f'{type(exc).__name__}: {exc}'
            results.append(record);log.write(json.dumps(record)+'\n');log.flush()
            if (i+1)%16==0: print(f'{i+1}/{len(rows)}',flush=True)
    dump(out/'summary.json',summarize(results))
    print(json.dumps({k:v for k,v in summarize(results).items() if k not in ('mate_cases','failures')}))


if __name__=='__main__': main()
