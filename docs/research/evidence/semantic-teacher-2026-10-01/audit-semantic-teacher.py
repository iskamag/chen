#!/usr/bin/env python3
"""Mechanical audit of one closed finite-search teacher attempt.

No engines, GPU, models, fitting, score distributions, policy quality or strength
metrics. Scores are read only to revalidate packet schema/sign/frame conversion.
"""
import argparse
from collections import Counter
from decimal import Decimal, localcontext
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import re
import sys
import time


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def equal(a,b):
    return json.dumps(a,sort_keys=True,allow_nan=False)==json.dumps(b,sort_keys=True,allow_nan=False)


def require(condition,message):
    if not condition: raise ValueError(message)


def validate_report_metadata(report):
    require(isinstance(report.get('worker_exitcodes'),list) and
            all(type(code) is int for code in report['worker_exitcodes']),'worker exit code type differs')
    wall=report.get('wall_seconds')
    require(type(wall) in (int,float) and math.isfinite(wall) and wall>=0,'invalid report wall telemetry')


def stats(values):
    return dict(count=len(values),minimum=min(values) if values else None,
                maximum=max(values) if values else None,sum=sum(values))


def audit(stage,attempt,sealed,admission,binary,oracle_source):
    start=time.monotonic(); stage=Path(stage).resolve(); attempt=Path(attempt).resolve()
    sys.path.insert(0,str(stage))
    import chess
    from neural.semantic_study import validate_declaration,query_plan,LabelBudget
    from neural.semantic_data import GamePosition,common_cp_labels,parse_pv
    declaration,declaration_sha,data=validate_declaration(sealed,admission,binary,repo=stage)
    report=json.loads((attempt/'report.json').read_text())
    validate_report_metadata(report)
    require(report['status']!='running','attempt is still running; audit forbidden')
    require(report['status']=='complete' or report['status'].startswith('partial_closed_'),'unknown closed status')
    require(report['declaration_sha256']==declaration_sha,'report declaration fingerprint differs')
    require(equal(json.loads((attempt/'declaration.json').read_text()),declaration),'attempt declaration differs')
    require(report['admission_sha256']==digest(admission),'report admission fingerprint differs')
    jobs,workload=query_plan(data,LabelBudget(**declaration['limits']))
    expected={row['id']:row for row in jobs}
    require(len(expected)==len(jobs),'query IDs duplicate in frozen plan')
    require(equal(report['workload'],workload),'report workload differs')
    require(report['requested_queries']==len(jobs),'query count differs')
    files={p.stem:p for p in attempt.glob('*.json') if re.fullmatch('[0-9a-f]{64}',p.stem)}
    other_json={p.name for p in attempt.glob('*.json')}- {p.name for p in files.values()}
    allowed_json={'report.json','declaration.json'} | {
        f'worker-{i}-{kind}.json' for i in range(declaration['limits']['workers']) for kind in ('setup','error')}
    require(other_json<=allowed_json,'unexpected JSON artifact outside frozen attempt schema')
    if report['status']=='complete':require(not list(attempt.glob('*.tmp')),'successful attempt retained incomplete atomic output')
    require(set(files)<=set(expected),'output contains an undeclared query')
    if report['status']=='complete': require(set(files)==set(expected),'successful attempt missing query files')
    listed={row['query_id']:row['sha256'] for row in report['source_records']}
    require(len(listed)==len(report['source_records']),'report duplicates query IDs')
    require(set(listed)<=set(files),'report references missing query files')
    if report['status']=='complete':require(set(listed)==set(files),'successful report excludes output records')
    require(report['completed_queries']==len(listed),'completed query count differs from reported inventory')
    spec=importlib.util.spec_from_file_location('_independent_chess_admission',oracle_source)
    oracle=importlib.util.module_from_spec(spec);spec.loader.exec_module(oracle)
    setup_commands=['uci','setoption name Threads value 1','setoption name Hash value 64',
        'setoption name NumaPolicy value none','setoption name Ponder value false',
        'setoption name UCI_LimitStrength value false','setoption name Skill Level value 20',
        'setoption name UCI_ShowWDL value false','setoption name UCI_Chess960 value false',
        'setoption name SyzygyPath value','setoption name SyzygyProbeLimit value 0','isready']
    setup=list(attempt.glob('worker-*-setup.json'))
    require(len(setup)==declaration['limits']['workers'],'missing worker setup')
    for file in setup:
        value=json.loads(file.read_text())
        require([line[2:] for line in value['trace'] if line.startswith('> ')]==setup_commands,'setup options/order differ')
        require(any(line.startswith('id name Stockfish 19') for line in value['identity']),'teacher identity differs')
        require(value['blocked_cancellation_signals']==[],'worker cancellation mask remains blocked')
    if report['status']=='complete':
        require(report['worker_exitcodes']==[0]*len(setup),'successful report contains failed worker')
        require(not list(attempt.glob('worker-*-error.json')),'successful report contains worker errors')
    cache={}; blocks={}; reasons={}; telemetry={}; full_root_rows=Counter(); inventory=[]
    frame=declaration['score_frame']
    weights=(1,3,3,5,9,0)*2
    with localcontext() as decimal_context:
        decimal_context.prec=45
        coeff=tuple(Decimal(str(x)) for x in frame['normalization_coefficients'])
        def a(material):
            m=Decimal(min(max(material,17),78))/Decimal(58)
            return ((coeff[0]*m+coeff[1])*m+coeff[2])*m+coeff[3]
        reference=a(58)
        for query_id,job in expected.items():
            split,kind=job['split'],'all_action' if job['all_legal'] else 'standalone'
            category=split+':'+kind
            telemetry.setdefault(category,dict(queries=0,reported_nodes=[],maximum_reported_depth=[],
                complete_packet_depth=Counter(),elapsed_seconds=[],score_kind_flags=Counter(),requested_nodes=0,
                rejected_packet_flags=Counter(),
                node_budget_overshoot_count=0))
            family=(split,job['position']['game_id']); blocks.setdefault(family,[])
            if query_id not in files:
                blocks[family].append('missing:'+kind);continue
            path=files[query_id]; fingerprint=digest(path)
            if query_id in listed:require(fingerprint==listed[query_id],'record byte SHA differs: '+query_id)
            record=json.loads(path.read_text())
            require(set(record)=={'query','labels','rejection','transcript','elapsed_seconds','declaration_sha256'},'record keys differ')
            require(equal(record['query'],job),'query/history/node selection differs: '+query_id)
            require(record['declaration_sha256']==declaration_sha,'record declaration differs')
            observed=job['position']; key=(observed['game_id'],observed['ply'])
            if key not in cache:
                board=chess.Board(); histories=[]
                for uci in observed['moves']:
                    histories.append(oracle.repetition_hash(board));move=chess.Move.from_uci(uci)
                    require(move in board.legal_moves,'illegal ingress prefix');board.push(move)
                require(board.fen(en_passant='fen')==observed['fen'],'prefix/raw EP/FEN differs')
                require(histories==observed['previous_hashes'],'independent repetition history differs')
                require(len(observed['moves'])==observed['ply'],'source ply differs')
                cache[key]=board
            board=cache[key]; legal={move.uci() for move in board.legal_moves}
            support=len(legal) if job['all_legal'] else 1
            commands=[line[2:] for line in record['transcript'] if line.startswith('> ')]
            require(commands==['setoption name MultiPV value '+str(support),'ucinewgame','isready',
                'position startpos'+(' moves '+' '.join(observed['moves']) if observed['moves'] else ''),
                'go nodes '+str(job['nodes'])],'query exact UCI commands differ')
            trace=record['transcript'];go=trace.index('> go nodes '+str(job['nodes']))
            lines=[line[2:] for line in trace[go+1:] if line.startswith('< ')]
            require(lines and lines[-1].startswith('bestmove '),'query did not complete with bestmove')
            require(sum(line.startswith('bestmove ') for line in lines)==1,'duplicate bestmove')
            require(lines[-1].split()[1] in legal,'bestmove is illegal')
            pos=GamePosition(observed['game_id'],observed['ply'],observed['fen'],tuple(observed['moves']),tuple(observed['previous_hashes']))
            labels,rejection=common_cp_labels(lines,pos,all_legal=job['all_legal'],min_depth=declaration['limits']['min_reported_depth'])
            require(equal(labels,record['labels']) and rejection==record['rejection'],'packet rejection/conversion differs')
            outcome=rejection or 'accepted'
            reasons.setdefault(category,Counter())[outcome]+=1
            if rejection:blocks[family].append(kind+':'+rejection)
            parsed=[value for line in lines if (value:=parse_pv(line)) is not None]
            packets={};current={};depth=None
            for row in parsed:
                if row.multipv==1:current={};depth=row.depth
                if row.depth!=depth or row.multipv!=len(current)+1:current={};depth=None;continue
                current[row.multipv]=row
                if len(current)==support:packets[row.depth]=tuple(current.values())
            packet=packets[max(packets)] if packets else ()
            details=telemetry[category];details['queries']+=1;details['requested_nodes']+=job['nodes']
            maximum_nodes=max((row.nodes for row in parsed),default=0)
            details['reported_nodes'].append(maximum_nodes)
            details['maximum_reported_depth'].append(max((row.depth for row in parsed),default=0))
            details['complete_packet_depth'][str(max(packets)) if packets else 'none']+=1
            if maximum_nodes>job['nodes']:details['node_budget_overshoot_count']+=1
            for row in packet:details['score_kind_flags'][row.score_kind+(':'+row.bound if row.bound else ':unbounded')]+=1
            if rejection=='mate_or_bound_common_packet':
                mate=any(row.score_kind=='mate' for row in packet)
                bound=any(row.bound is not None for row in packet)
                details['rejected_packet_flags']['mate_and_bound' if mate and bound else 'mate_only' if mate else 'bound_only']+=1
            elapsed=record['elapsed_seconds']
            require(type(elapsed) in (int,float) and math.isfinite(elapsed) and elapsed>=0,'invalid query wall telemetry')
            details['elapsed_seconds'].append(elapsed)
            if labels is not None:
                counts=[len(board.pieces(piece,color)) for color in (chess.WHITE,chess.BLACK) for piece in range(1,7)]
                material=sum(c*w for c,w in zip(counts,weights));factor=a(material)/reference
                require(labels['reference_piece_counts_absolute']==counts,'query root piece counts differ')
                require(labels['normalization_root_material']==material,'normalization used wrong root')
                require(labels['converted_perspective']==('child_mover' if job['all_legal'] else 'queried_root_mover'),'converted perspective differs')
                require(len(labels['rows'])==support,'label support size differs')
                if job['all_legal']:
                    require({row['uci'] for row in labels['rows']}==legal,'root legal support incomplete')
                    full_root_rows[split]+=support
                for row in labels['rows']:
                    move=chess.Move.from_uci(row['uci']);flip=0 if board.turn else 56
                    promo={None:0,chess.KNIGHT:1,chess.BISHOP:2,chess.ROOK:3,chess.QUEEN:4}[move.promotion]
                    action=(((move.from_square^flip)*64+(move.to_square^flip))*5+promo)
                    require(row['canonical_action']==action,'independent canonical action differs')
                    converted=row['converted_score'];cp=row['root_score_cp']
                    require(type(cp) is int,'ordinary CP is not integer')
                    sign=-1 if job['all_legal'] else 1
                    midpoint=Decimal(cp)*factor*sign;half=Decimal('.5')*factor
                    require(abs(Decimal(str(converted['value_cp']))-midpoint)<Decimal('1e-9')*(abs(midpoint)+1),'independent midpoint differs')
                    require(abs(Decimal(str(converted['quantization_halfwidth']))-half)<Decimal('1e-10'),'independent halfwidth differs')
                    require(Decimal(str(converted['lower_cp']))<=midpoint-half and
                            Decimal(str(converted['upper_cp']))>=midpoint+half,'reporting interval is not outward')
                    require(converted['normalization_root_material']==material and converted['negated'] is job['all_legal'],'conversion sign/material differs')
            inventory.append(dict(query_id=query_id,sha256=fingerprint,split=split,kind=kind,outcome=outcome))
    attempt_ok=report['status']=='complete' and report['worker_exitcodes']==[0]*declaration['limits']['workers']
    accepted={split:[] for split in ('train','validation','test')};rejected={split:[] for split in accepted}
    mechanical={split:[] for split in accepted};family_failures={split:Counter() for split in accepted}
    for (split,game_id),failures in blocks.items():
        if not failures:mechanical[split].append(game_id)
        if attempt_ok and not failures:accepted[split].append(game_id)
        else:rejected[split].append(game_id)
        family_failures[split]['standalone_failed_only' if failures and all(f.startswith('standalone') for f in failures)
            else 'root_failed_only' if failures and all(f.startswith('all_action') for f in failures)
            else 'both_or_missing' if failures else 'complete']+=1
    for name,computed in [('whole_family_accepted',accepted),('whole_family_rejected',rejected),
            ('mechanically_complete_families_not_admitted_if_attempt_failed',mechanical)]:
        require(equal(report[name],computed),'complete family coverage differs: '+name)
    for details in telemetry.values():
        for field in ('reported_nodes','maximum_reported_depth','elapsed_seconds'):details[field]=stats(details[field])
    require(all(len(data['source_records'][split])==len(accepted[split])+len(rejected[split]) for split in accepted),'game family inventory incomplete')
    return dict(kind='closed_teacher_attempt_mechanical_audit',status='pass',strength_evidence=False,
        scalar_score_error_or_strength_metrics_inspected=False,
        declaration_sha256=declaration_sha,admission_sha256=digest(admission),report_sha256=digest(attempt/'report.json'),
        auditor_sha256=digest(__file__),independent_history_oracle_sha256=digest(oracle_source),
        frozen_source_sha256=declaration['source_sha256'],attempt_status=report['status'],
        immutable_query_count=len(jobs),record_count=len(files),reported_record_count=len(listed),
        every_record_hash_query_command_packet_frame_and_history_checked=True,
        worker_exitcodes=report['worker_exitcodes'],adopted_children_reaped=report['adopted_children_reaped'],
        producer_wall_seconds=report['wall_seconds'],requested_workload=workload,
        query_outcomes=reasons,packet_and_node_and_elapsed_telemetry=telemetry,
        accepted_family_counts={k:len(v) for k,v in accepted.items()},rejected_family_counts={k:len(v) for k,v in rejected.items()},
        complete_root_query_legal_rows=dict(full_root_rows),family_failure_categories=family_failures,
        training_admission='inadmissible_missing_validation' if not accepted['validation'] else 'separate_learning_declaration_required',
        actual_max_reported_nodes_are_telemetry_not_exact_total_work=True,
        query_file_inventory=inventory,audit_wall_seconds=time.monotonic()-start)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('stage','attempt','sealed','admission','binary','oracle-source','output'):p.add_argument('--'+name,required=True)
    args=p.parse_args(); result=audit(args.stage,args.attempt,args.sealed,args.admission,args.binary,args.oracle_source)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('query_file_inventory','frozen_source_sha256')},indent=2))
