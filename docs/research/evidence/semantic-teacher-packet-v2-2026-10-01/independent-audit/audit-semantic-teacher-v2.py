#!/usr/bin/env python3
"""Independent packet-v2 mechanical audit of one CLOSED teacher attempt.

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
import platform
import selectors


class PacketSyntax(ValueError):
    pass


def independent_line(line):
    """Separate strict field parser: no producer parser or selector is called."""
    words = line.split()
    if not words or words[0] != 'info' or words[1:2] == ['string'] or 'score' not in words:
        return None
    required = ('depth', 'seldepth', 'multipv', 'score', 'nodes', 'tbhits', 'pv')
    if any(words.count(key) != 1 for key in required):
        raise PacketSyntax('required or duplicate scored field')
    boundary = words.index('pv')
    prefix = words[:boundary]
    if any(key not in prefix for key in required[:-1]) or boundary == len(words)-1:
        raise PacketSyntax('missing scored header or PV')
    try:
        number = lambda key: int(prefix[prefix.index(key)+1])
        score_at = prefix.index('score')
        kind, score = prefix[score_at+1], int(prefix[score_at+2])
        row = dict(depth=number('depth'), seldepth=number('seldepth'), rank=number('multipv'),
                   nodes=number('nodes'), hits=number('tbhits'), kind=kind, score=score,
                   pv=words[boundary+1:])
    except (ValueError, IndexError):
        raise PacketSyntax('invalid scored value') from None
    flags = [token for token in prefix if token in ('lowerbound', 'upperbound')]
    if (kind not in ('cp', 'mate') or len(flags)>1 or
            min(row['depth'],row['rank'],row['nodes'])<1 or min(row['seldepth'],row['hits'])<0):
        raise PacketSyntax('invalid scored telemetry or bounds')
    row['bound'] = flags[0] if flags else None
    return row


def independent_packets(lines, board, all_legal, min_depth):
    """Group complete rank-one delimited emissions, then verify and select.

    Error priority is intentionally not a second copy of producer early returns:
    acceptance and selected emission must agree; exact first-error serialization
    is separately checked by pinned producer recomputation.
    """
    import chess
    require(type(all_legal) is bool and type(min_depth) is int and min_depth>0,'invalid packet contract')
    legal={move.uci() for move in board.legal_moves}
    support=len(legal) if all_legal else 1
    groups=[]; malformed=[]; bestmoves=[]; parsed=[]; after_bestmove=False
    for offset,line in enumerate(lines):
        tokens=line.split()
        if tokens[:1]==['bestmove']:
            bestmoves.append(dict(line_offset=offset,tokens=tokens[1:]));continue
        try: row=independent_line(line)
        except PacketSyntax:
            malformed.append(offset);continue
        if row is None:continue
        if bestmoves:after_bestmove=True
        row['offset']=offset;parsed.append(row)
        if row['rank']==1:groups.append([])
        if not groups:groups.append([])  # illegal initial rank, still diagnostic
        groups[-1].append(row)
    fatal=bool(malformed or after_bestmove)
    if (len(bestmoves)>1 or any(len(b['tokens']) not in (1,3) or
            (len(b['tokens'])==3 and b['tokens'][1]!='ponder') or
            not b['tokens'] or b['tokens'][0] not in legal for b in bestmoves)):
        fatal=True
    decisive=any(row['kind']=='mate' or row['hits'] or abs(row['score'])>=19754 for row in parsed)
    trace=[]; eligible=[]; ordinal=0
    for group in groups:
        if not group:continue
        if group[0]['rank']==1:ordinal+=1
        ranks=[r['rank'] for r in group];depths=[r['depth'] for r in group];nodes=[r['nodes'] for r in group]
        rank_bad=ranks!=list(range(1,len(group)+1)) or len(group)>support
        complete=len(group)==support
        same_nodes=len(set(nodes))==1;same_depth=len(set(depths))==1
        bounded=any(r['bound'] is not None for r in group)
        local_decisive=any(r['kind']=='mate' or r['hits'] or abs(r['score'])>=19754 for r in group)
        classification=('decisive' if local_decisive else 'malformed_ranks' if rank_bad
            else 'incomplete' if not complete else 'inconsistent_nodes' if not same_nodes
            else 'mixed_depth' if not same_depth else 'bounded' if bounded
            else 'below_depth_floor' if depths[0]<min_depth else 'ordinary_complete_structure')
        meta=dict(ordinal=ordinal,line_start=group[0]['offset'],line_end=group[-1]['offset'],
                  ranks=ranks,depths=depths,nodes=nodes,classification=classification)
        trace.append(meta)
        fatal |= rank_bad
        for row in group:
            current=board.copy(stack=True)
            try:
                for uci in row['pv']:
                    move=chess.Move.from_uci(uci)
                    if move not in current.legal_moves:raise ValueError('illegal continuation')
                    current.push(move)
            except ValueError:fatal=True
        if complete:
            first=[r['pv'][0] for r in group]
            fatal |= len(set(first))!=support or any(m not in legal for m in first)
            fatal |= all_legal and set(first)!=legal
            fatal |= not same_nodes
            if same_depth and all(r['kind']=='cp' for r in group):
                fatal |= any(group[i]['score']<group[i+1]['score'] for i in range(len(group)-1))
            if not rank_bad and same_nodes and same_depth and not bounded and not local_decisive and depths[0]>=min_depth:
                eligible.append((depths[0],ordinal,group,meta))
    chosen=max(eligible,key=lambda row:(row[0],row[1])) if eligible and not fatal and not decisive else None
    return dict(selected=chosen,trace=trace,malformed=malformed,bestmoves=bestmoves,parsed=parsed,
                rejected=chosen is None,decisive=decisive)


def validate_runtime(path,binary):
    """Compare archived producer runtime with the auditor's actual CPU runtime."""
    import chess
    import torch
    archived=json.loads(Path(path).read_text())
    actual=dict(python_version=sys.version,python_executable=str(Path(sys.executable).resolve()),
        python_executable_sha256=digest(Path(sys.executable).resolve()),platform=platform.platform(),
        chess_version=chess.__version__,chess_module_sha256=digest(chess.__file__),
        torch_version=torch.__version__,torch_hip_build=torch.version.hip,
        torch_cuda_build=torch.version.cuda,stockfish_sha256=digest(binary),
        stdlib_selectors_sha256=digest(selectors.__file__),gpu_runtime_invoked=False)
    require(all(equal(archived.get(key),value) for key,value in actual.items()),'CPU runtime manifest differs')
    return digest(path),actual


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


def audit(stage,attempt,sealed,admission,binary,oracle_source,freeze,runtime_manifest):
    start=time.monotonic(); stage=Path(stage).resolve(); attempt=Path(attempt).resolve()
    frozen=json.loads(Path(freeze).read_text())
    require(frozen['seed']==20261008 and frozen['spent_position_count']==768311 and
            frozen['spent_report_count']==8,'unexpected independent-study freeze')
    require(len(frozen['source_sha256'])==16,'frozen source set differs')
    for name,expected_sha in frozen['source_sha256'].items():
        require(digest(stage/name)==expected_sha,'frozen source bytes differ: '+name)
    for name,expected_sha in frozen['files_sha256'].items():
        require(digest(stage/name)==expected_sha,'frozen preparation bytes differ: '+name)
    sys.path.insert(0,str(stage))
    import chess
    import neural.semantic_study as producer
    import neural.semantic_data as data_module
    require(Path(producer.__file__).resolve()==stage/'neural/semantic_study.py' and
            Path(data_module.__file__).resolve()==stage/'neural/semantic_data.py','wrong package source loaded')
    from neural.semantic_study import validate_declaration,query_plan,LabelBudget
    from neural.semantic_data import GamePosition,common_cp_labels
    declaration,declaration_sha,data=validate_declaration(sealed,admission,binary,repo=stage)
    require(equal(declaration['source_sha256'],frozen['source_sha256']),'seal and freeze source sets differ')
    require(data['seed']==20261008 and data['spent_position_count']==768311 and len(data['spent_reports'])==8,
            'fresh admission exclusions differ')
    runtime_sha,runtime=validate_runtime(runtime_manifest,binary)
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
            require(set(record)=={'query','labels','rejection','packet_provenance','transcript','elapsed_seconds','declaration_sha256'},'record keys differ')
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
            diagnostics={}
            labels,rejection=common_cp_labels(lines,pos,all_legal=job['all_legal'],
                min_depth=declaration['limits']['min_reported_depth'],diagnostics=diagnostics)
            require(equal(labels,record['labels']) and rejection==record['rejection'],'packet rejection/conversion differs')
            require(equal(diagnostics,record['packet_provenance']),'serialized packet provenance differs')
            independently=independent_packets(lines,board,job['all_legal'],declaration['limits']['min_reported_depth'])
            require(independently['rejected']==(labels is None),'independent packet admission decision differs')
            require(equal(independently['trace'],diagnostics['emission_trace']) and
                    equal(independently['malformed'],diagnostics['malformed_scored_line_offsets']) and
                    equal(independently['bestmoves'],diagnostics['final_bestmove_emissions']),
                    'independent full-transcript emission provenance differs')
            if labels is not None:
                depth,ordinal,selected,meta=independently['selected']
                require(diagnostics['selected_emission_ordinal']==ordinal and
                        diagnostics['selected_common_depth']==depth and
                        diagnostics['selected_line_start']==meta['line_start'] and
                        diagnostics['selected_line_end']==meta['line_end'],'independent selected packet differs')
                require(diagnostics['final_bestmove']==lines[-1].split()[1],'final choice provenance differs')
                require(diagnostics['final_bestmove_differs_from_selected_PV1'] is
                        (lines[-1].split()[1]!=selected[0]['pv'][0]),'selected/final distinction differs')
                for target,source in zip(labels['rows'],selected):
                    require(target['uci']==source['pv'][0] and target['pv']==source['pv'] and
                            target['root_score_cp']==source['score'] and target['nodes']==source['nodes'] and
                            target['reported_depth']==source['depth'] and target['seldepth']==source['seldepth'],
                            'selected row identity or telemetry differs')
            outcome=rejection or 'accepted'
            reasons.setdefault(category,Counter())[outcome]+=1
            if rejection:blocks[family].append(kind+':'+rejection)
            parsed=independently['parsed']
            details=telemetry[category];details['queries']+=1;details['requested_nodes']+=job['nodes']
            maximum_nodes=max((row['nodes'] for row in parsed),default=0)
            details['reported_nodes'].append(maximum_nodes)
            details['maximum_reported_depth'].append(max((row['depth'] for row in parsed),default=0))
            details['complete_packet_depth'][str(independently['selected'][0]) if labels is not None else 'none']+=1
            if maximum_nodes>job['nodes']:details['node_budget_overshoot_count']+=1
            for row in parsed:details['score_kind_flags'][row['kind']+(':'+row['bound'] if row['bound'] else ':unbounded')]+=1
            if rejection:details['rejected_packet_flags'][rejection]+=1
            details.setdefault('emission_classifications',Counter()).update(meta['classification'] for meta in independently['trace'])
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
                    require(row['negated_action_score_cp']==-cp,'raw action sign differs')
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
    return dict(kind='closed_packet_v2_teacher_attempt_independent_mechanical_audit',status='pass',strength_evidence=False,
        scalar_score_error_or_strength_metrics_inspected=False,
        declaration_sha256=declaration_sha,admission_sha256=digest(admission),report_sha256=digest(attempt/'report.json'),
        auditor_sha256=digest(__file__),independent_history_oracle_sha256=digest(oracle_source),
        preparation_freeze_sha256=digest(freeze),runtime_manifest_sha256=runtime_sha,runtime=runtime,
        frozen_source_sha256=declaration['source_sha256'],attempt_status=report['status'],
        immutable_query_count=len(jobs),record_count=len(files),reported_record_count=len(listed),
        every_record_hash_query_command_packet_frame_and_history_checked=True,
        independent_grammar_emission_selection_and_trace_checked=True,
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
    for name in ('stage','attempt','sealed','admission','binary','oracle-source','freeze','runtime-manifest','output'):
        p.add_argument('--'+name,required=True)
    args=p.parse_args(); result=audit(args.stage,args.attempt,args.sealed,args.admission,args.binary,args.oracle_source,
                                     args.freeze,args.runtime_manifest)
    output=Path(args.output);output.parent.mkdir(parents=True,exist_ok=True)
    tmp=output.with_name(output.name+'.tmp');tmp.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n');tmp.replace(output)
    print(json.dumps({k:v for k,v in result.items() if k not in ('query_file_inventory','frozen_source_sha256')},indent=2))
