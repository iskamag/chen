"""Exact CLOSED restricted legal destination-exchange targets, not chess value.

All traces/returns are supervision. No neural module or GPU is imported. A cap,
required off-destination check evasion, or real terminal supplies no CP target.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict, dataclass
import gzip
import hashlib
import json
from pathlib import Path
import time

import chess

from .tactical_inputs import (_terminal, action_id, canonical_square, position_hash, VALUES)

KIND = 'chen_closed_destination_exchange_v1'


@dataclass(frozen=True)
class Limits:
    max_plies: int = 16              # includes initial action
    nodes_per_action: int = 4096
    global_nodes: int = 250000
    wall_seconds: int = 180

    def __post_init__(self):
        if any(type(v) is not int or v < 1 for v in asdict(self).values()):
            raise ValueError('positive integer exchange limits required')


class MissingTarget(Exception):
    def __init__(self, reason, detail=None):
        self.reason, self.detail = reason, detail


def material(board, mover):
    return sum(VALUES[p.piece_type] * (1 if p.color == mover else -1)
               for p in board.piece_map().values())


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def complete_initial_actions(board):
    return sorted((m for m in board.legal_moves if board.is_capture(m) or m.promotion),
                  key=lambda m: action_id(m, board.turn))


def exchange_target(board, initial, *, limits=Limits(), global_charge=None, deadline=None,
                    reverse_order=False):
    """One whole action: closed exact material delta or explicitly missing.

    Board must retain its authoritative history. Authored FENs are mechanics
    fixtures only; the pilot supplies standard-start legal replay witnesses.
    """
    if not isinstance(board, chess.Board) or not board.is_valid() or board.chess960:
        raise ValueError('valid standard-chess board required')
    if (initial not in board.legal_moves or not (board.is_capture(initial) or initial.promotion) or
            _terminal(board) is not None or not isinstance(limits, Limits) or type(reverse_order) is not bool):
        raise ValueError('nonterminal legal initial capture/promotion and explicit limits required')
    parent = board.turn
    reference = material(board, parent)
    index = action_id(initial, parent)
    raw_destination = initial.to_square
    old = board.piece_map()
    ep = board.is_en_passant(initial)
    victim = initial.to_square + (-8 if parent else 8) if ep else initial.to_square if board.is_capture(initial) else -1
    state = board.copy(stack=True)
    state.push(initial)
    new = state.piece_map()
    code = lambda piece: 0 if piece is None else piece.piece_type + (0 if piece.color == parent else 6)
    edits = sorted((canonical_square(s,parent),code(old.get(s)),code(new.get(s)))
                   for s in set(old)|set(new) if old.get(s)!=new.get(s))
    result = dict(kind=KIND, initial_action_id=index, initial_uci=initial.uci(),
        score_frame='initial_parent_mover_material_delta_cp', reference_material_parent_cp=reference,
        occupied_destination=canonical_square(raw_destination,parent), en_passant=ep,
        victim_square=canonical_square(victim,parent) if victim>=0 else -1,
        promotion_suffix=index%5, initial_edits=edits,
        simple_material_delta_cp=material(state,parent)-reference, limits=asdict(limits),
        trace=[], entered_states=0, exact=False, target_cp=None, task='restricted_legal_destination_recaptures')
    trace = result['trace']

    def visit(path):
        if deadline is not None and time.monotonic()>=deadline:
            raise MissingTarget('wall_cap')
        if len(trace)>=limits.nodes_per_action:
            raise MissingTarget('node_cap')
        if global_charge is not None:
            global_charge()
        node_id=len(trace)
        checked=state.is_check()
        legal=list(state.legal_moves)
        recaptures=[m for m in legal if m.to_square==raw_destination and state.is_capture(m)]
        recaptures.sort(key=lambda m:action_id(m,parent),reverse=reverse_order)
        delta=material(state,parent)-reference
        terminal=_terminal(state)
        node=dict(id=node_id, branch_uci=list(path), fen=state.fen(en_passant='fen'),
            position_hash=f'{position_hash(state):016x}', mover_is_parent=state.turn==parent,
            checked=checked, decline_allowed=not checked and terminal is None,
            history=dict(halfmove_clock=state.halfmove_clock,current_repetition=state.is_repetition(3),
                         terminal=terminal), raw_material_delta_cp=delta,
            legal_action_ids_current=sorted(action_id(m,state.turn) for m in legal),
            recapture_action_ids_parent=sorted(action_id(m,parent) for m in recaptures), children=[])
        trace.append(node)
        if terminal:
            raise MissingTarget('typed_terminal',dict(terminal=terminal,branch_uci=list(path),
                result_current_mover=-1 if terminal=='checkmate' else 0,
                result_parent=(-1 if state.turn==parent else 1) if terminal=='checkmate' else 0))
        if state.piece_at(raw_destination) is None:
            raise ValueError('restricted destination unexpectedly empty')
        if checked:
            outside=[m for m in legal if m not in recaptures]
            if outside:
                raise MissingTarget('unsupported_check_evasion',dict(branch_uci=list(path),
                    outside_action_ids_current=sorted(action_id(m,state.turn) for m in outside)))
        if len(path)>=limits.max_plies and recaptures:
            raise MissingTarget('ply_cap')
        choices=[] if checked else [(delta,-1,'decline')]
        for move in recaptures:
            state.push(move)
            try:
                child_value,child_id=visit((*path,move.uci()))
            finally:
                state.pop()
            node['children'].append(dict(action_id_parent=action_id(move,parent),uci=move.uci(),child_id=child_id))
            choices.append((child_value,action_id(move,parent),'recapture'))
        # No pruning: every allowed branch is closed before any point target.
        if not choices:
            raise ValueError('nonterminal checked node has no legal task continuation')
        best=(max if state.turn==parent else min)(value for value,_,_ in choices)
        winner=min((choice for choice in choices if choice[0]==best),key=lambda choice:choice[1])
        node.update(material_return_parent_cp=best,
                    material_return_current_cp=best*(1 if state.turn==parent else -1),
                    selected_option=winner[2],selected_action_id_parent=winner[1],
                    dependency_round=1+max((trace[c['child_id']]['dependency_round'] for c in node['children']),default=0),
                    closed=True)
        return best,node_id

    try:
        value,_=visit((initial.uci(),))
        result.update(status='closed_exact',exact=True,target_cp=value,
                      correction_vs_immediate_cp=value-result['simple_material_delta_cp'],
                      dependency_rounds=trace[0]['dependency_round'])
    except MissingTarget as error:
        result.update(status='missing',missing_reason=error.reason,missing_detail=error.detail)
        # Avoid accidental point-loss admission from partially completed siblings.
        for node in trace:
            for name in ('material_return_parent_cp','material_return_current_cp',
                         'selected_option','selected_action_id_parent','dependency_round','closed'):
                node.pop(name,None)
    result['entered_states']=len(trace)
    return result


def generate(shard, output, *, limits=Limits()):
    """One frozen full TRAIN shard attempt; complete capture/promotion masks."""
    output=Path(output)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    deadline=started+limits.wall_seconds
    source_root=Path(__file__).resolve().parent
    source_pins={name:hashlib.sha256((source_root/name).read_bytes()).hexdigest()
                 for name in ('exchange_targets.py','tactical_inputs.py')}
    declaration=dict(kind=KIND,shard_sha256=hashlib.sha256(Path(shard).read_bytes()).hexdigest(),
                     source_sha256=source_pins,limits=asdict(limits),split='train',
                     plan='all_declared_nonterminal_roots_all_legal_captures_and_promotions_no_replacement',
                     trace_policy='canonical_DFS_no_pruning_all_branches_closed',
                     input_target_separation='all_trace_states_and_returns_supervision_only',
                     target_frame='initial_parent_mover_material_delta_cp',
                     training=False,gpu=False,strategic_cp=False)
    (output/'declaration.json').write_text(json.dumps(declaration,indent=2)+'\n')
    counts=Counter()
    returns=Counter()
    motifs=Counter()
    missing=Counter()
    elapsed_values=[]
    seen=set()
    def charge():
        if counts['entered_states']>=limits.global_nodes:
            raise MissingTarget('global_node_cap')
        counts['entered_states']+=1
    try:
        with gzip.open(shard,'rt') as inputs,gzip.open(output/'targets.jsonl.gz','wt') as sink:
            for line in inputs:
                if time.monotonic()>=deadline:raise TimeoutError('pilot wall cap')
                family=json.loads(line)
                if (family.get('schema')!='chen_reachable_mechanics_v1' or family.get('split')!='train' or
                    family.get('outcome_label') is not None or family['family_id'] in seen):
                    raise ValueError('unique whole-family TRAIN mechanics source required')
                seen.add(family['family_id'])
                source_roots={root['ply']:root for root in family['roots']}
                if len(source_roots)!=len(family['roots']):raise ValueError('duplicate source roots')
                replay=chess.Board()
                roots=[]
                for ply in range(len(family['moves'])+1):
                    if ply in source_roots:
                        supplied=source_roots[ply]
                        if replay.fen(en_passant='fen')!=supplied['fen']:
                            raise ValueError('source root FEN differs from full legal replay')
                        actions=complete_initial_actions(replay)
                        all_ids=sorted(action_id(m,replay.turn) for m in replay.legal_moves)
                        if all_ids!=[a['id'] for a in supplied['actions']]:raise ValueError('source complete support differs')
                        observations=[]
                        for action in actions:
                            tick=time.monotonic()
                            label=exchange_target(replay,action,limits=limits,global_charge=charge,deadline=deadline)
                            elapsed_values.append(time.monotonic()-tick)
                            observations.append(label)
                            counts['initial_actions']+=1
                            counts['ep_initial']+=label['en_passant']
                            counts['promotion_initial']+=bool(label['promotion_suffix'])
                            if label['exact']:
                                counts['closed_exact']+=1
                                correction=label['correction_vs_immediate_cp']
                                returns['nonzero_correction']+=correction!=0
                                returns['sum_abs_correction_cp']+=abs(correction)
                                returns['sum_return_cp']+=label['target_cp']
                                counts['closed_with_recapture']+=len(label['trace'])>1
                                counts['sum_dependency_rounds']+=label['dependency_rounds']
                                for motif in supplied['motifs']:motifs[motif]+=1
                            else:missing[label['missing_reason']]+=1
                        roots.append(dict(ply=ply,fen=supplied['fen'],root_context_sha256=_digest(
                            [family['family_id'],family['moves'][:ply],supplied['fen']]),
                            mover=supplied['mover'],motifs=supplied['motifs'],
                            complete_initial_action_ids=[action_id(m,replay.turn) for m in actions],targets=observations))
                        counts['roots']+=1
                    if ply<len(family['moves']):
                        if _terminal(replay):raise ValueError('family replay crosses terminal')
                        move=chess.Move.from_uci(family['moves'][ply])
                        if move not in replay.legal_moves:raise ValueError('illegal family replay')
                        replay.push(move)
                if len(roots)!=len(source_roots):raise ValueError('source roots outside game prefix')
                sink.write(json.dumps(dict(kind=KIND,family_id=family['family_id'],split='train',
                    moves=family['moves'],roots=roots),separators=(',',':'))+'\n')
                counts['families']+=1
                print(json.dumps(dict(families=counts['families'],roots=counts['roots'],actions=counts['initial_actions'])),flush=True)
        status='complete'
        if time.monotonic()>deadline:raise TimeoutError('final pilot wall cap')
    except BaseException as error:
        (output/'report.json').write_text(json.dumps(dict(status='failed_no_dataset_admission',
            error=dict(type=type(error).__name__,message=str(error)),counts=dict(counts)),indent=2)+'\n')
        raise
    report=dict(status=status,counts=dict(counts),missing=dict(missing),closed_motifs=dict(motifs),
                return_vs_immediate=dict(returns),generation_wall_seconds=time.monotonic()-started,
                max_action_seconds=max(elapsed_values,default=0),
                target_sha256=hashlib.sha256((output/'targets.jsonl.gz').read_bytes()).hexdigest(),
                strategic_quality_claim=False,training=False,gpu=False)
    (output/'report.json').write_text(json.dumps(report,indent=2)+'\n')
    return report


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--shard',required=True)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    print(json.dumps(generate(args.shard,args.output),indent=2))


if __name__=='__main__':main()
