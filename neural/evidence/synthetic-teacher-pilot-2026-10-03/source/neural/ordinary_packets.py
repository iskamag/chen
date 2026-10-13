"""Packet-v2: strict UCI emissions, never rows spliced across rank-one resets.

Pure CPU grammar/provenance. No engine process, score generation or model import.
"""
from dataclasses import dataclass

PROTOCOL_VERSION = 2
POLICY = {
    'version': PROTOCOL_VERSION,
    'emission_boundary': 'each_rank_one_starts_new_emission_finalize_at_next_rank_one_or_end',
    'selection': 'highest_eligible_common_reported_depth_latest_completion_ordinal_tiebreak',
    'eligible': 'ordered_1_to_K_same_depth_same_nodes_complete_unique_legal_support_ordinary_unbounded_cp_min_depth6',
    'ineligible': 'clean_incomplete_mixed_depth_bound_only_below_depth_floor',
    'query_rejection': 'malformed_scored_grammar_rank_duplicate_gap_excess_support_PV_rank_score_or_any_mate_tbhits_reserved_decisive_cp',
    'final_bestmove': 'record_separately_may_differ_from_selected_packet_PV1',
    'search_source_url': 'https://github.com/official-stockfish/Stockfish/blob/sf_19/src/search.cpp',
    'search_source_sha256': 'a934524dd2f386ec38cdf95b7b2e41cecdc85c9b17e12c0668621e6ae16be28a',
    'source_context': 'output_pv_2273_2342_final_inexact_250_264_completed_iteration_495_498',
}

@dataclass(frozen=True)
class PVLine:
    depth: int
    seldepth: int
    multipv: int
    score_kind: str
    score: int
    bound: str | None
    nodes: int
    pv: tuple[str, ...]
    tablebase_hits: int = 0


def parse_pv(line):
    tokens = line.split()
    if not tokens or tokens[0] != 'info' or (len(tokens)>1 and tokens[1]=='string'):
        return None
    if 'score' not in tokens:
        return None
    # Every scored line must be a fully parseable PV line, not silently skipped.
    keys = ('score','depth','seldepth','multipv','nodes','tbhits','pv')
    if any(tokens.count(key)>1 for key in keys) or 'pv' not in tokens:
        raise ValueError('malformed scored grammar')
    pi=tokens.index('pv'); header=tokens[:pi]
    def integer(key, default=None):
        if key not in header:
            if default is not None: return default
            raise ValueError('missing score telemetry '+key)
        try: return int(header[header.index(key)+1])
        except (ValueError,IndexError): raise ValueError('malformed score integer '+key) from None
    try:
        si=header.index('score'); kind=header[si+1]; score=int(header[si+2])
    except (ValueError,IndexError): raise ValueError('malformed score grammar') from None
    if kind not in ('cp','mate'): raise ValueError('unknown score kind')
    flags=[x for x in header if x in ('lowerbound','upperbound')]
    if len(flags)>1: raise ValueError('conflicting or repeated bound flags')
    depth,rank,nodes=integer('depth'),integer('multipv'),integer('nodes')
    seldepth,hits=integer('seldepth'),integer('tbhits')
    if min(depth,rank,nodes)<1 or min(seldepth,hits)<0 or not tokens[pi+1:]:
        raise ValueError('invalid score telemetry or empty PV')
    return PVLine(depth,seldepth,rank,kind,score,flags[0] if flags else None,nodes,tuple(tokens[pi+1:]),hits)


def _select_packet(lines, board, *, all_legal, min_depth):
    """Return (rows, provenance, rejection). Line offsets are zero-based inputs."""
    import chess
    legal={m.uci() for m in board.legal_moves}; wanted=len(legal) if all_legal else 1
    emissions=[]; pending=[]; ordinal=0; bestmove=None; decisive=False
    def finalize():
        if not pending: return None
        rows=[row for _,row in pending]
        meta=dict(ordinal=ordinal,line_start=pending[0][0],line_end=pending[-1][0],
                  ranks=[r.multipv for r in rows],depths=[r.depth for r in rows],
                  nodes=[r.nodes for r in rows],classification='incomplete')
        emissions.append(meta)
        # A clean prefix is incomplete; duplicate/gap/excess cannot become one.
        if [r.multipv for r in rows]!=list(range(1,len(rows)+1)) or len(rows)>wanted:
            return 'malformed_emission_ranks'
        for row in rows:
            copy=board.copy(stack=True)
            try:
                for uci in row.pv:
                    move=chess.Move.from_uci(uci)
                    if move not in copy.legal_moves: return 'illegal_packet_PV'
                    copy.push(move)
            except ValueError: return 'illegal_packet_PV'
        if len(rows)!=wanted: return None
        first=[r.pv[0] for r in rows]
        if len(set(first))!=wanted or any(m not in legal for m in first) or (all_legal and set(first)!=legal):
            return 'nonunique_or_incomplete_legal_actions'
        if len(set(r.nodes for r in rows))!=1: return 'inconsistent_emission_nodes'
        if len(set(r.depth for r in rows))!=1:
            meta['classification']='mixed_depth'; return None
        if any(r.score_kind!='cp' for r in rows): meta['classification']='decisive'; return None
        if any(rows[i].score<rows[i+1].score for i in range(len(rows)-1)):
            return 'nonmonotonic_common_rank_scores'
        if any(r.bound is not None for r in rows): meta['classification']='bounded'; return None
        if rows[0].depth<min_depth: meta['classification']='below_depth_floor'; return None
        meta['classification']='eligible'; meta['common_depth']=rows[0].depth
        meta['_rows']=rows
        return None
    for offset,line in enumerate(lines):
        tokens=line.split()
        if tokens and tokens[0]=='bestmove':
            if bestmove is not None or len(tokens) not in (2,4) or (len(tokens)==4 and tokens[2]!='ponder'):
                return None,{},'malformed_final_bestmove'
            bestmove=tokens[1]
            if bestmove not in legal: return None,{},'illegal_final_bestmove'
            continue
        try: row=parse_pv(line)
        except ValueError: return None,{},'malformed_scored_grammar'
        if row is None: continue
        if bestmove is not None: return None,{},'scored_output_after_bestmove'
        if row.score_kind=='mate' or row.tablebase_hits or abs(row.score)>=20000-246: decisive=True
        if row.multipv==1:
            error=finalize()
            if error: return None,{},error
            pending=[]; ordinal+=1
        elif not pending: return None,{},'rank_without_emission_start'
        pending.append((offset,row))
    error=finalize()
    if error: return None,{},error
    provenance=dict(protocol_version=2,selection=POLICY['selection'],emissions=emissions,
                    last_emitted_packet_classification=emissions[-1]['classification'] if emissions else 'none',
                    final_bestmove=bestmove)
    eligible=[e for e in emissions if e['classification']=='eligible']
    if decisive:
        for e in emissions: e.pop('_rows',None)
        return None,provenance,'decisive_mate_tablebase_or_extreme_evidence'
    if not eligible: return None,provenance,'no_eligible_ordinary_complete_packet'
    selected=max(eligible,key=lambda e:(e['common_depth'],e['ordinal']))
    rows=selected['_rows']
    provenance.update(selected_emission_ordinal=selected['ordinal'],selected_line_start=selected['line_start'],
                      selected_line_end=selected['line_end'],selected_common_depth=selected['common_depth'],
                      final_bestmove_differs_from_selected_PV1=(bestmove!=rows[0].pv[0]) if bestmove is not None else None)
    for e in emissions: e.pop('_rows',None)
    return rows,provenance,None


def select_packet(lines, board, *, all_legal, min_depth):
    """Strict selection plus full-transcript ordinals even for rejected queries."""
    if type(all_legal) is not bool or type(min_depth) is not int or min_depth<1:
        raise ValueError('explicit query kind and positive depth floor required')
    lines=list(lines)
    rows,provenance,error=_select_packet(lines,board,all_legal=all_legal,min_depth=min_depth)
    # Early fatal checks must not erase later emission/final-choice provenance.
    # No scalar score values are copied into this structural trace.
    wanted=board.legal_moves.count() if all_legal else 1
    trace=[]; pending=[]; ordinal=0; bestmoves=[]; malformed=[]
    def close_trace():
        if not pending:return
        parsed=[r for _,r in pending]
        ranks=[r.multipv for r in parsed]; depths=[r.depth for r in parsed];nodes=[r.nodes for r in parsed]
        kind=('decisive' if any(r.score_kind=='mate' or r.tablebase_hits or abs(r.score)>=20000-246 for r in parsed)
              else 'malformed_ranks' if ranks!=list(range(1,len(ranks)+1)) or len(ranks)>wanted
              else 'incomplete' if len(ranks)!=wanted
              else 'inconsistent_nodes' if len(set(nodes))!=1
              else 'mixed_depth' if len(set(depths))!=1
              else 'bounded' if any(r.bound for r in parsed)
              else 'below_depth_floor' if depths[0]<min_depth
              else 'ordinary_complete_structure')
        trace.append(dict(ordinal=ordinal,line_start=pending[0][0],line_end=pending[-1][0],
                          ranks=ranks,depths=depths,nodes=nodes,classification=kind))
    for offset,line in enumerate(lines):
        tokens=line.split()
        if tokens and tokens[0]=='bestmove':
            bestmoves.append(dict(line_offset=offset,tokens=tokens[1:]));continue
        try:parsed=parse_pv(line)
        except ValueError:
            malformed.append(offset);continue
        if parsed is None:continue
        if parsed.multipv==1:
            close_trace();pending=[];ordinal+=1
        pending.append((offset,parsed))
    close_trace()
    provenance.update(protocol_version=2,selection=POLICY['selection'],emission_trace=trace,
                      malformed_scored_line_offsets=malformed,final_bestmove_emissions=bestmoves,
                      query_rejection=error)
    if not provenance.get('emissions'):
        provenance['last_emitted_packet_classification']=trace[-1]['classification'] if trace else 'none'
    if 'final_bestmove' not in provenance:
        provenance['final_bestmove']=bestmoves[-1]['tokens'][0] if bestmoves and bestmoves[-1]['tokens'] else None
    return rows,provenance,error
