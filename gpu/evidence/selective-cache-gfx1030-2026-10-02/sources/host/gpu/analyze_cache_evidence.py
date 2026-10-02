#!/usr/bin/env python3
"""CPU-only raw cache evidence reconstruction; requires python-chess."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import statistics
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'scripts'))
from analyze_engine_profile import gpu
from selective_fusion_probe import normalized
from analyze_selective_sparse import audit_pv

def load(p): return json.loads(p.read_text())
def sha(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def main(root):
    declaration = load(root / 'exact-work/declaration.json')
    summaries = []
    for config in declaration['workloads']:
        comparable = None
        timing = {'baseline': [], 'candidate': []}
        rows_by_arm = {'baseline': [], 'candidate': []}
        for index, arm in enumerate(declaration['order'], 1):
            p = root / 'exact-work' / (config['name'] + '-' + str(index) + '-' + arm + '.stdout')
            rows = [json.loads(x) for x in p.read_text().splitlines() if x.startswith('{')]
            assert len(rows) == 6
            current = [normalized(x) for x in rows]
            if comparable is None: comparable = current
            assert comparable == current
            timing[arm].append(sum(x['elapsed_ns'] for x in rows) / 1e6)
            rows_by_arm[arm].append(rows)
        medians = {arm: statistics.median(values) for arm, values in timing.items()}
        positions = []
        for index in range(6):
            each = {arm: statistics.median(rows[index]['elapsed_ns'] for rows in values) / 1e6
                    for arm, values in rows_by_arm.items()}
            positions.append({'index': index+1, 'median_ms': each,
                              'reduction_percent': 100*(1-each['candidate']/each['baseline'])})
        summaries.append({'workload': config, 'median_six_search_ms': medians,
                          'baseline_over_candidate_ratio': medians['baseline']/medians['candidate'],
                          'reduction_percent':100*(1-medians['candidate']/medians['baseline']),
                          'positions': positions})
    native = load(root/'deep/result.json')
    assert native['status']=='passed' and len(native['cases'])==33
    native_steps = native_pvs = 0
    for case in native['cases']:
        assert case['exact_per_step_parity']
        for snapshot in case['snapshots']:
            assert snapshot['classified']<=case['absolute_limit'] and snapshot['arena']<=6000
            assert snapshot['maximum']<=case['max_ply']
            if 'pv' in snapshot:
                audit_pv(case['fen'],snapshot['pv']);native_pvs+=1
            native_steps+=1
        if case['repeated_root_history']:
            last=case['snapshots'][-1]
            assert last['root_flags']&2 and last['score']==0 and last['stop']==1
    fixture=[json.loads(x) for x in (root/'trace-fixture.stdout').read_text().splitlines()]
    assert len(fixture)==48 and all(x['status']=='passed' for x in fixture)
    for x in fixture:
        assert len(x['selected_indices'])==len(x['offsets'])==x['selected']
        assert x['selected']<=x['batch'] and x['examined']<=x['candidate_limit']
        assert x['added']<=min(x['remaining'],x['extra_capacity'])
    assert any(x['depth']==64 and x['candidate_limit']==1024 and x['examined']==1024 and x['selected']==0 for x in fixture)
    pv_count=run_count=0
    observations={}
    for mode in ['uci','trace']:
        record=load(root/mode/'result.json');assert record['status']=='passed'
        observations[mode]=[]
        for r in record['runs']:
            assert r['exit_code']==0 and r['process_group_absent'] and r['movetime_ms']==1000
            audit_pv(r['fen'],[r['bestmove']]);run_count+=1
            observations[mode].append({k:r[k] for k in ['arm','name','last_info','search_response_wall_seconds']})
        for stdout in (root/mode).rglob('1-*.stdout'):
            name=stdout.stem.removeprefix('1-');fen=next(r['fen'] for r in record['runs'] if r['name']==name)
            for line in stdout.read_text().splitlines():
                if line.startswith('info depth ') and ' pv ' in line:
                    audit_pv(fen,line.split(' pv ',1)[1].split());pv_count+=1
    assert pv_count==sum(load(root/mode/'result.json')['legal_reported_pvs'] for mode in ['uci','trace'])
    traces={arm:gpu(root/'trace'/arm) for arm in ['baseline','candidate']}
    for xs in traces.values():
        for x in xs:
            assert x['stream_fence_calls']==3*x['published_steps']+3 and x['worker_allocation_calls']==0
    matched={}
    for name in ['opening','kiwipete','rook-ending']:
        infos=[]
        for arm in ['baseline','candidate']:
            text=(root/'trace'/arm/'gpu-trace'/('1-'+name+'.stdout')).read_text()
            infos.append([re.sub(r' nps \d+| time \d+','',x) for x in text.splitlines() if x.startswith('info depth ')])
        matched[name]=infos[0]==infos[1]
    assert matched['opening'] and matched['kiwipete']
    isa=load(root/'isa/extraction-report.json')
    for x in isa['records']: assert sha(root/'isa'/x['archived_elf'])==x['sha256']
    return {'status':'passed','variant':declaration['variant'],'artifacts':declaration['artifacts'],
            'exact_work_rows':72,'equal_work':summaries,'native_cases':33,'native_per_step_snapshots':native_steps,
            'native_legal_pvs':native_pvs,'exact_selector_cases':fixture,'uci_workloads':run_count,
            'uci_legal_reported_pvs':pv_count,'one_second_observations':observations,'traces':traces,
            'profile_published_trajectory_exact_match':matched,'isa':isa,
            'limits':'Machine-local bounded authored suites, no significance/strength/rating/neural/fault-recovery claim. Trace instrumentation is separate from clean timings. Rook trace arms perform unequal work. API waits overlap kernels. Static ISA/resources are not dynamic instruction count or occupancy.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    result=main(a.root)
    with a.output.open('x') as f: json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print('Reconstructed',result['exact_work_rows'],'rows,',result['native_per_step_snapshots'],'native steps,',result['uci_legal_reported_pvs'],'legal UCI PVs')
