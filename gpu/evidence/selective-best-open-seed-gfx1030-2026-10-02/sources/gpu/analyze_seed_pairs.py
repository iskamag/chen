#!/usr/bin/env python3
"""CPU-only reconstruction of paired observations and exploratory intervals."""
import argparse
import json
import math
from pathlib import Path
import random
import statistics
from selective_cache_probe import parse_rows
from selective_fusion_probe import normalized, sha

BOOTSTRAP_SEED = 20261002
BOOTSTRAP_RESAMPLES = 10000


def quantile(values, p):
    values = sorted(values)
    at = (len(values)-1)*p
    lo = int(at)
    return values[lo] + (values[min(lo+1,len(values)-1)]-values[lo])*(at-lo)


def analyze(root):
    declaration = json.loads((root/'declaration.json').read_text())
    original = json.loads((root/'result.json').read_text())
    assert original['status']=='passed' and len(original['measured_pairs'])==12
    assert declaration['pairs']==12 and len(declaration['pair_order'])==12
    reference = None
    retired = 0
    for arm in declaration['warmup_order']:
        stem = 'warmup-' + arm
        transport = json.loads((root/(stem+'.json')).read_text())
        assert transport['exit_code']==0 and transport['process_group_absent'] and transport['owned_leader_retired']
        assert transport['requested_library']==declaration['artifacts'][arm]['path']
        actual = [normalized(row) for row in parse_rows((root/(stem+'.stdout')).read_text(),6,declaration['workload'])]
        if reference is None:
            reference = actual
        assert actual==reference
        retired += 1
    pairs = []
    checked_rows = 0
    for index, order in enumerate(declaration['pair_order'],1):
        rows = {}
        for arm in order:
            transport = json.loads((root/('pair-'+str(index)+'-'+arm+'.json')).read_text())
            assert transport['exit_code']==0 and transport['process_group_absent'] and transport['owned_leader_retired']
            assert transport['requested_library']==declaration['artifacts'][arm]['path']
            retired += 1
            rows[arm] = parse_rows((root/('pair-'+str(index)+'-'+arm+'.stdout')).read_text(),6,declaration['workload'])
            actual = [normalized(row) for row in rows[arm]]
            if reference is None:
                reference = actual
            assert actual==reference
            checked_rows += len(actual)
        positions = []
        for position in range(6):
            values = {arm:rows[arm][position]['elapsed_ns']/1e6 for arm in ('baseline','candidate')}
            positions.append(dict(position=position, fen=rows['baseline'][position]['fen'], search_ms=values,
                baseline_over_candidate_ratio=values['baseline']/values['candidate']))
        totals = {arm:sum(row['elapsed_ns'] for row in rows[arm])/1e6 for arm in ('baseline','candidate')}
        pairs.append(dict(pair=index, order=order, positions=positions, search_sum_ms=totals,
            baseline_over_candidate_ratio=totals['baseline']/totals['candidate']))
    assert pairs==original['measured_pairs']
    # Preserve the six AB/six BA design while resampling whole paired runs.
    strata = [[i for i,pair in enumerate(pairs) if pair['order'][0]==arm] for arm in ('baseline','candidate')]
    assert [len(group) for group in strata]==[6,6]
    rng = random.Random(BOOTSTRAP_SEED)
    samples = [[rng.choice(group) for group in strata for _ in group] for _ in range(BOOTSTRAP_RESAMPLES)]
    def summary(logratios):
        estimates = [math.exp(statistics.mean(logratios[i] for i in sample)) for sample in samples]
        ratio = math.exp(statistics.mean(logratios))
        interval = [quantile(estimates,.025),quantile(estimates,.975)]
        return dict(pairs=12, geometric_mean_paired_baseline_over_candidate_ratio=ratio,
            median_paired_ratio=statistics.median(math.exp(x) for x in logratios),
            observed_reduction_percent=100*(1-1/ratio),
            exploratory_stratified_bootstrap_ratio_interval95=interval,
            exploratory_reduction_percent_interval95=[100*(1-1/value) for value in interval])
    total = summary([math.log(pair['baseline_over_candidate_ratio']) for pair in pairs])
    per_position = [dict(position=i,fen=pairs[0]['positions'][i]['fen'],
        **summary([math.log(pair['positions'][i]['baseline_over_candidate_ratio']) for pair in pairs])) for i in range(6)]
    return dict(status='passed', kind='seed_balanced_pair_confirmation_reconstructed',
        analyzer_sha256=sha(__file__), declaration_sha256=sha(root/'declaration.json'), exact_work_rows=checked_rows,
        warmup_rows_checked=12, native_processes_retired=retired,
        measured_pairs=pairs, total=total, per_position=per_position,
        bootstrap=dict(seed=BOOTSTRAP_SEED,resamples=BOOTSTRAP_RESAMPLES,
            method='Percentile bootstrap of mean log paired ratios, resampling whole pairs within fixed AB/BA strata'),
        limits='Exploratory intervals describe this small machine-local process sample under a resampling assumption; time dependence, power/cache variation, repeatability on other days/devices and selection effects are uncontrolled. No hypothesis-test, population-confidence, global speedup, strength or rating claim.')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('root',type=Path)
    p.add_argument('--output',type=Path,required=True)
    args=p.parse_args()
    result=analyze(args.root)
    with args.output.open('x') as f:
        json.dump(result,f,indent=2,allow_nan=False);f.write('\n')
    print(json.dumps(result['total']))


if __name__=='__main__':
    main()
