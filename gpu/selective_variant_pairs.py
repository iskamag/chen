#!/usr/bin/env python3
"""Separately declared balanced fresh-process timing pairs; no correctness reruns."""
import argparse
import os
from pathlib import Path

from selective_cache_probe import parse_rows
from selective_fusion_probe import normalized, run, sha, write_json

PAIR_ORDER = (('baseline', 'candidate'), ('candidate', 'baseline'), ('candidate', 'baseline'),
              ('baseline', 'candidate'), ('baseline', 'candidate'), ('candidate', 'baseline')) * 2


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('engine', 'baseline', 'candidate', 'suite', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    for name in ('engine', 'baseline', 'candidate'):
        p.add_argument('--' + name + '-sha256', required=True)
    p.add_argument('--cpu', type=int, default=5)
    p.add_argument('--variant', required=True)
    p.add_argument('--depth', type=int, default=64)
    p.add_argument('--nodes', type=int, default=50000)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--capacity', type=int, default=250000)
    p.add_argument('--pairs', type=int, choices=(6, 12), default=12)
    args = p.parse_args()
    if (not 1 <= args.depth <= 64 or not 1 <= args.nodes <= 250000 or
            not 1 <= args.batch <= 256 or not 1 <= args.capacity <= 500000 or
            not args.variant or not args.variant.replace('_', '').isalnum()):
        p.error('invalid bounded workload or named variant')
    workload = dict(name=args.variant, depth=args.depth, nodes=args.nodes,
                    batch=args.batch, capacity=args.capacity)
    pair_order = PAIR_ORDER[:args.pairs]
    paths = {name: getattr(args, name).resolve(strict=True) for name in ('engine', 'baseline', 'candidate', 'suite')}
    for name in ('engine', 'baseline', 'candidate'):
        if sha(paths[name]) != getattr(args, name + '_sha256'):
            p.error(name + ' fingerprint mismatch')
    if sha(paths['baseline']) == sha(paths['candidate']):
        p.error('identical artifacts')
    if args.cpu not in os.sched_getaffinity(0):
        p.error('CPU not permitted')
    for key in ('HSA_OVERRIDE_GFX_VERSION', 'CHEN_NEURAL_MODEL', 'CHEN_NEURAL_LIBRARY', 'LD_PRELOAD'):
        if os.environ.get(key):
            p.error('unexpected inherited ' + key)
    count = sum(bool(line.strip()) and not line.lstrip().startswith('#') for line in paths['suite'].read_text().splitlines())
    if count != 6:
        p.error('this declaration requires original six-position suite')
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(kind='classical_selective_variant_balanced_pair_comparison', variant=args.variant,
        source_sha256=sha(__file__), helper_sha256=sha(Path(__file__).with_name('selective_fusion_probe.py')),
        parser_sha256=sha(Path(__file__).with_name('selective_cache_probe.py')), workload=workload,
        artifacts={name:dict(path=str(path), sha256=sha(path)) for name, path in paths.items()},
        cpu_affinity=[args.cpu], warmup_order=['baseline','candidate'], pair_order=pair_order,
        pairs=args.pairs, rows_per_run=count, timeout_per_child_seconds=30, correctness_reruns=False,
        concurrency='Exclusive GPU after coherence diagnostic; CPU5; shared cache/power uncontrolled',
        scope='Machine-local process timing confirmation, exact non-timing work checked in every run; no strength or significance claim')
    write_json(args.output/'declaration.json', declaration)
    reference = None
    records = []
    pairs = []
    try:
        def search(arm, stem):
            nonlocal reference
            for name, path in paths.items():
                if sha(path) != declaration['artifacts'][name]['sha256']:
                    raise ValueError('changed artifact ' + name)
            argv = ['taskset','-c',str(args.cpu),str(paths['engine']),'bench','--backend','hip-selective',
                '--eval','tapered','--depth',str(workload['depth']),'--nodes',str(workload['nodes']),'--selective-batch',str(workload['batch']),
                '--max-positions',str(workload['capacity']),'--suite',str(paths['suite'])]
            stdout, rec = run(argv, dict(os.environ, CHEN_GPU_LIBRARY=str(paths[arm])), args.output/stem, 30)
            rows = parse_rows(stdout, count, workload)
            comparable = [normalized(row) for row in rows]
            if reference is None:
                reference = comparable
            if comparable != reference:
                raise ValueError('non-timing exact-work parity failed')
            records.append(dict(stem=stem, arm=arm, search_sum_ms=sum(row['elapsed_ns'] for row in rows)/1e6,
                wall_seconds=rec['wall_seconds'], exact_work_parity=True))
            return rows
        for arm in declaration['warmup_order']:
            search(arm,'warmup-'+arm)
        for index, order in enumerate(pair_order,1):
            rows = {arm:search(arm,'pair-'+str(index)+'-'+arm) for arm in order}
            positions = []
            for position in range(count):
                values = {arm:rows[arm][position]['elapsed_ns']/1e6 for arm in ('baseline','candidate')}
                positions.append(dict(position=position, fen=rows['baseline'][position]['fen'], search_ms=values,
                    baseline_over_candidate_ratio=values['baseline']/values['candidate']))
            totals = {arm:sum(row['elapsed_ns'] for row in rows[arm])/1e6 for arm in ('baseline','candidate')}
            pairs.append(dict(pair=index, order=order, positions=positions, search_sum_ms=totals,
                baseline_over_candidate_ratio=totals['baseline']/totals['candidate']))
            print('completed pair',index,'ratio',pairs[-1]['baseline_over_candidate_ratio'],flush=True)
        write_json(args.output/'result.json', dict(status='passed', declaration_sha256=sha(args.output/'declaration.json'),
            measured_pairs=pairs, warmups_excluded_from_pair_analysis=True, runs=records))
    except Exception as error:
        write_json(args.output/'result.json', dict(status='failed_or_partial', error=str(error), measured_pairs=pairs,
            runs=records, timing_conclusion_admitted=False))
        raise


if __name__ == '__main__':
    main()
