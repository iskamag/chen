"""CPU-only reconstruction of a frozen whole-model INT8 diagnostic."""
import argparse
import json
import math
from pathlib import Path
import statistics

VARIANTS = ('fp32_full', 'fp32_value', 'int8_multi', 'int8_fused', 'int8_hybrid')


def analyze(root):
    complete = json.loads((root / 'attempt-1/complete.json').read_text())
    declaration = json.loads((root / 'control/declaration.json').read_text())
    terminal = json.loads((root / 'control/watchdog-terminal.json').read_text())
    assert complete['status'] == 'complete' and complete['split'] == 'train'
    assert terminal['status'] == 'terminal' and terminal['exitcode'] == 0
    assert terminal['child_retired'] and not terminal['timed_out'] and terminal['no_retry']
    assert complete['elapsed_seconds'] < declaration['config']['wall_seconds']
    assert not any(complete[name] for name in ('calibration', 'training', 'promotion', 'engine_latency_claim', 'graph_capture', 'teacher_quality_metrics'))
    records = complete['records']
    assert len(records) == 40
    groups = {}
    for key, row in records.items():
        phase = row['phase']
        assert phase in ('anchor', 'transition')
        filename = phase + '-' + key.replace('/', '-') + '.json'
        assert json.loads((root / 'attempt-1' / filename).read_text()) == row
        count = 1864 if phase == 'anchor' else 8477
        reference = row['fidelity']['fp32_value']
        assert reference['cp']['count'] == count and reference['latent']['count'] == count * 256
        assert all(reference[k]['maximum_absolute'] == reference[k]['mean_absolute'] == reference[k]['mean_squared'] == 0 for k in ('cp', 'latent'))
        if phase == 'transition':
            assert reference['roots'] == 233 and reference['first_canonical_action_agreement'] == 1
        encoder = key.split('/')[0]
        for variant, fidelity in row['fidelity'].items():
            assert variant in VARIANTS[1:] and fidelity['cp']['count'] == count
            for quantity in ('cp', 'latent'):
                assert all(math.isfinite(fidelity[quantity][name]) and fidelity[quantity][name] >= 0
                           for name in ('maximum_absolute', 'mean_absolute', 'mean_squared'))
            if phase == 'transition':
                assert fidelity['roots'] == 233
                changed = round((1 - fidelity['first_canonical_action_agreement']) * 233)
                assert changed == fidelity['changed_roots']
        for batch, timings in row['timings'].items():
            assert set(timings) == set(VARIANTS)
            for variant, timing in timings.items():
                assert len(timing['samples']) == declaration['config']['timing_rounds']
                for field in ('event_us', 'synchronized_wall_us'):
                    values = [s[field] for s in timing['samples']]
                    assert all(math.isfinite(v) and v > 0 for v in values)
                    assert timing[field + '_median'] == statistics.median(values)
                assert all(s['iterations'] == declaration['config']['timing_iterations'] for s in timing['samples'])
            name = phase + '/' + encoder + '/' + batch
            groups.setdefault(name, []).append(timings)
    timing_summary = {}
    for name, rows in sorted(groups.items()):
        timing_summary[name] = dict(models=len(rows), synchronized_wall_us_model_medians={
            variant: statistics.median(r[variant]['synchronized_wall_us_median'] for r in rows)
            for variant in VARIANTS}, geometric_paired_fp32_value_over_variant={
            variant: math.exp(statistics.mean(math.log(r['fp32_value']['synchronized_wall_us_median'] /
                           r[variant]['synchronized_wall_us_median']) for r in rows)) for variant in VARIANTS})
    fidelity_summary = {}
    for phase in ('anchor', 'transition'):
        selected = [r for r in records.values() if r['phase'] == phase]
        assert len(selected) == (8 if phase == 'anchor' else 32)
        fidelity_summary[phase] = {}
        for variant in VARIANTS[1:]:
            values = [r['fidelity'][variant] for r in selected]
            fidelity_summary[phase][variant] = dict(models=len(values),
                cp_mean_absolute_range=[min(v['cp']['mean_absolute'] for v in values), max(v['cp']['mean_absolute'] for v in values)],
                cp_maximum_absolute=max(v['cp']['maximum_absolute'] for v in values),
                changed_model_root_observations=sum(v['changed_roots'] for v in values),
                total_model_root_observations=sum(v['roots'] for v in values),
                changed_maximum_fp32_top_two_gap_cp=max((v['changed_fp32_best_second_gap_cp']['maximum'] or 0) for v in values))
    return dict(status='passed', models=40, elapsed_seconds=complete['elapsed_seconds'], fidelity=fidelity_summary,
                timings=timing_summary, scope='Same-checkpoint TRAIN fidelity and eager timing; model-root observations are repeated measurements of the same233roots, not independent chess games. No graph, teacher quality, deployment, strength or rating inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = analyze(args.root)
    with args.output.open('x') as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write('\n')
    print(json.dumps(result['fidelity'], sort_keys=True))
