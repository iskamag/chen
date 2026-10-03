"""Validate and summarize the bounded three-arm raw pipeline observations."""
import argparse
import json
import statistics

ARMS = ('abi1_E6144', 'api2_E6144', 'api2_buckets')


def analyze(result):
    assert result['status'] == 'PASS'
    assert {(x['width'], x['batch']) for x in result['rows']} == {
        (width, batch) for width in (32, 128, 256) for batch in (1, 20)}
    assert len(result['rows']) == 6
    rows, checkpoints = [], {}
    for row in result['rows']:
        width, batch = row['width'], row['batch']
        assert row['steps'] == 4 and row['dtype'] == 'float32'
        assert checkpoints.setdefault(width, row['checkpoint_sha256']) == row['checkpoint_sha256']
        assert len(row['samples']) == 6
        assert len(row['active_counts_cold_diagnostic']) == batch
        for index, sample in enumerate(row['samples']):
            assert sample['pair'] == index
            assert tuple(sample['order']) == (ARMS if index % 2 == 0 else ARMS[::-1])
            assert set(sample['arms']) == set(ARMS)
            for name, values in sample['arms'].items():
                assert values['edge_slots'] == (512 if name == 'api2_buckets' else 6144)
                assert values['retries'] == 0
                assert values['synchronized_pipeline_wall_ms'] > 0
                assert values['synchronized_export_wall_ms'] > 0
        median = lambda name, field: statistics.median(
            sample['arms'][name][field] for sample in row['samples'])
        ratios = [sample['arms']['abi1_E6144']['synchronized_pipeline_wall_ms'] /
                  sample['arms']['api2_buckets']['synchronized_pipeline_wall_ms'] for sample in row['samples']]
        assert ratios == row['paired_pipeline_control_over_bucket_ratios']
        for name in ARMS:
            assert row['setup'][name]['first_export_manifest']['edge_slots'] == (512 if name == 'api2_buckets' else 6144)
        rows.append(dict(width=width, batch=batch, pairs=6, bucket=512,
            checkpoint_sha256=row['checkpoint_sha256'], active_edges_total=sum(
                counts[2] for counts in row['active_counts_cold_diagnostic']),
            active_edges_max=max(counts[2] for counts in row['active_counts_cold_diagnostic']),
            pipeline_wall_median_ms={name:median(name, 'synchronized_pipeline_wall_ms') for name in ARMS},
            export_wall_median_ms={name:median(name, 'synchronized_export_wall_ms') for name in ARMS},
            native_event_median_ms={name:median(name, 'native_event_ms') for name in ARMS},
            inference_event_median_ms={name:median(name, 'captured_inference_event_ms') for name in ARMS},
            paired_control_over_bucket_ratios=ratios,
            median_paired_control_over_bucket_ratio=statistics.median(ratios),
            observed_paired_ratio_range=[min(ratios), max(ratios)],
            setup=row['setup'], model_setup_wall_ms=row['model_setup_wall_ms']))
    return dict(status='PASS', rows=rows, shapes=6, pipeline_observations=108,
        cold_prediction_checks='eager/captured/fresh-pipeline assertions passed during actual run',
        inference_only_ratios_used_as_pipeline=False,
        scope='Paired observations of one untrained checkpoint per width on one opening child set. '
              'No confidence interval, universal ratio, search throughput or strength inference.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('result')
    args = parser.parse_args()
    with open(args.result) as source:
        print(json.dumps(analyze(json.load(source)), indent=2))
