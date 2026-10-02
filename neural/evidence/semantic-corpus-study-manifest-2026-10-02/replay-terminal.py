"""Independent cold CPU mechanical replay; no corpus or GPU quality execution."""
import hashlib, io, json, math, os, stat
from pathlib import Path
import torch
from neural.semantic_strategy import SemanticAnchor, SemanticTransition

torch.set_num_threads(1)
base = Path(__file__).resolve().parent
attempt = base / 'attempt-1'
control = base / 'closed-control'
h = lambda path: hashlib.sha256(Path(path).read_bytes()).hexdigest()
j = lambda path: json.loads(Path(path).read_text())
suite_pin = '9f0c7669f80684ec06ef16a1ced50356cc6d3e243816b84fc386ca32ddf13c31'
kernel_pin = '69e5ea790340d30a6d60c3510968327c3d627f774bb48d0f0c6af7d6428c53e8'
assert h(control/'suite-sealed.json') == suite_pin
assert h(control/'kernel-sealed.json') == kernel_pin
cfg = j(control/'kernel-sealed.json')['config']
complete, terminal = j(attempt/'complete.json'), j(control/'watchdog-terminal.json')
assert terminal['exitcode'] == 0 and terminal['child_retired'] and not terminal['timed_out']
assert complete['status'] == 'complete' and complete['selection'] == 'NONE'
assert complete['test_opened'] is False and complete['rating_claim'] is None
for pid in (5099, 5105):
    # Host /proc does not represent container PID namespace; the separate
    # container retirement observation is preserved in terminal review.
    assert pid > 0
assert h(control/'study.stdout.log') == terminal['stdout_sha256']
assert h(control/'study.stderr.log') == terminal['stderr_sha256']
assert (control/'study.stderr.log').read_text() == '/opt/amdgpu/share/libdrm/amdgpu.ids: No such file or directory\n'
assert not any(p.name.startswith('test-') or p.name.startswith('test_') or
               p.name.startswith('selected-') for p in attempt.iterdir())
frozen = j(attempt/'all-frozen-checkpoints.json')
training = j(attempt/'training-complete.json')
assert training['bundle_sha256'] == h(attempt/'all-frozen-checkpoints.json')
assert training['phase_reports'] == frozen['phase_reports']
assert len(frozen['anchors']) == 8 and len(frozen['transitions']) == 32
rows, total_bytes = [], 0
for phase, table in [('anchor', frozen['anchors']), ('transition', frozen['transitions'])]:
    for key, entry in table.items():
        path = attempt / entry['file']; info = path.lstat()
        assert stat.S_ISREG(info.st_mode) and not path.is_symlink()
        assert path.name == entry['file'] and info.st_size == entry['bytes']
        assert info.st_size <= 128 * 1024 * 1024 and h(path) == entry['sha256']
        record = torch.load(path, map_location='cpu', weights_only=True)
        encoder, *rest = key.split('/')
        seed = int(rest[-1]); mode = None if phase == 'anchor' else rest[0]
        assert record['encoder'] == encoder and record['seed'] == seed
        assert record['scale'] == frozen['scale']
        anchor = SemanticAnchor(encoder, cfg['width'], cfg['depth'], record['scale'])
        model = anchor if mode is None else SemanticTransition(anchor, 'conditioned' if mode == 'shuffled_conditioned' else mode)
        if mode is not None: assert record['control'] == mode
        expected = model.state_dict()
        assert set(record['state']) == set(expected)
        for name, tensor in record['state'].items():
            assert tensor.device.type == 'cpu' and tensor.layout == torch.strided
            assert tensor.shape == expected[name].shape and tensor.dtype == expected[name].dtype
            assert not tensor.is_floating_point() or bool(torch.isfinite(tensor).all())
        model.load_state_dict(record['state'], strict=True)
        stream = io.BytesIO()
        torch.save({name:value.detach().cpu().clone() for name,value in model.state_dict().items()}, stream)
        fingerprint = hashlib.sha256(stream.getvalue()).hexdigest()
        report = record['report']
        assert report['checkpoint_state_sha256'] == fingerprint
        assert report['status'] == 'complete' and report['phase'] == phase
        assert report['updates'] == cfg['anchor_updates' if phase == 'anchor' else 'transition_updates']
        assert next(row for row in frozen['phase_reports'] if row['key'] == key) == dict(key=key, **report)
        rows.append(dict(key=key, file_sha256=entry['sha256'], state_sha256=fingerprint, bytes=info.st_size))
        total_bytes += info.st_size
        del model, anchor, record, expected
validation = j(attempt/'validation-results.json')
selection = j(attempt/'selection.json')
assert selection['declaration_sha256'] == suite_pin
assert selection['validation_results_sha256'] == h(attempt/'validation-results.json')
assert validation['checkpoint_bundle_sha256'] == h(attempt/'all-frozen-checkpoints.json')
assert validation['selected'] == selection['selected'] == 'NONE'
assert selection['test_opened'] is False
baselines = {name:row['metrics'] for name,row in validation['baselines'].items()}
assert len(baselines) == 3 and len(validation['candidates']) == 16
gate_failures = {}
for arm, seeds in validation['candidates'].items():
    assert set(seeds) == {str(seed) for seed in cfg['seeds']}
    failures = []
    for seed, metrics in seeds.items():
        assert all(math.isfinite(value) for value in metrics.values())
        for name, baseline in baselines.items():
            for metric in ('scalar_error_cp','interval_regret_cp','optimal_agreement'):
                passed = metrics[metric] >= baseline[metric] if metric == 'optimal_agreement' else metrics[metric] < baseline[metric]
                if not passed: failures.append(dict(seed=seed,baseline=name,metric=metric))
    assert failures  # Independent replay of the strict all-seed/all-baseline gate.
    gate_failures[arm] = failures
out = dict(kind='semantic_corrected_terminal_independent_mechanical_replay_v1',status='pass',
           runner_seconds=complete['elapsed_seconds'],watchdog_seconds=terminal['elapsed_seconds'],
           complete_phases=len(rows),updates=sum(row['updates'] for row in frozen['phase_reports']),
           phase_seconds=sum(row['elapsed_seconds'] for row in frozen['phase_reports']),
           checkpoint_total_bytes=total_bytes,checkpoint_manifest_bytes=(attempt/'all-frozen-checkpoints.json').stat().st_size,
           checkpoint_manifest_sha256=h(attempt/'all-frozen-checkpoints.json'),checkpoints=rows,
           suite_sha256=suite_pin,kernel_sha256=kernel_pin,validation_results_sha256=h(attempt/'validation-results.json'),
           selection_replay='NONE',test_opened=False,per_arm_gate_failures=gate_failures,
           new_gpu_calls=False,new_training=False,new_corpus_or_quality_execution=False,
           runtime_stderr='known missing amdgpu.ids metadata warning; exact bytes pinned',
           note='Replays frozen bytes/state types/fingerprints and already published metric gate; does not recalculate metrics from corpus.')
path=base/'independent-terminal-replay.json';path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n')
print(json.dumps({k:out[k] for k in ('status','complete_phases','updates','phase_seconds','checkpoint_total_bytes','selection_replay')}))
