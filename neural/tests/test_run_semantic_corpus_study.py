"""Synthetic orchestration checks. No actual corpus admission or training."""
import copy
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from neural import run_semantic_corpus_study as runner
from neural import semantic_corpus_study as study


class RunnerTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_json_duplicate_nonfinite_pins_and_exclusive_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'record.json'
            for raw in ('{"x":1,"x":2}', '{"x":1e309}', '{"x":NaN}'):
                path.write_text(raw)
                with self.assertRaises(ValueError): runner.read_json(path)
            path.unlink()
            pin = runner.publish_json(path, {'x':1})
            self.assertEqual(runner.read_json(path, pin), {'x':1})
            with self.assertRaises(ValueError): runner.read_json(path, '0'*64)
            with self.assertRaises(FileExistsError): runner.publish_json(path, {})

    def test_unequal_batch_aggregation_uses_root_and_observation_counts(self):
        row = runner.metric_row(18, 6, 21, 3, 7)
        self.assertEqual(row, dict(scalar_error_cp=3., interval_regret_cp=3., optimal_agreement=3/7))
        for args in ((1,0,1,1,1), (1,1,1,1,0), (float('nan'),1,1,1,1)):
            with self.assertRaises(ValueError): runner.metric_row(*args)

    def test_actual_sweep_masks_scalars_and_weights_unequal_root_batches(self):
        observation_batches = [SimpleNamespace(midpoint=torch.tensor([1.,1000.]),halfwidth=torch.zeros(2),
            ordinary=torch.tensor([True,False])), SimpleNamespace(midpoint=torch.tensor([5.]),halfwidth=torch.zeros(1),ordinary=torch.tensor([True]))]
        def roots(values):
            midpoint = torch.tensor([[0.,-value] for value in values])
            return SimpleNamespace(legal_actions=torch.tensor([[1,2]]*len(values)),
                targets=SimpleNamespace(midpoint=midpoint,halfwidth=torch.zeros_like(midpoint),
                    reported_cp=-midpoint,terminal=torch.zeros_like(midpoint,dtype=torch.bool)))
        root_batches = [roots([2.,4.]), roots([8.])]
        corpus=SimpleNamespace(prepare_structural_schedule=lambda:([0,1],[0,1]),
                observations=lambda plan:observation_batches[plan], roots=lambda plan:root_batches[plan])
        row=runner.evaluate(corpus,observer=lambda batch,plan:torch.zeros_like(batch.midpoint),
                            root_predictor=lambda batch:torch.zeros_like(batch.targets.midpoint),deadline=float('inf'))
        self.assertEqual(row['metrics']['scalar_error_cp'],3.)
        self.assertAlmostEqual(row['metrics']['interval_regret_cp'],14/3)
        self.assertEqual(row['support'],dict(ordinary_observations=2,roots=3,legal_candidates=6))
        with self.assertRaises(ValueError):
            runner.evaluate(corpus,observer=lambda batch,plan:torch.zeros_like(batch.midpoint),
                root_predictor=lambda batch:torch.full_like(batch.targets.midpoint,float('nan')),deadline=float('inf'))

    def test_checkpoint_has_owned_cpu_storage_and_exclusive_path(self):
        with tempfile.TemporaryDirectory() as directory:
            model = torch.nn.Linear(2, 1)
            state = runner.cpu_state(model)
            original = state['weight'].clone()
            model.weight.data.fill_(99)
            self.assertTrue(torch.equal(original, state['weight']))
            path = Path(directory) / 'checkpoint.pt'
            pin = runner.checkpoint(path, {'state':state})
            self.assertEqual(pin, runner.sha(path))
            loaded = torch.load(path, weights_only=True)
            self.assertTrue(torch.equal(loaded['state']['weight'], original))
            with self.assertRaises(FileExistsError): runner.checkpoint(path, {})

    def test_last_root_and_observation_work_cannot_cross_deadline(self):
        observation = SimpleNamespace(midpoint=torch.tensor([0.]), halfwidth=torch.zeros(1),
                                      ordinary=torch.tensor([True]))
        root = SimpleNamespace(legal_actions=torch.tensor([[1,2]]), targets=SimpleNamespace(
            midpoint=torch.tensor([[0.,-1.]]), halfwidth=torch.zeros(1,2),
            reported_cp=torch.tensor([[0,1]]), terminal=torch.zeros(1,2,dtype=torch.bool)))
        for final_stage in ('observation', 'root'):
            clock = [0.]
            corpus = SimpleNamespace(prepare_structural_schedule=lambda:([0] if final_stage == 'root' else [],[0]),
                observations=lambda plan:observation, roots=lambda plan:root)
            def observe(batch, plan):
                if final_stage == 'observation': clock[0] = 11.
                return torch.zeros_like(batch.midpoint)
            def predict(batch):
                clock[0] = 11.
                return torch.tensor([[1.,0.]])
            with patch.object(runner.time, 'monotonic', side_effect=lambda:clock[0]):
                with self.assertRaisesRegex(TimeoutError, 'completed evaluation'):
                    runner.evaluate(corpus, observer=observe, root_predictor=predict, deadline=10.)

    def test_kernel_and_suite_corpus_pins_must_match_before_proposal(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            packs = root / 'packs'
            for split in ('train', 'validation', 'test'):
                (packs / split).mkdir(parents=True)
                runner.publish_json(packs / split / 'manifest.json',
                                    {'counts':{}, 'source_sha256':'a'*64})
            baseline = root / 'baseline.json'
            baseline_pin = runner.publish_json(baseline, {'kind':'semantic_exact_baseline_witness_v1'})
            dependencies = {'/synthetic/frozen.py':'b'*64}
            kernel = {'dependencies':dependencies, 'config':{'total_wall_seconds':123},
                **{split + '_manifest_sha256':runner.sha(packs / split / 'manifest.json')
                   for split in ('train','validation')}}
            kernel_path = root / 'kernel.json'
            kernel_path.write_text(json.dumps(kernel))
            with patch.object(study, 'dependency_source_pins', return_value=dependencies.copy()):
                proposal = runner.suite_proposal(root, packs, kernel_path, baseline, baseline_pin, root/'attempt')
                self.assertTrue(proposal['wall_scope'].startswith('kernel_123_seconds'))
                for split in ('train', 'validation'):
                    bad = copy.deepcopy(kernel); bad[split + '_manifest_sha256'] = '0'*64
                    kernel_path.write_text(json.dumps(bad))
                    with self.assertRaisesRegex(ValueError, 'corpus identities differ'):
                        runner.suite_proposal(root, packs, kernel_path, baseline, baseline_pin, root/'attempt')

    def _orchestrate(self, directory, selected, fail_phase=None, corrupt_checkpoint=None):
        output = Path(directory) / 'attempt'
        sources = ('experiments/semantic_corpus/__init__.py', 'experiments/semantic_corpus/packing.py',
                   'experiments/semantic_corpus/resident.py')
        root = Path(directory) / 'main'
        declaration = dict(main_root=str(root), device='cpu', attempt_output=str(output),
            dependencies={str(root / name):'a'*64 for name in sources},
            baseline_witness={'path':'fixture', 'sha256':'a'*64},
            kernel_declaration={'path':'kernel', 'sha256':'a'*64},
            splits={name:dict(manifest_sha256='b'*64, packing_source_sha256='c'*64) for name in ('train','validation','test')})
        witness = dict(splits={name:{} for name in declaration['splits']})
        config = study.Configuration(width=8, anchor_updates=1, transition_updates=1)
        resident = SimpleNamespace(EvaluationGate=lambda *args: args)
        corpus = SimpleNamespace(device=torch.device('cpu'))
        calls = []
        original_read_json = runner.read_json
        original_checkpoint = runner.checkpoint
        def save_checkpoint(path, record):
            if corrupt_checkpoint and Path(path).name == f'transition-square-transport-{config.seeds[0]}.pt':
                record = copy.deepcopy(record)
                if corrupt_checkpoint == 'identity': record['seed'] = True
                else:
                    name = next(name for name,value in record['state'].items() if value.is_floating_point())
                    record['state'][name] = record['state'][name] + 1
            return original_checkpoint(path, record)
        def phase(model, corpus, **kwargs):
            calls.append(('phase', kwargs['phase'], kwargs.get('control')))
            if fail_phase is not None and len(calls) == fail_phase:
                raise TimeoutError('synthetic bounded phase failure')
            return dict(status='complete', phase=kwargs['phase'], updates=1,
                        checkpoint_state_sha256=study.fingerprint(model))
        def load(declaration, resident, split, output, gate=None):
            calls.append(('load', split))
            if split == 'validation':
                self.assertTrue((output/'training-complete.json').exists())
                bundle = runner.read_json(output/'all-frozen-checkpoints.json')
                self.assertEqual(len(bundle['anchors']), 8)
                self.assertEqual(len(bundle['transitions']), 32)
            if split == 'test':
                self.assertTrue((output/'selection.json').exists())
                self.assertTrue((output/'selected-paired-checkpoints.pt').exists())
                self.assertTrue((output/'test-pass-consumed.json').exists())
            return corpus
        baseline_row = dict(scalar_error_cp=10., interval_regret_cp=10., optimal_agreement=0.)
        candidate_row = dict(scalar_error_cp=1. if selected else 11., interval_regret_cp=1., optimal_agreement=1.)
        with patch.object(runner,'verify_suite',return_value=declaration), \
             patch.object(runner,'read_json',side_effect=lambda path,*a,**kw: witness if path == 'fixture' else original_read_json(path,*a,**kw)), \
             patch.object(runner,'validate_witness',return_value=witness), \
             patch.object(study,'admit_resident_dependency',return_value=resident), \
             patch.object(runner,'load_split',side_effect=load), \
             patch.object(runner,'admit_terminal_support'), \
             patch.object(runner,'checkpoint',side_effect=save_checkpoint), \
             patch.object(study,'require_gpu_learning_authority',return_value=config), \
             patch.object(study,'fit_train_material',return_value=1.), \
             patch.object(study,'train_phase',side_effect=phase), \
             patch.object(runner,'baseline_metrics',return_value={name:{'metrics':baseline_row} for name in ('raw_material100','scaled_material','raw_tapered')}), \
             patch.object(runner,'evaluate',return_value={'metrics':candidate_row,'diagnostics':{},'support':{}}) as evaluation:
            if fail_phase:
                with self.assertRaises(TimeoutError): runner.run('fixture','d'*64,output)
                self.assertFalse((output/'training-complete.json').exists())
                self.assertFalse((output/'validation-checkpoint-gate.json').exists())
                return calls, None
            if corrupt_checkpoint:
                with self.assertRaises(ValueError): runner.run('fixture','d'*64,output)
                self.assertFalse((output/'validation-checkpoint-gate.json').exists())
                self.assertNotIn(('load','validation'), calls)
                self.assertNotIn(('load','test'), calls)
                return calls, None
            selection = runner.run('fixture','d'*64,output)
            expected_eval = 34 if selected else 32
            self.assertEqual(evaluation.call_count, expected_eval)
            with self.assertRaises(FileExistsError): runner.run('fixture','d'*64,output)
        return calls, selection

    def test_complete_8_anchors_32_controls_then_none_keeps_test_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, selection = self._orchestrate(directory, False)
            self.assertEqual(sum(row[:2] == ('phase','anchor') for row in calls), 8)
            self.assertEqual(sum(row[:2] == ('phase','transition') for row in calls), 32)
            self.assertEqual(selection['selected'],'NONE')
            self.assertNotIn(('load','test'), calls)

    def test_selected_pair_is_frozen_before_one_test_open(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, selection = self._orchestrate(directory, True)
            self.assertNotEqual(selection['selected'],'NONE')
            self.assertEqual(calls.count(('load','test')),1)
            self.assertFalse(selection['test_opened']) # pre-test selection artifact
            complete = json.loads((Path(directory)/'attempt/complete.json').read_text())
            self.assertTrue(complete['test_opened'])

    def test_failed_phase_has_no_validation_or_test_promotion(self):
        with tempfile.TemporaryDirectory() as directory:
            calls, _ = self._orchestrate(directory,False,fail_phase=5)
            self.assertNotIn(('load','validation'),calls)
            self.assertNotIn(('load','test'),calls)

    def test_malformed_frozen_record_or_state_fails_before_validation_admission(self):
        for corruption in ('identity', 'fingerprint'):
            with tempfile.TemporaryDirectory() as directory:
                self._orchestrate(directory, False, corrupt_checkpoint=corruption)

    def test_native_witness_rejects_mates_missing_support_and_boolean_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)/'source'; source.write_text('synthetic native source')
            pin = runner.sha(source)
            provenance = dict(sources_sha256={str(source):pin})
            for key in ('cpu_binary','terminal_oracle','terminal_oracle_build'):
                provenance[key+'_path']=str(source); provenance[key+'_sha256']=pin
            info = {'manifest_sha256':'a'*64,'counts':{'states':2,'observations':1}}
            row = dict(manifest_sha256='a'*64,raw_tapered_state_cp=[1,-1],native_material_state_cp=[1,-1],
                       native_hashes=[1,2],observation_state_ids=[0],
                       terminal_states=[dict(state_id=1,kind='draw',in_check=False,legal_count=0)])
            witness = dict(kind='semantic_exact_baseline_witness_v1',splits={'train':row},provenance=provenance)
            runner.validate_witness(witness,{'train':info})
            for change in ('mate','bool','missing','hash'):
                bad = copy.deepcopy(witness)
                if change=='mate': bad['splits']['train']['terminal_states'][0]['in_check']=True
                if change=='bool': bad['splits']['train']['observation_state_ids']=[True]
                if change=='missing': bad['splits']['train']['raw_tapered_state_cp']=[1]
                if change=='hash': bad['splits']['train']['native_hashes']=[-1,2]
                with self.assertRaises(ValueError): runner.validate_witness(bad,{'train':info})

    def _freeze_fixture(self, root):
        config = study.Configuration(anchor_updates=1, transition_updates=1)
        anchors, transitions, phases = {}, {}, []
        for seed in config.seeds:
            for encoder in runner.ENCODERS:
                key = f'{encoder}/{seed}'
                anchors[key] = dict(file=f'anchor-{encoder}-{seed}.pt', sha256='a'*64, bytes=100 << 20)
                phases.append(dict(key=key, status='complete', phase='anchor', updates=1))
                for control in study.CONTROLS:
                    key = f'{encoder}/{control}/{seed}'
                    transitions[key] = dict(file=f'transition-{encoder}-{control}-{seed}.pt', sha256='a'*64, bytes=100 << 20)
                    phases.append(dict(key=key, status='complete', phase='transition', updates=1))
        record = dict(kind=runner.FREEZE_KIND, version=1, declaration_sha256='b'*64,
                      scale=1., anchors=anchors, transitions=transitions, phase_reports=phases)
        path = root/'freeze.json'; pin = runner.publish_json(path, record)
        return config, record, path, pin

    def test_freeze_manifest_bounds_each_file_without_a_combined_tensor_cap(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder); config, record, path, pin = self._freeze_fixture(root)
            self.assertGreater(sum(row['bytes'] for table in ('anchors','transitions') for row in record[table].values()), runner.CHECKPOINT_LIMIT)
            # No giant fixture allocation: the known producer files are each
            # bounded; this metadata test verifies all 40 joins are admitted.
            with patch.object(runner, 'checkpoint_bytes', return_value=b'') as reader, \
                 patch.object(runner, 'load_frozen_record') as loader:
                frozen = runner.read_checkpoint_manifest(root, path, pin, 'b'*64, config)
                self.assertEqual(reader.call_count,40)
                self.assertEqual(loader.call_count,40)
                self.assertEqual(len(frozen['transitions']),32)
            for mutation in ('missing','path','version_bool','incomplete_phase'):
                bad = copy.deepcopy(record)
                if mutation == 'missing': bad['anchors'].pop(next(iter(bad['anchors'])))
                elif mutation == 'path': bad['transitions'][next(iter(bad['transitions']))]['file']='../escape.pt'
                elif mutation == 'version_bool': bad['version']=True
                else: bad['phase_reports'][0]['status']='partial'
                changed = root/(mutation+'.json'); changed_pin=runner.publish_json(changed,bad)
                with patch.object(runner,'checkpoint_bytes',return_value=b''):
                    with self.assertRaises(ValueError):runner.read_checkpoint_manifest(root,changed,changed_pin,'b'*64,config)

    def test_individual_file_hash_and_type_are_checked_before_tensor_loading(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder); path=root/'transition-square-transport-20261021.pt'
            path.write_bytes(b'mechanical invalid tensor archive')
            entry=runner.checkpoint_entry(path,runner.sha(path))
            self.assertEqual(runner.checkpoint_bytes(root,entry),path.read_bytes())
            with patch.object(runner.torch,'load') as loader:
                path.write_bytes(b'changed bytes')
                frozen={'transitions':{'square/transport/20261021':entry}}
                with self.assertRaises(ValueError):runner.load_transition_file(root,'square/transport/20261021',frozen,study.Configuration())
                loader.assert_not_called()
            for value in (True, runner.CHECKPOINT_LIMIT+1):
                bad=entry.copy();bad['bytes']=value
                with self.assertRaises(ValueError):runner.checkpoint_bytes(root,bad)
            path.unlink(); target=root/'target';target.write_bytes(b'abc');path.symlink_to(target)
            with self.assertRaises(OSError):runner.checkpoint_bytes(root,entry)
            path.unlink();import os;os.mkfifo(path)
            with self.assertRaises(ValueError):runner.checkpoint_bytes(root,entry)


if __name__ == '__main__':
    unittest.main()
