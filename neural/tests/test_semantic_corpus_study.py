import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.features import encode_fen, move_to_index
from neural.semantic_strategy import anchor_losses
from neural.semantic_corpus_study import (Configuration, CONTROLS, COEFFICIENTS,
    admit_resident_dependency, branch, complete_root_metrics, fingerprint,
    fit_positive_material, proposal, require_gpu_learning_authority,
    profile_branch_on_train,
    scalar_interval_error, select_or_none, select_complete_study_or_none,
    shared_anchors, transition_controls,
    transition_loss)
from neural.value_heads import material_cp


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError('host scalar extraction in resident kernel')
        return function(*args, **(kwargs or {}))


def fixture(fen=chess.STARTING_FEN):
    board = chess.Board(fen)
    moves = sorted(board.legal_moves, key=lambda m: move_to_index(m.uci(), 'w' if board.turn else 'b'))
    actions = torch.tensor([[move_to_index(m.uci(), 'w' if board.turn else 'b') for m in moves]])
    children = []
    for move in moves:
        child = board.copy(); child.push(move)
        children.append(encode_fen(child.fen(en_passant='fen')))
    features = torch.stack(children)[None]
    k = len(moves)
    return SimpleNamespace(features=encode_fen(fen)[None], legal_actions=actions,
        targets=SimpleNamespace(features=features, midpoint=torch.zeros(1, k),
            halfwidth=torch.ones(1, k), ordinary=torch.ones(1, k, dtype=torch.bool),
            terminal=torch.zeros(1, k, dtype=torch.bool), reported_cp=torch.zeros(1, k, dtype=torch.int32)))


class StudyTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.config = Configuration(width=8, anchor_updates=1, transition_updates=1)

    def test_protocol_is_concrete_unsealed_and_bounded(self):
        p = proposal(train_manifest_sha256='a'*64, validation_manifest_sha256='b'*64,
                     dependencies={'example':'c'*64})
        self.assertEqual(p['status'], 'unsealed')
        self.assertEqual(p['config']['anchor_updates'], 384)
        self.assertEqual(p['config']['transition_updates'], 192)
        self.assertEqual(p['selection']['failed_gate'], 'NONE')
        self.assertEqual(p['rollout']['plies'], 1)
        self.assertFalse(p['equal_compute_claim'])
        for invalid in (dict(seeds=(1,1)), dict(width=True), dict(anchor_updates=0),
                        dict(material_lower=0), dict(learning_rate=float('nan'))):
            with self.assertRaises(ValueError): Configuration(**invalid)

    def test_scale_known_optimum_bounds_mask_sign_and_flat_tie(self):
        x = torch.tensor([100., -200., 50., 999.])
        y = torch.tensor([200., -400., 100., -50000.])
        width = torch.zeros(4)
        mask = torch.tensor([True, True, True, False])
        with NoHostScalar(): scale = fit_positive_material(x,y,width,mask,self.config)
        self.assertAlmostEqual(float(scale), 2., places=10)
        self.assertEqual(float(fit_positive_material(torch.zeros(4),y,width,mask,self.config)), 1.)
        self.assertAlmostEqual(float(fit_positive_material(x,y*100,width,mask,self.config)), 5., places=8)
        self.assertAlmostEqual(float(fit_positive_material(x,-y,width,mask,self.config)), .05, places=8)
        self.assertEqual(float(fit_positive_material(x,x,torch.ones(4)*10,mask,self.config)), 1.)

    def test_common_initialization_all_four_encoders_and_pair_replay(self):
        suite = shared_anchors(12,1.2,self.config)
        again = shared_anchors(12,1.2,self.config)
        f = fixture().features
        for name, anchor in suite.items():
            self.assertEqual(fingerprint(anchor), fingerprint(again[name]))
            torch.testing.assert_close(anchor.board.weight, suite['square'].board.weight, atol=0, rtol=0)
            torch.testing.assert_close(anchor.queries, suite['square'].queries, atol=0, rtol=0)
            self.assertEqual(anchor(f)['latent'].shape, (1,8,8))
        self.assertNotEqual(fingerprint(suite['square']),fingerprint(shared_anchors(13,1.2,self.config)['square']))

    def test_anchor_semantic_and_interval_gradient_without_host_reads(self):
        anchor = shared_anchors(6,1.,self.config)['threat']
        b = fixture(); target=torch.tensor([100.]); width=torch.tensor([.51]); mask=torch.tensor([True])
        with NoHostScalar():
            output=anchor(b.features)
            losses=anchor_losses(anchor,output,b.features,target,mask,teacher_halfwidth=width)
            sum(COEFFICIENTS[name]*value for name,value in losses.items()).backward()
        self.assertGreater(float(anchor.cp.weight.grad.abs().sum()), 0)
        self.assertIsNotNone(anchor.messages[0].update[1].weight.grad)

    def test_transition_initialization_and_detached_exact_child_loss(self):
        anchor = shared_anchors(6,1.,self.config)['threat']
        controls = transition_controls(anchor,77)
        before = fingerprint(anchor)
        reference = controls['transport'].state_dict()
        for name, model in controls.items():
            for parameter, value in model.state_dict().items():
                torch.testing.assert_close(value,reference[parameter],atol=0,rtol=0)
            b=fixture(); b.targets.features.requires_grad_(True); b.targets.midpoint.requires_grad_(True)
            with NoHostScalar():
                output=branch(model,name,b.features,b.legal_actions)
                loss,parts=transition_loss(model,output,b)
                loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertGreater(float(model.update[0].weight.grad.abs().sum()),0)
            self.assertIsNone(b.targets.features.grad)
            self.assertIsNone(b.targets.midpoint.grad)
            self.assertEqual(fingerprint(anchor),before)
            self.assertTrue(all(p.grad is None for p in anchor.parameters()))

    def test_action_blind_exact_same_material_ties_and_shuffle_true_prior(self):
        # Startpos legal siblings all preserve material exactly.
        controls=transition_controls(shared_anchors(3,1.4,self.config)['relation'],12)
        b=fixture()
        output=branch(controls['action_blind'],'action_blind',b.features,b.legal_actions)
        self.assertTrue(torch.equal(output['value_cp'],output['value_cp'][:,:1].expand_as(output['value_cp'])))
        b=fixture('4k3/8/8/3p4/4Q3/8/8/4K3 w - - 0 1')
        m=controls['shuffled_conditioned']
        output=branch(m,'shuffled_conditioned',b.features,b.legal_actions)
        expected=material_cp(b.targets.features.reshape(-1,18,8,8)).reshape_as(output['value_cp'])*1.4
        torch.testing.assert_close(output['material_cp'],expected,atol=0,rtol=0)
        self.assertGreater(float(expected.max()-expected.min()),0)
        with self.assertRaises(ValueError): branch(m,'transport',b.features,b.legal_actions)

    def test_complete_root_interval_metrics_canonical_ties_terminal_required(self):
        b=fixture(); k=b.legal_actions.shape[1]
        b.targets.midpoint=torch.arange(k,dtype=torch.float32)[None]*10
        b.targets.halfwidth=torch.ones(1,k)
        b.targets.reported_cp=-torch.arange(k,dtype=torch.int32)[None]*10
        metrics=complete_root_metrics(torch.zeros(1,k),b)
        self.assertEqual(float(metrics['interval_regret_cp']),0)
        self.assertEqual(float(metrics['optimal_agreement']),1)
        self.assertEqual(int(metrics['selected_canonical_action'][0]),int(b.legal_actions[0,0]))
        prediction=torch.zeros(1,k); prediction[0,1]=-1
        self.assertEqual(float(complete_root_metrics(prediction,b)['interval_regret_cp']),8)
        b.targets.terminal[0,0]=True
        with self.assertRaises(ValueError): complete_root_metrics(prediction,b)
        terminal_cp=torch.zeros(1,k)
        draw_prediction=torch.ones(1,k)
        self.assertEqual(float(complete_root_metrics(draw_prediction,b,terminal_cp=terminal_cp)['optimal_agreement']),1)
        terminal_cp[0,0]=-30000
        with self.assertRaises(ValueError):complete_root_metrics(prediction,b,terminal_cp=terminal_cp)
        self.assertEqual(float(scalar_interval_error(torch.tensor([4.,0.]),torch.tensor([0.,0.]),torch.tensor([1.,1.]))),1.5)

    def test_joint_gate_requires_every_seed_every_baseline_and_none(self):
        row=dict(scalar_error_cp=10.,interval_regret_cp=20.,optimal_agreement=.5)
        baselines={name:row.copy() for name in ('raw_material100','scaled_material','raw_tapered')}
        good=dict(scalar_error_cp=9.,interval_regret_cp=19.,optimal_agreement=.5)
        seeds=self.config.seeds
        candidate={s:good.copy() for s in seeds}
        self.assertEqual(select_or_none({'threat/transport':candidate},baselines),'threat/transport')
        candidate[seeds[1]]['scalar_error_cp']=10.
        self.assertEqual(select_or_none({'threat/transport':candidate},baselines),'NONE')
        candidate[seeds[1]]['scalar_error_cp']=float('nan')
        with self.assertRaises(ValueError):select_or_none({'threat/transport':candidate},baselines)
        with self.assertRaises(ValueError):select_or_none({},dict(raw_material100=row))
        with self.assertRaises(ValueError):select_complete_study_or_none({'threat/transport':candidate},baselines)

    def test_shared_loader_uses_pinned_normal_import_and_source_mismatch_rejected(self):
        main=Path(__file__).resolve().parents[3]/'chen'
        names=('experiments/semantic_corpus/__init__.py','experiments/semantic_corpus/packing.py','experiments/semantic_corpus/resident.py')
        pins={name:hashlib.sha256((main/name).read_bytes()).hexdigest() for name in names}
        module=admit_resident_dependency(main,pins)
        self.assertEqual(Path(module.__file__).resolve(),main/names[-1])
        pins[names[-1]]='0'*64
        with self.assertRaises(ValueError):admit_resident_dependency(main,pins)

    def test_unsealed_declaration_never_authorizes_real_training(self):
        import neural.semantic_corpus_study as study
        path=Path(study.__file__).resolve()
        p=proposal(train_manifest_sha256='a'*64,validation_manifest_sha256='b'*64,
                   dependencies={str(path):hashlib.sha256(path.read_bytes()).hexdigest()})
        corpus=SimpleNamespace(stage='train',manifest_sha256='a'*64,device=torch.device('cpu'))
        with tempfile.TemporaryDirectory() as temp:
            f=Path(temp)/'unsealed.json';f.write_text(json.dumps(p))
            pin=hashlib.sha256(f.read_bytes()).hexdigest()
            with self.assertRaises(ValueError):require_gpu_learning_authority(f,pin,corpus)
            with self.assertRaises(ValueError):require_gpu_learning_authority(f,'0'*64,corpus)
        with self.assertRaises(ValueError):profile_branch_on_train(None,None,corpus,None,expected_train_manifest_sha256='a'*64)


if __name__=='__main__': unittest.main()
