import copy
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess
import torch
from torch.nn import functional as F
from torch.utils._python_dispatch import TorchDispatchMode

from neural.dynamics import batch_effects, generate_counterfactuals, move_effects
from neural.features import encode_fen, move_to_index
from neural.models import MODELS, build_model
from neural.strategic_transport import (MODES, StrategicPredictor, derive_effects,
    load_strategic_checkpoint, save_strategic_checkpoint, successor_pieces)
from neural.value_heads import material_cp


def row(fen, uci):
    root = chess.Board(fen)
    move = chess.Move.from_uci(uci)
    assert move in root.legal_moves
    features = encode_fen(fen)[None]
    action = torch.tensor([move_to_index(uci, fen.split()[1])])
    expected_effects = batch_effects([move_effects(root, move)])
    root.push(move)
    successor = encode_fen(root.fen(en_passant='fen'))[None]
    return features, action, expected_effects, successor


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten._local_scalar_dense.default:
            raise RuntimeError('host scalar extraction in prediction')
        return function(*args, **(kwargs or {}))


class StrategicTransportTests(unittest.TestCase):
    def test_effects_exact_on_special_moves_both_colors_and_random_roots(self):
        fixtures = [
            ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1'),
            ('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1c1'),
            ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', 'e8g8'),
            ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', 'e8c8'),
            ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'e5d6'),
            ('4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1', 'e4d3'),
            ('4k3/8/8/8/8/2p5/3P4/4K3 w - - 0 1', 'd2c3'),
        ]
        for promotion in 'nbrq':
            fixtures += [
                ('4k3/P7/8/8/8/8/8/4K3 w - - 0 1', 'a7a8' + promotion),
                ('4k3/8/8/8/8/8/p7/4K3 b - - 0 1', 'a2a1' + promotion),
                ('1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1', 'a7b8' + promotion),
            ]
        corpus = generate_counterfactuals(12, seed=902, max_plies=60)
        fixtures += [(r['root_fen'], move) for r in corpus.records for move in r['moves']]
        model = StrategicPredictor(build_model('square', width=8, depth=1))
        for fen, uci in fixtures:
            with self.subTest(fen=fen, move=uci):
                features, action, oracle, successor = row(fen, uci)
                effects = derive_effects(features, action)
                for key, expected in oracle.items():
                    torch.testing.assert_close(effects[key], expected, atol=0, rtol=0)
                pieces = successor_pieces(features, effects)
                torch.testing.assert_close(pieces, successor[:, :12], atol=0, rtol=0)
                torch.testing.assert_close(model.material(pieces), material_cp(successor), atol=0, rtol=0)

    def test_initial_anchor_modes_common_state_shapes_and_canonical_color_symmetry(self):
        root = chess.Board()
        f, a, _, successor = row(root.fen(), 'g1f3')
        for architecture in MODELS:
            states = []
            for mode in MODES:
                torch.manual_seed(711)
                model = StrategicPredictor(build_model(architecture, width=8, depth=1), mode=mode)
                states.append(model.state_dict())
                out = model(f, a)
                self.assertEqual(out['latent'].shape, (1, 64, 8))
                self.assertEqual(out['pieces'].shape, (1, 12, 8, 8))
                torch.testing.assert_close(out['residual_cp'], torch.zeros(1), atol=0, rtol=0)
                torch.testing.assert_close(out['value_cp'], material_cp(successor), atol=0, rtol=0)
                mirror = root.mirror()
                mirror_f = encode_fen(mirror.fen())[None]
                mirror_action = torch.tensor([move_to_index('g8f6', 'b')])
                self.assertTrue(torch.equal(f, mirror_f))
                self.assertTrue(torch.equal(a, mirror_action))
                torch.testing.assert_close(model(mirror_f, mirror_action)['latent'], out['latent'], atol=0, rtol=0)
            for key, value in states[0].items():
                self.assertTrue(all(torch.equal(value, state[key]) for state in states[1:]))
            zero = StrategicPredictor(build_model(architecture, width=8, depth=1), anchor_kind='zero')
            torch.testing.assert_close(zero(f, a)['value_cp'], torch.zeros(1), atol=0, rtol=0)

    def test_pseudoattacks_match_oracle_and_blocker_interventions(self):
        model = StrategicPredictor(build_model('square', width=8, depth=1))
        corpus = generate_counterfactuals(3, seed=890)
        for fen in [r['root_fen'] for r in corpus.records]:
            board = chess.Board(fen)
            flip = 0 if board.turn else 56
            observed = model.attack_counts(encode_fen(fen)[None, :12])[0]
            expected = torch.zeros(2, 64)
            for source, piece in board.piece_map().items():
                side = int(piece.color != board.turn)
                for target in board.attacks(source):
                    expected[side, target ^ flip] += 1
            torch.testing.assert_close(observed, expected, atol=0, rtol=0)
        open_fen = '4k3/8/8/8/8/8/8/R3K3 w - - 0 1'
        blocked_fen = '4k3/8/8/8/8/8/P7/R3K3 w - - 0 1'
        opened = model.attack_counts(encode_fen(open_fen)[None, :12])
        blocked = model.attack_counts(encode_fen(blocked_fen)[None, :12])
        self.assertEqual(opened[0, 0, chess.A8].item(), 1)
        self.assertEqual(blocked[0, 0, chess.A8].item(), 0)
        self.assertEqual(blocked[0, 0, chess.A2].item(), 1)

    def test_no_successor_ingress_action_blind_control_cached_latent_and_host_reads(self):
        root = encode_fen(chess.STARTING_FEN)[None].repeat(2, 1, 1, 1)
        actions = torch.tensor([move_to_index('g1f3'), move_to_index('b1c3')])
        model = StrategicPredictor(build_model('relation', width=8, depth=1))
        with patch.object(model, 'refresh', side_effect=RuntimeError('successor shortcut')):
            with NoHostScalar():
                out = model.predict(root, actions)
                cached = model.predict(root, actions, root_latent=model.encode(root))
        torch.testing.assert_close(out['latent'], cached['latent'], atol=0, rtol=0)
        self.assertGreater((out['latent'][0] - out['latent'][1]).abs().max().item(), .01)
        blind = StrategicPredictor(copy.deepcopy(model.base), mode='action_blind')
        blind.load_state_dict(model.state_dict())
        with torch.no_grad():
            blind.head.weight.fill_(.03)
        observed = blind(root, actions)
        torch.testing.assert_close(observed['residual_cp'][0], observed['residual_cp'][1], atol=0, rtol=0)
        torch.testing.assert_close(observed['value_cp'][0], observed['value_cp'][1], atol=0, rtol=0)
        with self.assertRaises(ValueError): model(root.double(), actions)
        with self.assertRaises(ValueError): model(root, actions.int())
        with self.assertRaises(ValueError): model(root, actions, root_latent=root)
        with torch.no_grad(): model.head.bias.fill_(100)
        self.assertGreater(model(root, actions)['value_cp'].min().item(), 29000)
        torch.testing.assert_close(model.static_scores(root, actions),
                                   torch.full((2,), 29000, dtype=torch.int32), atol=0, rtol=0)

    def test_learning_semantic_sibling_delta_gradients_and_frozen_encoder(self):
        torch.manual_seed(53)
        roots = encode_fen(chess.STARTING_FEN)[None].repeat(2, 1, 1, 1)
        actions = torch.tensor([move_to_index('g1f3'), move_to_index('b1c3')])
        _, _, _, a = row(chess.STARTING_FEN, 'g1f3')
        _, _, _, b = row(chess.STARTING_FEN, 'b1c3')
        successors = torch.cat((a, b))
        for frozen in (False, True):
            model = StrategicPredictor(build_model('square', width=8, depth=1), freeze_encoder=frozen)
            target_encoder = copy.deepcopy(model.base).requires_grad_(False)
            target = target_encoder.encode(successors).detach()
            before = {k: v.clone() for k, v in model.base.state_dict().items()}
            optimizer = torch.optim.AdamW(model.active_parameters(), lr=.005)
            # Deliberately synthetic 2-action semantic labels exercise optimizer
            # direction; they are NOT engine teacher/outcome or strength evidence.
            scores = torch.tensor([-120., 120.])
            def loss():
                out = model(roots, actions)
                delta = out['value_cp'][0] - out['value_cp'][1]
                return (F.smooth_l1_loss(out['value_cp'] / 800, scores / 800) +
                        F.smooth_l1_loss(delta / 800, (scores[0] - scores[1]) / 800) +
                        .05 * F.mse_loss(F.normalize(out['latent'], dim=-1), F.normalize(target, dim=-1)))
            initial = loss().detach()
            for _ in range(30):
                optimizer.zero_grad(set_to_none=True)
                value = loss(); value.backward()
                self.assertTrue(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.active_parameters()))
                optimizer.step()
            self.assertLess(loss().item(), initial.item())
            self.assertLess(model(roots, actions)['value_cp'][0].item(), model(roots, actions)['value_cp'][1].item())
            if frozen:
                self.assertTrue(all(p.grad is None for p in model.base.parameters()))
                self.assertTrue(all(torch.equal(v, model.base.state_dict()[k]) for k, v in before.items()))
            else:
                self.assertGreater(model.base.input.weight.grad.abs().sum().item(), 0)
                self.assertIsNone(model.base.wdl.weight.grad)
                self.assertIsNone(model.base.dynamics[1].weight.grad)

    def test_strict_checkpoint_roundtrip_and_negative_semantics_buffers(self):
        roots, actions, _, _ = row(chess.STARTING_FEN, 'g1f3')
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'model.pt'
            for architecture in MODELS:
                model = StrategicPredictor(build_model(architecture, width=8, depth=1), freeze_encoder=True)
                save_strategic_checkpoint(model, path, {'purpose': 'mechanical fixture', 'strength_evidence': False})
                restored, metadata = load_strategic_checkpoint(path)
                self.assertIs(metadata['strength_evidence'], False)
                expected, actual = model(roots, actions), restored(roots, actions)
                for key, value in expected.items():
                    torch.testing.assert_close(value, actual[key], atol=0, rtol=0)
            pristine = torch.load(path, weights_only=True)
            variants = [
                ('bool_version', lambda p: p['predictor_config'].__setitem__('strategic_version', True)),
                ('wrong_frame', lambda p: p['predictor_config'].__setitem__('perspective', 'root_side_to_move')),
                ('bool_freeze', lambda p: p['predictor_config'].__setitem__('freeze_encoder', 1)),
                ('bool_material', lambda p: p['predictor_config']['material_values'].__setitem__(5, False)),
                ('orientation', lambda p: p['state_dict']['base.next_orientation'].__setitem__(0, 64)),
                ('attacks', lambda p: p['state_dict']['base.attack_patterns'].__setitem__((0, 0, 0), 1)),
                ('dtype', lambda p: p['state_dict'].__setitem__('head.weight', p['state_dict']['head.weight'].double())),
                ('nonfinite', lambda p: p['state_dict']['head.weight'].__setitem__((0, 0), float('nan'))),
            ]
            for name, mutate in variants:
                with self.subTest(corruption=name):
                    payload = copy.deepcopy(pristine); mutate(payload); torch.save(payload, path)
                    with self.assertRaises(ValueError): load_strategic_checkpoint(path)

    @unittest.skipUnless(os.environ.get('CHEN_STRATEGIC_TEST_GPU') == '1', 'GPU slot not requested')
    def test_gpu_prediction_and_gradients_scoped_parity(self):
        f, a, _, _ = row(chess.STARTING_FEN, 'g1f3')
        cpu = StrategicPredictor(build_model('square', width=8, depth=1))
        gpu = copy.deepcopy(cpu).cuda()
        with torch.no_grad(): cpu.head.weight.fill_(.01); gpu.head.weight.copy_(cpu.head.weight.cuda())
        left, right = cpu(f, a), gpu(f.cuda(), a.cuda())
        for key in left:
            torch.testing.assert_close(left[key], right[key].cpu(), atol=3e-4, rtol=3e-4)
        left['value_cp'].sum().backward(); right['value_cp'].sum().backward()
        torch.testing.assert_close(cpu.correction[-1].weight.grad, gpu.correction[-1].weight.grad.cpu(), atol=3e-4, rtol=3e-4)


if __name__ == '__main__':
    unittest.main()
