import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.features import encode_fen, move_to_index
from neural.semantic_strategy import (ENCODERS, MODES, Diagram, SemanticAnchor,
    SemanticTransition, anchor_losses, load_semantic_checkpoint, save_semantic_checkpoint)


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError('host scalar extraction')
        return function(*args, **(kwargs or {}))


def feature(board):
    return encode_fen(board.fen(en_passant='fen'))[None]


def action(board, uci):
    move = chess.Move.from_uci(uci)
    assert move in board.legal_moves
    return torch.tensor([move_to_index(uci, 'w' if board.turn else 'b')])


class SemanticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(991)

    def test_diagram_oracle_color_blockers_pins_and_pawn_destinations(self):
        diagram = Diagram()
        fixtures = [chess.Board(), chess.Board('4k3/8/8/8/8/2q5/3P4/4K3 w - - 0 1'),
                    chess.Board('4r1k1/8/8/8/8/8/4N3/4K3 w - - 0 1'),
                    chess.Board('4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1')]
        for board in fixtures:
            f = feature(board)
            canonical = board if board.turn else board.mirror()
            expected = torch.zeros(2, 64, 64)
            for square, piece in canonical.piece_map().items():
                side = 0 if piece.color else 1
                for target in canonical.attacks(square):
                    expected[side, square, target] = 1
            torch.testing.assert_close(diagram.edges(f[:, :12])[0], expected, atol=0, rtol=0)
            maps = diagram.maps(f[:, :12])[0].flatten(1)
            counts = expected.sum(1)
            for side, index in [(True, 6), (False, 7)]:
                advances = torch.zeros(64)
                for source in canonical.pieces(chess.PAWN, side):
                    target = source + (8 if side else -8)
                    if 0 <= target < 64 and canonical.piece_at(target) is None:
                        advances[target] = 1
                torch.testing.assert_close(maps[index], advances, atol=0, rtol=0)
            torch.testing.assert_close(maps[4:6], counts.clamp_max(8) / 8, atol=0, rtol=0)
            torch.testing.assert_close(diagram.maps(feature(board.mirror())[:, :12]),
                                       diagram.maps(f[:, :12]), atol=0, rtol=0)
        # The pinned knight has diagram attacks even when the legal move is absent.
        pinned = fixtures[2]
        self.assertTrue(pinned.is_pinned(chess.WHITE, chess.E2))
        self.assertGreater(float(diagram.edges(feature(pinned)[:, :12])[0, 0, chess.E2].sum()), 0)

    def test_all_backends_shapes_semantics_gradients_and_no_host_reads(self):
        features = feature(chess.Board()).expand(2, -1, -1, -1).clone()
        for backend in ENCODERS:
            with self.subTest(backend=backend):
                anchor = SemanticAnchor(backend, width=8, depth=1, material_scale=1.3)
                with NoHostScalar():
                    out = anchor(features)
                    losses = anchor_losses(anchor, out, features, torch.tensor([120., -80.]), torch.tensor([True, True]))
                    sum(losses.values()).backward()
                self.assertEqual(out['latent'].shape, (2, 8, 8))
                self.assertEqual(out['board_logits'].shape, (2, 13, 8, 8))
                self.assertEqual(out['map_logits'].shape, (2, 8, 8, 8))
                self.assertEqual(out['ep_logits'].shape, (2, 65))
                self.assertGreater(float(anchor.queries.grad.abs().sum()), 0)
                self.assertGreater(float(anchor.cp.weight.grad.abs().sum()), 0)
                if backend == 'threat':
                    self.assertGreater(float(anchor.messages[0].messages[1].weight.grad.abs().sum()), 0)
                else:
                    self.assertIsNone(anchor.base.wdl.weight.grad)
                    self.assertIsNone(anchor.base.dynamics[1].weight.grad)
                self.assertEqual(len(list(anchor.active_parameters())), len({id(p) for p in anchor.active_parameters()}))
                targets = anchor.semantic_targets(features)
                self.assertTrue(torch.equal(targets['ep'], torch.full((2,), 64)))
                torch.testing.assert_close(out['value_cp'], torch.zeros(2), atol=0, rtol=0)
                for invalid in ([0., 0.], False, torch.zeros(2, dtype=torch.float64), torch.zeros(1)):
                    with self.assertRaises(ValueError):
                        anchor_losses(anchor, out, features, torch.tensor([120., -80.]),
                                      torch.tensor([True, True]), teacher_halfwidth=invalid)

    def test_successor_prior_oracle_special_moves_and_no_successor_encoder(self):
        fixtures = [('r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1', 'e1g1'),
                    ('r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1', 'e8c8'),
                    ('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1', 'e5d6'),
                    ('4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 1', 'e4d3'),
                    ('1r2k3/P7/8/8/8/8/8/4K3 w - - 0 1', 'a7b8n')]
        predictor = SemanticTransition(SemanticAnchor('threat', width=8), 'transport')
        for fen, uci in fixtures:
            board = chess.Board(fen)
            f, a = feature(board), action(board, uci)
            latent = predictor.encode(f)
            with patch.object(predictor.anchor, 'encode', side_effect=AssertionError('hidden refresh')):
                with NoHostScalar():
                    out = predictor.predict(f, a, latent)
            board.push_uci(uci)
            torch.testing.assert_close(out['pieces'], feature(board)[:, :12], atol=0, rtol=0)
            torch.testing.assert_close(out['material_cp'], predictor.anchor.diagram.material(out['pieces']), atol=0, rtol=0)
            # Learned scalar starts zero; exact prior uses CHILD mover perspective.
            torch.testing.assert_close(out['value_cp'], out['material_cp'], atol=0, rtol=0)
        with self.assertRaises(TypeError):
            predictor.predict(f, a, successor_features=feature(board))

    def test_controls_identical_state_and_blind_exact_cached_sibling_ties(self):
        snapshots = []
        for mode in MODES:
            torch.manual_seed(981)
            model = SemanticTransition(SemanticAnchor('plan', width=8), mode)
            snapshots.append(model.state_dict())
        for key, value in snapshots[0].items():
            self.assertTrue(all(torch.equal(value, state[key]) for state in snapshots[1:]))
        board = chess.Board()
        f = feature(board).expand(3, -1, -1, -1)
        actions = torch.tensor([[move_to_index(m, 'w') for m in ['g1f3', 'b1c3', 'e2e4']] for _ in range(3)])
        blind = SemanticTransition(SemanticAnchor('plan', width=8), 'action_blind')
        with torch.no_grad():
            blind.anchor.cp.weight.normal_(0, .1)  # emulate an admitted nonzero pretrained readout
        with NoHostScalar():
            out = blind.branch(f, actions)
        self.assertTrue(torch.equal(out['value_cp'][:, :1].expand_as(out['value_cp']), out['value_cp']))
        self.assertTrue(torch.equal(out['latent'][:, :1].expand_as(out['latent']), out['latent']))
        latent = blind.encode(f)
        with patch.object(blind.anchor, 'encode', side_effect=AssertionError('cache ignored')):
            cached = blind.branch(f, actions, latent)
        torch.testing.assert_close(cached['value_cp'], out['value_cp'], atol=0, rtol=0)

    def test_action_sensitivity_fixed_readout_gradient_and_hybrid_rollout(self):
        anchor = SemanticAnchor('threat', width=8)
        root = chess.Board()
        f = feature(root)
        # Toy semantic pretraining is a mechanics check, never strength evidence.
        optimizer = torch.optim.Adam(anchor.active_parameters(), lr=.01)
        for _ in range(3):
            optimizer.zero_grad()
            out = anchor(f)
            losses = anchor_losses(anchor, out, f, torch.tensor([100.]), torch.tensor([True]))
            sum(losses.values()).backward()
            optimizer.step()
        frozen = copy.deepcopy(anchor.state_dict())
        predictor = SemanticTransition(anchor)
        anchor.zero_grad(set_to_none=True)
        actions = torch.tensor([[move_to_index('g1f3'), move_to_index('b1c3')]])
        out = predictor.branch(f, actions)
        self.assertGreater(float((out['latent'][:, 0] - out['latent'][:, 1]).detach().abs().sum()), 0)
        optimizer = torch.optim.Adam(predictor.active_parameters(), lr=.01)
        optimizer.zero_grad()
        loss = ((out['value_cp'] / 800 - torch.tensor([[.2, -.2]])) ** 2).mean()
        loss.backward()
        self.assertGreater(float(predictor.event[0].weight.grad.abs().sum()), 0)
        self.assertTrue(all(p.grad is None for p in anchor.parameters()))
        optimizer.step()
        for key, value in anchor.state_dict().items():
            self.assertTrue(torch.equal(value, frozen[key]))
        observations, moves = [], []
        for uci in ['g1f3', 'g8f6', 'b1c3', 'b8c6']:
            observations.append(feature(root))
            moves.append(action(root, uci))
            root.push_uci(uci)
        with patch.object(anchor, 'encode', wraps=anchor.encode) as encoder:
            with NoHostScalar():
                rollout = predictor.rollout(torch.stack(observations), torch.stack(moves))
            self.assertEqual(encoder.call_count, 1)
        torch.testing.assert_close(rollout[-1]['pieces'], feature(root)[:, :12], atol=0, rtol=0)
        self.assertEqual(len(rollout), 4)
        self.assertFalse(torch.equal(rollout[-1]['latent'], predictor.refresh(feature(root))['latent']))
        with self.assertRaises(ValueError):
            predictor.rollout(torch.empty(0, 1, 18, 8, 8), torch.empty(0, 1, dtype=torch.int64))

    def test_restricted_checkpoints_and_config_buffer_mutations(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'model.pt'
            for backend in ENCODERS:
                for phase in ['anchor', 'transition']:
                    model = SemanticAnchor(backend, width=8, material_scale=.91)
                    if phase == 'transition':
                        model = SemanticTransition(model, 'conditioned')
                    save_semantic_checkpoint(model, path, {'scope': 'untrained_unit_fixture', 'train_only': True})
                    restored, metadata = load_semantic_checkpoint(path)
                    self.assertEqual(restored.config(), model.config())
                    self.assertEqual(metadata['phase'], phase)
                    for key, value in model.state_dict().items():
                        self.assertTrue(torch.equal(value, restored.state_dict()[key]))
            original = torch.load(path, weights_only=True)
            corruptions = [('config_bool', lambda p: p['config'].update(transition_version=True)),
                           ('material_bool', lambda p: p['config']['anchor']['material_values'].__setitem__(0, True)),
                           ('frame_bool', lambda p: p['config']['anchor']['score_frame'].update(frame_version=True)),
                           ('frame_coefficient', lambda p: p['config']['anchor']['score_frame']['normalization_coefficients'].__setitem__(0, 0.)),
                           ('buffer', lambda p: p['state_dict']['anchor.diagram.role_flip'].__setitem__(0, 0)),
                           ('nan', lambda p: p['state_dict']['anchor.queries'].__setitem__((0, 0), float('nan'))),
                           ('extra', lambda p: p.update(extra=True))]
            for name, change in corruptions:
                payload = copy.deepcopy(original)
                change(payload)
                torch.save(payload, path)
                with self.subTest(corruption=name), self.assertRaises(ValueError):
                    load_semantic_checkpoint(path)
        for kwargs in [{'width': True}, {'material_scale': True}, {'material_scale': float('nan')}, {'depth': 0}]:
            with self.assertRaises(ValueError):
                SemanticAnchor(**kwargs)


if __name__ == '__main__':
    unittest.main()
