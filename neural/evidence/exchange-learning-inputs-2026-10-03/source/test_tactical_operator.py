"""CPU mechanics only: sparse oracle, shared controls, admission and causality."""
from dataclasses import replace
import unittest
from unittest.mock import patch

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.features import encode_fen, move_to_index
from neural.tactical_operator import (Relation, TacticalBatch, TacticalConfig, TacticalOperator,
                                     admit_child_mapping, admit_contents, incoming_reductions)


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError('forward extracted a host scalar')
        return function(*args, **(kwargs or {}))


def fixture(batch_size=2):
    """Declared synthetic arithmetic graph, not a claimed legal chess position."""
    config = TacticalConfig(width=8, piece_channels=3, square_channels=4,
                            action_channels=5, edge_channels=2, feature_schema='synthetic_fixture')
    b, p, k, e = batch_size, 3, 4, 8
    edge = [[0, p], [1, p + 4], [0, p + 64], [p + 1, p + 64],
            [p + 64, p + 1], [0, 1], [1, p + 7], [-800, 900]]
    kinds = [Relation.OCCUPIES, Relation.OCCUPIES, Relation.ACTION_SOURCE,
             Relation.ACTION_DESTINATION, Relation.ACTION_EFFECT,
             Relation.OWN_ATTACK_ENEMY, Relation.PSEUDOATTACK, -5]
    return config, TacticalBatch(
        pieces=torch.randn(b, p, 3), squares=torch.randn(b, 64, 4), actions=torch.randn(b, k, 5),
        piece_mask=torch.tensor([[True, True, False]]).expand(b, -1).clone(),
        piece_squares=torch.tensor([[0, 4, -1]]).expand(b, -1).clone(),
        action_mask=torch.tensor([[True, True, True, False]]).expand(b, -1).clone(),
        action_ids=torch.tensor([[5, 10, 15, -1]]).expand(b, -1).clone(),
        edge_index=torch.tensor(edge)[None].expand(b, -1, -1).clone(),
        edge_kind=torch.tensor(kinds)[None].expand(b, -1).clone(),
        edge_mask=torch.tensor([[True] * 7 + [False]]).expand(b, -1).clone(),
        edge_features=torch.randn(b, e, 2), edge_overflow=torch.zeros(b, dtype=torch.bool),
        value_prior_cp=torch.tensor([20.]).expand(b).clone(),
        action_prior_cp=torch.tensor([[8., 8., 3., 0.]]).expand(b, -1).clone(),
        feature_schema=config.feature_schema)


def board_fixture(board):
    """Independent CPU producer for ONE test; no production topology builder.

    Only occupancy/action endpoint relations are supplied. This deliberately
    does not pretend that a partial attack graph is complete exact topology.
    Complete LEGAL action coverage is checked against python-chess separately.
    """
    config = TacticalConfig(width=8, piece_channels=20, square_channels=18,
                            action_channels=6, edge_channels=2, feature_schema='oracle_fixture_v1')
    planes = encode_fen(board.fen()).flatten(1).T
    flip = 0 if board.turn else 56
    locations = sorted(square ^ flip for square in board.piece_map())
    p = len(locations)
    slots = {square: index for index, square in enumerate(locations)}
    moves = {move_to_index(move.uci(), 'w' if board.turn else 'b'): move for move in board.legal_moves}
    legal = sorted(moves)
    k = len(legal)
    piece = torch.stack([torch.cat((planes[square], torch.tensor([square % 8 / 7, square // 8 / 7])))
                         for square in locations])
    actions, edges, kinds = [], [], []
    for index, square in enumerate(locations):
        edges.extend([[index, p + square], [p + square, index]])
        kinds.extend([Relation.OCCUPIES, Relation.OCCUPIED_BY])
    for slot, action in enumerate(legal):
        pair, promotion = divmod(action, 5)
        source, target = divmod(pair, 64)
        actions.append([source % 8 / 7, source // 8 / 7, target % 8 / 7, target // 8 / 7,
                        promotion / 4, float(board.is_capture(moves[action]))])
        edges.extend([[slots[source], p + 64 + slot], [p + target, p + 64 + slot]])
        kinds.extend([Relation.ACTION_SOURCE, Relation.ACTION_DESTINATION])
    e = len(edges)
    batch = TacticalBatch(piece[None], planes[None], torch.tensor(actions).reshape(1, k, 6),
                          torch.ones(1, p, dtype=torch.bool), torch.tensor(locations)[None],
                          torch.ones(1, k, dtype=torch.bool), torch.tensor(legal, dtype=torch.int64)[None],
                          torch.tensor(edges, dtype=torch.int64).reshape(1, e, 2),
                          torch.tensor(kinds, dtype=torch.int64)[None], torch.ones(1, e, dtype=torch.bool),
                          torch.zeros(1, e, 2), torch.zeros(1, dtype=torch.bool), torch.zeros(1),
                          torch.zeros(1, k), feature_schema=config.feature_schema)
    return config, batch, legal


class TacticalMechanics(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5729)

    def test_sparse_reductions_match_independent_loop_and_empty_receivers(self):
        for e in (0, 1, 17):
            messages = torch.randn(2, e, 5)
            receiver = torch.randint(0, 7, (2, e))
            valid = torch.rand(2, e) > .35
            messages[~valid] = torch.nan
            receiver[~valid] = -999
            expected = [torch.zeros(2, 7, 5) for _ in range(3)]
            for b in range(2):
                for node in range(7):
                    rows = messages[b, valid[b] & (receiver[b] == node)]
                    if len(rows):
                        expected[0][b, node] = rows.mean(0)
                        expected[1][b, node] = rows.max(0).values
                        expected[2][b, node] = rows.min(0).values
            with NoHostScalar():
                actual = incoming_reductions(messages, receiver, valid, 7)
            for value, reference in zip(actual, expected):
                torch.testing.assert_close(value, reference, rtol=1e-6, atol=1e-7)

    def test_shared_recurrence_is_manual_cell_unroll_not_new_weights(self):
        config, batch = fixture()
        admit_contents(batch, config, expected_legal_actions=[[5, 10, 15]] * 2)
        model = TacticalOperator(config)
        before = {name: parameter.detach().clone() for name, parameter in model.named_parameters()}
        mask = torch.cat((batch.piece_mask, torch.ones(2, 64, dtype=torch.bool), batch.action_mask), 1)
        with NoHostScalar():
            one = model(batch, steps=1, return_trace=True)
        for steps in (2, 4, 8):
            with NoHostScalar():
                output = model(batch, steps=steps, return_trace=True)
                manual = one['latent']
                for _ in range(steps - 1):
                    manual = model.cell(manual, mask, batch)
            torch.testing.assert_close(output['trace'][0], one['latent'], rtol=0, atol=0)
            torch.testing.assert_close(output['latent'], manual, rtol=0, atol=0)
        self.assertFalse(torch.equal(one['latent'], output['latent']))
        for name, parameter in model.named_parameters():
            torch.testing.assert_close(parameter, before[name], rtol=0, atol=0)

    def test_node_projections_once_are_equivalent_to_per_edge_projection(self):
        config, batch = fixture()
        cell = TacticalOperator(config).cell
        latent = torch.randn(2, 71, 8)
        normalized = cell.norm(latent)
        endpoints = batch.edge_index.clamp(0, 70)
        index = endpoints[:, :, 0, None].expand(-1, -1, 8)
        # Independent arithmetic control for the optimization, with GEMM-order tolerance.
        torch.testing.assert_close(cell.source(normalized).gather(1, index),
                                   cell.source(normalized.gather(1, index)), rtol=2e-6, atol=2e-7)
        index = endpoints[:, :, 1, None].expand(-1, -1, 8)
        torch.testing.assert_close(cell.receiver(normalized).gather(1, index),
                                   cell.receiver(normalized.gather(1, index)), rtol=2e-6, atol=2e-7)
        calls = []
        hooks = [module.register_forward_pre_hook(lambda module, args: calls.append(tuple(args[0].shape)))
                 for module in (cell.source, cell.receiver)]
        try:
            mask = torch.ones(2, 71, dtype=torch.bool)
            cell(latent, mask, batch)
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(calls, [(2, 71, 8), (2, 71, 8)])

    def test_typed_control_removes_only_relation_information_and_gradients_flow(self):
        config, batch = fixture()
        model = TacticalOperator(config)
        other_kinds = batch.edge_kind.clone()
        other_kinds[:, 5] = Relation.ENEMY_ATTACK_OWN
        changed = replace(batch, edge_kind=other_kinds)
        with torch.no_grad():
            model.value.weight.normal_(0, .1)
            model.action.weight.normal_(0, .1)
        with NoHostScalar():
            plain = model(batch, typed=False)
            plain_changed = model(changed, typed=False)
            typed = model(batch)
            typed_changed = model(changed)
            (typed['value_cp'].sum() + typed['action_scores_cp'][batch.action_mask].sum()).backward()
        torch.testing.assert_close(plain['latent'], plain_changed['latent'], rtol=0, atol=0)
        self.assertFalse(torch.equal(typed['latent'], typed_changed['latent']))
        for parameter in (model.piece_input.weight, model.square_input.weight, model.action_input.weight,
                          model.cell.relation.weight, model.cell.evidence.weight, model.cell.update[0].weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)

    def test_padding_is_ignored_even_nan_payload_and_invalid_indices(self):
        config, batch = fixture()
        model = TacticalOperator(config)
        poisoned = replace(batch, pieces=batch.pieces.clone(), actions=batch.actions.clone(),
                           edge_features=batch.edge_features.clone(), action_prior_cp=batch.action_prior_cp.clone())
        poisoned.pieces[~batch.piece_mask] = torch.nan
        poisoned.actions[~batch.action_mask] = torch.nan
        poisoned.edge_features[~batch.edge_mask] = torch.nan
        poisoned.action_prior_cp[~batch.action_mask] = torch.nan
        admit_contents(poisoned, config)
        with NoHostScalar():
            expected, actual = model(batch), model(poisoned)
        for name in ('latent', 'value_cp', 'action_scores_cp', 'action_probabilities'):
            torch.testing.assert_close(actual[name], expected[name], rtol=0, atol=0)
        self.assertEqual(float(actual['latent'][:, 2].detach().abs().sum()), 0)
        self.assertEqual(float(actual['latent'][:, -1].detach().abs().sum()), 0)

    def test_zero_heads_preserve_priors_canonical_ties_and_terminal_rows(self):
        config, batch = fixture()
        # All action-touching edges in row1 must disappear for this terminal fixture.
        mask = batch.action_mask.clone()
        mask[1] = False
        edges = batch.edge_mask.clone()
        edges[1, 2:5] = False
        terminal = replace(batch, action_mask=mask, edge_mask=edges)
        admit_contents(terminal, config)
        model = TacticalOperator(config)
        with NoHostScalar():
            output = model(terminal)
        torch.testing.assert_close(output['value_cp'], batch.value_prior_cp, rtol=0, atol=0)
        torch.testing.assert_close(output['action_scores_cp'][mask], batch.action_prior_cp[mask], rtol=0, atol=0)
        self.assertEqual(output['canonical_action'].tolist(), [5, -1])
        self.assertEqual(float(output['action_probabilities'][1].detach().sum()), 0)
        self.assertTrue(torch.isneginf(output['action_scores_cp'][1]).all())
        empty = replace(terminal, actions=terminal.actions[:, :0], action_mask=mask[:, :0],
                        action_ids=terminal.action_ids[:, :0], action_prior_cp=terminal.action_prior_cp[:, :0],
                        edge_index=terminal.edge_index[:, :0], edge_kind=terminal.edge_kind[:, :0],
                        edge_mask=terminal.edge_mask[:, :0], edge_features=terminal.edge_features[:, :0])
        with NoHostScalar():
            empty_output = model(empty)
        self.assertEqual(empty_output['canonical_action'].tolist(), [-1, -1])
        self.assertEqual(tuple(empty_output['action_probabilities'].shape), (2, 0))

    def test_exact_child_control_uses_same_model_and_correct_parent_negation(self):
        config, parent = fixture()
        _, children = fixture(batch_size=6)
        children = replace(children, value_prior_cp=torch.tensor([4., -7., -7., 10., 20., 30.]))
        mapping = torch.tensor([[0, 1, 2, -1], [3, 4, 5, -1]])
        admit_child_mapping(parent, children, mapping, config)
        model = TacticalOperator(config)
        with NoHostScalar():
            expected = model(children)['value_cp']
            output = model.exact_child_control(parent, children, mapping)
        torch.testing.assert_close(output['action_scores_cp'][parent.action_mask],
                                   -expected[mapping[parent.action_mask]], rtol=0, atol=0)
        self.assertEqual(output['canonical_action'].tolist(), [10, 5])
        # Parent predicted latent/prior is not an exact-child input or substitute.
        changed_parent = replace(parent, value_prior_cp=torch.full((2,), 1e6), pieces=parent.pieces + 100)
        torch.testing.assert_close(model.exact_child_control(changed_parent, children, mapping)['action_scores_cp'],
                                   output['action_scores_cp'], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            admit_child_mapping(parent, children, torch.zeros_like(mapping), config)

    def test_cold_admission_rejects_content_errors_and_hot_metadata_errors(self):
        config, batch = fixture()
        model = TacticalOperator(config)
        mutations = []
        for name, index, value in (('edge_index', (0, 0, 1), 900),
                                   ('edge_kind', (0, 0), len(Relation)),
                                   ('piece_squares', (0, 1), 0), ('edge_overflow', (0,), True),
                                   ('action_ids', (0, 1), 5), ('actions', (0, 0, 0), torch.nan)):
            tensor = getattr(batch, name).clone()
            tensor[index] = value
            mutations.append(replace(batch, **{name: tensor}))
        # Node category valid but canonical source/destination wrong.
        for edge_slot, endpoint, bad_value in ((2, 0, 1), (3, 0, 3 + 2), (0, 1, 3 + 8)):
            edges = batch.edge_index.clone()
            edges[0, edge_slot, endpoint] = bad_value
            mutations.append(replace(batch, edge_index=edges))
        for malformed in mutations:
            with self.assertRaises(ValueError):
                admit_contents(malformed, config)
        with self.assertRaisesRegex(ValueError, 'coverage'):
            admit_contents(batch, config, expected_legal_actions=[[5, 10], [5, 10, 15]])
        for malformed in (replace(batch, squares=batch.squares.double()),
                          replace(batch, score_frame='child_mover_cp'),
                          replace(batch, prior_kind='teacher_cp'),
                          replace(batch, feature_schema='wrong')):
            with self.assertRaises(ValueError):
                model(malformed)
        for settings in (dict(steps=3), dict(steps=True), dict(width=7), dict(edge_channels=0),
                         dict(cp_scale=float('nan'))):
            with self.assertRaises(ValueError):
                TacticalConfig(**settings)
        with self.assertRaises(ValueError):
            model(batch, typed=1)

    def test_real_complete_legal_sets_black_orientation_ep_castling_promotion_terminal(self):
        fens = [chess.STARTING_FEN, 'r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1',
                '4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1',
                '4k3/P7/8/8/8/8/7p/4K3 b - - 0 1',
                '7k/6Q1/5K2/8/8/8/8/8 b - - 0 1']
        for fen in fens:
            board = chess.Board(fen)
            config, batch, legal = board_fixture(board)
            admit_contents(batch, config, expected_legal_actions=[legal])
            with NoHostScalar():
                output = TacticalOperator(config)(batch)
            self.assertEqual(output['canonical_action'].tolist(), [min(legal) if legal else -1])
            if legal:
                trimmed = replace(batch, action_mask=torch.zeros_like(batch.action_mask),
                                  edge_mask=torch.zeros_like(batch.edge_mask))
                with self.assertRaisesRegex(ValueError, 'coverage'):
                    admit_contents(trimmed, config, expected_legal_actions=[legal])

    def test_allocation_free_scaling_and_explicit_caller_budget(self):
        config, _ = fixture()
        small = TacticalOperator(config)
        self.assertEqual(config.estimate()['parameters'], sum(p.numel() for p in small.parameters()))
        self.assertEqual(config.estimate()['processor_parameters'], sum(p.numel() for p in small.cell.parameters()))
        large = replace(config, width=4096)
        with patch('neural.tactical_operator.nn.Linear', side_effect=AssertionError('allocated')):
            descriptor = large.estimate()
            self.assertFalse(descriptor['allocated'])
            self.assertGreater(descriptor['parameters'], 250000000)
            with self.assertRaisesRegex(ValueError, 'budget exceeded'):
                TacticalOperator(large, max_parameters=1000)
        exact = config.estimate()['parameters']
        self.assertEqual(TacticalOperator(config, max_parameters=exact).parameter_report()['max_parameters'], exact)


if __name__ == '__main__':
    unittest.main()
