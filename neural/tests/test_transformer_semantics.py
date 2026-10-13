"""CPU mechanics and shared-control tests, without training or strength claims."""
import copy
import math
import unittest
from unittest.mock import patch

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.features import encode_fen
from neural.semantic_strategy import anchor_losses
from neural.transformer_semantics import (MultiheadAttention, TransformerConfig,
                                         TransformerSemantics, scaling_profiles)
from neural.value_heads import material_cp


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError('host scalar extraction')
        return function(*args, **(kwargs or {}))


class MatmulMACs(TorchDispatchMode):
    """Independent executed shape accounting; no performance measurement."""
    def __init__(self):
        super().__init__()
        self.macs = 0

    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten.addmm.default:
            left, right = args[1:3]
            self.macs += left.shape[0] * left.shape[1] * right.shape[1]
        elif function is torch.ops.aten.mm.default:
            left, right = args[:2]
            self.macs += left.shape[0] * left.shape[1] * right.shape[1]
        elif function is torch.ops.aten.bmm.default:
            left, right = args[:2]
            self.macs += left.shape[0] * left.shape[1] * left.shape[2] * right.shape[2]
        return function(*args, **(kwargs or {}))


def features():
    return torch.stack([encode_fen(fen) for fen in (
        chess.STARTING_FEN, '4r1k1/8/8/8/8/8/4N3/4K3 w - - 17 1')])


class TransformerFixtures(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(8731)

    def model(self, **settings):
        return TransformerSemantics(TransformerConfig(width=8, heads=2, depth=1, **settings))

    def test_independent_heads_match_loop_oracle_and_exclude_masked_keys(self):
        module = MultiheadAttention(8, 2)
        query, context = torch.randn(2, 3, 8), torch.randn(2, 5, 8)
        bias = torch.randn(2, 2, 3, 5)
        keys = torch.tensor([[True, False, True, False, False],
                             [False, True, True, True, False]])
        expected = []
        for b in range(2):
            heads = []
            for h in range(2):
                part = slice(h * 4, (h + 1) * 4)
                q = module.query(query[b])[:, part]
                k = module.key(context[b])[:, part]
                v = module.value(context[b])[:, part]
                logits = (q @ k.T / math.sqrt(4) + bias[b, h])[:, keys[b]]
                heads.append(logits.softmax(-1) @ v[keys[b]])
            expected.append(module.output(torch.cat(heads, -1)))
        torch.testing.assert_close(module(query, context, bias=bias, keys=keys),
                                   torch.stack(expected), rtol=2e-6, atol=2e-6)
        changed = context.clone()
        changed[~keys] = 1e4
        torch.testing.assert_close(module(query, context, bias=bias, keys=keys),
                                   module(query, changed, bias=bias, keys=keys), rtol=0, atol=0)
        # Finite logits below a fixed mask sentinel must still exclude keys.
        extreme = torch.full_like(bias, -20000.)
        torch.testing.assert_close(module(query, context, bias=extreme, keys=keys),
                                   module(query, changed, bias=extreme, keys=keys), rtol=0, atol=0)
        with NoHostScalar():
            empty = module(query, context, keys=torch.zeros_like(keys))
        torch.testing.assert_close(empty, module.output.bias[None, None].expand_as(empty), rtol=0, atol=0)
        with self.assertRaises(ValueError):
            module(query, context[:1])

    def test_semantics_gradients_no_host_reads_and_material_prior(self):
        f = features()
        model = self.model(rule_bias=True)
        with NoHostScalar():
            result = model(f)
            losses = anchor_losses(model, result, f, torch.tensor([120., -80.]),
                                   torch.tensor([True, True]))
            sum(losses.values()).backward()
        torch.testing.assert_close(result['value_cp'], material_cp(f), rtol=0, atol=0)
        self.assertEqual(result['latent'].shape, (2, 8, 8))
        self.assertEqual(result['board_logits'].shape, (2, 13, 8, 8))
        for parameter in (model.queries, model.cp.weight,
                          model.layers[0].rule_weight, model.layers[0].geometry_weight):
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(float(parameter.grad.abs().sum()), 0)
        self.assertEqual(len(list(model.active_parameters())), len(list(model.parameters())))

    def test_value_only_is_exact_and_never_calls_semantic_decoders(self):
        model, f = self.model(), features()
        with torch.no_grad():
            model.cp.weight.normal_(0, .01)
            full = model(f)
            with patch.object(model.board, 'forward', side_effect=AssertionError('decoder ran')):
                value = model(f, value_only=True)
        for key in ('latent', 'pieces', 'value_cp', 'residual_cp', 'material_cp'):
            torch.testing.assert_close(value[key], full[key], rtol=0, atol=0)
        self.assertNotIn('board_logits', value)

    def test_symbolic_shared_zero_control_then_actual_residual_injection(self):
        f = features()
        base, injected = self.model(), self.model(symbolic_channels=3)
        with torch.no_grad():
            base.cp.weight.normal_(0, .01)
        injected.copy_shared_state(base)
        symbols = torch.randn(2, 64, 3, requires_grad=True)
        with NoHostScalar():
            baseline = base(f)
            controlled = injected(f, symbols)
        for key in baseline:
            torch.testing.assert_close(controlled[key], baseline[key], rtol=0, atol=0)
        with torch.no_grad():
            injected.symbolic_square.weight.normal_(0, .1)
            injected.symbolic_value.weight.normal_(0, .1)
        with NoHostScalar():
            output = injected(f, symbols)
            output['value_cp'].sum().backward()
        self.assertGreater(float((output['latent'] - baseline['latent']).detach().abs().max()), 1e-4)
        self.assertGreater(float(symbols.grad.abs().sum()), 0)
        self.assertGreater(float(injected.symbolic_square.weight.grad.abs().sum()), 0)
        self.assertGreater(float(injected.symbolic_value.weight.grad.abs().sum()), 0)
        with self.assertRaises(ValueError):
            injected(f)
        with self.assertRaises(ValueError):
            base(f, symbols)

    def test_piece_keys_preserve_empty_square_ep_context(self):
        model = self.model(piece_keys_only=True)
        f = encode_fen('4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 1')[None]
        absent = f.clone()
        absent[:, 16] = 0
        # Same pieces/rights/clock; only the genuinely empty EP target differs.
        self.assertEqual(float(f[:, :12, 5, 3].sum()), 0)
        self.assertFalse(torch.equal(model.encode(f), model.encode(absent)))
        with patch.object(model.ep_context, 'forward', side_effect=lambda x: torch.zeros_like(x)):
            torch.testing.assert_close(model.encode(f), model.encode(absent), rtol=0, atol=0)
        self.assertFalse(model.config()['token_compaction'])

    def test_shared_state_rejects_incompatible_controls_before_copy(self):
        target, source = self.model(), self.model()
        before = copy.deepcopy(target.state_dict())
        # A late tensor mismatch must not overwrite earlier shared parameters.
        source.cp = torch.nn.Linear(64, 2)
        with self.assertRaisesRegex(ValueError, 'shape or dtype'):
            target.copy_shared_state(source)
        for key, value in target.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        with self.assertRaisesRegex(ValueError, 'configuration'):
            target.copy_shared_state(TransformerSemantics(TransformerConfig(width=16)))

    def test_invalid_configuration_and_input_metadata(self):
        for settings in (dict(width=9, heads=2), dict(heads=0), dict(width=True),
                         dict(depth=0), dict(symbolic_channels=65), dict(rule_bias=1)):
            with self.assertRaises(ValueError):
                TransformerConfig(**settings)
        with self.assertRaises(ValueError):
            MultiheadAttention(8, 0)
        model = self.model()
        for invalid in (features().double(), torch.zeros(0, 18, 8, 8), torch.zeros(1, 17, 8, 8)):
            with self.assertRaises(ValueError):
                model(invalid)
        with self.assertRaises(ValueError):
            self.model(symbolic_channels=2)(features(), torch.zeros(2, 64, 3))

    def test_exact_size_reports_and_executed_matmul_counts_on_small_fixtures(self):
        for settings in (dict(width=8, heads=2, depth=1),
                         dict(width=12, heads=3, depth=2, expansion=2, geometry_bias=False),
                         dict(width=8, heads=1, depth=2, rule_bias=True, symbolic_channels=3)):
            config = TransformerConfig(**settings)
            model = TransformerSemantics(config)
            symbols = torch.randn(2, 64, 3) if config.symbolic_channels else None
            estimate = config.estimate(batch_size=2)
            self.assertEqual(estimate['stored_parameters'], sum(p.numel() for p in model.parameters()))
            self.assertEqual(estimate['value_path_parameters'],
                             sum(p.numel() for name, p in model.named_parameters()
                                 if not name.startswith(('board.', 'maps.', 'rights.', 'ep.', 'clock.'))))
            for value_only in (False, True):
                # Topology production is expressly excluded; substitute supplied zero edges.
                with patch.object(model.diagram, 'typed_edges', return_value=torch.zeros(2, 4, 64, 64)):
                    counter = MatmulMACs()
                    with counter:
                        model(features(), symbols, value_only=value_only)
                predicted = config.estimate(batch_size=2, value_only=value_only)
                self.assertEqual(counter.macs, predicted['forward_matmul_macs'])
            report = model.parameter_report(batch_size=2)
            self.assertTrue(report['allocated'])
            self.assertIsNone(report['measured_runtime'])

    def test_large_descriptors_allocate_nothing_and_explicit_budget_precedes_modules(self):
        with patch('neural.transformer_semantics.Diagram', side_effect=AssertionError('allocated')):
            profiles = scaling_profiles()
            large = TransformerConfig(width=2048, heads=32, depth=48)
            estimate = large.estimate(batch_size=32)
            self.assertGreater(estimate['value_path_parameters'], 1000000000)
            self.assertFalse(estimate['allocated'])
            self.assertEqual(len(profiles), 5)
            for profile in profiles:
                self.assertFalse(profile['allocated'])
                self.assertLess(abs(profile['value_path_parameters'] / profile['target_value_parameters'] - 1), .15)
            with self.assertRaisesRegex(ValueError, 'budget exceeded'):
                TransformerSemantics(large, max_parameters=1000)
        config = TransformerConfig(width=8, heads=2, depth=1)
        exact = config.estimate()['stored_parameters']
        model = TransformerSemantics(config, max_parameters=exact)
        self.assertEqual(model.parameter_report()['max_parameters'], exact)
        for invalid in (0, -1, True, 1.5):
            with self.assertRaises(ValueError):
                TransformerSemantics(config, max_parameters=invalid)

    def test_heads_separate_attention_workspace_from_dense_work(self):
        one = TransformerConfig(width=16, heads=1, depth=2, geometry_bias=False).estimate(batch_size=3)
        four = TransformerConfig(width=16, heads=4, depth=2, geometry_bias=False).estimate(batch_size=3)
        self.assertEqual(one['stored_parameters'], four['stored_parameters'])
        self.assertEqual(one['forward_matmul_macs'], four['forward_matmul_macs'])
        self.assertEqual(four['summed_attention_logit_elements'], 4 * one['summed_attention_logit_elements'])


if __name__ == '__main__':
    unittest.main()
