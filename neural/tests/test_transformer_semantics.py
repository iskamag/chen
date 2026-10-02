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
                                         TransformerSemantics)
from neural.value_heads import material_cp


class NoHostScalar(TorchDispatchMode):
    def __torch_dispatch__(self, function, types, args=(), kwargs=None):
        if function is torch.ops.aten._local_scalar_dense.default:
            raise AssertionError('host scalar extraction')
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


if __name__ == '__main__':
    unittest.main()
