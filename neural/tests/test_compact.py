"""Exact-state and resident-hot-path proofs for compact replay; no strength test."""
import os
import unittest

import chess
import torch
from torch.utils._python_dispatch import TorchDispatchMode

from neural.compact import (CompactReplay, benchmark, compact_corpus, decode_batch,
                            decode_features, decode_legal_mask, pack_fens, unsigned_clock)
from neural.data import DeviceReplay, generate_synthetic
from neural.features import ACTION_COUNT, encode_fen, index_to_move
from neural.models import MODELS, build_model
from neural.train import loss_terms


DEVICE = os.environ.get("CHEN_TEST_DEVICE", "cpu")


class ResidentOnly(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = kwargs or {}
        if "_local_scalar_dense" in str(func):
            raise AssertionError("hot path requested a scalar host read")
        if "_to_copy" in str(func) and str(kwargs.get("device", "")) == "cpu":
            raise AssertionError("hot path requested a device-to-host transfer")
        return func(*args, **kwargs)


class CompactTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cpu = generate_synthetic(24, seed=203, max_plies=8)
        cls.source = compact_corpus(cls.cpu).to(DEVICE)

    def subset(self, first, last):
        return {key: value[first:last] for key, value in self.source.tensors.items()}

    def test_gpu_decoder_exact_all_clocks_and_signed_bitboards(self):
        fens = []
        board = chess.Board()
        for side in (chess.WHITE, chess.BLACK):
            board.turn = side
            for clock in range(101):
                board.halfmove_clock = clock
                fens.append(board.fen(en_passant="fen"))
        board.halfmove_clock = 65535
        board.fullmove_number = 0xFFFFFFFF
        fens.append(board.fen(en_passant="fen"))
        pieces, state = pack_fens(fens)
        self.assertTrue(bool((pieces[:, 9] < 0).all()))  # Black h8 rook sets uint64 bit63
        self.assertEqual(unsigned_clock(state[-1:, 4]).item(), 0xFFFFFFFF)
        self.assertEqual(unsigned_clock(state[-1:, 3]).item(), 65535)
        expected = torch.stack([encode_fen(fen) for fen in fens])
        pieces, state = pieces.to(DEVICE), state.to(DEVICE)
        for dtype in (torch.float32, torch.float16, torch.bfloat16):
            with ResidentOnly():
                decoded = decode_features(pieces, state, dtype)
            self.assertTrue(torch.equal(decoded.cpu(), expected.to(dtype)))
            self.assertEqual(decoded.dtype, dtype)
            self.assertEqual(decoded.device, pieces.device)
        for field, bad in (("halfmove", "65536"), ("fullmove", "4294967296")):
            fields = chess.Board().fen().split()
            fields[4 if field == "halfmove" else 5] = bad
            with self.assertRaises(ValueError):
                pack_fens([" ".join(fields)])

    def test_rule_fixtures_raw_ep_castles_promotions_and_color_canonicalization(self):
        fens = [
            "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 19 47",
            "r3k2r/8/8/8/8/8/8/R3K2R b Kq - 20 47",
            "4k3/8/8/3pP3/8/8/8/4K3 w - d6 0 19",
            "4k3/8/8/8/3Pp3/8/8/4K3 b - d3 0 19",
            "1r5k/P7/8/8/8/8/8/7K w - - 0 1",
            "8/8/8/r4pPK/8/8/8/7k w - f6 0 1",  # pinned EP marker retained
        ]
        board = chess.Board()
        board.push_uci("e2e4")
        fens.append(board.fen(en_passant="fen"))  # ineffective raw e3 still encoded
        for rights in ("-", "K", "Q", "k", "q", "KQkq"):
            row = chess.Board().fen().split()
            row[2] = rights
            fens.append(" ".join(row))
        fens += [chess.Board(fen).mirror().fen(en_passant="fen") for fen in fens[:6]]
        pieces, state = pack_fens(fens)
        decoded = decode_features(pieces.to(DEVICE), state.to(DEVICE))
        expected = torch.stack([encode_fen(fen) for fen in fens]).to(DEVICE)
        self.assertTrue(torch.equal(decoded, expected))

    def test_dense_support_all_word_boundaries_sentinels_and_terminal(self):
        # Supports arbitrary valid labels to test decoding independent of rules.
        labels = [0, 1, 31, 32, 63, 64, 255, 256, ACTION_COUNT - 1]
        listed = torch.full((2, 16), -1, dtype=torch.int16, device=DEVICE)
        listed[0, :len(labels)] = torch.tensor(labels, dtype=torch.int16, device=DEVICE)
        counts = torch.tensor([len(labels), 0], dtype=torch.int32, device=DEVICE)
        with ResidentOnly():
            mask = decode_legal_mask(listed, counts)
        self.assertEqual(mask[0].nonzero().flatten().tolist(), labels)
        self.assertFalse(bool(mask[1].any()))

    def test_generated_support_each_action_and_successor_matches_rules_oracle(self):
        decoded = decode_batch(self.source.tensors)
        for key in ("features", "next_features", "legal_mask", "actions", "target_actions"):
            self.assertTrue(torch.equal(decoded[key].cpu(), self.cpu.tensors[key]), key)
        for row, record in enumerate(self.cpu.records):
            board = chess.Board(record["fen"])
            side = "w" if board.turn else "b"
            active = self.source.tensors["legal_actions"][row, :self.source.tensors["legal_counts"][row]].tolist()
            moves = {index_to_move(action, side) for action in active}
            self.assertEqual(moves, {move.uci() for move in board.legal_moves})
            board.push_uci(record["action"])
            self.assertEqual(board.fen(en_passant="fen"), record["next_fen"])

    def test_ring_wrap_and_oversized_transaction_keep_newest_records(self):
        replay = CompactReplay(7, DEVICE)
        replay.append(self.subset(0, 5))
        replay.append(self.subset(5, 10))
        self.assertEqual((replay.size, replay.cursor, replay.total_inserted), (7, 3, 10))
        self.assertEqual(sorted(replay.sequence_ids.tolist()), list(range(3, 10)))
        replay.append(self.subset(10, 24))
        self.assertEqual((replay.size, replay.cursor, replay.total_inserted), (7, 3, 24))
        self.assertEqual(sorted(replay.sequence_ids.tolist()), list(range(17, 24)))
        for row, identity in enumerate(replay.sequence_ids.tolist()):
            for key in ("pieces", "state", "next_pieces", "next_state"):
                self.assertTrue(torch.equal(replay.storage[key][row], self.source.tensors[key][identity]))

    def test_clear_keeps_ids_monotonic_and_gathered_leases_immutable(self):
        replay = CompactReplay(7, DEVICE)
        replay.append(self.subset(0, 7))
        pointers = {key: value.data_ptr() for key, value in replay.storage.items()}
        generator = torch.Generator(device=replay.device).manual_seed(5)
        with ResidentOnly():
            lease = replay.sample_compact(4, generator)
            decoded = decode_batch(lease)
        snapshot = {key: value.clone() for key, value in lease.items()}
        replay.clear()
        with self.assertRaises(ValueError):
            replay.sample(1)
        replay.append(self.subset(7, 14))
        self.assertEqual(sorted(replay.sequence_ids.tolist()), list(range(7, 14)))
        self.assertEqual(pointers, {key: value.data_ptr() for key, value in replay.storage.items()})
        for key in lease:
            self.assertTrue(torch.equal(lease[key], snapshot[key]))
        self.assertTrue(torch.equal(decoded["features"], decode_batch(lease)["features"]))

    def test_invalid_ingress_rejected_atomically_before_any_ring_mutation(self):
        replay = CompactReplay(7, DEVICE)
        replay.append(self.subset(0, 3))
        before = {key: value[:replay.size].clone() for key, value in replay.storage.items()}
        old_ids = replay.sequence_ids[:replay.size].clone()
        old_state = (replay.size, replay.cursor, replay.total_inserted)
        mutations = (
            lambda b: b.update(pieces=b["pieces"].float()),
            lambda b: b.update(state=b["state"][:, :4]),
            lambda b: b["state"][:, 0].fill_(2),
            lambda b: b["state"][:, 4].zero_(),
            lambda b: b["state"][:, 3].fill_(65536),
            lambda b: b["state"][:, 2].fill_(1),
            lambda b: b["pieces"][:, 5].fill_(1),  # overlaps rook on a1
            lambda b: b["next_pieces"][:, 11].zero_(),
            lambda b: b["history_handles"].fill_(-2),
            lambda b: b["legal_counts"].fill_(257),
            lambda b: b["legal_counts"].zero_(),
            lambda b: b["legal_actions"][:, 0].fill_(-1),
            lambda b: b["legal_actions"][:, 1].copy_(b["legal_actions"][:, 0]),
            lambda b: b["legal_actions"][:, -1].zero_(),
            lambda b: b["actions"].fill_(ACTION_COUNT - 1),
            lambda b: b["material_proxy"].fill_(float("nan")),
        )
        for mutate in mutations:
            bad = {key: value.clone() for key, value in self.subset(3, 5).items()}
            mutate(bad)
            with self.assertRaises(ValueError):
                replay.append(bad)
            self.assertEqual(old_state, (replay.size, replay.cursor, replay.total_inserted))
            self.assertTrue(torch.equal(replay.sequence_ids[:replay.size], old_ids))
            for key in before:
                self.assertTrue(torch.equal(replay.storage[key][:replay.size], before[key]), key)
        with self.assertRaises(ValueError):
            CompactReplay(7, DEVICE, max_bytes=1)
        with self.assertRaises(ValueError):
            compact_corpus(self.cpu, max_slots=1)
        replay.total_inserted = torch.iinfo(torch.int64).max - 1
        with self.assertRaises(ValueError):
            replay.append(self.subset(3, 5))

    def test_hot_sampling_device_dtype_and_seed_match_dense_reference(self):
        dense = DeviceReplay(24, DEVICE)
        dense.append({key: value.to(DEVICE) for key, value in self.cpu.tensors.items()})
        compact = CompactReplay(24, DEVICE)
        compact.append(self.source.tensors)
        dense_rng = torch.Generator(device=compact.device).manual_seed(82)
        compact_rng = torch.Generator(device=compact.device).manual_seed(82)
        expected = dense.sample(12, dense_rng)
        with ResidentOnly():
            actual = compact.sample(12, compact_rng)
        self.assertEqual(set(expected), set(actual))
        for key, value in actual.items():
            self.assertEqual(value.device, compact.device)
            self.assertEqual(value.dtype, expected[key].dtype)
            self.assertTrue(torch.equal(value, expected[key]), key)
        self.assertEqual(compact.bytes_per_record, 801)
        self.assertLess(compact.bytes_per_record * 35, dense.bytes_per_record)
        actual_bytes = sum(t.numel() * t.element_size() for t in compact.storage.values())
        actual_bytes += compact.sequence_ids.numel() * compact.sequence_ids.element_size()
        self.assertEqual(actual_bytes, compact.capacity * compact.bytes_per_record)

    def test_nonzero_history_handles_preserved_without_claiming_context(self):
        replay = CompactReplay(2, DEVICE)
        batch = {key: value.clone() for key, value in self.subset(0, 2).items()}
        batch["history_handles"][:] = torch.tensor([77, 88], device=DEVICE)
        batch["next_history_handles"][:] = torch.tensor([78, 89], device=DEVICE)
        replay.append(batch)
        sample = replay.sample_compact(8, torch.Generator(device=replay.device).manual_seed(2))
        for index, handle in zip(sample["sequence_ids"].tolist(), sample["history_handles"].tolist()):
            self.assertEqual(handle, [77, 88][index])
        self.assertIn("no repetition", replay.manifest()["history_scope"])

    def test_exact_unclipped_clock_bitpatterns_and_autograd_free_admission(self):
        replay = CompactReplay(2, DEVICE)
        batch = {key: value.clone() for key, value in self.subset(0, 2).items()}
        batch["state"][:, 3] = 65535
        batch["state"][:, 4] = -1  # exact uint32 maximum, not an invalid negative clock
        batch["next_state"][:, 4] = -2147483648  # exact uint32 2**31
        batch["material_proxy"].requires_grad_()
        replay.append(batch)
        self.assertFalse(replay.storage["material_proxy"].requires_grad)
        self.assertIsNone(replay.storage["material_proxy"].grad_fn)
        self.assertTrue(bool((unsigned_clock(replay.storage["state"][:, 3]) == 65535).all()))
        self.assertTrue(bool((unsigned_clock(replay.storage["state"][:, 4]) == 0xFFFFFFFF).all()))
        self.assertTrue(bool((unsigned_clock(replay.storage["next_state"][:, 4]) == 1 << 31).all()))
        with ResidentOnly():
            decoded = replay.sample(2, torch.Generator(device=replay.device).manual_seed(19))
        self.assertTrue(bool((decoded["features"][:, 17] == 1).all()))
        if replay.device.type == "cuda":
            bad = {key: value.clone() for key, value in self.subset(0, 2).items()}
            bad["state"] = bad["state"].cpu()
            with self.assertRaises(ValueError):
                replay.append(bad)

    def test_model_outputs_losses_and_gradients_equal_dense_all_architectures(self):
        batch = {key: value[:4].to(DEVICE) for key, value in self.cpu.tensors.items()}
        actual = decode_batch(self.subset(0, 4))
        for name in MODELS:
            torch.manual_seed(8)
            reference = build_model(name, width=8, depth=1).to(DEVICE)
            candidate = build_model(name, width=8, depth=1).to(DEVICE)
            candidate.load_state_dict(reference.state_dict())
            output_a = reference(batch["features"], batch["legal_mask"])
            output_b = candidate(actual["features"], actual["legal_mask"])
            for key in output_a:
                torch.testing.assert_close(output_a[key], output_b[key], atol=1e-6, rtol=1e-5)
            a, b = loss_terms(reference, batch), loss_terms(candidate, actual)
            for key in a:
                torch.testing.assert_close(a[key], b[key], atol=1e-6, rtol=1e-5)
            a["total"].backward()
            b["total"].backward()
            for (key, value), (_, other) in zip(reference.named_parameters(), candidate.named_parameters()):
                if value.grad is None:
                    self.assertIsNone(other.grad)
                else:
                    torch.testing.assert_close(value.grad, other.grad, atol=1e-6, rtol=1e-5, msg=lambda message: key + ": " + message)

    def test_benchmark_same_loss_schedule_no_persistent_duplicate(self):
        report = benchmark(device=DEVICE, capacity=24, source_samples=8, repeats=1,
                           batch_size=2, steps=1)
        self.assertFalse(report["strength_evidence"])
        a, b = report["results"]
        self.assertEqual(a["sample_sequence_sha256"], b["sample_sequence_sha256"])
        self.assertEqual(a["first_loss"], b["first_loss"])
        self.assertEqual(a["final_loss"], b["final_loss"])
        self.assertEqual(b["replay"]["allocated_payload_bytes"], 24 * 801)


if __name__ == "__main__":
    unittest.main()
