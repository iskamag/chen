"""Focused actual GPU complete-bucket/retry oracle; no training or rating."""
import ctypes as c
import unittest

import torch

from neural.bridge import BridgeError
from neural.native_symbolic import SymbolicExport
from neural.native_tactical import (FIELDS, NativeTacticalBucketProducer,
                                   TacticalSink, bucket_schema)
from neural.tactical_inputs import ReplaySpec, to_tactical_batch
from gpu import test_native_tactical as reference


class TacticalBucketGPU(reference.TacticalGPU):
    def setUp(self):
        super().setUp()
        self.control = self.producer
        self.producer = NativeTacticalBucketProducer(self.arena, self.pin)

    def test_preflight_preserves_outputs(self):
        # Reuse old ABI1 admission fixtures against the unchanged old entry.
        bucket = self.producer
        self.producer = self.control
        try:
            super().test_preflight_preserves_outputs()
        finally:
            self.producer = bucket

    def test_api2_admission_and_partial_capacity_metadata(self):
        _, span = self.reset(ReplaySpec())
        schema = bucket_schema(1)
        outputs = {name: torch.full((1, *shape), 1, dtype=dtype, device=self.arena.device)
                   for name, (shape, dtype) in schema.items()}
        sink = TacticalSink(2, 192, 0, 0, 1, 256, 1,
                            *(outputs[name].data_ptr() for name in FIELDS))
        for field, value in (('abi_version', 1), ('struct_size', 8), ('reserved', 1),
                             ('device', 1), ('legal_slots', 255), ('edge_slots', 0),
                             ('capacity', 0), ('actions', sink.pieces),
                             ('edge_kind', sink.edge_kind + 1)):
            bad = TacticalSink.from_buffer_copy(sink)
            setattr(bad, field, value)
            metadata = SymbolicExport(123, 123, 123)
            with self.assertRaises(BridgeError) as failure:
                self.arena._call('export_tactical_bucket', 0, 1, span.epoch,
                                 c.byref(bad), c.byref(metadata))
            self.assertEqual(failure.exception.code, -1)
            self.assertEqual((metadata.rows, metadata.root_epoch, metadata.kernel_ms), (123, 123, 123))
            for tensor in outputs.values():
                self.assertTrue(bool((tensor == 1).all()))
        metadata = SymbolicExport(123, 123, 123)
        torch.cuda.synchronize(self.arena.device)
        with self.assertRaises(BridgeError) as failure:
            self.arena._call('export_tactical_bucket', 0, 1, span.epoch,
                             c.byref(sink), c.byref(metadata))
        self.assertEqual(failure.exception.code, -2)
        self.assertEqual((metadata.rows, metadata.root_epoch, metadata.kernel_ms), (123, 123, 123))
        self.assertTrue(bool(outputs['edge_overflow'][0]))
        # This is a discarded packet. No contents or counts are admitted.
        self.assertFalse(self.arena.closed)

    def test_overflow_retry_exact_boundary_and_owned_snapshot(self):
        state, span = self.reset(ReplaySpec())
        producer = NativeTacticalBucketProducer(self.arena, self.pin, buckets=(1, 128, 512))
        with reference.ResidentOnly():
            snapshot = producer.export(span)
            control = self.control.export(span)
        self.compare(snapshot, [state], span)
        self.assertEqual([x['status'] for x in snapshot.manifest['attempts']], [-2, -2, 0])
        self.assertEqual(snapshot.manifest['edge_slots'], 512)
        self.assertFalse(snapshot.manifest['count_download'])
        # Oracle cardinality is a cold test input, not the producer's admission.
        exact_capacity = int(to_tactical_batch([state]).edge_mask.sum())
        exact = NativeTacticalBucketProducer(self.arena, self.pin, buckets=(exact_capacity,))
        with reference.ResidentOnly():
            exact_snapshot = exact.export(span)
        self.compare(exact_snapshot, [state], span)
        too_small = NativeTacticalBucketProducer(self.arena, self.pin, buckets=(exact_capacity-1,))
        with reference.ResidentOnly(), self.assertRaises(BridgeError) as failure:
            too_small.export(span)
        self.assertEqual(failure.exception.code, -2)
        exhausted = NativeTacticalBucketProducer(self.arena, self.pin, buckets=(1, 2, 3))
        with reference.ResidentOnly(), self.assertRaises(BridgeError) as failure:
            exhausted.export(span)
        self.assertEqual(failure.exception.code, -2)
        # Previous owned snapshots remain complete and unchanged after failures.
        self.compare(snapshot, [state], span)
        self.compare(control, [state], span)
        for name in FIELDS[:14]:
            a, b = getattr(snapshot.batch, name), getattr(control.batch, name)
            if name in ('edge_index', 'edge_kind', 'edge_mask', 'edge_features'):
                b = b[:, :512]
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        self.reset(ReplaySpec(moves=('e2e4',)))
        self.compare(snapshot, [state], span)
        with self.assertRaises(ValueError):
            producer.export(span)


if __name__ == '__main__':
    unittest.main()
