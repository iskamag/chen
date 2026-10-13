"""CPU-only wrapper admission/retry/ownership; not native GPU correctness."""
import ctypes as c
from types import SimpleNamespace
import threading
import unittest
from unittest.mock import Mock, patch

import torch

from neural.bridge import BridgeError
from neural.native_symbolic import SymbolicExport
from neural.native_tactical import (NativeTacticalBucketProducer, TacticalSink,
                                   _UNRETIRED_BUCKETS, bucket_schema, sink_schema)


class BucketWrapperCPU(unittest.TestCase):
    def producer(self, buckets=(512, 1024, 6144)):
        producer = object.__new__(NativeTacticalBucketProducer)
        producer.buckets = buckets
        producer.arena = SimpleNamespace(_lock=threading.RLock())
        producer._source_locked = Mock(return_value=SimpleNamespace(start=0, count=1, epoch=7))
        producer._snapshot_locked = Mock(side_effect=lambda *args, **kw: kw)
        return producer

    def test_old_layout_and_capacity_are_preserved(self):
        self.assertEqual(c.sizeof(TacticalSink), 192)
        with self.assertRaises(ValueError):
            sink_schema(512)
        self.assertEqual(sink_schema(6144), bucket_schema(6144))
        self.assertEqual(bucket_schema(1)['edge_features'][0], (1, 4))
        for value in (0, -1, 65537, True, 512.0):
            with self.assertRaises(ValueError):
                bucket_schema(value)

    def test_policy_admission(self):
        with patch.object(NativeTacticalBucketProducer, '_initialize') as initialize:
            for invalid in ((), [], (512, 512), (1024, 512), (True,), (65537,)):
                with self.assertRaises(ValueError):
                    NativeTacticalBucketProducer(None, '', buckets=invalid)
            NativeTacticalBucketProducer(None, '', buckets=(1, 512, 6144))
            self.assertEqual(initialize.call_count, 1)

    def test_complete_retry_same_source_under_lock(self):
        producer = self.producer()
        sources = []
        def emit(source, size):
            self.assertTrue(producer.arena._lock._is_owned())
            sources.append(source)
            if size == 512:
                raise BridgeError('export_tactical_bucket', -2, 'edge capacity')
            return {'complete': True}, SymbolicExport(1, 7, .5)
        producer._emit_locked = Mock(side_effect=emit)
        result = producer.export('span')
        self.assertEqual([x['status'] for x in result['attempts']], [-2, 0])
        self.assertEqual([x['edge_slots'] for x in result['attempts']], [512, 1024])
        self.assertEqual(result['retries'], 1)
        self.assertFalse(result['count_download'])
        self.assertIs(sources[0], sources[1])
        self.assertEqual(producer._snapshot_locked.call_count, 1)

    def test_noncapacity_and_unretired_never_retry(self):
        for code in (-1, -3):
            producer = self.producer()
            producer._emit_locked = Mock(side_effect=BridgeError('export_tactical_bucket', code, 'failure'))
            with self.assertRaises(BridgeError) as failure:
                producer.export('span')
            self.assertEqual(failure.exception.code, code)
            self.assertEqual(producer._emit_locked.call_count, 1)
            producer._snapshot_locked.assert_not_called()

    def test_exhausted_buckets_publish_nothing(self):
        producer = self.producer()
        producer._emit_locked = Mock(side_effect=BridgeError('export_tactical_bucket', -2, 'capacity'))
        with self.assertRaises(BridgeError):
            producer.export('span')
        self.assertEqual(producer._emit_locked.call_count, 3)
        producer._snapshot_locked.assert_not_called()

    def test_unretired_storage_quarantined_and_arena_closed(self):
        producer = self.producer()
        producer.abi, producer.device, producer.operation = 2, SimpleNamespace(index=0), 'export_tactical_bucket'
        producer.arena._ctx, producer.arena._root_manifest = 42, {'root_epoch': 7}
        producer.arena._call = Mock(side_effect=BridgeError('export_tactical_bucket', -3, 'unretired'))
        source = SimpleNamespace(start=0, count=1, epoch=7)
        before = len(_UNRETIRED_BUCKETS)
        empty = torch.empty
        try:
            with patch('neural.native_tactical.torch.empty', side_effect=lambda shape, **kw:
                       empty(shape, dtype=kw['dtype'], device='cpu')), \
                 patch('neural.native_tactical.validate_output_tensors'), \
                 patch('neural.native_tactical.torch.cuda.synchronize'):
                with self.assertRaises(BridgeError):
                    producer._emit_locked(source, 1)
            self.assertIsNone(producer.arena._ctx)
            self.assertIsNone(producer.arena._root_manifest)
            self.assertEqual(len(_UNRETIRED_BUCKETS), before + 1)
            arena, context, outputs = _UNRETIRED_BUCKETS[-1]
            self.assertIs(arena, producer.arena)
            self.assertEqual(context, 42)
            self.assertEqual(len(outputs), 20)
        finally:
            # CPU fixture has no live native writes; actual unretired storage
            # must remain quarantined until process retirement.
            del _UNRETIRED_BUCKETS[before:]


if __name__ == '__main__':
    unittest.main()
