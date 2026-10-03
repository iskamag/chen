"""Whole native producer + resident copy + captured inference comparison.

All active evidence comes directly from complete native child graphs. Payload
downloads/checkpoint hashing are cold checks, outside timing. Captured input
buffers are owned snapshots; each pipeline call exports a fresh snapshot, copies
its resident tensors into those static buffers, and replays the captured model.
"""
import argparse
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch

from neural.bridge import ResidentSpan, RuleArena, position_from_fen
from neural.native_tactical import (NativeTacticalProducer, NativeTacticalBucketProducer,
                                   TENSOR_FIELDS)
from neural.tactical_operator import TacticalConfig, TacticalOperator
from gpu.profile_tactical_padding import START, capture, compare
from gpu.test_native_tactical import ResidentOnly


def pipeline(producer, span, static, graph, events):
    torch.cuda.synchronize()
    started = time.perf_counter()
    snapshot = producer.export(span)
    exported = time.perf_counter()
    begin, copied, finished = events
    begin.record()
    for name in TENSOR_FIELDS:
        getattr(static.batch, name).copy_(getattr(snapshot.batch, name))
    copied.record()
    graph.replay()
    finished.record()
    torch.cuda.synchronize()
    ended = time.perf_counter()
    return dict(synchronized_pipeline_wall_ms=(ended-started)*1000,
                synchronized_export_wall_ms=(exported-started)*1000,
                copy_and_replay_wall_ms=(ended-exported)*1000,
                copy_event_ms=begin.elapsed_time(copied),
                captured_inference_event_ms=copied.elapsed_time(finished),
                native_event_ms=snapshot.manifest['kernel_event_ms'],
                edge_slots=snapshot.manifest['edge_slots'],
                retries=snapshot.manifest.get('retries', 0))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--library', required=True)
    parser.add_argument('--library-sha256', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--pairs', type=int, default=6)
    args = parser.parse_args()
    if not 2 <= args.pairs <= 12 or args.pairs % 2:
        parser.error('even pairs in [2,12] required')
    args.output.mkdir(exist_ok=False)
    write = lambda name, value: (args.output/name).write_text(json.dumps(value, indent=2)+'\n')
    torch.set_num_threads(1)
    rows = []
    started = time.monotonic()
    with RuleArena(args.library, capacity=128, device='cuda:0') as arena:
        producers = dict(abi1_E6144=NativeTacticalProducer(arena, args.library_sha256),
                         api2_E6144=NativeTacticalBucketProducer(arena, args.library_sha256, buckets=(6144,)),
                         api2_buckets=NativeTacticalBucketProducer(arena, args.library_sha256))
        reset_started = time.perf_counter()
        root = arena.reset(position_from_fen(START))
        children = arena.expand(ResidentSpan(arena.context_uuid, root['root_epoch'], 0, 1))['span']
        arena_setup_ms = (time.perf_counter()-reset_started)*1000
        if children.count != 20:
            raise RuntimeError('expected all20 native initial-board children')
        write('runtime.json', dict(library_sha256=args.library_sha256,
            runtime=producers['api2_buckets'].runtime_identity,
            device=producers['api2_buckets'].device_identity, arena_reset_expand_wall_ms=arena_setup_ms,
            policy='complete native graphs, no shape/payload download for admission',
            model='untrained FP32, random nonzero value/action residual heads', quality=False))
        for width in (32, 128, 256):
            torch.manual_seed(20261003+width)
            model_started = time.perf_counter()
            model = TacticalOperator(TacticalConfig(width=width, steps=4,
                feature_schema='chen_tactical_exact_v1')).to(arena.device).eval().requires_grad_(False)
            with torch.no_grad():
                model.value.weight.normal_(std=.01)
                model.action.weight.normal_(std=.01)
            torch.cuda.synchronize()
            model_setup_ms = (time.perf_counter()-model_started)*1000
            checkpoint = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                checkpoint.update(name.encode())
                checkpoint.update(tensor.cpu().contiguous().numpy().tobytes())
            for batch_size in (1, 20):
                span = ResidentSpan(arena.context_uuid, children.epoch, children.start, batch_size)
                static, captured, eager, setup, meters = {}, {}, {}, {}, {}
                for name, producer in producers.items():
                    export_started = time.perf_counter()
                    with ResidentOnly():
                        static[name] = producer.export(span)
                    first_export_ms = (time.perf_counter()-export_started)*1000
                    with torch.inference_mode(), ResidentOnly():
                        eager[name] = model(static[name].batch)
                        torch.cuda.synchronize()
                    capture_started = time.perf_counter()
                    with torch.inference_mode(), ResidentOnly():
                        captured[name] = capture(lambda state=static[name]: model(state.batch))
                    meters[name] = [torch.cuda.Event(enable_timing=True) for _ in range(3)]
                    setup[name] = dict(first_export_wall_ms=first_export_ms,
                        capture_warmup_and_setup_wall_ms=(time.perf_counter()-capture_started)*1000,
                        first_export_manifest=static[name].manifest)
                parity = {name:compare(eager['abi1_E6144'], eager[name]) for name in producers}
                for name, (graph, output) in captured.items():
                    with torch.inference_mode(), ResidentOnly():
                        graph.replay(); torch.cuda.synchronize()
                    parity[name+'_captured'] = compare(eager[name], output)
                record = dict(width=width, batch=batch_size, steps=4, dtype='float32',
                    checkpoint_sha256=checkpoint.hexdigest(), model_setup_wall_ms=model_setup_ms,
                    active_counts_cold_diagnostic=static['api2_buckets'].sidecar['slot_counts'].cpu().tolist(),
                    setup=setup, prediction_parity_max_abs=parity, samples=[])
                # Warm the complete measured pipeline, then balanced reverse-order
                # paired rounds. All three arms use the SAME pinned native binary.
                with torch.inference_mode(), ResidentOnly():
                    for name in producers:
                        pipeline(producers[name], span, static[name], captured[name][0], meters[name])
                    names = tuple(producers)
                    for pair in range(args.pairs):
                        order = names if pair % 2 == 0 else names[::-1]
                        sample = dict(pair=pair, order=order, arms={})
                        for name in order:
                            sample['arms'][name] = pipeline(producers[name], span, static[name], captured[name][0], meters[name])
                        record['samples'].append(sample)
                # Verify that fresh native outputs/copies, not just initial capture
                # inputs, fed the final replay of every arm.
                record['pipeline_prediction_parity_max_abs'] = {
                    name:compare(eager[name], captured[name][1]) for name in producers}
                fields = ('synchronized_pipeline_wall_ms', 'synchronized_export_wall_ms',
                          'copy_and_replay_wall_ms', 'copy_event_ms',
                          'captured_inference_event_ms', 'native_event_ms')
                record['median_ms'] = {name:{field:statistics.median(
                    sample['arms'][name][field] for sample in record['samples']) for field in fields}
                    for name in producers}
                record['paired_pipeline_control_over_bucket_ratios'] = [
                    sample['arms']['abi1_E6144']['synchronized_pipeline_wall_ms'] /
                    sample['arms']['api2_buckets']['synchronized_pipeline_wall_ms'] for sample in record['samples']]
                rows.append(record)
                write('results.json', dict(status='RUNNING', rows=rows))
                print(json.dumps(dict(width=width, batch=batch_size, median_ms=record['median_ms'],
                    median_paired_pipeline_ratio=statistics.median(record['paired_pipeline_control_over_bucket_ratios']))), flush=True)
                del static, captured, eager, setup, meters
                torch.cuda.synchronize(); torch.cuda.empty_cache()
            del model
    write('results.json', dict(status='PASS', rows=rows, elapsed_seconds=time.monotonic()-started,
        scope='Producer allocations/export/admission plus resident copy and captured inference; reset/expand/model/capture/checkpoint hashing/cold validation separately excluded. Untrained same-opening diagnostic, six paired rounds, shared clocks/power uncontrolled; no training, search throughput, strength or general speed claim.'))


if __name__ == '__main__':
    main()
