"""Causal padding intervention on resident tactical graph inference; no learning.

Counts/payload assertions are cold diagnostics. Timing excludes export, compact
copy, checkpoint hashing and validation. Tight storage is a measured intervention,
not an implemented native dynamic-capacity/capture policy.
"""
import argparse
from dataclasses import fields, replace
import hashlib
import json
from pathlib import Path
import statistics
import time

import torch
from neural.bridge import ResidentSpan, RuleArena, position_from_fen
from neural.native_tactical import NativeTacticalProducer
from neural.tactical_operator import TacticalConfig, TacticalOperator
from gpu.test_native_tactical import ResidentOnly

START = 'rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1'
EDGE_FIELDS = ('edge_index', 'edge_kind', 'edge_mask', 'edge_features')


def subset(batch, count, edge_capacity=None):
    changes = {}
    for field in fields(batch):
        tensor = getattr(batch, field.name)
        if isinstance(tensor, torch.Tensor):
            tensor = tensor[:count]
            if field.name in EDGE_FIELDS and edge_capacity is not None:
                tensor = tensor[:, :edge_capacity]
            changes[field.name] = tensor.contiguous()
    return replace(batch, **changes)


def measure(call, samples, inner):
    for _ in range(3):
        call()
    torch.cuda.synchronize()
    events, walls = [], []
    begin, end = [torch.cuda.Event(enable_timing=True) for _ in range(2)]
    for _ in range(samples):
        torch.cuda.synchronize()
        started = time.perf_counter()
        begin.record()
        for _ in range(inner):
            call()
        end.record()
        torch.cuda.synchronize()
        walls.append((time.perf_counter() - started)*1000/inner)
        events.append(begin.elapsed_time(end)/inner)
    return dict(event_ms=events, synchronized_wall_ms=walls,
                event_median_ms=statistics.median(events), wall_median_ms=statistics.median(walls))


def capture(call):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            call()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        result = call()
    torch.cuda.synchronize()
    return graph, result


def compare(reference, candidate):
    errors = {}
    for name in reference:
        a, b = reference[name], candidate[name]
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-4)
        if a.dtype.is_floating_point:
            valid = torch.isfinite(a) & torch.isfinite(b)
            errors[name] = (a[valid]-b[valid]).abs().max().item() if bool(valid.any()) else 0.
        else:
            errors[name] = 0.
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--library', required=True)
    p.add_argument('--library-sha256', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--samples', type=int, default=3)
    p.add_argument('--inner', type=int, default=3)
    args = p.parse_args()
    if not 1 <= args.samples <= 31 or not 1 <= args.inner <= 31:
        p.error('positive bounded samples/inner required')
    args.output.mkdir(exist_ok=False)
    torch.set_num_threads(1)
    started = time.monotonic()
    rows = []
    with RuleArena(args.library, capacity=128, device='cuda:0') as arena:
        producer = NativeTacticalProducer(arena, args.library_sha256)
        root = arena.reset(position_from_fen(START))
        children = arena.expand(ResidentSpan(arena.context_uuid, root['root_epoch'], 0, 1))['span']
        snapshot = producer.export(children)
        if children.count != 20:
            raise RuntimeError('complete standard-start child support must contain20 rows')
        # Cold shape intervention; no teacher labels or reconstructed graph input.
        counts = snapshot.sidecar['slot_counts'].cpu().tolist()
        write = lambda name, x: (args.output/name).write_text(json.dumps(x,indent=2)+'\n')
        write('runtime.json', dict(runtime=producer.runtime_identity, device=producer.device_identity,
            native_manifest=snapshot.manifest, roots='all20 native children of standard initial board',
            slot_counts=counts, training=False, quality=False, graph_capacity_policy='cold diagnostic intervention only'))
        for width in (32, 128, 256):
            torch.manual_seed(20261003+width)
            model = TacticalOperator(TacticalConfig(width=width, steps=4,
                feature_schema=snapshot.batch.feature_schema)).to(arena.device).eval().requires_grad_(False)
            with torch.no_grad():
                model.value.weight.normal_(std=.01)
                model.action.weight.normal_(std=.01)
            checkpoint = hashlib.sha256()
            for name, tensor in model.state_dict().items():
                checkpoint.update(name.encode())
                checkpoint.update(tensor.cpu().contiguous().numpy().tobytes())
            for count in (1, 20):
                edges = max(row[2] for row in counts[:count])
                batches = dict(full=subset(snapshot.batch,count), tight=subset(snapshot.batch,count,edges))
                record = dict(width=width, steps=4, batch=count, edge_capacity_full=6144,
                    edge_capacity_tight=edges, active_edges=sum(r[2] for r in counts[:count]),
                    parameters=model.parameter_report()['actual_parameters'], checkpoint_sha256=checkpoint.hexdigest(),
                    repetitions=args.samples, calls_per_sample=args.inner, timing={})
                with torch.inference_mode(), ResidentOnly():
                    eager = {name:model(batch) for name,batch in batches.items()}
                record['eager_parity_max_abs'] = compare(eager['full'],eager['tight'])
                with torch.inference_mode(), ResidentOnly():
                    captured = {name:capture(lambda batch=batch:model(batch)) for name,batch in batches.items()}
                record['captured_parity_max_abs'] = {}
                for name,(graph,output) in captured.items():
                    with torch.inference_mode(), ResidentOnly():
                        graph.replay()
                        torch.cuda.synchronize()
                    record['captured_parity_max_abs'][name] = compare(eager[name],output)
                # Alternate policy order by shape; raw small-sample timing retained.
                order = ('full','tight') if (width == 128) != (count == 20) else ('tight','full')
                for execution in ('eager','captured'):
                    for name in order:
                        call = (lambda batch=batches[name]:model(batch)) if execution=='eager' else captured[name][0].replay
                        with torch.inference_mode(), ResidentOnly():
                            record['timing'][execution+'_'+name] = measure(call,args.samples,args.inner)
                for execution in ('eager','captured'):
                    times=record['timing']
                    record[execution+'_padding_speed_ratio']=times[execution+'_full']['event_median_ms']/times[execution+'_tight']['event_median_ms']
                rows.append(record)
                write('results.json',dict(status='RUNNING',rows=rows))
                print(json.dumps(dict(width=width,batch=count,edges=edges,
                    eager_ratio=record['eager_padding_speed_ratio'],captured_ratio=record['captured_padding_speed_ratio'])),flush=True)
                del eager, captured, batches
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            del model
    write('results.json',dict(status='PASS',rows=rows,elapsed_seconds=time.monotonic()-started,
        limitation='Inference-only cold padding intervention; native export/copy/setup costs excluded; random untrained weights; three small samples, clocks/power uncontrolled; no strength or native compaction implementation.'))


if __name__ == '__main__':
    main()
