#!/usr/bin/env python3
"""
Benchmark an ONNX model or the full PlantDiagEdge pipeline on an edge device.

Two modes, same methodology as benchmark_full.py (warmup excluded, garbage
collection disabled around the timed call only, fixed input seed, process RSS,
per-run latency distribution):

  model     one ONNX model through ONNX Runtime on a seeded input; records an
            output signature (top-5 indices + checksum) like the TFLite path.
  pipeline  the whole PlantDiagEdge pipeline of one configuration (C1–C6) from an
            artefact bundle, per stage: backbone inference, domain adaptation,
            gallery retrieval, optional heatmap, end to end. Inputs are the
            bundle's sample photographs (or --images), decoded before timing.

`pipeline` needs the plantdiag_edge package: installed, or its source passed with
--plantdiag-src. Both modes run in the agent environment of benchmark_full.py, whose
helpers they reuse (latency statistics, device info, system monitor) — including its
import-time check for a TFLite runtime, present on every edge-bench device.

Usage:
    python3 benchmark_pipeline.py model --model c6_mobilenet_v2_int8.onnx --runs 100
    python3 benchmark_pipeline.py pipeline --bundle ~/plantdiag-bundle --config c6 --runs 100 \\
        --plantdiag-src ~/plantdiag-edge
"""

import argparse
from datetime import UTC, datetime
import gc
import json
import os
from pathlib import Path
import sys
import time

from benchmark_full import (
    SystemMonitor,
    compute_latency_stats,
    file_hash,
    get_device_info,
)
import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def rss_mb() -> float | None:
    return round(psutil.Process().memory_info().rss / 1e6, 1) if HAS_PSUTIL else None


def timed_runs(fn, inputs: list, warmup: int, runs: int) -> tuple[list, list]:
    """Call fn on inputs (cycled); return (warmup outputs+latencies, measured outputs+latencies)."""
    warm, meas = [], []
    for i in range(warmup + runs):
        x = inputs[i % len(inputs)]
        gc_was_enabled = gc.isenabled()
        gc.disable()
        t0 = time.perf_counter()
        out = fn(x)
        ms = (time.perf_counter() - t0) * 1000
        if gc_was_enabled:
            gc.enable()
        (warm if i < warmup else meas).append((out, ms))
    return warm, meas


def run_model(args) -> dict:
    import onnxruntime as ort

    opts = ort.SessionOptions()
    if args.threads:
        opts.intra_op_num_threads = args.threads
    t0 = time.perf_counter()
    sess = ort.InferenceSession(
        args.model, sess_options=opts, providers=['CPUExecutionProvider']
    )
    load_ms = (time.perf_counter() - t0) * 1000
    inp = sess.get_inputs()[0]
    shape = [d if isinstance(d, int) else 1 for d in inp.shape]
    rng = np.random.default_rng(args.seed)
    x = rng.standard_normal(shape).astype(np.float32)

    monitor = SystemMonitor()
    monitor.start()
    warm, meas = timed_runs(
        lambda v: sess.run(None, {inp.name: v})[0], [x], args.warmup, args.runs
    )
    system = monitor.stop()
    lat = np.array([ms for _, ms in meas])
    out = np.asarray(meas[-1][0]).ravel()
    return {
        'mode': 'model',
        'model': {
            'path': args.model,
            'sha256_16': file_hash(args.model),
            'size_mb': round(os.path.getsize(args.model) / 1e6, 3),
            'input': {'name': inp.name, 'shape': shape},
        },
        'runtime': {'onnxruntime': ort.__version__, 'providers': sess.get_providers()},
        'params': {
            'warmup_runs': args.warmup,
            'benchmark_runs': args.runs,
            'threads': args.threads,
            'input_seed': args.seed,
        },
        'latency': {
            **compute_latency_stats(lat),
            'all_values_ms': [round(v, 3) for v in lat.tolist()],
        },
        'throughput': {'fps': round(1000.0 / float(np.mean(lat)), 2)},
        'cold_start': {
            'model_load_ms': round(load_ms, 2),
            'first_inference_ms': round(warm[0][1], 2) if warm else None,
        },
        'output_signature': {
            'top5_indices': [int(i) for i in np.argsort(-out)[:5]],
            'checksum': round(float(out.sum()), 4),
        },
        'process_rss_mb': rss_mb(),
        'system': system.to_dict(),
    }


def load_images(args, bundle) -> list:
    from PIL import Image

    folder = Path(args.images) if args.images else bundle.samples_dir
    if folder is None or not Path(folder).is_dir():
        raise SystemExit('no images: bundle has no samples/, pass --images DIR')
    files = sorted(
        p
        for p in Path(folder).iterdir()
        if p.suffix.lower() in {'.jpg', '.jpeg', '.png'}
    )
    if not files:
        raise SystemExit(f'no .jpg/.png images in {folder}')
    return [(p.name, Image.open(p).convert('RGB')) for p in files]


def run_pipeline(args) -> dict:
    if args.plantdiag_src:
        sys.path.insert(0, str(Path(args.plantdiag_src).expanduser()))
    import plantdiag_edge
    from plantdiag_edge.bundle import load_bundle
    from plantdiag_edge.config import get_config
    from plantdiag_edge.pipeline import Pipeline

    cfg = get_config(args.config)
    primary = load_bundle(Path(args.bundle).expanduser())
    bundle = next(
        (
            b
            for b in (primary, *primary.extra_bundles())
            if b.missing_for_config(cfg) is None
        ),
        None,
    )
    if bundle is None:
        raise SystemExit(f'{args.config}: {primary.missing_for_config(cfg)}')

    t0 = time.perf_counter()
    pipe = Pipeline.from_bundle(bundle, cfg, enable_xai=args.xai, top_k=5)
    load_ms = (time.perf_counter() - t0) * 1000
    images = load_images(args, primary)

    monitor = SystemMonitor()
    monitor.start()
    warm, meas = timed_runs(
        lambda im: (im[0], pipe.run(im[1], instrument=True)),
        images,
        args.warmup,
        args.runs,
    )
    system = monitor.stop()

    results = [r for (_, r), _ in meas]
    stages = {
        'inference': [r.timings.inference_ms for r in results],
        'domain_adaptation': [r.timings.domain_adapt_ms for r in results],
        'retrieval': [r.timings.retrieval_ms for r in results],
        'xai': [r.timings.xai_ms for r in results],
    }
    e2e = np.array([ms for _, ms in meas])
    model = bundle.int8_onnx if cfg.quantization.model_bits == 8 else bundle.fp32_onnx
    return {
        'mode': 'pipeline',
        'config': {
            'key': args.config,
            'backbone': cfg.backbone.name,
            'model_bits': cfg.quantization.model_bits,
            'polarquant': cfg.quantization.use_polarquant,
            'domain_adapter': cfg.domain_adapter.method,
            'xai': args.xai,
        },
        'bundle': {
            'root': str(bundle.root),
            'version': primary.version,
            'model': model.name,
            'model_sha256_16': file_hash(str(model)),
            'model_size_mb': round(model.stat().st_size / 1e6, 3),
        },
        'plantdiag_edge_version': getattr(plantdiag_edge, '__version__', None),
        'params': {
            'warmup_runs': args.warmup,
            'benchmark_runs': args.runs,
            'images': [n for n, _ in images],
        },
        'latency': {
            'end_to_end': compute_latency_stats(e2e),
            **{k: compute_latency_stats(np.array(v)) for k, v in stages.items()},
        },
        'throughput': {'fps': round(1000.0 / float(np.mean(e2e)), 2)},
        'cold_start': {
            'pipeline_load_ms': round(load_ms, 2),
            'first_run_ms': round(warm[0][1], 2) if warm else None,
        },
        'predictions': dict(
            sorted({n: r.predicted_label for (n, r), _ in meas}.items())
        ),
        'process_rss_mb': rss_mb(),
        'system': system.to_dict(),
        'note': 'end_to_end excludes image decoding (images are decoded before timing)',
    }


def main():
    parser = argparse.ArgumentParser(
        description='ONNX model / PlantDiagEdge pipeline benchmark'
    )
    sub = parser.add_subparsers(dest='mode', required=True)
    for name in ('model', 'pipeline'):
        p = sub.add_parser(name)
        p.add_argument('--warmup', type=int, default=10)
        p.add_argument('--runs', type=int, default=100)
        p.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    m = sub.choices['model']
    m.add_argument('--model', '-m', required=True, help='Path to .onnx model')
    m.add_argument(
        '--threads', type=int, default=0, help='intra-op threads (0 = runtime default)'
    )
    m.add_argument(
        '--seed', type=int, default=int(os.environ.get('EDGEBENCH_INPUT_SEED', '42'))
    )
    pl = sub.choices['pipeline']
    pl.add_argument(
        '--bundle', required=True, help='PlantDiagEdge artefact bundle directory'
    )
    pl.add_argument('--config', default='c6', help='c1 … c6')
    pl.add_argument(
        '--images', help='folder of .jpg/.png inputs (default: bundle samples/)'
    )
    pl.add_argument('--xai', action='store_true', help='include the heatmap stage')
    pl.add_argument('--plantdiag-src', help='path to a plantdiag-edge source checkout')
    args = parser.parse_args()

    start = time.perf_counter()
    results = {'timestamp': datetime.now(UTC).isoformat(), 'device': get_device_info()}
    try:
        results.update(run_model(args) if args.mode == 'model' else run_pipeline(args))
        results['status'] = 'completed'
    except Exception as e:  # noqa: BLE001 - reported in the JSON like benchmark_full.py
        results['status'] = 'failed'
        results['error'] = f'{type(e).__name__}: {e}'
    results['duration_seconds'] = round(time.perf_counter() - start, 2)
    output = json.dumps(results, indent=2, default=str)
    if args.output:
        Path(args.output).write_text(output)
    else:
        print(output)
    return 0 if results['status'] == 'completed' else 1


if __name__ == '__main__':
    sys.exit(main())
