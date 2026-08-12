#!/usr/bin/env python3
"""Short end-to-end validation of the real benchmark pipeline.

This is a *smoke* benchmark, not a scientific measurement: it uses few
iterations so it finishes in seconds. Its purpose is to prove that

    model loading -> device detection -> inference -> timing
    -> memory measurement -> result serialization

works on the current host with a real TFLite runtime and a real model file.
It drives ``agent.executor.BenchmarkExecutor`` — the same code path the server
uses — so a green smoke run exercises production code, not a stand-in.

Never run this on a device whose numbers are going into a paper: iteration
counts are far too low. Use agent/benchmark_full.py for that.

Usage:
    python scripts/benchmark_smoke.py --model data/models/foo.tflite
    python scripts/benchmark_smoke.py            # auto-discovers a model
"""

import argparse
import asyncio
from datetime import UTC, datetime
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
# The agent is deployed flat on the Pi and uses top-level imports.
sys.path.insert(0, str(ROOT / 'agent'))

DEFAULT_MODEL_DIRS = ('data/models', 'models')


def discover_model(root: Path) -> Path | None:
    """Return the smallest .tflite in the usual model directories."""
    candidates: list[Path] = []
    for rel in DEFAULT_MODEL_DIRS:
        directory = root / rel
        if directory.is_dir():
            candidates.extend(directory.glob('*.tflite'))
    if not candidates:
        return None
    # Smallest model keeps the smoke run fast.
    return min(candidates, key=lambda p: p.stat().st_size)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--model', help='Path to a .tflite model (auto-discovered if omitted)'
    )
    parser.add_argument(
        '--backend',
        default='cpu',
        choices=('cpu', 'edgetpu'),
        help='Inference backend (default: cpu)',
    )
    parser.add_argument(
        '--threads', type=int, default=4, help='CPU threads (default: 4)'
    )
    parser.add_argument(
        '--warmup', type=int, default=5, help='Warmup iterations (default: 5)'
    )
    parser.add_argument(
        '--runs', type=int, default=30, help='Measured iterations (default: 30)'
    )
    parser.add_argument('--seed', type=int, default=42, help='Input seed (default: 42)')
    parser.add_argument(
        '--output-dir',
        default='results/smoke',
        help='Directory for the result JSON (default: results/smoke)',
    )
    parser.add_argument(
        '--no-write',
        action='store_true',
        help='Print the result without writing a file',
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.model:
        model_path = Path(args.model).expanduser()
        if not model_path.is_absolute():
            model_path = (Path.cwd() / model_path).resolve()
    else:
        found = discover_model(ROOT)
        if found is None:
            print(
                'No .tflite model found in '
                f'{" or ".join(DEFAULT_MODEL_DIRS)}. '
                'Pass --model /path/to/model.tflite.',
                file=sys.stderr,
            )
            return 2
        model_path = found
        print(f'Auto-selected model: {model_path.relative_to(ROOT)}', file=sys.stderr)

    if not model_path.is_file():
        print(f'Model not found: {model_path}', file=sys.stderr)
        return 2

    try:
        from tflite_backend import TFLiteBackendError, resolve_backend
    except ImportError as exc:  # pragma: no cover - import wiring failure
        print(f'Cannot import agent modules: {exc}', file=sys.stderr)
        return 2

    try:
        _, _, source = resolve_backend()
    except TFLiteBackendError as exc:
        print(f'{exc}\nInstall one with: make install-hardware', file=sys.stderr)
        return 3

    print(f'TFLite runtime: {source}', file=sys.stderr)

    from executor import BenchmarkExecutor

    executor = BenchmarkExecutor()
    experiment_id = f'smoke_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}'

    result = asyncio.run(
        executor.run_benchmark(
            experiment_id=experiment_id,
            model_path=str(model_path),
            params={
                'backend': args.backend,
                'num_threads': args.threads,
                'warmup_runs': args.warmup,
                'benchmark_runs': args.runs,
                'input_seed': args.seed,
                'smoke': True,
            },
        )
    )

    if result.get('status') != 'completed':
        print(json.dumps(result, indent=2, default=str))
        print(f'\nSmoke benchmark FAILED: {result.get("error")}', file=sys.stderr)
        return 1

    if not args.no_write:
        out_dir = Path(args.output_dir)
        if not out_dir.is_absolute():
            out_dir = ROOT / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f'{experiment_id}.json'
        out_path.write_text(json.dumps(result, indent=2, default=str))
        print(f'Result written to {out_path}', file=sys.stderr)

    latency = result['latency']
    throughput = result['throughput']
    runtime = result.get('runtime', {})
    device = result.get('device_info', {})

    print()
    print('=== Smoke benchmark ===')
    print(
        f'model          : {result["model"]["name"]} ({result["model"]["size_bytes"] / 1e6:.2f} MB)'
    )
    print(f'quantization   : {result["model"]["quantization"] or "unknown"}')
    print(f'backend        : {args.backend}')
    print(
        f'runtime        : {runtime.get("tflite_source")} {runtime.get("tflite_version")}'
    )
    print(
        'device         : '
        f'{device.get("device_model") or device.get("platform") or device.get("hostname")}'
    )
    print(f'iterations     : {args.warmup} warmup + {args.runs} measured')
    print(f'latency mean   : {latency["mean_ms"]:.3f} ms (std {latency["std_ms"]:.3f})')
    print(f'latency p50/p95: {latency["p50_ms"]:.3f} / {latency["p95_ms"]:.3f} ms')
    print(f'throughput     : {throughput["fps_from_mean"]:.2f} fps (from mean)')
    print(f'peak RSS       : {result["system"].get("process_rss_mb_max", "n/a")} MB')
    if result.get('warnings'):
        print('warnings       : ' + '; '.join(result['warnings']))
    print()
    print('Smoke benchmark PASSED (validation run — not a scientific result).')
    return 0


if __name__ == '__main__':
    sys.exit(main())
