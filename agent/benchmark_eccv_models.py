#!/usr/bin/env python3
"""
ECCV 2026 Models Benchmark Script - Agent Version

Benchmarks all ECCV2026 models (MobileNetV2, MobileNetV1, EfficientNet-Lite0,
EfficientNet-B3, ResNet50) on CPU and EdgeTPU, generating Table T4 data.

Can be run:
1. Directly on Raspberry Pi: python benchmark_eccv_models.py --local
2. Via edge-bench server: uploaded and executed remotely

Usage:
    # Local execution on RPi
    python benchmark_eccv_models.py --local --models-dir /path/to/models --output results.json

    # List available models
    python benchmark_eccv_models.py --list-models --models-dir /path/to/models
"""

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys

# Import benchmark_tflite functions
try:
    from benchmark_tflite import run_benchmark
except ImportError:
    # If running standalone, add parent to path
    sys.path.insert(0, str(Path(__file__).parent))
    from benchmark_tflite import run_benchmark


# ECCV Model definitions
# All architectures x 3 strategies (hybrid / Fuzzy / sbert)
_STRATEGIES = ['hybrid', 'Fuzzy', 'sbert']
_ARCHITECTURES = {
    'mobilenetv2': 'MobileNetV2',
    'mobilenetv1': 'MobileNetV1',
    'efficientnet_lite0': 'EfficientNet-Lite0',
    'efficientnet': 'EfficientNet-B3',
    'resnet50': 'ResNet-50',
}

ECCV_MODELS = {}
for _arch_key, _arch_label in _ARCHITECTURES.items():
    for _strat in _STRATEGIES:
        _name = f'{_arch_key}_int8_ptq_{_strat}'
        ECCV_MODELS[_name] = {
            'strategy': _strat,
            'quantization': 'int8',
            'architecture': _arch_label,
        }


def find_models(models_dir: Path) -> dict:
    """Find ECCV models in directory."""
    found = {}

    for name, meta in ECCV_MODELS.items():
        # Check for regular INT8 model
        int8_path = models_dir / f'{name}.tflite'
        if int8_path.exists():
            found[name] = {
                'path': str(int8_path),
                'edgetpu_path': None,
                **meta,
            }

        # Check for EdgeTPU compiled model
        edgetpu_path = models_dir / f'{name}_edgetpu.tflite'
        if edgetpu_path.exists():
            if name in found:
                found[name]['edgetpu_path'] = str(edgetpu_path)
            else:
                found[name] = {
                    'path': None,
                    'edgetpu_path': str(edgetpu_path),
                    **meta,
                }

    return found


def _get_model_path(meta: dict, backend: str, name: str) -> str | None:
    """Get appropriate model path for given backend."""
    if backend == 'edgetpu':
        model_path = meta.get('edgetpu_path')
        if not model_path:
            print(f'  [SKIP] No EdgeTPU model for {name}')
        return model_path
    else:
        model_path = meta.get('path') or meta.get('edgetpu_path')
        if not model_path:
            print(f'  [SKIP] No model file for {name}')
        return model_path


def _run_single_model(
    model_path: str, backend: str, name: str, meta: dict, warmup: int, runs: int
) -> dict:
    """Run benchmark for a single model/backend combination."""
    print(f'  [{backend.upper()}] {model_path}')

    class Args:
        pass

    args = Args()
    args.model = model_path
    args.backend = backend
    args.threads = 4
    args.warmup = warmup
    args.runs = runs

    try:
        result = run_benchmark(args)
        result['eccv'] = {
            'model_name': name,
            'strategy': meta['strategy'],
            'quantization': meta['quantization'],
            'architecture': meta['architecture'],
        }

        if result['status'] == 'completed':
            lat = result['latency']
            print(f'      Latency: {lat["mean_ms"]:.2f}ms (p95: {lat["p95_ms"]:.2f}ms)')
            print(f'      FPS: {result["throughput"]["fps"]:.1f}')
        else:
            print(f'      [FAILED] {result.get("error", "Unknown error")}')

        return result

    except Exception as e:
        print(f'      [ERROR] {e}')
        return {
            'model_path': model_path,
            'backend': backend,
            'status': 'failed',
            'error': str(e),
            'eccv': {'model_name': name, 'strategy': meta['strategy']},
        }


def _print_found_models(models: dict):
    """Print discovered models info."""
    print(f'[INFO] Found {len(models)} model(s)')
    for name, meta in models.items():
        print(f'  - {name}')
        if meta['path']:
            print(f'      INT8: {meta["path"]}')
        if meta['edgetpu_path']:
            print(f'      EdgeTPU: {meta["edgetpu_path"]}')


def run_eccv_benchmark(
    models_dir: Path,
    output_path: Path,
    backends: list = None,
    runs: int = 100,
    warmup: int = 20,
):
    """Run benchmarks for all ECCV models."""
    if backends is None:
        backends = ['cpu', 'edgetpu']

    models = find_models(models_dir)

    if not models:
        print(f'[ERROR] No ECCV models found in {models_dir}')
        print('Expected models:')
        for name in ECCV_MODELS:
            print(f'  - {name}.tflite')
            print(f'  - {name}_edgetpu.tflite')
        return None

    _print_found_models(models)

    results = {
        'experiment': 'eccv2026_t4_benchmark',
        'timestamp': datetime.utcnow().isoformat(),
        'config': {'runs': runs, 'warmup': warmup, 'backends': backends},
        'benchmarks': [],
    }

    for name, meta in models.items():
        print(f'\n[INFO] Benchmarking: {name}')

        for backend in backends:
            model_path = _get_model_path(meta, backend, name)
            if not model_path:
                continue

            result = _run_single_model(model_path, backend, name, meta, warmup, runs)
            results['benchmarks'].append(result)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f'\n[OK] Results saved: {output_path}')
    return results


def generate_t4_csv(results: dict, output_csv: Path):
    """Generate T4 table CSV from benchmark results."""
    rows = [
        [
            'strategy',
            'model',
            'backend',
            'latency_mean_ms',
            'latency_p95_ms',
            'fps',
            'model_size_mb',
        ]
    ]

    for bench in results.get('benchmarks', []):
        if bench['status'] != 'completed':
            continue

        eccv = bench.get('eccv', {})
        rows.append(
            [
                eccv.get('strategy', 'unknown'),
                eccv.get('architecture', 'MobileNetV2'),
                bench['params']['backend'].upper(),
                round(bench['latency']['mean_ms'], 2),
                round(bench['latency']['p95_ms'], 2),
                round(bench['throughput']['fps'], 1),
                round(bench['model']['size_bytes'] / 1024 / 1024, 2),
            ]
        )

    with open(output_csv, 'w') as f:
        for row in rows:
            f.write(','.join(map(str, row)) + '\n')

    print(f'[OK] T4 CSV saved: {output_csv}')


def main():
    parser = argparse.ArgumentParser(description='ECCV 2026 Model Benchmark')
    parser.add_argument(
        '--models-dir',
        '-m',
        type=Path,
        default=Path('models'),
        help='Directory containing TFLite models',
    )
    parser.add_argument(
        '--output',
        '-o',
        type=Path,
        default=Path('results/eccv_benchmark.json'),
        help='Output JSON file',
    )
    parser.add_argument(
        '--csv', type=Path, default=None, help='Also generate T4 CSV table'
    )
    parser.add_argument(
        '--backends',
        '-b',
        nargs='+',
        default=['cpu', 'edgetpu'],
        choices=['cpu', 'edgetpu'],
        help='Backends to benchmark',
    )
    parser.add_argument(
        '--runs', '-r', type=int, default=100, help='Number of benchmark runs'
    )
    parser.add_argument(
        '--warmup', '-w', type=int, default=20, help='Number of warmup runs'
    )
    parser.add_argument(
        '--list-models', action='store_true', help='List available models and exit'
    )
    parser.add_argument(
        '--local', action='store_true', help='Run in local mode (direct execution)'
    )

    args = parser.parse_args()

    if args.list_models:
        models = find_models(args.models_dir)
        if models:
            print('Available ECCV models:')
            for name, meta in models.items():
                print(f'  {name}:')
                print(f'    Strategy: {meta["strategy"]}')
                if meta['path']:
                    print(f'    INT8: {meta["path"]}')
                if meta['edgetpu_path']:
                    print(f'    EdgeTPU: {meta["edgetpu_path"]}')
        else:
            print(f'No ECCV models found in {args.models_dir}')
        return

    results = run_eccv_benchmark(
        models_dir=args.models_dir,
        output_path=args.output,
        backends=args.backends,
        runs=args.runs,
        warmup=args.warmup,
    )

    if results and args.csv:
        generate_t4_csv(results, args.csv)


if __name__ == '__main__':
    main()
