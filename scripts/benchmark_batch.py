#!/usr/bin/env python3
"""
Batch Benchmark Script for Edge Devices

Automatically discovers and benchmarks all TFLite models in specified directory.
Runs both CPU and EdgeTPU backends where applicable.
Generates aggregated CSV and JSON reports.

Usage:
    python3 benchmark_batch.py --models-dir ~/models --output-dir ./results
    python3 benchmark_batch.py --models-dir ~/models --backends cpu edgetpu --runs 100
"""

import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

# Import benchmark function
try:
    from benchmark_full import check_edgetpu_available, get_device_info, run_benchmark
except ImportError:
    # Running standalone
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from benchmark_full import check_edgetpu_available, get_device_info, run_benchmark


class Args:
    """Arguments container for benchmark function."""

    def __init__(self, model: str, backend: str, threads: int, warmup: int, runs: int):
        self.model = model
        self.backend = backend
        self.threads = threads
        self.warmup = warmup
        self.runs = runs


def discover_models(models_dir: Path) -> list[dict]:
    """Discover TFLite models in directory."""
    models = []

    for ext in ['*.tflite', '*.TFLITE']:
        for path in models_dir.glob(ext):
            name = path.name
            size_mb = path.stat().st_size / 1024 / 1024

            # Determine compatible backends
            backends = ['cpu']
            if '_edgetpu' in name.lower():
                backends = ['edgetpu']  # EdgeTPU models only work on TPU
            elif '_int8' in name.lower() or '_quant' in name.lower():
                backends = ['cpu', 'edgetpu']  # INT8 can run on both

            models.append(
                {
                    'path': str(path),
                    'name': name,
                    'size_mb': round(size_mb, 2),
                    'backends': backends,
                }
            )

    # Sort by name
    models.sort(key=lambda x: x['name'])
    return models


def generate_csv_report(results: list[dict], output_path: Path):
    """Generate aggregated CSV report."""
    headers = [
        'model_name',
        'backend',
        'size_mb',
        'latency_mean_ms',
        'latency_std_ms',
        'latency_p50_ms',
        'latency_p95_ms',
        'latency_p99_ms',
        'fps',
        'model_load_ms',
        'first_inference_ms',
        'cpu_percent_mean',
        'memory_mb_mean',
        'cpu_temp_mean',
        'status',
    ]

    with open(output_path, 'w') as f:
        f.write(','.join(headers) + '\n')

        for r in results:
            if r.get('status') != 'completed':
                # Failed benchmark
                row = [
                    r.get('model', {}).get('name', 'unknown'),
                    r.get('params', {}).get('actual_backend', 'unknown'),
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    '',
                    r.get('status', 'failed'),
                ]
            else:
                latency = r.get('latency', {})
                throughput = r.get('throughput', {})
                cold_start = r.get('cold_start', {})
                system = r.get('system', {})

                row = [
                    r.get('model', {}).get('name', ''),
                    r.get('params', {}).get('actual_backend', ''),
                    r.get('model', {}).get('size_mb', ''),
                    latency.get('mean_ms', ''),
                    latency.get('std_ms', ''),
                    latency.get('p50_ms', ''),
                    latency.get('p95_ms', ''),
                    latency.get('p99_ms', ''),
                    throughput.get('fps', ''),
                    cold_start.get('model_load_ms', ''),
                    cold_start.get('first_inference_ms', ''),
                    system.get('cpu_percent', {}).get('mean', ''),
                    system.get('memory_mb', {}).get('mean', ''),
                    system.get('cpu_temp_celsius', {}).get('mean', ''),
                    r.get('status', ''),
                ]

            f.write(','.join(str(x) for x in row) + '\n')

    print(f'CSV report saved: {output_path}')


def generate_comparison_report(results: list[dict]) -> dict:
    """Generate comparison statistics between backends."""
    comparison = {}

    # Group by model
    by_model = {}
    for r in results:
        if r.get('status') != 'completed':
            continue

        model_name = r.get('model', {}).get('name', '')
        backend = r.get('params', {}).get('actual_backend', '')

        # Normalize name (remove _edgetpu suffix for comparison)
        base_name = model_name.replace('_edgetpu', '')

        if base_name not in by_model:
            by_model[base_name] = {}

        by_model[base_name][backend] = {
            'latency_ms': r.get('latency', {}).get('mean_ms'),
            'fps': r.get('throughput', {}).get('fps'),
            'p95_ms': r.get('latency', {}).get('p95_ms'),
        }

    # Calculate speedups
    speedups = []
    for model, backends in by_model.items():
        if 'cpu' in backends and 'edgetpu' in backends:
            cpu_lat = backends['cpu']['latency_ms']
            tpu_lat = backends['edgetpu']['latency_ms']

            if cpu_lat and tpu_lat and tpu_lat > 0:
                speedup = cpu_lat / tpu_lat
                speedups.append(
                    {
                        'model': model,
                        'cpu_latency_ms': cpu_lat,
                        'edgetpu_latency_ms': tpu_lat,
                        'speedup': round(speedup, 2),
                    }
                )

    comparison['speedups'] = speedups

    if speedups:
        comparison['summary'] = {
            'avg_speedup': round(np.mean([s['speedup'] for s in speedups]), 2),
            'max_speedup': round(max(s['speedup'] for s in speedups), 2),
            'min_speedup': round(min(s['speedup'] for s in speedups), 2),
        }

    return comparison


def _parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Batch TFLite Benchmark',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        '--models-dir', '-d', required=True, help='Directory containing TFLite models'
    )
    parser.add_argument(
        '--output-dir',
        '-o',
        default='./benchmark_results',
        help='Output directory for results',
    )
    parser.add_argument(
        '--backends',
        '-b',
        nargs='+',
        default=['cpu', 'edgetpu'],
        choices=['cpu', 'edgetpu'],
        help='Backends to test',
    )
    parser.add_argument(
        '--threads', '-t', type=int, default=4, help='Number of threads (CPU only)'
    )
    parser.add_argument(
        '--warmup', '-w', type=int, default=20, help='Warmup iterations'
    )
    parser.add_argument(
        '--runs', '-r', type=int, default=100, help='Benchmark iterations per model'
    )
    parser.add_argument(
        '--filter', '-f', help='Filter models by name (substring match)'
    )
    parser.add_argument(
        '--skip-existing',
        action='store_true',
        help='Skip models that already have results',
    )
    return parser.parse_args()


def _print_header(args, models_dir, output_dir, device_info, edgetpu_info):
    """Print benchmark header information."""
    print('=' * 60)
    print('Batch TFLite Benchmark')
    print('=' * 60)
    print(f'Date: {datetime.now().isoformat()}')
    print(f'Models directory: {models_dir}')
    print(f'Output directory: {output_dir}')
    print(f'Backends: {", ".join(args.backends)}')
    print(f'Runs per model: {args.runs}')
    print()
    print(f'Device: {device_info.get("hostname", "unknown")}')
    print(
        f'Platform: {device_info.get("device_model", device_info.get("platform", "unknown"))}'
    )

    if edgetpu_info['available']:
        print(f'EdgeTPU: {edgetpu_info.get("device", "Available")}')
    else:
        print('EdgeTPU: Not detected')
        if 'edgetpu' in args.backends:
            print('Warning: EdgeTPU requested but not available')
    print()


def _run_single_benchmark(model, backend, args, output_dir, current, total):
    """Run benchmark for a single model/backend combination."""
    result_file = output_dir / f'benchmark_{model["name"]}_{backend}.json'

    if args.skip_existing and result_file.exists():
        print(f'[{current}/{total}] Skipping {model["name"]} ({backend}) - exists')
        with open(result_file) as f:
            return json.load(f)

    print()
    print('-' * 60)
    print(f'[{current}/{total}] {model["name"]} ({backend})')
    print('-' * 60)

    bench_args = Args(
        model=model['path'],
        backend=backend,
        threads=args.threads,
        warmup=args.warmup,
        runs=args.runs,
    )
    result = run_benchmark(bench_args)

    # Save individual result
    result_compact = result.copy()
    if 'latency' in result_compact:
        result_compact['latency'] = {
            k: v for k, v in result_compact['latency'].items() if k != 'all_values_ms'
        }
    with open(result_file, 'w') as f:
        json.dump(result_compact, f, indent=2, default=str)

    time.sleep(1)
    return result


def _run_all_benchmarks(models, args, output_dir, edgetpu_info):
    """Run benchmarks for all models."""
    all_results = []
    total = sum(len(set(m['backends']) & set(args.backends)) for m in models)
    current = 0

    for model in models:
        model_backends = set(model['backends']) & set(args.backends)

        for backend in model_backends:
            current += 1

            if backend == 'edgetpu' and not edgetpu_info['available']:
                print(
                    f'[{current}/{total}] Skipping {model["name"]} ({backend}) - TPU not available'
                )
                continue

            result = _run_single_benchmark(
                model, backend, args, output_dir, current, total
            )
            all_results.append(result)

    return all_results


def _generate_reports(all_results, models, args, device_info, edgetpu_info, output_dir):
    """Generate CSV and JSON reports."""
    print()
    print('=' * 60)
    print('Generating Reports')
    print('=' * 60)

    csv_path = (
        output_dir / f'benchmark_summary_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    )
    generate_csv_report(all_results, csv_path)

    json_path = (
        output_dir / f'benchmark_all_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    )
    report = {
        'timestamp': datetime.now().isoformat(),
        'device': device_info,
        'edgetpu': edgetpu_info,
        'params': {'warmup': args.warmup, 'runs': args.runs, 'threads': args.threads},
        'models_count': len(models),
        'benchmarks_count': len(all_results),
        'results': [
            {k: v for k, v in r.items() if k != 'latency' or 'all_values_ms' not in v}
            for r in all_results
        ],
        'comparison': generate_comparison_report(all_results),
    }

    with open(json_path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f'JSON report saved: {json_path}')

    return report


def _print_summary(all_results, report, args, output_dir):
    """Print benchmark summary."""
    print()
    print('=' * 60)
    print('Summary')
    print('=' * 60)

    completed = [r for r in all_results if r.get('status') == 'completed']
    failed = [r for r in all_results if r.get('status') != 'completed']

    print(f'Completed: {len(completed)}')
    print(f'Failed: {len(failed)}')

    if completed:
        print()
        print('Results by backend:')
        for backend in args.backends:
            backend_results = [
                r
                for r in completed
                if r.get('params', {}).get('actual_backend') == backend
            ]
            if backend_results:
                latencies = [r['latency']['mean_ms'] for r in backend_results]
                fps_values = [r['throughput']['fps'] for r in backend_results]
                print(f'\n  {backend.upper()}:')
                print(f'    Models: {len(backend_results)}')
                print(
                    f'    Latency: {np.mean(latencies):.2f} ms (avg), {np.min(latencies):.2f} - {np.max(latencies):.2f} ms'
                )
                print(
                    f'    FPS: {np.mean(fps_values):.1f} (avg), {np.min(fps_values):.1f} - {np.max(fps_values):.1f}'
                )

    speedups = report.get('comparison', {}).get('speedups', [])
    if speedups:
        print()
        print('EdgeTPU Speedup vs CPU:')
        for s in speedups:
            print(
                f'  {s["model"]}: {s["speedup"]:.1f}x ({s["cpu_latency_ms"]:.1f}ms -> {s["edgetpu_latency_ms"]:.1f}ms)'
            )
        summary = report.get('comparison', {}).get('summary', {})
        if summary:
            print(f'\n  Average speedup: {summary.get("avg_speedup", 0):.1f}x')

    print()
    print(f'Results saved to: {output_dir}')


def main():
    args = _parse_args()
    models_dir = Path(args.models_dir).expanduser()
    output_dir = Path(args.output_dir)

    if not models_dir.exists():
        print(f'Error: Models directory not found: {models_dir}', file=sys.stderr)
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    device_info = get_device_info()
    edgetpu_info = check_edgetpu_available()

    _print_header(args, models_dir, output_dir, device_info, edgetpu_info)

    print('Discovering models...')
    models = discover_models(models_dir)
    if args.filter:
        models = [m for m in models if args.filter.lower() in m['name'].lower()]

    if not models:
        print('No models found!')
        sys.exit(1)

    print(f'Found {len(models)} models:')
    for m in models:
        print(f'  - {m["name"]} ({m["size_mb"]} MB) [{", ".join(m["backends"])}]')
    print()

    all_results = _run_all_benchmarks(models, args, output_dir, edgetpu_info)
    report = _generate_reports(
        all_results, models, args, device_info, edgetpu_info, output_dir
    )
    _print_summary(all_results, report, args, output_dir)


if __name__ == '__main__':
    main()
