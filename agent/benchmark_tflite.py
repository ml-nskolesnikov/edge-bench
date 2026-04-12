#!/usr/bin/env python3
"""
TFLite Benchmark Script

Standalone benchmark script that can be run on Raspberry Pi directly.
Results are printed as JSON for easy parsing.

Usage:
    python3 benchmark_tflite.py --model model.tflite --backend cpu --runs 100
"""

import argparse
from datetime import datetime
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter

    TFLITE_SOURCE = 'tflite_runtime'
except ImportError:
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter
    TFLITE_SOURCE = 'tensorflow'


def load_interpreter(model_path: str, backend: str, num_threads: int):
    """Load TFLite interpreter with specified backend."""
    if backend == 'edgetpu':
        try:
            if TFLITE_SOURCE == 'tflite_runtime':
                from tflite_runtime.interpreter import load_delegate

                delegates = [load_delegate('libedgetpu.so.1')]
            else:
                delegates = [tf.lite.experimental.load_delegate('libedgetpu.so.1')]

            return Interpreter(
                model_path=model_path,
                experimental_delegates=delegates,
            )
        except Exception as e:
            print(f'Warning: Could not load Edge TPU delegate: {e}', file=sys.stderr)
            print('Falling back to CPU', file=sys.stderr)

    return Interpreter(model_path=model_path, num_threads=num_threads)


def get_cpu_temp():
    """Get CPU temperature on Raspberry Pi."""
    try:
        with open('/sys/class/thermal/thermal_zone0/temp') as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None


def file_hash(path: str) -> str:
    """Calculate SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def run_benchmark(args):
    """Run the benchmark and return results as dict."""
    results = {
        'model_path': args.model,
        'backend': args.backend,
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'running',
    }

    try:
        # Load model
        t0 = time.perf_counter()
        interpreter = load_interpreter(args.model, args.backend, args.threads)
        interpreter.allocate_tensors()
        model_load_time = (time.perf_counter() - t0) * 1000

        # Get input details
        input_details = interpreter.get_input_details()
        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']

        # Generate input
        if input_dtype == np.float32:
            input_data = np.random.rand(*input_shape).astype(np.float32)
        elif input_dtype == np.uint8:
            input_data = np.random.randint(0, 255, input_shape, dtype=np.uint8)
        else:
            input_data = np.random.rand(*input_shape).astype(input_dtype)

        # Warmup
        first_inference = None
        for i in range(args.warmup):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t0 = time.perf_counter()
            interpreter.invoke()
            if i == 0:
                first_inference = (time.perf_counter() - t0) * 1000

        # Benchmark
        latencies = []
        for _ in range(args.runs):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t0 = time.perf_counter()
            interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)

        latencies = np.array(latencies)

        # Results
        results['model'] = {
            'name': os.path.basename(args.model),
            'hash': file_hash(args.model),
            'size_bytes': os.path.getsize(args.model),
            'input_shape': input_shape.tolist(),
        }

        results['params'] = {
            'backend': args.backend,
            'num_threads': args.threads,
            'warmup_runs': args.warmup,
            'benchmark_runs': args.runs,
        }

        results['latency'] = {
            'mean_ms': round(float(np.mean(latencies)), 3),
            'std_ms': round(float(np.std(latencies)), 3),
            'min_ms': round(float(np.min(latencies)), 3),
            'max_ms': round(float(np.max(latencies)), 3),
            'p50_ms': round(float(np.percentile(latencies, 50)), 3),
            'p90_ms': round(float(np.percentile(latencies, 90)), 3),
            'p95_ms': round(float(np.percentile(latencies, 95)), 3),
            'p99_ms': round(float(np.percentile(latencies, 99)), 3),
        }

        results['throughput'] = {
            'fps': round(1000.0 / results['latency']['mean_ms'], 2),
        }

        results['cold_start'] = {
            'model_load_ms': round(model_load_time, 2),
            'first_inference_ms': round(first_inference, 2)
            if first_inference
            else None,
        }

        results['system'] = {
            'hostname': platform.node(),
            'platform': platform.platform(),
            'python': platform.python_version(),
            'tflite_source': TFLITE_SOURCE,
            'cpu_temp_celsius': get_cpu_temp(),
        }

        results['status'] = 'completed'

    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)

    return results


def main():
    parser = argparse.ArgumentParser(description='TFLite Benchmark')
    parser.add_argument('--model', '-m', required=True, help='Path to TFLite model')
    parser.add_argument('--backend', '-b', default='cpu', choices=['cpu', 'edgetpu'])
    parser.add_argument(
        '--threads', '-t', type=int, default=4, help='Number of threads (CPU only)'
    )
    parser.add_argument(
        '--warmup', '-w', type=int, default=10, help='Warmup iterations'
    )
    parser.add_argument(
        '--runs', '-r', type=int, default=100, help='Benchmark iterations'
    )
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'Error: Model not found: {args.model}', file=sys.stderr)
        sys.exit(1)

    results = run_benchmark(args)

    output = json.dumps(results, indent=2)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f'Results saved to {args.output}', file=sys.stderr)
    else:
        print(output)

    sys.exit(0 if results['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
