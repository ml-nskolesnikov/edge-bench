#!/usr/bin/env python3
"""
Full Benchmark Script for Edge Devices

Extended benchmark with:
- CPU/Memory monitoring during inference
- Detailed latency distribution
- Power consumption estimation (if available)
- Multiple runs with statistics
- JSON output with all metrics

Usage:
    python3 benchmark_full.py --model model.tflite --backend cpu --runs 100
    python3 benchmark_full.py --model model_edgetpu.tflite --backend edgetpu --runs 100
"""

import argparse
from dataclasses import dataclass, field
from datetime import datetime
import hashlib
import json
import os
import platform
import subprocess
import sys
import threading
import time

import numpy as np

try:
    import psutil

    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    from tflite_runtime.interpreter import Interpreter

    TFLITE_SOURCE = 'tflite_runtime'
except ImportError:
    try:
        import tensorflow as tf

        Interpreter = tf.lite.Interpreter
        TFLITE_SOURCE = 'tensorflow'
    except ImportError:
        print('Error: No TFLite runtime found', file=sys.stderr)
        sys.exit(1)


@dataclass
class SystemMetrics:
    """System metrics during benchmark."""

    cpu_percent_samples: list = field(default_factory=list)
    memory_mb_samples: list = field(default_factory=list)
    cpu_temp_samples: list = field(default_factory=list)
    timestamps: list = field(default_factory=list)

    def add_sample(self, cpu: float, memory: float, temp: float | None):
        self.timestamps.append(time.time())
        self.cpu_percent_samples.append(cpu)
        self.memory_mb_samples.append(memory)
        if temp is not None:
            self.cpu_temp_samples.append(temp)

    def to_dict(self) -> dict:
        result = {}

        if self.cpu_percent_samples:
            result['cpu_percent'] = {
                'mean': round(np.mean(self.cpu_percent_samples), 2),
                'max': round(np.max(self.cpu_percent_samples), 2),
                'min': round(np.min(self.cpu_percent_samples), 2),
                'samples': len(self.cpu_percent_samples),
            }

        if self.memory_mb_samples:
            result['memory_mb'] = {
                'mean': round(np.mean(self.memory_mb_samples), 2),
                'max': round(np.max(self.memory_mb_samples), 2),
                'min': round(np.min(self.memory_mb_samples), 2),
            }

        if self.cpu_temp_samples:
            result['cpu_temp_celsius'] = {
                'mean': round(np.mean(self.cpu_temp_samples), 2),
                'max': round(np.max(self.cpu_temp_samples), 2),
                'start': round(self.cpu_temp_samples[0], 2),
                'end': round(self.cpu_temp_samples[-1], 2),
            }

        return result


class SystemMonitor:
    """Background system monitoring during benchmark."""

    def __init__(self, interval: float = 0.1):
        self.interval = interval
        self.metrics = SystemMetrics()
        self._running = False
        self._thread = None
        self._process = psutil.Process() if HAS_PSUTIL else None

    def _get_cpu_temp(self) -> float | None:
        """Get CPU temperature on Raspberry Pi."""
        try:
            with open('/sys/class/thermal/thermal_zone0/temp') as f:
                return int(f.read().strip()) / 1000.0
        except Exception:
            return None

    def _monitor_loop(self):
        """Background monitoring loop."""
        while self._running:
            try:
                if HAS_PSUTIL:
                    cpu = psutil.cpu_percent(interval=None)
                    mem = self._process.memory_info().rss / 1024 / 1024
                else:
                    cpu = 0
                    mem = 0

                temp = self._get_cpu_temp()
                self.metrics.add_sample(cpu, mem, temp)
            except Exception:
                pass

            time.sleep(self.interval)

    def start(self):
        """Start monitoring."""
        if HAS_PSUTIL:
            psutil.cpu_percent()  # Initialize
        self._running = True
        self._thread = threading.Thread(target=self._monitor_loop, daemon=True)
        self._thread.start()

    def stop(self) -> SystemMetrics:
        """Stop monitoring and return metrics."""
        self._running = False
        if self._thread:
            self._thread.join(timeout=1.0)
        return self.metrics


def load_interpreter(model_path: str, backend: str, num_threads: int):
    """Load TFLite interpreter with specified backend."""
    delegates = []
    tpu_detected = False

    if backend == 'edgetpu':
        try:
            if TFLITE_SOURCE == 'tflite_runtime':
                from tflite_runtime.interpreter import load_delegate

                delegates = [load_delegate('libedgetpu.so.1')]
            else:
                import tensorflow as tf

                delegates = [tf.lite.experimental.load_delegate('libedgetpu.so.1')]
            tpu_detected = True
        except Exception as e:
            print(f'Warning: Could not load Edge TPU: {e}', file=sys.stderr)
            print('Falling back to CPU', file=sys.stderr)
            backend = 'cpu'

    if delegates:
        interpreter = Interpreter(
            model_path=model_path,
            experimental_delegates=delegates,
        )
    else:
        interpreter = Interpreter(model_path=model_path, num_threads=num_threads)

    return interpreter, tpu_detected, backend


def file_hash(path: str) -> str:
    """Calculate SHA256 hash of file."""
    sha256 = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            sha256.update(chunk)
    return sha256.hexdigest()[:16]


def get_device_info() -> dict:
    """Get detailed device information."""
    info = {
        'hostname': platform.node(),
        'platform': platform.platform(),
        'machine': platform.machine(),
        'python': platform.python_version(),
        'tflite_source': TFLITE_SOURCE,
    }

    # CPU info
    if HAS_PSUTIL:
        info['cpu_count'] = psutil.cpu_count(logical=True)
        info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        mem = psutil.virtual_memory()
        info['memory_total_mb'] = round(mem.total / 1024 / 1024, 0)
        info['memory_available_mb'] = round(mem.available / 1024 / 1024, 0)

    # Raspberry Pi specific
    try:
        with open('/proc/device-tree/model') as f:
            info['device_model'] = f.read().strip().rstrip('\x00')
    except Exception:
        pass

    # CPU frequency
    try:
        with open('/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq') as f:
            info['cpu_freq_mhz'] = int(f.read().strip()) // 1000
    except Exception:
        pass

    return info


def check_edgetpu_available() -> dict:
    """Check if Edge TPU is available."""
    result = {'available': False, 'device': None, 'runtime': None}

    try:
        # Check USB devices
        lsusb = subprocess.run(['lsusb'], capture_output=True, text=True)
        if 'Google' in lsusb.stdout or 'Global Unichip' in lsusb.stdout:
            result['available'] = True
            if 'Google' in lsusb.stdout:
                result['device'] = 'Coral USB Accelerator'
            elif 'Global Unichip' in lsusb.stdout:
                result['device'] = 'Coral USB Accelerator (GUC)'
    except Exception:
        pass

    # Check runtime version
    try:
        from pycoral.utils import edgetpu

        result['runtime'] = edgetpu.get_runtime_version()
    except Exception:
        pass

    return result


def compute_latency_stats(latencies: np.ndarray) -> dict:
    """Compute detailed latency statistics."""
    return {
        'mean_ms': round(float(np.mean(latencies)), 3),
        'std_ms': round(float(np.std(latencies)), 3),
        'min_ms': round(float(np.min(latencies)), 3),
        'max_ms': round(float(np.max(latencies)), 3),
        'median_ms': round(float(np.median(latencies)), 3),
        'p50_ms': round(float(np.percentile(latencies, 50)), 3),
        'p75_ms': round(float(np.percentile(latencies, 75)), 3),
        'p90_ms': round(float(np.percentile(latencies, 90)), 3),
        'p95_ms': round(float(np.percentile(latencies, 95)), 3),
        'p99_ms': round(float(np.percentile(latencies, 99)), 3),
        'variance_ms2': round(float(np.var(latencies)), 4),
        'iqr_ms': round(
            float(np.percentile(latencies, 75) - np.percentile(latencies, 25)), 3
        ),
    }


def run_benchmark(args) -> dict:
    """Run the full benchmark and return detailed results."""
    results = {
        'benchmark_id': f'bench_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}',
        'timestamp': datetime.utcnow().isoformat(),
        'status': 'running',
    }

    monitor = SystemMonitor(interval=0.1)

    try:
        # Device info
        results['device'] = get_device_info()
        results['edgetpu'] = check_edgetpu_available()

        # Load model
        print(f'Loading model: {args.model}', file=sys.stderr)
        t0 = time.perf_counter()
        interpreter, tpu_detected, actual_backend = load_interpreter(
            args.model, args.backend, args.threads
        )
        interpreter.allocate_tensors()
        model_load_time = (time.perf_counter() - t0) * 1000

        # Get tensor details
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        input_shape = input_details[0]['shape']
        input_dtype = input_details[0]['dtype']
        output_shape = output_details[0]['shape']

        # Model info
        results['model'] = {
            'path': args.model,
            'name': os.path.basename(args.model),
            'hash': file_hash(args.model),
            'size_bytes': os.path.getsize(args.model),
            'size_mb': round(os.path.getsize(args.model) / 1024 / 1024, 3),
            'input_shape': input_shape.tolist(),
            'input_dtype': str(input_dtype),
            'output_shape': output_shape.tolist(),
            'output_dtype': str(output_details[0]['dtype']),
        }

        # Parameters
        results['params'] = {
            'requested_backend': args.backend,
            'actual_backend': actual_backend,
            'tpu_detected': tpu_detected,
            'num_threads': args.threads if actual_backend == 'cpu' else None,
            'warmup_runs': args.warmup,
            'benchmark_runs': args.runs,
            'batch_size': int(input_shape[0]) if len(input_shape) > 0 else 1,
        }

        # Generate input data
        if input_dtype == np.float32:
            input_data = np.random.rand(*input_shape).astype(np.float32)
        elif input_dtype == np.uint8:
            input_data = np.random.randint(0, 255, input_shape, dtype=np.uint8)
        elif input_dtype == np.int8:
            input_data = np.random.randint(-128, 127, input_shape, dtype=np.int8)
        else:
            input_data = np.random.rand(*input_shape).astype(input_dtype)

        # Warmup
        print(f'Warmup: {args.warmup} runs', file=sys.stderr)
        first_inference = None
        warmup_latencies = []

        for i in range(args.warmup):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t0 = time.perf_counter()
            interpreter.invoke()
            lat = (time.perf_counter() - t0) * 1000
            warmup_latencies.append(lat)
            if i == 0:
                first_inference = lat

        # Start monitoring
        monitor.start()

        # Benchmark
        print(f'Benchmark: {args.runs} runs', file=sys.stderr)
        latencies = []

        for i in range(args.runs):
            interpreter.set_tensor(input_details[0]['index'], input_data)
            t0 = time.perf_counter()
            interpreter.invoke()
            latencies.append((time.perf_counter() - t0) * 1000)

            if (i + 1) % 50 == 0:
                print(f'  Progress: {i + 1}/{args.runs}', file=sys.stderr)

        # Stop monitoring
        system_metrics = monitor.stop()

        latencies = np.array(latencies)
        warmup_latencies = np.array(warmup_latencies)

        # Results
        results['latency'] = compute_latency_stats(latencies)
        results['latency']['all_values_ms'] = [round(x, 3) for x in latencies.tolist()]

        results['warmup'] = {
            'mean_ms': round(float(np.mean(warmup_latencies)), 3),
            'first_inference_ms': round(first_inference, 3)
            if first_inference
            else None,
        }

        results['throughput'] = {
            'fps': round(1000.0 / results['latency']['mean_ms'], 2),
            'fps_p95': round(1000.0 / results['latency']['p95_ms'], 2),
            'inferences_per_second': round(1000.0 / results['latency']['mean_ms'], 2),
        }

        results['cold_start'] = {
            'model_load_ms': round(model_load_time, 2),
            'first_inference_ms': round(first_inference, 2)
            if first_inference
            else None,
            'total_cold_start_ms': round(model_load_time + (first_inference or 0), 2),
        }

        results['system'] = system_metrics.to_dict()

        # Stability metrics
        results['stability'] = {
            'cv_percent': round(
                float(np.std(latencies) / np.mean(latencies) * 100), 2
            ),  # Coefficient of variation
            'outliers_count': int(
                np.sum(np.abs(latencies - np.mean(latencies)) > 3 * np.std(latencies))
            ),
        }

        results['status'] = 'completed'
        results['duration_seconds'] = round(
            time.time() - datetime.fromisoformat(results['timestamp']).timestamp(), 2
        )

        print('\nResults:', file=sys.stderr)
        print(
            f'  Latency: {results["latency"]["mean_ms"]:.2f} ms (p95: {results["latency"]["p95_ms"]:.2f} ms)',
            file=sys.stderr,
        )
        print(f'  Throughput: {results["throughput"]["fps"]:.1f} FPS', file=sys.stderr)

    except Exception as e:
        results['status'] = 'failed'
        results['error'] = str(e)
        import traceback

        results['traceback'] = traceback.format_exc()
        print(f'Error: {e}', file=sys.stderr)

    finally:
        monitor.stop()

    return results


def main():
    parser = argparse.ArgumentParser(
        description='Full TFLite Benchmark with System Monitoring',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', '-m', required=True, help='Path to TFLite model')
    parser.add_argument(
        '--backend',
        '-b',
        default='cpu',
        choices=['cpu', 'edgetpu'],
        help='Inference backend',
    )
    parser.add_argument(
        '--threads', '-t', type=int, default=4, help='Number of threads (CPU only)'
    )
    parser.add_argument(
        '--warmup', '-w', type=int, default=20, help='Warmup iterations'
    )
    parser.add_argument(
        '--runs', '-r', type=int, default=100, help='Benchmark iterations'
    )
    parser.add_argument('--output', '-o', help='Output JSON file (default: stdout)')
    parser.add_argument(
        '--compact',
        action='store_true',
        help='Compact JSON output (no latency values array)',
    )

    args = parser.parse_args()

    if not os.path.exists(args.model):
        print(f'Error: Model not found: {args.model}', file=sys.stderr)
        sys.exit(1)

    results = run_benchmark(args)

    # Remove raw latency values if compact mode
    if args.compact and 'latency' in results:
        results['latency'].pop('all_values_ms', None)

    output = json.dumps(results, indent=2, default=str)

    if args.output:
        with open(args.output, 'w') as f:
            f.write(output)
        print(f'\nResults saved to {args.output}', file=sys.stderr)
    else:
        print(output)

    sys.exit(0 if results['status'] == 'completed' else 1)


if __name__ == '__main__':
    main()
