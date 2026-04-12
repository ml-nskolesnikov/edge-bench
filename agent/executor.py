"""
Benchmark Executor for Edge Devices
"""

import asyncio
from datetime import datetime
import hashlib
import os
import time
from typing import Any

from metrics import SystemMetrics
import numpy as np
from result_cache import result_cache


class BenchmarkExecutor:
    """Execute ML benchmarks on Raspberry Pi."""

    def __init__(self):
        self.current_task: str | None = None
        self.metrics = SystemMetrics()

    async def run_benchmark(
        self,
        experiment_id: str,
        model_path: str,
        params: dict[str, Any],
    ) -> dict[str, Any]:
        """Run a TFLite benchmark."""
        # Always resolve ~ and relative paths
        model_path = os.path.expanduser(model_path)
        model_path = os.path.abspath(model_path)

        self.current_task = experiment_id
        start_time = time.time()
        logs = []

        try:
            # Parameters
            backend = params.get('backend', 'cpu')
            num_threads = params.get('num_threads', 4)
            warmup_runs = params.get('warmup_runs', 10)
            benchmark_runs = params.get('benchmark_runs', 100)

            logs.append(f'Loading model: {model_path}')
            logs.append(f'Backend: {backend}, Threads: {num_threads}')

            # Load interpreter
            interpreter, model_load_time = self._load_interpreter(
                model_path, backend, num_threads
            )
            logs.append(f'Model loaded in {model_load_time:.2f}ms')

            # Get input details
            input_details = interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']

            logs.append(f'Input shape: {input_shape}, dtype: {input_dtype}')

            # Generate dummy input
            if input_dtype == np.float32:
                input_data = np.random.rand(*input_shape).astype(np.float32)
            elif input_dtype == np.uint8:
                input_data = np.random.randint(0, 255, input_shape, dtype=np.uint8)
            else:
                input_data = np.random.rand(*input_shape).astype(input_dtype)

            # Warmup
            logs.append(f'Running {warmup_runs} warmup iterations...')
            first_inference_time = None

            for i in range(warmup_runs):
                interpreter.set_tensor(input_details[0]['index'], input_data)
                t0 = time.perf_counter()
                interpreter.invoke()
                t1 = time.perf_counter()

                if i == 0:
                    first_inference_time = (t1 - t0) * 1000

            # Benchmark
            logs.append(f'Running {benchmark_runs} benchmark iterations...')
            latencies = []

            # Start system metrics collection in background
            metrics_task = asyncio.create_task(
                self._collect_metrics_async(benchmark_runs * 0.015)  # Estimate duration
            )

            for _ in range(benchmark_runs):
                interpreter.set_tensor(input_details[0]['index'], input_data)
                t0 = time.perf_counter()
                interpreter.invoke()
                t1 = time.perf_counter()
                latencies.append((t1 - t0) * 1000)  # Convert to ms

            # Wait for metrics
            system_metrics = await metrics_task

            # Calculate statistics
            latencies = np.array(latencies)

            latency_stats = {
                'mean_ms': round(float(np.mean(latencies)), 3),
                'std_ms': round(float(np.std(latencies)), 3),
                'min_ms': round(float(np.min(latencies)), 3),
                'max_ms': round(float(np.max(latencies)), 3),
                'p50_ms': round(float(np.percentile(latencies, 50)), 3),
                'p90_ms': round(float(np.percentile(latencies, 90)), 3),
                'p95_ms': round(float(np.percentile(latencies, 95)), 3),
                'p99_ms': round(float(np.percentile(latencies, 99)), 3),
            }

            throughput = {
                'fps': round(1000.0 / latency_stats['mean_ms'], 2),
                'images_per_second': round(1000.0 / latency_stats['mean_ms'], 2),
            }

            cold_start = {
                'model_load_ms': round(model_load_time, 2),
                'first_inference_ms': round(first_inference_time, 2)
                if first_inference_time
                else 0,
            }

            # Model info
            model_info = {
                'name': os.path.basename(model_path),
                'hash': self._file_hash(model_path),
                'size_bytes': os.path.getsize(model_path),
                'quantization': self._detect_quantization(model_path),
            }

            duration = time.time() - start_time
            logs.append(f'Benchmark completed in {duration:.1f}s')
            logs.append(
                f'Mean latency: {latency_stats["mean_ms"]:.2f}ms, FPS: {throughput["fps"]:.1f}'
            )

            result = {
                'experiment_id': experiment_id,
                'device': self.metrics.get_device_info()['hostname'],
                'model': model_info,
                'params': params,
                'latency': latency_stats,
                'throughput': throughput,
                'cold_start': cold_start,
                'system': system_metrics,
                'device_info': self.metrics.get_device_info(),
                'timestamp': datetime.utcnow().isoformat(),
                'duration_seconds': round(duration, 2),
                'status': 'completed',
                'logs': '\n'.join(logs),
            }

            # Persist to local cache before returning
            # If server is down, the result is safe on disk
            try:
                cache_path = result_cache.save(experiment_id, result)
                logs.append(f'Result cached: {cache_path}')
            except Exception as cache_err:
                logs.append(f'Cache warning: {cache_err}')

            return result

        except Exception as e:
            logs.append(f'ERROR: {str(e)}')
            return {
                'experiment_id': experiment_id,
                'status': 'failed',
                'error': str(e),
                'logs': '\n'.join(logs),
                'timestamp': datetime.utcnow().isoformat(),
            }
        finally:
            self.current_task = None

    async def run_script(
        self,
        script_path: str,
        args: list[str],
        timeout: int = 600,
    ) -> dict[str, Any]:
        """Run a custom Python script."""
        self.current_task = f'script:{script_path}'
        start_time = time.time()

        try:
            cmd = ['python3', script_path] + args

            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(),
                timeout=timeout,
            )

            duration = time.time() - start_time

            return {
                'script': script_path,
                'args': args,
                'exit_code': process.returncode,
                'stdout': stdout.decode(),
                'stderr': stderr.decode(),
                'duration_seconds': round(duration, 2),
                'status': 'completed' if process.returncode == 0 else 'failed',
            }

        except TimeoutError:
            return {
                'script': script_path,
                'status': 'timeout',
                'error': f'Script timed out after {timeout}s',
            }
        except Exception as e:
            return {
                'script': script_path,
                'status': 'failed',
                'error': str(e),
            }
        finally:
            self.current_task = None

    def _load_interpreter(
        self,
        model_path: str,
        backend: str,
        num_threads: int,
    ):
        """Load TFLite interpreter."""
        t0 = time.perf_counter()

        # Check Edge TPU availability first
        if backend == 'edgetpu':
            if not self.metrics.check_tpu():
                raise RuntimeError(
                    'Edge TPU not detected. Check USB connection and run: '
                    'lsusb | grep -i google'
                )

        # Try tflite_runtime first (lighter)
        try:
            from tflite_runtime.interpreter import Interpreter

            if backend == 'edgetpu':
                from tflite_runtime.interpreter import load_delegate

                # Try different library paths
                lib_paths = [
                    'libedgetpu.so.1',
                    '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1',
                    '/usr/lib/arm-linux-gnueabihf/libedgetpu.so.1',
                ]
                delegate = None
                last_error = None

                for lib_path in lib_paths:
                    try:
                        delegate = load_delegate(lib_path)
                        break
                    except (ValueError, OSError) as e:
                        last_error = e
                        continue

                if delegate is None:
                    raise RuntimeError(
                        f'Cannot load Edge TPU delegate. '
                        f'Install libedgetpu: sudo apt install libedgetpu1-std. '
                        f'Last error: {last_error}'
                    )

                interpreter = Interpreter(
                    model_path=model_path,
                    experimental_delegates=[delegate],
                )
            else:
                interpreter = Interpreter(
                    model_path=model_path,
                    num_threads=num_threads,
                )
        except ImportError:
            # Fall back to full TensorFlow
            import tensorflow as tf

            if backend == 'edgetpu':
                # Edge TPU delegate
                try:
                    delegate = tf.lite.experimental.load_delegate('libedgetpu.so.1')
                except (ValueError, OSError) as e:
                    raise RuntimeError(
                        f'Cannot load Edge TPU delegate: {e}. '
                        f'Install: sudo apt install libedgetpu1-std'
                    )
                interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    experimental_delegates=[delegate],
                )
            else:
                interpreter = tf.lite.Interpreter(
                    model_path=model_path,
                    num_threads=num_threads,
                )

        interpreter.allocate_tensors()
        load_time = (time.perf_counter() - t0) * 1000

        return interpreter, load_time

    async def _collect_metrics_async(self, duration: float) -> dict[str, Any]:
        """Collect system metrics asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None,
            self.metrics.collect_during_benchmark,
            duration,
        )

    def _file_hash(self, path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return f'sha256:{sha256.hexdigest()[:16]}'

    def _detect_quantization(self, model_path: str) -> str | None:
        """Detect model quantization type."""
        name = os.path.basename(model_path).lower()

        if 'int8' in name or '_quant' in name:
            return 'int8'
        elif 'fp16' in name:
            return 'fp16'
        elif 'edgetpu' in name:
            return 'int8_edgetpu'

        return None
