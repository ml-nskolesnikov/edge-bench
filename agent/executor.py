"""
Benchmark Executor for Edge Devices
"""

import asyncio
from datetime import UTC, datetime
import gc
import hashlib
import os
import platform
import time
from typing import Any

from metrics import SystemMetrics
import numpy as np
from result_cache import result_cache
from tflite_backend import backend_version, resolve_backend


class BenchmarkExecutor:
    """Execute ML benchmarks on Raspberry Pi."""

    def __init__(self):
        self.current_task: str | None = None
        self.metrics = SystemMetrics()
        # Set by _load_interpreter; recorded in the result for reproducibility.
        self.tflite_source: str | None = None

    async def _send_metric(
        self,
        callback_url: str,
        payload: dict[str, Any],
    ) -> None:
        """Fire-and-forget metric push to server callback URL."""
        try:
            import httpx

            async with httpx.AsyncClient(timeout=2) as client:
                await client.post(callback_url, json=payload)
        except Exception:
            pass  # Never block benchmark on callback errors

    async def run_benchmark(
        self,
        experiment_id: str,
        model_path: str,
        params: dict[str, Any],
        stream_callback_url: str | None = None,
    ) -> dict[str, Any]:
        """Run a TFLite benchmark."""
        # Always resolve ~ and relative paths
        model_path = os.path.expanduser(model_path)
        model_path = os.path.abspath(model_path)

        self.current_task = experiment_id
        start_time = time.perf_counter()
        logs = []

        try:
            # Parameters
            backend = params.get('backend', 'cpu')
            num_threads = params.get('num_threads', 4)
            warmup_runs = params.get('warmup_runs', 10)
            benchmark_runs = params.get('benchmark_runs', 100)
            tpu_device_index = int(params.get('tpu_index', 0))

            logs.append(f'Loading model: {model_path}')
            logs.append(
                f'Backend: {backend}, Threads: {num_threads}, TPU index: {tpu_device_index}'
            )

            # Load interpreter
            interpreter, model_load_time = self._load_interpreter(
                model_path, backend, num_threads, tpu_device_index
            )
            logs.append(f'Model loaded in {model_load_time:.2f}ms')

            # Get input details
            input_details = interpreter.get_input_details()
            input_shape = input_details[0]['shape']
            input_dtype = input_details[0]['dtype']

            logs.append(f'Input shape: {input_shape}, dtype: {input_dtype}')

            # Seed before generating input — quantized kernels can select different
            # paths depending on input distribution, so reproducibility matters.
            input_seed = int(
                params.get('input_seed', os.environ.get('EDGEBENCH_INPUT_SEED', '42'))
            )
            np.random.seed(input_seed)

            # Generate dummy input
            input_data = self._generate_input(input_shape, input_dtype)

            # Capture baseline CPU frequency + governor for throttle detection
            try:
                import psutil as _psutil

                _freq = _psutil.cpu_freq()
                initial_cpu_freq_mhz: float | None = (
                    round(_freq.current, 0) if _freq else None
                )
            except Exception:
                initial_cpu_freq_mhz = None

            try:
                with open(
                    '/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor'
                ) as _gf:
                    _cpu_governor: str | None = _gf.read().strip()
            except Exception:
                _cpu_governor = None

            # Warmup
            logs.append(f'Running {warmup_runs} warmup iterations...')
            first_inference_time = None

            for i in range(warmup_runs):
                interpreter.set_tensor(input_details[0]['index'], input_data)
                _gc_was = gc.isenabled()
                gc.disable()
                t0 = time.perf_counter()
                interpreter.invoke()
                t1 = time.perf_counter()
                if _gc_was:
                    gc.enable()

                if i == 0:
                    first_inference_time = (t1 - t0) * 1000

            # Benchmark
            logs.append(f'Running {benchmark_runs} benchmark iterations...')
            latencies = []

            # Start system metrics collection in background
            metrics_task = asyncio.create_task(
                self._collect_metrics_async(benchmark_runs * 0.015)  # Estimate duration
            )

            stream_interval = max(1, benchmark_runs // 20)  # ~20 updates total

            for run_idx in range(benchmark_runs):
                interpreter.set_tensor(input_details[0]['index'], input_data)
                # GC disabled only for the timed invoke(); restored before the rest
                # of the loop body so memory metrics are not distorted.
                _gc_was = gc.isenabled()
                gc.disable()
                t0 = time.perf_counter()
                interpreter.invoke()
                t1 = time.perf_counter()
                if _gc_was:
                    gc.enable()
                latencies.append((t1 - t0) * 1000)  # Convert to ms

                # Stream live metric update every stream_interval runs
                if stream_callback_url and (run_idx + 1) % stream_interval == 0:
                    current_mean = float(np.mean(latencies))
                    current_fps = (
                        round(1000.0 / current_mean, 2) if current_mean > 0 else 0.0
                    )
                    asyncio.create_task(
                        self._send_metric(
                            stream_callback_url,
                            {
                                'type': 'metric',
                                'latency_ms': round(current_mean, 3),
                                'fps': current_fps,
                                'run': run_idx + 1,
                                'total_runs': benchmark_runs,
                            },
                        )
                    )

            # Wait for metrics
            system_metrics = await metrics_task

            # Thermal throttle warnings — do not abort; flag in results
            _THROTTLE_TEMP_C = 80.0
            _FREQ_DROP_THRESHOLD = 0.15  # 15 % drop triggers warning
            warnings: list[str] = []

            cpu_temp_max = system_metrics.get('cpu_temp_max')
            if cpu_temp_max is not None and cpu_temp_max >= _THROTTLE_TEMP_C:
                warnings.append(
                    f'Thermal throttle risk: CPU peaked at {cpu_temp_max:.1f}°C '
                    f'(threshold {_THROTTLE_TEMP_C:.0f}°C) — results may be degraded'
                )

            freq_min = system_metrics.get('cpu_freq_mhz_min')
            if freq_min is not None and initial_cpu_freq_mhz is not None:
                drop = (initial_cpu_freq_mhz - freq_min) / initial_cpu_freq_mhz
                if drop >= _FREQ_DROP_THRESHOLD:
                    # Under 'performance' governor the CPU stays at max, so any
                    # drop is almost certainly thermal throttling.  Under
                    # 'ondemand' / 'powersave' the kernel can scale down
                    # between invocations — that is normal idle scaling, not
                    # throttling.  We reflect this in the warning text to
                    # reduce false positives.
                    if _cpu_governor == 'performance':
                        freq_cause = 'thermal throttling confirmed (governor=performance, drop is hardware-enforced)'
                    elif _cpu_governor in (
                        'ondemand',
                        'powersave',
                        'schedutil',
                        'conservative',
                    ):
                        freq_cause = (
                            f'possible idle frequency scaling (governor={_cpu_governor}); '
                            'may NOT be thermal throttling — consider governor=performance for benchmarks'
                        )
                    else:
                        freq_cause = f'possible thermal throttling (governor={_cpu_governor or "unknown"})'
                    warnings.append(
                        f'CPU frequency drop: {initial_cpu_freq_mhz:.0f} → {freq_min:.0f} MHz '
                        f'({drop * 100:.0f}% drop) — {freq_cause}'
                    )

            for w in warnings:
                logs.append(f'WARNING: {w}')

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

            params['input_seed'] = input_seed  # record used seed in result
            _fps_from_mean = round(1000.0 / latency_stats['mean_ms'], 2)
            _fps_from_median = (
                round(1000.0 / latency_stats['p50_ms'], 2)
                if latency_stats['p50_ms'] > 0
                else 0.0
            )
            throughput = {
                'fps_from_mean': _fps_from_mean,
                'fps_from_median': _fps_from_median,
                # Backward-compatible aliases (templates and API consumers use these)
                'fps': _fps_from_mean,
                'images_per_second': _fps_from_mean,
            }

            cold_start = {
                'model_load_ms': round(model_load_time, 2),
                'first_inference_ms': round(first_inference_time, 2)
                if first_inference_time
                else 0,
            }

            # Output signature — captured once, after the timed loop, so it
            # costs nothing in the measurement. Without it a benchmark can
            # report excellent latency for a model that computes nonsense on
            # this backend; this is what makes CPU/Edge TPU results comparable.
            output_signature = self._output_signature(interpreter)

            # Model info
            model_info = {
                'name': os.path.basename(model_path),
                'hash': self._file_hash(model_path),
                'size_bytes': os.path.getsize(model_path),
                'quantization': self._detect_quantization(model_path),
                'input_shape': [int(d) for d in input_shape],
                'input_dtype': np.dtype(input_dtype).name,
            }

            # Runtime provenance: which inference stack actually produced these
            # numbers. Additive — existing consumers are unaffected.
            runtime_info = {
                'tflite_source': self.tflite_source,
                'tflite_version': backend_version(self.tflite_source)
                if self.tflite_source
                else None,
                'numpy_version': np.__version__,
                'python_version': platform.python_version(),
                'cpu_governor': _cpu_governor,
            }

            duration = time.perf_counter() - start_time
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
                'runtime': runtime_info,
                'output': output_signature,
                'timestamp': datetime.now(UTC).isoformat(),
                'duration_seconds': round(duration, 2),
                'status': 'completed',
                # Non-empty when thermal throttle or frequency drop detected.
                # Results are still valid but should be treated with caution.
                'warnings': warnings,
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
                'timestamp': datetime.now(UTC).isoformat(),
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
        start_time = time.perf_counter()

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

            duration = time.perf_counter() - start_time

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
        tpu_device_index: int = 0,
    ):
        """Load TFLite interpreter with optional Edge TPU delegate."""
        t0 = time.perf_counter()

        # Check Edge TPU availability first
        if backend == 'edgetpu':
            tpu_devices = self.metrics.detect_tpu_devices()
            if not tpu_devices:
                raise RuntimeError(
                    'Edge TPU not detected. Check USB connection and run: '
                    'lsusb | grep -i google'
                )
            if tpu_device_index >= len(tpu_devices):
                raise RuntimeError(
                    f'TPU index {tpu_device_index} out of range '
                    f'({len(tpu_devices)} device(s) found)'
                )

        def _build_edgetpu_delegate(load_delegate_fn):
            """Build Edge TPU delegate with optional device path."""
            lib_paths = [
                'libedgetpu.so.1',
                '/usr/lib/aarch64-linux-gnu/libedgetpu.so.1',
                '/usr/lib/arm-linux-gnueabihf/libedgetpu.so.1',
            ]
            # Build device option for multi-TPU selection
            tpu_devices = self.metrics.detect_tpu_devices()
            options = {}
            if tpu_devices and tpu_device_index < len(tpu_devices):
                dev = tpu_devices[tpu_device_index]
                if dev.startswith('/dev/apex_'):
                    options['device'] = dev

            last_error = None
            for lib_path in lib_paths:
                try:
                    return load_delegate_fn(lib_path, options)
                except (ValueError, OSError) as e:
                    last_error = e
            raise RuntimeError(
                f'Cannot load Edge TPU delegate. '
                f'Install libedgetpu: sudo apt install libedgetpu1-std. '
                f'Last error: {last_error}'
            )

        # Resolve whichever TFLite runtime is installed (see tflite_backend).
        interpreter_cls, load_delegate, self.tflite_source = resolve_backend()

        if backend == 'edgetpu':
            if load_delegate is None:
                raise RuntimeError(
                    f'TFLite runtime "{self.tflite_source}" exposes no load_delegate; '
                    'Edge TPU is unavailable with this runtime.'
                )
            delegate = _build_edgetpu_delegate(load_delegate)
            interpreter = interpreter_cls(
                model_path=model_path,
                experimental_delegates=[delegate],
            )
        else:
            interpreter = interpreter_cls(
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

    @staticmethod
    def _generate_input(input_shape, input_dtype) -> np.ndarray:
        """Build a synthetic input tensor of the model's own dtype.

        Integer dtypes must be filled over their real range. The previous
        version fell through to ``np.random.rand(...).astype(dtype)`` for any
        dtype that was not float32/uint8 — and since ``rand`` returns values in
        [0, 1), casting to int8 truncated every element to zero. Every INT8
        model in this project therefore benchmarked an all-zero tensor: the
        input seed had no effect, and the all-zero input drove the quantized
        network into a degenerate state whose output varied between
        interpreter instances, making results irreproducible.
        """
        dtype = np.dtype(input_dtype)

        if np.issubdtype(dtype, np.integer):
            info = np.iinfo(dtype)
            return np.random.randint(
                info.min, info.max + 1, size=input_shape, dtype=dtype
            )

        if np.issubdtype(dtype, np.floating):
            return np.random.rand(*input_shape).astype(dtype)

        raise ValueError(f'Unsupported model input dtype: {dtype}')

    def _output_signature(self, interpreter) -> dict[str, Any] | None:
        """Summarise the model's output tensor for cross-platform comparison.

        Two devices running the same model on the same seeded input must
        agree on what the model computed, not merely on how fast it ran.
        The full tensor is not stored — a compact signature is enough to
        detect a miscompiled Edge TPU model or a broken delegate:

        - ``top_k``      : indices of the largest values (classifier verdict)
        - ``checksum``   : sha256 over the dequantised values, exact match test
        - ``sum``/``mean``/``norm`` : tolerant comparison for float paths

        Values are dequantised first, so an INT8 CPU model and its Edge TPU
        build are compared in the same units.
        """
        try:
            details = interpreter.get_output_details()
            if not details:
                return None
            spec = details[0]
            raw = interpreter.get_tensor(spec['index'])

            values = np.asarray(raw).astype(np.float64).reshape(-1)
            # Undo affine quantisation when the tensor is quantised.
            scale, zero_point = spec.get('quantization', (0.0, 0)) or (0.0, 0)
            if scale:
                values = (values - float(zero_point)) * float(scale)

            k = int(min(5, values.size))
            top_k = np.argsort(values)[::-1][:k]

            return {
                'shape': [int(d) for d in np.asarray(raw).shape],
                'dtype': np.asarray(raw).dtype.name,
                'quantization': {'scale': float(scale), 'zero_point': int(zero_point)},
                'top_k_indices': [int(i) for i in top_k],
                'top_k_values': [round(float(values[i]), 6) for i in top_k],
                'sum': round(float(values.sum()), 6),
                'mean': round(float(values.mean()), 6),
                'l2_norm': round(float(np.linalg.norm(values)), 6),
                'checksum': hashlib.sha256(np.round(values, 6).tobytes()).hexdigest()[
                    :16
                ],
            }
        except Exception as exc:
            # Never fail a benchmark because the signature could not be taken.
            return {'error': str(exc)}

    def _file_hash(self, path: str) -> str:
        """Calculate SHA256 hash of file."""
        sha256 = hashlib.sha256()
        with open(path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return f'sha256:{sha256.hexdigest()[:16]}'

    def _detect_quantization(self, model_path: str) -> str | None:
        """Detect model quantization type from the filename.

        Edge TPU must be checked first: every Edge TPU build in this project
        is named `*_int8_edgetpu.tflite`, so testing `int8` first made the
        `edgetpu` branch unreachable and recorded plain `int8` in results
        while the models page showed `int8_edgetpu` for the same file.

        Kept byte-for-byte in step with the server-side classifier in
        `server/routes/ui.py` — see `tests/test_model_platforms.py`.
        """
        name = os.path.basename(model_path).lower()

        if 'edgetpu' in name:
            return 'int8_edgetpu'
        elif 'int8' in name or '_quant' in name:
            return 'int8'
        elif 'fp16' in name:
            return 'fp16'
        elif 'fp32' in name:
            return 'fp32'

        return None
