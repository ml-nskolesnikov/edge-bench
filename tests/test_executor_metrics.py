"""Tests for BenchmarkExecutor and SystemMetrics computation logic.

Hardware dependencies (TFLite interpreter, real psutil hardware) are mocked
so these tests run on any machine without a Raspberry Pi or Edge TPU.
"""

import asyncio
import gc
from pathlib import Path
import sys
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

AGENT_DIR = Path(__file__).resolve().parents[1] / 'agent'
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# ── Helpers ────────────────────────────────────────────────────────────────────


def _make_mock_interpreter(shape=(1, 4), dtype=np.float32):
    """Minimal TFLite interpreter mock that satisfies BenchmarkExecutor."""
    mock = MagicMock()
    mock.get_input_details.return_value = [
        {'shape': np.array(shape), 'dtype': dtype, 'index': 0}
    ]
    mock.invoke = MagicMock()
    return mock


def _fake_system_metrics(cpu_temp_max=50.0, cpu_freq_mhz_min=None):
    return {
        'cpu_percent_mean': 10.0,
        'cpu_percent_max': 20.0,
        'process_rss_mb_mean': 100.0,
        'process_rss_mb_max': 110.0,
        'cpu_temp_celsius': cpu_temp_max,
        'cpu_temp_max': cpu_temp_max,
        'cpu_freq_mhz_min': cpu_freq_mhz_min,
        'tpu_detected': False,
    }


def _run_benchmark(bench, params=None, system_metrics=None, freq_mock=None):
    """Run executor.run_benchmark() with all hardware mocked.

    Creates a real temp file so _file_hash() and os.path.getsize() succeed.
    """
    import executor as _exe_mod

    params = params or {'warmup_runs': 2, 'benchmark_runs': 5, 'backend': 'cpu'}
    mock_interp = _make_mock_interpreter()
    metrics = system_metrics if system_metrics is not None else _fake_system_metrics()

    with tempfile.NamedTemporaryFile(suffix='.tflite', delete=False) as tmp:
        tmp.write(b'fake tflite model content for testing')
        tmp.flush()
        model_path = tmp.name

    async def _go():
        with (
            patch.object(bench, '_load_interpreter', return_value=(mock_interp, 1.0)),
            patch.object(
                bench.metrics, 'collect_during_benchmark', return_value=metrics
            ),
            patch.object(
                bench.metrics, 'get_device_info', return_value={'hostname': 'test-host'}
            ),
            patch.object(_exe_mod.result_cache, 'save', return_value='/tmp/test.json'),
            patch('psutil.cpu_freq', return_value=freq_mock),
        ):
            return await bench.run_benchmark('exp_test', model_path, params)

    return asyncio.run(_go())


# ── Latency statistics ─────────────────────────────────────────────────────────


def test_latency_p50_is_median():
    """p50_ms == median of the latency array."""
    latencies = np.array([30.0, 10.0, 50.0, 20.0, 40.0])
    p50 = float(np.percentile(latencies, 50))
    assert p50 == pytest.approx(30.0)


def test_latency_p99_elevated_by_tail():
    """p99 is clearly above the bulk when 10% of samples are 10× slower."""
    # 90 fast samples + 10 slow ones → p99 lands in the slow tail
    latencies = np.array([10.0] * 90 + [100.0] * 10)
    p99 = float(np.percentile(latencies, 99))
    assert p99 == pytest.approx(100.0)


def test_p99_greater_than_mean_greater_than_p50_on_skewed_data():
    """p99 ≥ mean ≥ p50 must hold for right-skewed latencies."""
    latencies = np.array([10.0] * 9 + [1000.0])
    mean_ms = float(np.mean(latencies))
    p50_ms = float(np.percentile(latencies, 50))
    p99_ms = float(np.percentile(latencies, 99))
    assert p99_ms >= mean_ms >= p50_ms


def test_latency_stats_present_in_result():
    """Full result dict must contain the expected latency keys."""
    from executor import BenchmarkExecutor

    result = _run_benchmark(BenchmarkExecutor())
    assert result['status'] == 'completed', f'failed: {result.get("error")}'
    lat = result['latency']
    for key in (
        'mean_ms',
        'std_ms',
        'min_ms',
        'max_ms',
        'p50_ms',
        'p90_ms',
        'p95_ms',
        'p99_ms',
    ):
        assert key in lat, f'Missing latency key: {key}'


def test_fps_from_median_and_mean_both_present():
    """throughput dict must include both fps_from_mean and fps_from_median."""
    from executor import BenchmarkExecutor

    result = _run_benchmark(BenchmarkExecutor())
    assert result['status'] == 'completed', f'failed: {result.get("error")}'
    thr = result['throughput']
    assert 'fps_from_mean' in thr
    assert 'fps_from_median' in thr
    # Backward-compat aliases must still be present
    assert 'fps' in thr
    assert 'images_per_second' in thr
    # fps == fps_from_mean (alias guarantee)
    assert thr['fps'] == thr['fps_from_mean']


# ── GC restoration ────────────────────────────────────────────────────────────


def test_gc_restored_after_benchmark_loop():
    """GC must be re-enabled after each invoke() — disabled only around the timed call."""
    from executor import BenchmarkExecutor

    gc.enable()
    assert gc.isenabled(), 'precondition: GC enabled before benchmark'

    result = _run_benchmark(
        BenchmarkExecutor(),
        params={'warmup_runs': 2, 'benchmark_runs': 6, 'backend': 'cpu'},
    )
    assert gc.isenabled(), 'GC must be enabled after benchmark loop completes'
    assert result['status'] == 'completed', f'failed: {result.get("error")}'


def test_gc_restored_when_starting_disabled():
    """If GC was off before the loop, it must stay off after (no accidental enable)."""
    from executor import BenchmarkExecutor

    gc.disable()
    try:
        result = _run_benchmark(
            BenchmarkExecutor(),
            params={'warmup_runs': 1, 'benchmark_runs': 3, 'backend': 'cpu'},
        )
        assert not gc.isenabled(), (
            'GC must remain disabled if it was disabled before the loop'
        )
        assert result['status'] == 'completed', f'failed: {result.get("error")}'
    finally:
        gc.enable()  # always restore for subsequent tests


# ── Thermal warnings ──────────────────────────────────────────────────────────


def test_thermal_warning_when_temp_exceeds_80():
    """Warning appended to result when cpu_temp_max >= 80°C."""
    from executor import BenchmarkExecutor

    result = _run_benchmark(
        BenchmarkExecutor(),
        system_metrics=_fake_system_metrics(cpu_temp_max=85.0),
    )
    assert result['status'] == 'completed', f'failed: {result.get("error")}'
    assert len(result['warnings']) >= 1
    assert any(
        '85' in w or 'Thermal' in w or 'thermal' in w for w in result['warnings']
    )


def test_no_thermal_warning_when_temp_safe():
    """No thermal warning when CPU temp is safely below 80°C."""
    from executor import BenchmarkExecutor

    result = _run_benchmark(
        BenchmarkExecutor(),
        system_metrics=_fake_system_metrics(cpu_temp_max=55.0),
    )
    assert result['status'] == 'completed', f'failed: {result.get("error")}'
    assert result['warnings'] == []


def test_freq_drop_warning_when_cpu_throttled():
    """Frequency-drop warning fires when CPU freq drops >= 15% from baseline."""
    from executor import BenchmarkExecutor

    # Baseline capture: psutil.cpu_freq().current = 1500 MHz
    # Benchmark min: 1200 MHz → 20% drop (> 15% threshold) → warning expected
    freq_mock = MagicMock()
    freq_mock.current = 1500.0

    result = _run_benchmark(
        BenchmarkExecutor(),
        system_metrics=_fake_system_metrics(cpu_freq_mhz_min=1200.0),
        freq_mock=freq_mock,
    )
    assert result['status'] == 'completed', f'failed: {result.get("error")}'
    assert any(
        '1500' in w or '1200' in w or 'frequency' in w.lower()
        for w in result['warnings']
    )
