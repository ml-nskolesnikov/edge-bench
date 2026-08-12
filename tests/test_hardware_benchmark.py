"""Real-hardware benchmark tests.

Excluded from the default run (`addopts = -m 'not hardware'`). Run with:

    make test-hardware        # or: poetry run pytest -m hardware

These tests require a real TFLite runtime and a real model file — they are
skipped, never faked, when either is missing. No mock interpreter stands in
for the hardware path: the point is to prove the real one works.
"""

import asyncio
import json
from pathlib import Path
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / 'agent'
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

pytestmark = pytest.mark.hardware


def _find_model() -> Path | None:
    for rel in ('data/models', 'models'):
        directory = ROOT / rel
        if directory.is_dir():
            models = sorted(directory.glob('*.tflite'), key=lambda p: p.stat().st_size)
            if models:
                return models[0]
    return None


@pytest.fixture(scope='module')
def tflite_source() -> str:
    from tflite_backend import TFLiteBackendError, resolve_backend

    try:
        _, _, source = resolve_backend()
    except TFLiteBackendError as exc:
        pytest.skip(f'No TFLite runtime installed: {exc}')
    return source


@pytest.fixture(scope='module')
def model_path() -> Path:
    model = _find_model()
    if model is None:
        pytest.skip('No .tflite model available in data/models or models')
    return model


@pytest.fixture(scope='module')
def benchmark_result(tflite_source: str, model_path: Path) -> dict:
    """One real inference run, shared by the assertions below."""
    from executor import BenchmarkExecutor

    executor = BenchmarkExecutor()
    return asyncio.run(
        executor.run_benchmark(
            experiment_id='pytest_hardware_smoke',
            model_path=str(model_path),
            params={
                'backend': 'cpu',
                'num_threads': 2,
                'warmup_runs': 2,
                'benchmark_runs': 10,
                'input_seed': 42,
            },
        )
    )


def test_benchmark_completes(benchmark_result: dict):
    assert benchmark_result['status'] == 'completed', benchmark_result.get('error')


def test_latency_is_measured(benchmark_result: dict):
    latency = benchmark_result['latency']
    assert latency['mean_ms'] > 0
    assert latency['min_ms'] <= latency['p50_ms'] <= latency['max_ms']
    assert latency['p50_ms'] <= latency['p95_ms'] <= latency['p99_ms']


def test_throughput_matches_latency(benchmark_result: dict):
    mean_ms = benchmark_result['latency']['mean_ms']
    fps = benchmark_result['throughput']['fps_from_mean']
    assert fps == pytest.approx(1000.0 / mean_ms, rel=0.01)
    # Backward-compatible alias must keep tracking fps_from_mean.
    assert benchmark_result['throughput']['fps'] == fps


def test_memory_is_measured(benchmark_result: dict):
    assert benchmark_result['system']['process_rss_mb_max'] > 0


def test_runtime_provenance_recorded(benchmark_result: dict, tflite_source: str):
    runtime = benchmark_result['runtime']
    assert runtime['tflite_source'] == tflite_source
    assert runtime['tflite_version']
    assert runtime['python_version']


def test_result_is_json_serializable(benchmark_result: dict):
    """A result that cannot be stored is not a result."""
    encoded = json.dumps(benchmark_result)
    assert json.loads(encoded)['experiment_id'] == 'pytest_hardware_smoke'


def test_seed_makes_input_reproducible(model_path: Path, tflite_source: str):
    """Same seed, same model, same host -> identical model hash and input shape."""
    from executor import BenchmarkExecutor

    def run() -> dict:
        return asyncio.run(
            BenchmarkExecutor().run_benchmark(
                experiment_id='pytest_seed_check',
                model_path=str(model_path),
                params={
                    'backend': 'cpu',
                    'num_threads': 2,
                    'warmup_runs': 1,
                    'benchmark_runs': 3,
                    'input_seed': 1234,
                },
            )
        )

    first, second = run(), run()
    assert first['status'] == 'completed'
    assert first['model']['hash'] == second['model']['hash']
    assert first['model']['input_shape'] == second['model']['input_shape']
    assert first['params']['input_seed'] == second['params']['input_seed'] == 1234
