"""Model behaviour across runtimes, backends and devices.

Split in two:

* Module-level tests are pure and run in normal CI — they cover the logic that
  *decides* how a model is executed and classified, with no runtime installed.
* Tests marked `hardware` run real inference and are excluded by default.

The central idea is that a benchmark which only measures time is not enough:
two devices must agree on *what the model computed*, otherwise a miscompiled
Edge TPU build reports excellent latency for nonsense.
"""

from pathlib import Path
import sys
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT / 'agent'
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))


# ---------------------------------------------------------------------------
# Quantization classification — agent and server must never disagree
# ---------------------------------------------------------------------------

# Real filenames from data/models plus the edge cases around them.
MODEL_NAMES = [
    'mobilenetv2_int8_ptq_hybrid.tflite',
    'mobilenetv2_int8_ptq_hybrid_edgetpu.tflite',
    'efficientnet_lite0_int8_ptq_Fuzzy_edgetpu_1.tflite',
    'resnet50_int8_ptq_sbert.tflite',
    'c6_mobilenet_v2_int8.tflite',
    'model_fp32_hybrid.tflite',
    'model_fp16.tflite',
    'weights_quant.tflite',
    'MobileNetV2_INT8_EDGETPU.TFLITE',
    'plain_model.tflite',
]


@pytest.mark.parametrize('name', MODEL_NAMES)
def test_quantization_classification_agrees_between_agent_and_server(name):
    """The result JSON and the models page must label a file identically.

    They previously did not: the agent tested `int8` before `edgetpu`, making
    its `edgetpu` branch unreachable, so a `*_int8_edgetpu.tflite` file was
    recorded as `int8` in results while the UI displayed `int8_edgetpu`.
    """
    from executor import BenchmarkExecutor

    from server.routes.ui import detect_quantization as server_detect

    agent_verdict = BenchmarkExecutor()._detect_quantization(name)
    assert agent_verdict == server_detect(name), (
        f'{name}: agent says {agent_verdict!r}, server says {server_detect(name)!r}'
    )


@pytest.mark.parametrize(
    'name,expected',
    [
        ('mobilenetv2_int8_edgetpu.tflite', 'int8_edgetpu'),
        ('mobilenetv2_int8.tflite', 'int8'),
        ('model_fp32.tflite', 'fp32'),
        ('model_fp16.tflite', 'fp16'),
        ('weights_quant.tflite', 'int8'),
        ('plain.tflite', None),
    ],
)
def test_quantization_classification_values(name, expected):
    """Edge TPU builds must not be recorded as plain int8."""
    from executor import BenchmarkExecutor

    assert BenchmarkExecutor()._detect_quantization(name) == expected


def test_quantization_uses_basename_not_path():
    """A directory called 'edgetpu' must not reclassify a CPU model."""
    from executor import BenchmarkExecutor

    verdict = BenchmarkExecutor()._detect_quantization('/models/edgetpu/plain.tflite')
    assert verdict is None


# ---------------------------------------------------------------------------
# Runtime resolution — which inference stack a platform ends up using
# ---------------------------------------------------------------------------


def test_runtime_preference_order_is_documented_and_real():
    """tflite_runtime must win over ai_edge_litert where both exist.

    The Pi ships `tflite_runtime`; changing which stack it picks would
    silently change measured latency on the reference device.
    """
    import tflite_backend

    candidates = [entry[0] for entry in tflite_backend._CANDIDATES]
    assert candidates.index('tflite_runtime.interpreter') < candidates.index(
        'ai_edge_litert.interpreter'
    )


def test_resolve_backend_falls_back_to_tensorflow(monkeypatch):
    """With no slim runtime installed, full TensorFlow is used."""
    import tflite_backend

    fake_tf = mock.Mock()
    fake_tf.lite.Interpreter = 'TF_INTERPRETER'
    fake_tf.lite.experimental.load_delegate = 'TF_DELEGATE'

    def only_tensorflow(path):
        if path == 'tensorflow':
            return fake_tf
        raise ImportError(path)

    monkeypatch.setattr(tflite_backend, '_import_module', only_tensorflow)
    monkeypatch.setitem(sys.modules, 'tensorflow', fake_tf)

    interpreter, load_delegate, source = tflite_backend.resolve_backend()
    assert source == 'tensorflow'
    assert interpreter == 'TF_INTERPRETER'
    assert load_delegate == 'TF_DELEGATE'


def test_resolve_backend_raises_when_nothing_installed(monkeypatch):
    """A missing runtime must produce an actionable error, not an AttributeError."""
    import tflite_backend

    def nothing_importable(path):
        raise ImportError(path)

    monkeypatch.setattr(tflite_backend, '_import_module', nothing_importable)
    monkeypatch.setitem(sys.modules, 'tensorflow', None)
    monkeypatch.delitem(sys.modules, 'tensorflow')

    with mock.patch.dict(sys.modules, {'tensorflow': None}):
        with pytest.raises(tflite_backend.TFLiteBackendError) as excinfo:
            tflite_backend.resolve_backend()

    message = str(excinfo.value)
    for hint in ('tflite-runtime', 'ai-edge-litert', 'tensorflow'):
        assert hint in message


def test_backend_version_unknown_source_returns_none():
    import tflite_backend

    assert tflite_backend.backend_version('not-a-runtime') is None


# ---------------------------------------------------------------------------
# Edge TPU selection — refuse clearly, rather than silently measuring CPU
# ---------------------------------------------------------------------------


def test_edgetpu_requested_without_device_raises():
    """Asking for Edge TPU with no TPU present must fail loudly.

    Falling back to CPU here would publish a CPU number under an Edge TPU
    label — the worst possible failure mode for a comparison table.
    """
    import asyncio

    from executor import BenchmarkExecutor

    executor = BenchmarkExecutor()
    with mock.patch.object(executor.metrics, 'detect_tpu_devices', return_value=[]):
        result = asyncio.run(
            executor.run_benchmark(
                experiment_id='no_tpu',
                model_path=str(ROOT / 'nonexistent.tflite'),
                params={'backend': 'edgetpu', 'warmup_runs': 1, 'benchmark_runs': 1},
            )
        )

    assert result['status'] == 'failed'
    assert 'Edge TPU not detected' in result['error']


def test_edgetpu_index_out_of_range_raises():
    """Selecting TPU #2 on a single-TPU host must be rejected."""
    import asyncio

    from executor import BenchmarkExecutor

    executor = BenchmarkExecutor()
    with mock.patch.object(
        executor.metrics, 'detect_tpu_devices', return_value=['/dev/apex_0']
    ):
        result = asyncio.run(
            executor.run_benchmark(
                experiment_id='bad_tpu_index',
                model_path=str(ROOT / 'nonexistent.tflite'),
                params={
                    'backend': 'edgetpu',
                    'tpu_index': 5,
                    'warmup_runs': 1,
                    'benchmark_runs': 1,
                },
            )
        )

    assert result['status'] == 'failed'
    assert 'out of range' in result['error']


def test_missing_model_file_fails_cleanly():
    """A missing model must produce a failed result, not an unhandled crash."""
    import asyncio

    from executor import BenchmarkExecutor

    result = asyncio.run(
        BenchmarkExecutor().run_benchmark(
            experiment_id='missing_model',
            model_path='/definitely/not/here.tflite',
            params={'backend': 'cpu', 'warmup_runs': 1, 'benchmark_runs': 1},
        )
    )
    assert result['status'] == 'failed'
    assert result['error']
    assert 'logs' in result
