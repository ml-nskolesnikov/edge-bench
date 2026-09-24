"""agent/benchmark_pipeline.py: timing loop and the ONNX model mode (no device needed)."""

import argparse
import gc
from pathlib import Path
import sys

import pytest

AGENT_DIR = Path(__file__).resolve().parents[1] / 'agent'
if str(AGENT_DIR) not in sys.path:
    sys.path.insert(0, str(AGENT_DIR))

# benchmark_full.py (imported by benchmark_pipeline) exits at import without a TFLite runtime.
try:
    import benchmark_pipeline as bp
except SystemExit:
    pytest.skip(
        'no TFLite runtime: benchmark_full.py cannot be imported',
        allow_module_level=True,
    )


def test_timed_runs_excludes_warmup_and_cycles_inputs():
    seen = []
    warm, meas = bp.timed_runs(
        lambda x: seen.append(x) or x * 2, [1, 2, 3], warmup=2, runs=5
    )
    assert len(warm) == 2 and len(meas) == 5
    assert seen == [1, 2, 3, 1, 2, 3, 1]
    assert [out for out, _ in meas] == [6, 2, 4, 6, 2]
    assert all(ms >= 0 for _, ms in warm + meas)
    assert gc.isenabled(), 'garbage collection must be restored after timing'


def test_model_mode_on_a_tiny_onnx_graph(tmp_path):
    onnx = pytest.importorskip('onnx')
    pytest.importorskip('onnxruntime')
    from onnx import TensorProto, helper

    x = helper.make_tensor_value_info('input', TensorProto.FLOAT, [1, 4])
    y = helper.make_tensor_value_info('output', TensorProto.FLOAT, [1, 4])
    graph = helper.make_graph(
        [helper.make_node('Relu', ['input'], ['output'])], 'tiny', [x], [y]
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid('', 13)])
    model.ir_version = (
        8  # readable by any onnxruntime >= 1.10, whatever the onnx default
    )
    path = tmp_path / 'tiny.onnx'
    onnx.save(model, str(path))

    args = argparse.Namespace(model=str(path), threads=1, seed=42, warmup=2, runs=10)
    first, second = bp.run_model(args), bp.run_model(args)
    assert first['mode'] == 'model' and first['params']['benchmark_runs'] == 10
    assert len(first['latency']['all_values_ms']) == 10
    assert first['output_signature'] == second['output_signature'], (
        'same seed must give the same output'
    )
