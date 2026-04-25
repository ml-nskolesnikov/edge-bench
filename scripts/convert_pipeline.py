#!/usr/bin/env python3
"""
Model Conversion Pipeline: PyTorch/ONNX -> TFLite -> Edge TPU TFLite

Usage:
    python scripts/convert_pipeline.py --input model.pt --input-shape 1 224 224 3 --target edgetpu
    python scripts/convert_pipeline.py --input model.onnx --input-shape 1 224 224 3 --target tflite
"""

import argparse
from pathlib import Path
import subprocess
import sys


def convert_pt_to_onnx(pt_path: Path, output_dir: Path, input_shape: list[int]) -> Path:
    """Convert PyTorch model to ONNX."""
    try:
        import torch

        onnx_path = output_dir / (pt_path.stem + '.onnx')
        model = torch.load(pt_path, map_location='cpu')
        model.eval()

        dummy = torch.randn(*input_shape)
        torch.onnx.export(
            model,
            dummy,
            str(onnx_path),
            opset_version=12,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={'input': {0: 'batch_size'}, 'output': {0: 'batch_size'}},
        )
        print(f'[convert] ONNX saved: {onnx_path}')
        return onnx_path
    except ImportError:
        raise RuntimeError('PyTorch not installed. Install: pip install torch')


def convert_onnx_to_tflite(onnx_path: Path, output_dir: Path, input_shape: list[int]) -> Path:
    """Convert ONNX model to TFLite INT8 via onnx-tf + TFLiteConverter."""
    tflite_path = output_dir / (onnx_path.stem + '_int8.tflite')

    try:
        import numpy as np
        import onnx
        from onnx_tf.backend import prepare
        import tensorflow as tf

        # ONNX -> SavedModel
        onnx_model = onnx.load(str(onnx_path))
        tf_rep = prepare(onnx_model)
        saved_model_dir = str(output_dir / (onnx_path.stem + '_savedmodel'))
        tf_rep.export_graph(saved_model_dir)

        # SavedModel -> TFLite INT8 with PTQ
        converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        # Calibration dataset (random for generic conversion)
        def representative_dataset():
            for _ in range(100):
                yield [np.random.rand(*input_shape).astype(np.float32)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8

        tflite_model = converter.convert()
        tflite_path.write_bytes(tflite_model)
        print(f'[convert] TFLite INT8 saved: {tflite_path}')
        return tflite_path

    except ImportError as e:
        raise RuntimeError(
            f'Missing dependency: {e}. Install: pip install onnx onnx-tf tensorflow'
        )


def convert_tflite_to_edgetpu(tflite_path: Path, output_dir: Path) -> Path:
    """Compile TFLite model for Edge TPU using edgetpu_compiler."""
    # Try local edgetpu_compiler first
    compiler = 'edgetpu_compiler'
    try:
        result = subprocess.run(
            [compiler, '--version'],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if result.returncode != 0:
            raise FileNotFoundError
    except (FileNotFoundError, subprocess.TimeoutExpired):
        raise RuntimeError(
            'edgetpu_compiler not found locally. '
            'Install from https://coral.ai/docs/edgetpu/compiler/ '
            'or use --rpi-host to compile on RPi.'
        )

    result = subprocess.run(
        [compiler, '-s', '-o', str(output_dir), str(tflite_path)],
        capture_output=True,
        text=True,
        timeout=120,
    )

    if result.returncode != 0:
        raise RuntimeError(f'edgetpu_compiler failed:\n{result.stderr}')

    # Compiler adds _edgetpu suffix
    compiled = output_dir / (tflite_path.stem + '_edgetpu.tflite')
    if not compiled.exists():
        raise RuntimeError(f'Expected output not found: {compiled}')

    print(f'[convert] Edge TPU TFLite saved: {compiled}')
    return compiled


def convert_tflite_to_edgetpu_via_ssh(
    tflite_path: Path, output_dir: Path, rpi_host: str
) -> Path:
    """Compile TFLite for Edge TPU via SSH to Raspberry Pi."""
    remote_tmp = f'/tmp/{tflite_path.name}'
    edgetpu_name = tflite_path.stem + '_edgetpu.tflite'
    remote_out = f'/tmp/{edgetpu_name}'
    local_out = output_dir / edgetpu_name

    # Upload
    subprocess.run(['scp', str(tflite_path), f'{rpi_host}:{remote_tmp}'], check=True)

    # Compile
    subprocess.run(
        ['ssh', rpi_host, f'edgetpu_compiler -s -o /tmp {remote_tmp}'],
        check=True,
    )

    # Download
    subprocess.run(['scp', f'{rpi_host}:{remote_out}', str(local_out)], check=True)

    print(f'[convert] Edge TPU TFLite (via SSH) saved: {local_out}')
    return local_out


def run_pipeline(
    input_path: Path,
    output_dir: Path,
    input_shape: list[int],
    target: str = 'edgetpu',
    rpi_host: str | None = None,
) -> dict:
    """Run full conversion pipeline. Returns dict with output paths."""
    output_dir.mkdir(parents=True, exist_ok=True)
    results = {'input': str(input_path), 'steps': []}

    suffix = input_path.suffix.lower()

    # Step 1: to ONNX if PyTorch
    if suffix == '.pt':
        onnx_path = convert_pt_to_onnx(input_path, output_dir, input_shape)
        results['steps'].append({'step': 'pt_to_onnx', 'output': str(onnx_path)})
    elif suffix == '.onnx':
        onnx_path = input_path
    elif suffix == '.tflite':
        onnx_path = None
    else:
        raise ValueError(f'Unsupported input format: {suffix}')

    if target == 'onnx':
        results['final_output'] = str(onnx_path)
        return results

    # Step 2: to TFLite
    if suffix != '.tflite':
        tflite_path = convert_onnx_to_tflite(onnx_path, output_dir, input_shape)
        results['steps'].append({'step': 'onnx_to_tflite', 'output': str(tflite_path)})
    else:
        tflite_path = input_path

    if target == 'tflite':
        results['final_output'] = str(tflite_path)
        return results

    # Step 3: to Edge TPU TFLite
    if rpi_host:
        edgetpu_path = convert_tflite_to_edgetpu_via_ssh(tflite_path, output_dir, rpi_host)
    else:
        edgetpu_path = convert_tflite_to_edgetpu(tflite_path, output_dir)

    results['steps'].append({'step': 'tflite_to_edgetpu', 'output': str(edgetpu_path)})
    results['final_output'] = str(edgetpu_path)
    return results


def main():
    parser = argparse.ArgumentParser(description='Model Conversion Pipeline')
    parser.add_argument('--input', required=True, help='Input model path (.pt, .onnx, .tflite)')
    parser.add_argument(
        '--input-shape',
        nargs='+',
        type=int,
        default=[1, 224, 224, 3],
        help='Input tensor shape (default: 1 224 224 3)',
    )
    parser.add_argument(
        '--target',
        choices=['onnx', 'tflite', 'edgetpu'],
        default='edgetpu',
        help='Target format (default: edgetpu)',
    )
    parser.add_argument(
        '--output-dir',
        default='converted_models',
        help='Output directory (default: converted_models)',
    )
    parser.add_argument(
        '--rpi-host',
        default=None,
        help='SSH host for RPi edgetpu_compiler (e.g. pi@192.168.1.100)',
    )
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f'Error: {input_path} not found', file=sys.stderr)
        sys.exit(1)

    try:
        result = run_pipeline(
            input_path=input_path,
            output_dir=Path(args.output_dir),
            input_shape=args.input_shape,
            target=args.target,
            rpi_host=args.rpi_host,
        )
        print(f'\n[convert] Done. Final output: {result["final_output"]}')
    except Exception as e:
        print(f'Error: {e}', file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()
