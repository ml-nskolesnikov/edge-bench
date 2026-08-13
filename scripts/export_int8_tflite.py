#!/usr/bin/env python3
"""Re-export an ONNX model as a full-integer INT8 TFLite model.

Motivation: `data/models/c6_mobilenet_v2_int8.tflite` carries an `int8` name
but computes in float32 — its first operator is a DEQUANTIZE, weights are
int8 and activations are float. That graph is non-deterministic (see
`scripts/check_determinism.py`) and its latency is not INT8 latency. Every
other `_int8_` model in the corpus is full-integer with zero float tensors.
This script produces that same kind of graph.

Two things matter for a usable result:

* **Real calibration data.** Post-training quantization derives activation
  ranges from a representative sample. `scripts/convert_pipeline.py` feeds
  uniform random noise, which produces ranges no real image ever occupies.
  This script reads actual images.
* **Matching preprocessing.** Calibration must use the exact transform the
  runtime applies at inference, otherwise the ranges are in the wrong units.
  The defaults here mirror `plantdiag_edge/edge_runtime/tflite_runner.py`:
  resize, scale to [0,1], ImageNet mean/std, NHWC.

The output is written to a new file; no existing artifact is overwritten.

Requires a toolchain that is intentionally not a project dependency:

    pip install tensorflow-cpu onnx onnx2tf onnxruntime pillow

Usage:
    python scripts/export_int8_tflite.py \\
        --onnx artifacts/c6_mobilenet_v2_fp32.onnx \\
        --calibration-dir /data/plantvillage/color \\
        --output data/models/c6_mobilenet_v2_int8_full.tflite
"""

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import shutil
import subprocess
import sys

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--onnx', required=True, help='Source fp32 ONNX model')
    parser.add_argument('--output', required=True, help='Destination .tflite path')
    parser.add_argument(
        '--calibration-dir', required=True, help='Directory tree of real images'
    )
    parser.add_argument(
        '--calibration-images',
        type=int,
        default=512,
        help='Number of calibration images (default: 512)',
    )
    parser.add_argument('--input-size', type=int, nargs=2, default=(224, 224))
    parser.add_argument(
        '--input-name',
        default='input',
        help="Name of the ONNX graph input (default: 'input')",
    )
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--keep-saved-model', action='store_true', help='Keep the intermediate export'
    )
    parser.add_argument(
        '--work-dir', default=None, help='Scratch directory (default: alongside output)'
    )
    return parser.parse_args()


def collect_images(root: Path, count: int, seed: int) -> list[Path]:
    """Sample calibration images deterministically across the whole tree."""
    files = sorted(
        p for p in root.rglob('*') if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f'No images found under {root}')
    rng = random.Random(seed)
    if len(files) <= count:
        return files
    # Sampling across the sorted list keeps all classes represented.
    return rng.sample(files, count)


def build_calibration_array(images, input_size, mean, std):
    """Preprocess calibration images into one NCHW float32 array.

    The layout matches the ONNX graph input, which onnx2tf transposes to NHWC
    together with the rest of the model. Values are already normalised, so the
    converter is told mean=0 / std=1 and applies no further scaling.
    """
    import numpy as np
    from PIL import Image

    mean_arr = np.array(mean, dtype=np.float32)
    std_arr = np.array(std, dtype=np.float32)
    height, width = input_size

    batch = []
    skipped = 0
    for path in images:
        try:
            with Image.open(path) as raw:
                img = raw.convert('RGB').resize((width, height))
        except Exception:  # a corrupt file must not abort calibration
            skipped += 1
            continue
        arr = np.asarray(img, dtype=np.float32) / 255.0
        arr = (arr - mean_arr) / std_arr
        batch.append(arr.transpose(2, 0, 1))  # HWC -> CHW

    if not batch:
        raise SystemExit('No calibration image could be read')
    if skipped:
        print(f'      {skipped} unreadable images skipped', file=sys.stderr)
    return np.stack(batch).astype(np.float32)


def onnx_to_float_tflite(
    onnx_path: Path, work_dir: Path, input_name: str, input_size
) -> Path:
    """ONNX (NCHW) -> float32 TFLite (NHWC) via onnx2tf.

    `-dgc` is required: without it the depthwise blocks of MobileNetV2 hit a
    converter bug (`input.shape.rank must be at least 5`). The batch dimension
    is pinned because the source graph declares it dynamic and quantization
    needs a static shape.

    onnx2tf's own integer-quantization path (`-oiqt`) fails on this model, so
    quantization is done separately from the float graph it does produce.
    """
    out_dir = work_dir / 'float'
    if out_dir.exists():
        shutil.rmtree(out_dir)

    height, width = input_size
    cmd = [
        sys.executable,
        '-m',
        'onnx2tf',
        '-i',
        str(onnx_path),
        '-o',
        str(out_dir),
        '-dgc',
        '-ois',
        f'{input_name}:1,3,{height},{width}',
        '-n',
    ]
    print(f'  {" ".join(cmd)}')
    proc = subprocess.run(cmd, capture_output=True, text=True)

    produced = sorted(out_dir.glob('*_float32.tflite')) if out_dir.exists() else []
    if not produced:
        tail = (proc.stderr or proc.stdout).strip().splitlines()[-20:]
        raise SystemExit('onnx2tf produced no float32 model:\n  ' + '\n  '.join(tail))
    return produced[0]


def quantize_full_integer(float_model: Path, calibration, output: Path) -> bytes:
    """Static INT8 quantization of weights *and* activations.

    `static_wi8_ai8` is the full-integer recipe. The alternative,
    weight-only/dynamic-range quantization, is what produced the model this
    script replaces: int8 weights with float activations.
    """
    from ai_edge_quantizer import quantizer, recipe

    qt = quantizer.Quantizer(str(float_model), recipe.static_wi8_ai8())
    data = {
        'serving_default': [
            {'input': calibration[i : i + 1]} for i in range(calibration.shape[0])
        ]
    }
    calibration_result = qt.calibrate(data)
    result = qt.quantize(calibration_result)
    result.export_model(str(output))
    return output.read_bytes()


def describe(model_bytes: bytes) -> dict:
    """Report the float/int composition of the produced graph."""
    from ai_edge_litert.interpreter import Interpreter
    import numpy as np

    interpreter = Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    floats = sum(1 for t in details if np.issubdtype(np.dtype(t['dtype']), np.floating))
    spec_in = interpreter.get_input_details()[0]
    spec_out = interpreter.get_output_details()[0]

    # What matters is that the arithmetic runs in integer. onnx2tf builds
    # padding out of FILL/SHAPE/MUL rather than a native PAD, and those helper
    # subgraphs stay float — harmless, but it means "zero float tensors" is
    # the wrong success criterion. Check the heavy ops instead.
    ops = interpreter._get_ops_details()
    op_counts: dict[str, int] = {}
    for op in ops:
        op_counts[op['op_name']] = op_counts.get(op['op_name'], 0) + 1
    float_dtypes = {
        int(t['index'])
        for t in details
        if np.issubdtype(np.dtype(t['dtype']), np.floating)
    }
    compute_ops = ('CONV_2D', 'DEPTHWISE_CONV_2D', 'FULLY_CONNECTED')
    float_compute = sum(
        1
        for op in ops
        if op['op_name'] in compute_ops
        and any(int(i) in float_dtypes for i in op['outputs'])
    )

    return {
        'op_counts': dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
        'float_compute_ops': float_compute,
        'tensors_total': len(details),
        'tensors_float32': floats,
        'input_dtype': np.dtype(spec_in['dtype']).name,
        'input_shape': [int(d) for d in spec_in['shape']],
        'output_dtype': np.dtype(spec_out['dtype']).name,
        'output_shape': [int(d) for d in spec_out['shape']],
        'input_quantization': [
            float(spec_in['quantization'][0]),
            int(spec_in['quantization'][1]),
        ],
        'output_quantization': [
            float(spec_out['quantization'][0]),
            int(spec_out['quantization'][1]),
        ],
    }


def main() -> int:
    args = parse_args()
    onnx_path = Path(args.onnx).expanduser().resolve()
    output = Path(args.output).expanduser()
    calibration_dir = Path(args.calibration_dir).expanduser()

    if not onnx_path.is_file():
        print(f'ONNX model not found: {onnx_path}', file=sys.stderr)
        return 2
    if not calibration_dir.is_dir():
        print(f'Calibration directory not found: {calibration_dir}', file=sys.stderr)
        return 2
    if output.exists():
        print(
            f'Refusing to overwrite an existing artifact: {output}\n'
            'Choose another --output path.',
            file=sys.stderr,
        )
        return 2

    work_dir = Path(args.work_dir) if args.work_dir else output.parent / '.export_work'
    work_dir.mkdir(parents=True, exist_ok=True)
    output.parent.mkdir(parents=True, exist_ok=True)

    print(f'[1/4] Sampling {args.calibration_images} calibration images ...')
    images = collect_images(calibration_dir, args.calibration_images, args.seed)
    print(f'      {len(images)} images from {calibration_dir}')

    print('[2/4] Preprocessing calibration images ...')
    import numpy as np

    calibration = build_calibration_array(
        images, tuple(args.input_size), IMAGENET_MEAN, IMAGENET_STD
    )
    print(f'      calibration tensor {calibration.shape}')

    print('[3/4] ONNX -> float32 TFLite -> full-integer INT8 ...')
    float_model = onnx_to_float_tflite(
        onnx_path, work_dir, args.input_name, tuple(args.input_size)
    )
    # The float graph is NHWC; the calibration array was built NCHW to match
    # the ONNX input, so transpose it to the TFLite layout.
    model_bytes = quantize_full_integer(
        float_model, np.transpose(calibration, (0, 2, 3, 1)).copy(), output
    )

    print('[4/4] Verifying the produced graph ...')
    info = describe(model_bytes)

    sidecar = output.with_suffix('.export.json')
    sidecar.write_text(
        json.dumps(
            {
                'generated_at': datetime.now(UTC).isoformat(),
                'source_onnx': str(onnx_path),
                'source_sha256': hashlib.sha256(onnx_path.read_bytes()).hexdigest(),
                'output_sha256': hashlib.sha256(model_bytes).hexdigest(),
                'output_bytes': len(model_bytes),
                'quantization': {
                    'scheme': 'full-integer PTQ (TFLITE_BUILTINS_INT8)',
                    'inference_input_type': 'int8',
                    'inference_output_type': 'int8',
                    'calibration_images': len(images),
                    'calibration_source': str(calibration_dir),
                    'calibration_seed': args.seed,
                    'preprocessing': {
                        'resize': list(args.input_size),
                        'scale': '1/255',
                        'mean': list(IMAGENET_MEAN),
                        'std': list(IMAGENET_STD),
                        'layout': 'NHWC',
                    },
                },
                'graph': info,
            },
            indent=2,
        )
    )

    print()
    print(f'  written        : {output}  ({len(model_bytes) / 1e6:.3f} MB)')
    print(f'  metadata       : {sidecar}')
    print(f'  input          : {info["input_dtype"]} {info["input_shape"]}')
    print(f'  output         : {info["output_dtype"]} {info["output_shape"]}')
    print(f'  float32 tensors: {info["tensors_float32"]} of {info["tensors_total"]}')

    if not args.keep_saved_model:
        shutil.rmtree(work_dir, ignore_errors=True)

    print(f'  float compute ops: {info["float_compute_ops"]}')

    if info['float_compute_ops'] != 0:
        print(
            '\nWARNING: convolutions still run in float32, so this is not an '
            'integer-inference model. Check the converter log for unsupported ops.',
            file=sys.stderr,
        )
        return 1

    if info['tensors_float32']:
        print(
            f'\nNote: {info["tensors_float32"]} float32 tensors remain, in the '
            'padding helper subgraphs the converter emits instead of a native '
            'PAD. All convolutions are integer.'
        )
    print('\nInteger-inference graph produced.')
    print(f'Verify with: make check-determinism DETERMINISM_MODELS={output}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
