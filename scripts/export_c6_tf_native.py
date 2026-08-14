#!/usr/bin/env python3
"""Export C6 through TensorFlow so the Edge TPU compiler accepts it.

Background — why not the obvious routes:

* **ONNX (`onnx2tf`)** cannot produce a compilable graph. MobileNetV2 has five
  stride-2 convolutions with symmetric `pad=1`, which TFLite's conv op cannot
  express (only SAME/VALID), so onnx2tf emulates padding with `SHAPE`/`FILL`.
  That yields dynamic tensors and `edgetpu_compiler` refuses the model.
* **PyTorch (`litert-torch`)** compiles, but its PT2E path leaves nine of the
  ten residual `ADD`s in float32, fragmenting the graph: 19 of 94 ops mapped.
* **`tf.keras.applications.MobileNetV2`** cannot be used as-is: its ImageNet
  weights come from a different training run than torchvision's, and it pads
  stride-2 convolutions asymmetrically. Measured cosine similarity to the
  torchvision model: 0.91 — a different network.

So this script rebuilds the torchvision graph in Keras *exactly*, with
symmetric `ZeroPadding2D` + `valid` convolutions, and ports the weights. The
port is verified numerically (cosine 0.99999988 against torchvision).

Two quantizations are produced from that one clone, because the choice is a
genuine trade-off measured on 64 held-out images:

    TFLiteConverter    cosine 0.9692   68/68 ops on Edge TPU
    ai-edge-quantizer  cosine 0.9912    4/68 ops on Edge TPU

Both graphs are structurally identical (175 tensors, 68 ops, zero float, zero
dynamic). `edgetpu_compiler` 16.0 simply cannot parse the parameter encoding
ai-edge-quantizer emits, reporting "Filter, bias, or other param is not
constant at compile-time" for 48 convolutions whose weights are ordinary int8
constants. 16.0 is the final public release, so this is not a version to wait
out.

Runs in two stages because torch and TensorFlow do not coexist comfortably:

    # stage 1, in an environment with torch + torchvision
    python scripts/export_c6_tf_native.py dump --work-dir /tmp/c6

    # stage 2, in an environment with tensorflow + ai-edge-quantizer
    python scripts/export_c6_tf_native.py build --work-dir /tmp/c6 \\
        --calibration-dir /data/plantvillage/color \\
        --output-dir data/models --compile-edgetpu
"""

import argparse
from datetime import UTC, datetime
import hashlib
import json
from pathlib import Path
import random
import subprocess
import sys

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_SUFFIXES = {'.jpg', '.jpeg', '.png', '.bmp'}


# ---------------------------------------------------------------------------
# Stage 1 — read the reference model out of torchvision
# ---------------------------------------------------------------------------


def stage_dump(work_dir: Path, seed: int) -> int:
    """Serialise C6's structure and weights into a framework-neutral form."""
    import numpy as np
    import torch
    from torch import nn
    from torchvision import models
    from torchvision.ops.misc import Conv2dNormActivation

    torch.manual_seed(seed)
    features = (
        models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
        .eval()
        .features
    )

    spec: list[dict] = []
    arrays: dict[str, object] = {}

    def emit(prefix: str, conv: nn.Conv2d, bn: nn.BatchNorm2d, act: bool) -> None:
        depthwise = conv.groups > 1 and conv.groups == conv.in_channels
        spec.append(
            {
                'type': 'dwconv' if depthwise else 'conv',
                'name': prefix,
                'kernel': list(conv.kernel_size),
                'stride': list(conv.stride),
                'pad': list(conv.padding),
                'out_ch': conv.out_channels,
                'act': 'relu6' if act else None,
                'has_bias': conv.bias is not None,
                'bn_eps': float(bn.eps),
            }
        )
        weight = conv.weight.detach().numpy()
        # torch conv   (out, in, kh, kw) -> keras (kh, kw, in, out)
        # torch dwconv (in*m, 1, kh, kw) -> keras (kh, kw, in, m)
        arrays[prefix + '.w'] = (
            weight.transpose(2, 3, 0, 1) if depthwise else weight.transpose(2, 3, 1, 0)
        )
        if conv.bias is not None:
            arrays[prefix + '.b'] = conv.bias.detach().numpy()
        arrays[prefix + '.gamma'] = bn.weight.detach().numpy()
        arrays[prefix + '.beta'] = bn.bias.detach().numpy()
        arrays[prefix + '.mean'] = bn.running_mean.detach().numpy()
        arrays[prefix + '.var'] = bn.running_var.detach().numpy()

    for i, module in enumerate(features):
        if isinstance(module, Conv2dNormActivation):
            emit(f'f{i}', module[0], module[1], act=len(module) > 2)
            continue
        spec.append(
            {
                'type': 'block_start',
                'name': f'f{i}',
                'residual': bool(module.use_res_connect),
            }
        )
        for j, sub in enumerate(module.conv):
            if isinstance(sub, Conv2dNormActivation):
                emit(f'f{i}.c{j}', sub[0], sub[1], act=len(sub) > 2)
            elif isinstance(sub, nn.Conv2d):  # linear projection, BN follows
                emit(f'f{i}.c{j}', sub, module.conv[j + 1], act=False)
        spec.append(
            {
                'type': 'block_end',
                'name': f'f{i}',
                'residual': bool(module.use_res_connect),
            }
        )
    spec.append({'type': 'gap'})

    # Reference pair so stage 2 can prove the port is faithful.
    sample = torch.randn(1, 3, 224, 224)
    with torch.no_grad():
        reference = torch.flatten(nn.AdaptiveAvgPool2d(1)(features(sample)), 1)
    arrays['__ref_in'] = sample.numpy()
    arrays['__ref_out'] = reference.numpy()

    work_dir.mkdir(parents=True, exist_ok=True)
    np.savez(work_dir / 'c6_weights.npz', **arrays)
    (work_dir / 'c6_spec.json').write_text(json.dumps(spec, indent=1))

    convs = sum(1 for e in spec if e['type'] in ('conv', 'dwconv'))
    print(f'dumped {convs} convolutions, {len(arrays)} arrays -> {work_dir}')
    return 0


# ---------------------------------------------------------------------------
# Stage 2 — rebuild in Keras, quantize, optionally compile
# ---------------------------------------------------------------------------


def build_keras(work_dir: Path):
    """Reconstruct the torchvision graph as Keras layers and load the weights."""
    import numpy as np
    from tensorflow.keras import Model, layers

    spec = json.loads((work_dir / 'c6_spec.json').read_text())
    weights = np.load(work_dir / 'c6_weights.npz')

    inputs = layers.Input(shape=(224, 224, 3), batch_size=1, name='input')
    x = inputs
    residual_stack: list = []

    for entry in spec:
        kind = entry['type']
        if kind == 'block_start':
            residual_stack.append((x, entry['residual']))
            continue
        if kind == 'block_end':
            source, residual = residual_stack.pop()
            if residual:
                x = layers.Add()([source, x])
            continue
        if kind == 'gap':
            x = layers.GlobalAveragePooling2D()(x)
            continue

        name = entry['name']
        pad_h, pad_w = entry['pad']
        if pad_h or pad_w:
            # Symmetric, matching torchvision. Keras' own MobileNetV2 pads
            # asymmetrically here, which is what makes it a different network.
            x = layers.ZeroPadding2D(padding=((pad_h, pad_h), (pad_w, pad_w)))(x)

        shared = {
            'strides': tuple(entry['stride']),
            'padding': 'valid',
            'use_bias': entry['has_bias'],
        }
        if kind == 'dwconv':
            layer = layers.DepthwiseConv2D(kernel_size=tuple(entry['kernel']), **shared)
        else:
            layer = layers.Conv2D(
                filters=entry['out_ch'], kernel_size=tuple(entry['kernel']), **shared
            )
        x = layer(x)
        loaded = [weights[name + '.w']]
        if entry['has_bias']:
            loaded.append(weights[name + '.b'])
        layer.set_weights(loaded)

        norm = layers.BatchNormalization(epsilon=entry['bn_eps'])
        x = norm(x)
        norm.set_weights(
            [
                weights[name + '.gamma'],
                weights[name + '.beta'],
                weights[name + '.mean'],
                weights[name + '.var'],
            ]
        )
        if entry['act'] == 'relu6':
            x = layers.ReLU(max_value=6.0)(x)

    model = Model(inputs, x, name='c6_features')
    model.trainable = False
    return model, weights


def verify_port(model, weights) -> float:
    """Cosine similarity between the Keras clone and the torchvision reference."""
    import numpy as np

    reference = weights['__ref_out'].reshape(-1)
    produced = (
        model(weights['__ref_in'].transpose(0, 2, 3, 1), training=False)
        .numpy()
        .reshape(-1)
    )
    return float(
        reference @ produced / (np.linalg.norm(reference) * np.linalg.norm(produced))
    )


def calibration_batch(directory: Path, count: int, seed: int):
    import numpy as np
    from PIL import Image

    files = sorted(
        p
        for p in directory.rglob('*')
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    )
    if not files:
        raise SystemExit(f'No images under {directory}')
    chosen = files if len(files) <= count else random.Random(seed).sample(files, count)

    mean = np.array(IMAGENET_MEAN, dtype=np.float32)
    std = np.array(IMAGENET_STD, dtype=np.float32)
    batch = []
    for path in chosen:
        try:
            with Image.open(path) as raw:
                img = raw.convert('RGB').resize((224, 224))
        except Exception:
            continue
        batch.append((np.asarray(img, dtype=np.float32) / 255.0 - mean) / std)
    return np.stack(batch).astype(np.float32)


def describe(model_bytes: bytes) -> dict:
    import collections

    from ai_edge_litert.interpreter import Interpreter
    import numpy as np

    interpreter = Interpreter(model_content=model_bytes)
    interpreter.allocate_tensors()
    details = interpreter.get_tensor_details()
    return {
        'tensors_total': len(details),
        'tensors_float32': sum(
            1 for t in details if np.issubdtype(np.dtype(t['dtype']), np.floating)
        ),
        'tensors_dynamic': sum(
            1
            for t in details
            if len(t['shape']) == 0 or any(int(d) <= 0 for d in t['shape'])
        ),
        'op_counts': dict(
            collections.Counter(
                o['op_name'] for o in interpreter._get_ops_details()
            ).most_common()
        ),
    }


def stage_build(args) -> int:
    import tensorflow as tf

    work_dir = Path(args.work_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print('[1/4] Rebuilding the torchvision graph in Keras ...')
    model, weights = build_keras(work_dir)
    cosine = verify_port(model, weights)
    print(f'      port fidelity vs torchvision: cosine {cosine:.8f}')
    if cosine < 0.9999:
        print('      port is not faithful; aborting', file=sys.stderr)
        return 1

    saved_model = work_dir / 'c6_keras_saved_model'
    model.export(str(saved_model))

    print(f'[2/4] Loading {args.calibration_images} calibration images ...')
    batch = calibration_batch(
        Path(args.calibration_dir), args.calibration_images, args.seed
    )
    print(f'      calibration tensor {batch.shape}')

    results = {}

    print('[3/4] Quantizing — TFLiteConverter (Edge-TPU-compatible) ...')
    converter = tf.lite.TFLiteConverter.from_saved_model(str(saved_model))
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = lambda: (
        [batch[i : i + 1]] for i in range(batch.shape[0])
    )
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8
    tpu_blob = converter.convert()
    tpu_path = out_dir / 'c6_mobilenet_v2_int8_tpu.tflite'
    tpu_path.write_bytes(tpu_blob)
    results['tpu'] = (tpu_path, describe(tpu_blob))

    print('[4/4] Quantizing — ai-edge-quantizer (accuracy build) ...')
    float_path = work_dir / 'c6_keras_float32.tflite'
    float_path.write_bytes(
        tf.lite.TFLiteConverter.from_saved_model(str(saved_model)).convert()
    )
    from ai_edge_litert.interpreter import Interpreter
    from ai_edge_quantizer import quantizer, recipe

    probe = Interpreter(model_path=str(float_path))
    probe.allocate_tensors()
    signature = probe.get_signature_list()
    key = next(iter(signature))
    input_name = signature[key]['inputs'][0]

    engine = quantizer.Quantizer(str(float_path), recipe.static_wi8_ai8())
    calibrated = engine.calibrate(
        {key: [{input_name: batch[i : i + 1]} for i in range(batch.shape[0])]}
    )
    accurate_path = out_dir / 'c6_mobilenet_v2_int8_accurate.tflite'
    engine.quantize(calibrated).export_model(str(accurate_path))
    results['accurate'] = (accurate_path, describe(accurate_path.read_bytes()))

    compiled = None
    if args.compile_edgetpu:
        print('      compiling the Edge TPU build ...')
        proc = subprocess.run(
            ['edgetpu_compiler', '-s', '-o', str(out_dir), str(tpu_path)],
            capture_output=True,
            text=True,
        )
        for line in proc.stdout.splitlines():
            if 'will run on' in line or 'subgraphs' in line:
                print('      ' + line.strip())
        candidate = out_dir / (tpu_path.stem + '_edgetpu.tflite')
        compiled = candidate if candidate.exists() else None

    for label, (path, info) in results.items():
        sidecar = path.with_suffix('.export.json')
        sidecar.write_text(
            json.dumps(
                {
                    'generated_at': datetime.now(UTC).isoformat(),
                    'build': label,
                    'source': 'torchvision mobilenet_v2 IMAGENET1K_V1, features + avgpool',
                    'route': 'torchvision weights -> Keras clone -> TFLite',
                    'port_fidelity_cosine': round(cosine, 8),
                    'quantizer': (
                        'TFLiteConverter (TFLITE_BUILTINS_INT8)'
                        if label == 'tpu'
                        else 'ai-edge-quantizer static_wi8_ai8'
                    ),
                    'calibration': {
                        'images': int(batch.shape[0]),
                        'source': str(args.calibration_dir),
                        'seed': args.seed,
                        'preprocessing': {
                            'resize': [224, 224],
                            'scale': '1/255',
                            'mean': list(IMAGENET_MEAN),
                            'std': list(IMAGENET_STD),
                            'layout': 'NHWC',
                        },
                    },
                    'output_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
                    'output_bytes': path.stat().st_size,
                    'graph': info,
                    # Basename only: the build directory is a scratch path and
                    # must not leak into a committed artifact.
                    'edgetpu_compiled': compiled.name
                    if (label == 'tpu' and compiled)
                    else None,
                },
                indent=2,
            )
        )
        print(f'  {label:9s} -> {path}  ({path.stat().st_size / 1e6:.3f} MB)')

    if compiled:
        print(f'  edgetpu   -> {compiled}')
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest='stage', required=True)

    dump = sub.add_parser('dump', help='stage 1: read weights out of torchvision')
    dump.add_argument('--work-dir', required=True)
    dump.add_argument('--seed', type=int, default=42)

    build = sub.add_parser('build', help='stage 2: rebuild, quantize, compile')
    build.add_argument('--work-dir', required=True)
    build.add_argument('--output-dir', required=True)
    build.add_argument('--calibration-dir', required=True)
    build.add_argument('--calibration-images', type=int, default=512)
    build.add_argument('--seed', type=int, default=42)
    build.add_argument('--compile-edgetpu', action='store_true')

    args = parser.parse_args()
    if args.stage == 'dump':
        return stage_dump(Path(args.work_dir), args.seed)
    return stage_build(args)


if __name__ == '__main__':
    sys.exit(main())
