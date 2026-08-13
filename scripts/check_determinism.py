#!/usr/bin/env python3
"""Check whether a TFLite model returns the same output for the same input.

A benchmark measures how long a model takes. It cannot tell you whether the
model computes a stable answer — and a model that does not is unusable for
cross-device comparison, because a difference between two devices can no
longer be attributed to the devices.

The check feeds a byte-identical seeded tensor to several freshly constructed
interpreters and compares the outputs. It also reports whether repeated
invocations on a *single* interpreter agree, which separates two causes:

    fresh interpreters differ, same interpreter agrees
        -> state established at construction/allocation differs between
           instances; the model depends on something not fixed by its inputs

    same interpreter also differs
        -> non-determinism inside execution (threading, scheduling)

Usage:
    python scripts/check_determinism.py data/models/foo.tflite
    python scripts/check_determinism.py data/models/*.tflite --runs 8

Exit code is non-zero when any model is non-deterministic, so this can gate a
release or a results publication.
"""

import argparse
import hashlib
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'agent'))


def _digest(array: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()[:12]


def _make_input(spec, seed: int) -> np.ndarray:
    """Same generation rule the benchmark executor uses."""
    np.random.seed(seed)
    dtype = np.dtype(spec['dtype'])
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype)
        return np.random.randint(
            info.min, info.max + 1, size=spec['shape'], dtype=dtype
        )
    return np.random.rand(*spec['shape']).astype(dtype)


def check_model(interpreter_cls, model: Path, runs: int, seed: int) -> dict:
    fresh_digests = []
    for _ in range(runs):
        interpreter = interpreter_cls(model_path=str(model))
        interpreter.allocate_tensors()
        spec = interpreter.get_input_details()[0]
        data = _make_input(spec, seed)
        interpreter.set_tensor(spec['index'], data)
        interpreter.invoke()
        output = interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
        fresh_digests.append(_digest(output))

    # Repeated invocations on one interpreter, same input.
    interpreter = interpreter_cls(model_path=str(model))
    interpreter.allocate_tensors()
    spec = interpreter.get_input_details()[0]
    data = _make_input(spec, seed)
    repeat_digests = []
    for _ in range(runs):
        interpreter.set_tensor(spec['index'], data)
        interpreter.invoke()
        repeat_digests.append(
            _digest(
                interpreter.get_tensor(interpreter.get_output_details()[0]['index'])
            )
        )

    details = interpreter.get_tensor_details()
    float_tensors = sum(
        1 for t in details if np.issubdtype(np.dtype(t['dtype']), np.floating)
    )

    return {
        'model': model.name,
        'input_digest': _digest(data),
        'fresh_unique': len(set(fresh_digests)),
        'repeat_unique': len(set(repeat_digests)),
        'runs': runs,
        'float_tensors': float_tensors,
        'total_tensors': len(details),
        'deterministic': len(set(fresh_digests)) == 1,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('models', nargs='+', help='Paths to .tflite models')
    parser.add_argument('--runs', type=int, default=6)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    try:
        from tflite_backend import TFLiteBackendError, resolve_backend
    except ImportError as exc:  # pragma: no cover
        print(f'Cannot import agent modules: {exc}', file=sys.stderr)
        return 2

    try:
        interpreter_cls, _, source = resolve_backend()
    except TFLiteBackendError as exc:
        print(f'{exc}\nInstall one with: make install-hardware', file=sys.stderr)
        return 3

    print(f'runtime: {source}   runs per model: {args.runs}   seed: {args.seed}')
    print()
    header = f'{"model":<44} {"fresh":>7} {"repeat":>7} {"float/total":>12}  verdict'
    print(header)
    print('-' * len(header))

    failed = []
    for raw in args.models:
        model = Path(raw).expanduser()
        if not model.is_file():
            print(f'{model.name:<44} {"—":>7} {"—":>7} {"—":>12}  NOT FOUND')
            failed.append(model.name)
            continue

        try:
            report = check_model(interpreter_cls, model, args.runs, args.seed)
        except (RuntimeError, ValueError) as exc:
            # Edge TPU builds cannot be loaded without the delegate; that is a
            # skip, not a verdict.
            reason = 'needs Edge TPU' if 'edgetpu' in str(exc).lower() else 'load error'
            print(
                f'{model.name[:44]:<44} {"—":>7} {"—":>7} {"—":>12}  SKIPPED ({reason})'
            )
            continue
        verdict = 'deterministic' if report['deterministic'] else 'NON-DETERMINISTIC'
        mix = f'{report["float_tensors"]}/{report["total_tensors"]}'
        print(
            f'{report["model"][:44]:<44} '
            f'{report["fresh_unique"]:>3}/{report["runs"]:<3} '
            f'{report["repeat_unique"]:>3}/{report["runs"]:<3} '
            f'{mix:>12}  {verdict}'
        )
        if not report['deterministic']:
            failed.append(report['model'])
            if report['repeat_unique'] == 1:
                print(
                    '      identical input, identical results within one '
                    'interpreter, different across fresh ones'
                )

    print()
    print('fresh  = distinct outputs across freshly built interpreters (1 is good)')
    print('repeat = distinct outputs across repeated invokes on one interpreter')
    print('float/total = float32 tensors vs all; a full-integer model has 0 float')

    if failed:
        print(f'\nNon-deterministic or missing: {", ".join(failed)}', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
