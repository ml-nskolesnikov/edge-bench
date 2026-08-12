"""TFLite runtime resolution shared by the agent and the standalone scripts.

Three runtimes provide the same `Interpreter` / `load_delegate` API:

1. ``tflite_runtime``  — the classic slim wheel shipped for Raspberry Pi OS.
2. ``ai_edge_litert``  — Google's maintained successor; the only one with
   wheels for current CPython on x86_64, so it is what makes the pipeline
   runnable on a workstation or a CI-like host.
3. ``tensorflow``      — full TF, used when neither slim runtime is present.

Resolution order prefers the slim runtimes so measurements on the Pi are
unaffected: ``tflite_runtime`` keeps winning wherever it is installed.
"""

from typing import Any

# (module path, attribute holding Interpreter, source label)
_CANDIDATES = (
    ('tflite_runtime.interpreter', 'Interpreter', 'tflite_runtime'),
    ('ai_edge_litert.interpreter', 'Interpreter', 'ai_edge_litert'),
)


class TFLiteBackendError(ImportError):
    """No usable TFLite runtime is installed."""


def _import_module(path: str):
    return __import__(path, fromlist=['_'])


def resolve_backend() -> tuple[Any, Any, str]:
    """Return ``(Interpreter, load_delegate, source_name)``.

    Raises:
        TFLiteBackendError: when no TFLite runtime can be imported.
    """
    for module_path, attr, label in _CANDIDATES:
        try:
            module = _import_module(module_path)
        except ImportError:
            continue
        interpreter = getattr(module, attr, None)
        if interpreter is None:
            continue
        return interpreter, getattr(module, 'load_delegate', None), label

    try:
        import tensorflow as tf
    except ImportError:
        raise TFLiteBackendError(
            'No TFLite runtime found. Install one of: '
            'tflite-runtime (Raspberry Pi), ai-edge-litert (x86_64), tensorflow.'
        )

    return (
        tf.lite.Interpreter,
        tf.lite.experimental.load_delegate,
        'tensorflow',
    )


def backend_version(source: str) -> str | None:
    """Best-effort version string for the resolved runtime."""
    module_by_source = {
        'tflite_runtime': 'tflite_runtime',
        'ai_edge_litert': 'ai_edge_litert',
        'tensorflow': 'tensorflow',
    }
    module_name = module_by_source.get(source)
    if not module_name:
        return None
    try:
        module = _import_module(module_name)
    except ImportError:
        return None
    return getattr(module, '__version__', 'unknown')
