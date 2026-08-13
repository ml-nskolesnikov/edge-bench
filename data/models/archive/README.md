# archive/

Superseded model artifacts, kept for traceability. Nothing here is a usable
model — the directory is deliberately outside the discovery path used by
`scripts/benchmark_smoke.py`, `scripts/platform_matrix.py` and the hardware
tests, all of which glob `data/models/*.tflite` one level only.

## c6_mobilenet_v2_int8_hybrid_broken.tflite

`sha256:1ffd174380c64787…` — the original `c6_mobilenet_v2_int8.tflite`,
replaced on 2026-08-13.

Despite the `int8` name it computed in float32: the first operator is a
`DEQUANTIZE` and 139 of its 234 tensors are float. Two measured consequences:

- it returned a different result on every freshly built interpreter
  (reproduced on `ai-edge-litert`/x86_64 and `tflite_runtime`/aarch64);
- its features barely tracked the fp32 reference — mean cosine similarity
  0.501 over 64 held-out images, against 0.991 for the replacement.

Any measurement taken with this file is invalid, including
`results/2026-06-06_163338_c6_cpu/` (which records this hash).

The replacement was produced by `scripts/export_int8_tflite.py` from the same
fp32 ONNX named in the C6 bundle manifest. See `data/models/*.export.json`.
