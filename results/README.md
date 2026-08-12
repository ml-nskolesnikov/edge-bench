# results/

Benchmark output. Three kinds of content live here, and they are **not**
interchangeable.

| Directory | Tracked in git | What it is |
|---|---|---|
| `<date>_<time>_<scenario>/` | yes | Measured experiment runs. Research artifacts. |
| `smoke/` | no (gitignored) | Throwaway output of `make benchmark-smoke`. |
| anything else | decide per case | Ask before adding. |

## Measured runs

Each directory holds the raw agent output plus a normalized summary:

```
2026-06-06_163338_c6_cpu/
├── c6_cpu_latency.json   # full result from the benchmark script
└── normalized.json       # scenario / machine / git_sha + headline metrics
```

These are inputs to the dissertation and to `~/aspirantura/RESULTS_INDEX.md`,
which is the single source of truth for any number that gets published.
**Do not edit, regenerate or delete them** without checking RESULTS_INDEX first.

## Smoke runs

`make benchmark-smoke` writes to `results/smoke/`. Those files prove the
pipeline executes on the current host — model loading, inference, timing,
memory measurement, serialization. They use a handful of iterations and are
**validation output, not measurements**. Never cite them.

## What a result should contain

The agent records the following, and anything added later should preserve
these keys (existing analysis scripts read them):

- `model` — name, sha256 prefix, size, quantization, input shape/dtype
- `params` — backend, thread count, warmup/measured iteration counts, input seed
- `latency` — mean, std, min, max, p50/p90/p95/p99 in ms
- `throughput` — `fps_from_mean`, `fps_from_median` (`fps` aliases the mean)
- `cold_start` — model load time, first inference time
- `system` — CPU percent, process RSS, temperature, frequency series
- `device_info` — hostname, platform, kernel, CPU/RAM, TPU presence
- `runtime` — TFLite source and version, numpy, Python, CPU governor
- `warnings` — thermal throttle / frequency drop flags (empty when clean)
- `timestamp`, `duration_seconds`, `status`

## Reproducing a run

```bash
make install-hardware
poetry run python agent/benchmark_full.py \
    --model data/models/<model>.tflite \
    --backend cpu --runs 100 --seed 42
```

Set `EDGEBENCH_INPUT_SEED` to change the seed globally. On a Raspberry Pi,
put the CPU governor in `performance` mode first — otherwise idle frequency
scaling is indistinguishable from thermal throttling in the warnings.
