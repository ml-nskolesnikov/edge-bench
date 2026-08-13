# Edge-Bench

Remote ML benchmarking for Raspberry Pi and the Google Coral Edge TPU.

You run a server on a workstation; lightweight agents run on the edge devices.
The server queues experiments, dispatches them to a device, collects latency,
throughput, memory and thermal metrics, and stores everything in SQLite with a
web dashboard on top.

---

## Table of contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project structure](#project-structure)
- [Requirements](#requirements)
- [Quick start](#quick-start)
- [Configuration](#configuration)
- [Running benchmarks](#running-benchmarks)
- [Web interface](#web-interface)
- [Results](#results)
- [Tests](#tests)
- [Docker](#docker)
- [CI and quality checks](#ci-and-quality-checks)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Known limitations](#known-limitations)
- [Future work](#future-work)
- [License](#license)

---

## Overview

Benchmarking edge inference by hand does not scale: you SSH into a Pi, copy a
model, run a script, copy the JSON back, and lose track of which numbers came
from which model on which device with which thread count.

Edge-Bench turns that into a queue plus a dashboard. Every result carries the
provenance needed to reproduce it — model hash, input seed, iteration counts,
runtime versions, CPU governor, and thermal warnings when the device throttled
mid-run.

## Features

- **Remote execution** — register devices, queue experiments, run them one at a
  time with automatic retry and backoff.
- **Zero-touch agent install** — `curl -sSL http://<server>:8000/install | bash`
  on the Pi sets up a venv, a systemd unit and registers the device.
- **CPU and Edge TPU backends** — TFLite via `tflite-runtime`,
  `ai-edge-litert` or full TensorFlow, with an Edge TPU delegate when a Coral
  is attached.
- **Reproducible measurements** — seeded synthetic input, GC disabled around
  the timed `invoke()`, warmup separated from measurement, full latency
  percentiles.
- **Throttle detection** — flags thermal events and CPU frequency drops, and
  distinguishes real throttling from ordinary `powersave` idle scaling.
- **Live metrics** — WebSocket streaming of latency while a run is in flight.
- **Scheduling** — cron-style nightly benchmark schedules.
- **Web dashboard** — metric cards, comparison charts, CSV/JSON export.
- **Result cache** — the agent persists results locally and syncs them when the
  server comes back.
- **MLflow integration** — optional, off by default.

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                     HOST MACHINE (Server)                        │
│  ┌────────────┐  ┌────────────┐  ┌───────────────────────────┐   │
│  │  FastAPI   │  │ SQLite DB  │  │  Web UI (Jinja2)          │   │
│  │  + queue   │──│  results   │──│  dashboard / compare      │   │
│  │  + cron    │  │  schedules │  │  results / devices        │   │
│  └─────┬──────┘  └────────────┘  └───────────────────────────┘   │
└────────┼─────────────────────────────────────────────────────────┘
         │ HTTP  (X-Agent-Secret)      ▲ WebSocket (live metrics)
         ▼                             │
┌──────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI (Agent)                          │
│  ┌────────────┐  ┌────────────┐  ┌───────────────────────────┐   │
│  │ HTTP agent │  │  metrics   │  │  BenchmarkExecutor        │   │
│  │ (uvicorn)  │──│  (psutil)  │──│  TFLite inference + timing│   │
│  └────────────┘  └────────────┘  └───────────────────────────┘   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │  Google Coral Edge TPU (USB, optional)                     │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

The server never runs inference itself. It orchestrates; the agent measures.

## Project structure

```
edge-bench/
├── server/                  # FastAPI application (the host side)
│   ├── main.py              # app wiring, /api/health, /install, WebSocket
│   ├── api/                 # REST endpoints, one module per resource
│   ├── core/                # config, auth, task queue, scheduler, models
│   ├── db/                  # SQLite schema and connection helper
│   ├── routes/ui.py         # HTML page routes
│   ├── integrations/        # MLflow logger
│   ├── templates/           # Jinja2 pages
│   └── static/              # CSS design system, i18n, vendored Chart.js
├── agent/                   # Deployed flat onto the Raspberry Pi
│   ├── main.py              # agent HTTP API
│   ├── executor.py          # BenchmarkExecutor — the measurement path
│   ├── metrics.py           # CPU / RAM / temperature / TPU detection
│   ├── tflite_backend.py    # runtime resolution (tflite_runtime → litert → TF)
│   ├── result_cache.py      # offline result persistence + sync
│   └── benchmark_*.py       # standalone benchmark scripts (canonical copies)
├── scripts/                 # Host-side utilities
│   ├── benchmark_smoke.py   # short end-to-end pipeline validation
│   └── convert_*.py         # model conversion / Edge TPU compilation
├── tests/                   # pytest suite (hardware tests marked separately)
├── docs/CI_READINESS.md     # how to wire this into CI
├── results/                 # benchmark output — see results/README.md
└── data/                    # runtime storage (DB, models, uploads) — gitignored
```

> `agent/benchmark_*.py` are the canonical benchmark implementations. Do not
> copy them elsewhere — divergent copies are how two "identical" runs end up
> producing different numbers.

## Requirements

**Server (workstation):**

- Python 3.11–3.13
- [Poetry](https://python-poetry.org/) 2.x
- or just Docker

**Agent (Raspberry Pi):**

- Raspberry Pi 4 or newer, Raspberry Pi OS 64-bit
- Python 3.9+
- A TFLite runtime (the installer handles this)
- Optional: Coral USB Accelerator + `libedgetpu1-std`

Nothing in the default test suite or CI needs a Pi, a Coral, a GPU, a dataset
or model weights.

## Quick start

```bash
git clone <repository-url>
cd edge-bench

make install          # poetry install --with dev + create data dirs
make run              # http://localhost:8000
```

Then open <http://localhost:8000>.

`make help` lists every target.

### Install an agent on a Raspberry Pi

One command on the Pi:

```bash
curl -sSL http://<SERVER_IP>:8000/install | bash
```

This downloads the agent, creates a venv, installs a systemd unit, starts it
and registers the device with the server. To remove it:

```bash
curl -sSL http://<SERVER_IP>:8000/uninstall | bash
```

Manual alternative, from a checkout:

```bash
make agent-deploy RPI_HOST=pi@192.168.1.100
```

### Register a device manually

```bash
curl -X POST http://localhost:8000/api/devices \
  -H "Content-Type: application/json" \
  -d '{"name": "rpi4-lab", "ip": "192.168.1.100", "port": 8001}'
```

## Configuration

All settings are environment variables prefixed with `EDGEBENCH_`, with working
defaults — the app starts with no configuration at all.

```bash
cp .env.example .env      # then edit
```

See [`.env.example`](.env.example) for the annotated list. The ones that matter
most:

| Variable | Default | Purpose |
|---|---|---|
| `EDGEBENCH_HOST` / `EDGEBENCH_PORT` | `0.0.0.0` / `8000` | Bind address |
| `EDGEBENCH_DATABASE_PATH` | `data/edgebench.db` | SQLite file |
| `EDGEBENCH_MODELS_DIR` | `data/models` | Uploaded models |
| `EDGEBENCH_AGENT_SECRET` | *(empty)* | Shared secret for `/api/*`; empty disables auth |
| `EDGEBENCH_PROXY_TRUST` | `127.0.0.1` | Which host may set `X-Forwarded-*` |
| `EDGEBENCH_INPUT_SEED` | `42` | Seed for synthetic benchmark input |
| `EDGEBENCH_DEBUG` | `false` | Enables the agent's `/execute/code` endpoint — never in production |

Relative paths resolve against the working directory: the repo root locally,
`/app` inside the container. Directories are created on startup.

Set the same `EDGEBENCH_AGENT_SECRET` on the server and on every agent, or on
neither.

## Running benchmarks

### From the web UI

**Devices** → register a Pi → **Models** → upload a `.tflite` →
**+ New** → pick model, device, backend, thread count and iteration counts →
watch it run on **Experiments** → read **Results**.

### From the API

```bash
curl -X POST http://localhost:8000/api/experiments \
  -H "Content-Type: application/json" \
  -d '{
        "name": "mobilenet-v2 int8 on TPU",
        "device_id": "dev_abc123",
        "model_path": "/home/pi/models/mobilenetv2_int8_edgetpu.tflite",
        "params": {"backend": "edgetpu", "num_threads": 4,
                   "warmup_runs": 10, "benchmark_runs": 100}
      }'
```

### Directly on a device

```bash
python3 agent/benchmark_full.py \
    --model ~/models/mobilenetv2_int8_edgetpu.tflite \
    --backend edgetpu --runs 100 --seed 42
```

### Pipeline smoke run

Proves the whole measurement path works on the current host — model loading,
device detection, inference, timing, memory, serialization — in a few seconds:

```bash
make install-hardware        # installs the right TFLite runtime for this host
make benchmark-smoke         # auto-selects the smallest model in data/models
make benchmark-smoke BENCH_MODEL=data/models/foo.tflite BENCH_RUNS=100
```

Before trusting any comparison, check that the models are reproducible at all:

```bash
make check-determinism                                     # every model in data/models
make check-determinism DETERMINISM_MODELS=path/to/m.tflite
```

It feeds a byte-identical seeded tensor to several freshly built interpreters
and reports whether they agree. It also shows how many float32 tensors a model
has: a full-integer INT8 graph has zero, so a model labelled `int8` with a
large float count is not doing integer inference.

Smoke output lands in `results/smoke/` and is **validation evidence, not a
measurement** — the iteration count is far too low to cite.

### Comparing devices and backends

Running the same model on several platforms and checking that they agree:

```bash
make platform-matrix \
    MATRIX_MODEL=data/models/mobilenetv1_int8_ptq_Fuzzy.tflite \
    MATRIX_TARGETS="x86=:cpu rpi-cpu=rpi:cpu rpi-tpu=rpi:edgetpu@~/models/mobilenetv1_int8_ptq_Fuzzy_edgetpu.tflite"
```

Targets are `name=host:backend[@model]`; an empty host means this machine, and
`@model` is required for Edge TPU because it needs its own compiled
`*_edgetpu.tflite` build. Remote targets need SSH and a TFLite runtime; the
agent sources are copied to a temporary directory and removed afterwards.

Besides latency and throughput per platform, the run compares **output
signatures** — top-k indices and a dequantised checksum of the output tensor.
This is the part a timing-only benchmark cannot do: an Edge TPU build that
computes nonsense still reports excellent latency. The script exits non-zero
when a backend disagrees on top-1.

Measured example (MobileNetV1 INT8, 50 iterations, seed 42):

| target | device | runtime | backend | mean | fps | agreement |
|---|---|---|---|---|---|---|
| x86 | workstation | ai-edge-litert | cpu | 0.99 ms | 1011 | reference |
| rpi-cpu | Raspberry Pi 4 | tflite_runtime | cpu | 33.52 ms | 29.8 | same top-5, ΔL2 0.17% |
| rpi-tpu | Raspberry Pi 4 + Coral | tflite_runtime | edgetpu | 4.72 ms | 211.9 | same top-5, ΔL2 0.17% |

All three agree on the ranking; the sub-percent numeric spread is the expected
consequence of different kernels and hardware. The Edge TPU build ran 7.1×
faster than the same network on the Pi's CPU.

### Methodology notes

- Warmup iterations are excluded from statistics.
- Garbage collection is disabled around the timed `invoke()` only, then
  restored, so it cannot distort either latency or memory figures.
- Input is generated from a fixed seed, because quantized kernels can take
  different code paths depending on input distribution.
- `fps_from_mean` and `fps_from_median` are both reported; `fps` aliases the
  mean for backward compatibility.
- For publishable numbers, put the CPU governor in `performance` mode first.
  Under `powersave`, idle frequency scaling is indistinguishable from thermal
  throttling and the run gets flagged.

## Web interface

| Page | What it is for |
|---|---|
| `/` | Dashboard: device/experiment counts, average and best latency |
| `/devices` | Register devices, check status, deploy agents |
| `/models` | Upload models, inspect quantization, convert for Edge TPU |
| `/experiments` | Queue, live status, retry, reassign |
| `/experiments/{id}` | Single run: percentiles, charts, logs, warnings |
| `/results` | All measured runs, comparative bars, CSV/JSON export |
| `/compare` | Side-by-side charts for several runs |
| `/benchmark` | Batch benchmark tools |
| `/scripts` | Remote script execution and device inspection |
| `/schedules` | Cron-style recurring benchmarks |
| `/settings` | Paths, timeouts, dependencies, integrations |
| `/docs`, `/redoc` | Auto-generated OpenAPI documentation |

The UI is server-rendered Jinja2 with a plain-CSS design system — no frontend
build step, no framework. Chart.js is vendored under
`server/static/js/vendor/`, so the dashboard works on an isolated lab network
with no internet access. It has light and dark themes and a Russian/English
toggle.

## Results

Results live in `results/`. Read [`results/README.md`](results/README.md) before
touching anything there — measured runs are research artifacts and are tracked
in git; `results/smoke/` is throwaway and is not.

Every result records what is needed to reproduce it:

```json
{
  "experiment_id": "exp_20260204_143022_a1b2",
  "device": "raspberrypi",
  "model": {
    "name": "mobilenetv2_int8_edgetpu.tflite",
    "hash": "sha256:abc123def456",
    "size_bytes": 3456789,
    "quantization": "int8_edgetpu",
    "input_shape": [1, 224, 224, 3],
    "input_dtype": "uint8"
  },
  "params": {
    "backend": "edgetpu", "num_threads": 4,
    "warmup_runs": 10, "benchmark_runs": 100, "input_seed": 42
  },
  "latency": {
    "mean_ms": 12.34, "std_ms": 1.23, "min_ms": 10.12, "max_ms": 18.45,
    "p50_ms": 12.01, "p90_ms": 14.56, "p95_ms": 15.23, "p99_ms": 17.89
  },
  "throughput": {
    "fps_from_mean": 81.03, "fps_from_median": 83.26,
    "fps": 81.03, "images_per_second": 81.03
  },
  "cold_start": { "model_load_ms": 234.5, "first_inference_ms": 45.6 },
  "system": {
    "cpu_percent": {"mean": 45.2, "max": 78.9},
    "process_rss_mb_mean": 123.4, "process_rss_mb_max": 156.7,
    "cpu_temp_celsius": 52.3, "cpu_temp_max": 61.0,
    "cpu_freq_mhz_min": 1500
  },
  "device_info": {
    "hostname": "raspberrypi", "platform": "Linux-6.12.75-aarch64",
    "kernel_version": "6.12.75", "python_version": "3.11.2",
    "cpu_count": 4, "memory_total_mb": 7819.8,
    "tpu_detected": true, "tflite_version": "2.14.0",
    "libedgetpu_version": "16.0", "cpu_governor": "performance"
  },
  "runtime": {
    "tflite_source": "tflite_runtime", "tflite_version": "2.14.0",
    "numpy_version": "1.26.4", "python_version": "3.11.2",
    "cpu_governor": "performance"
  },
  "warnings": [],
  "timestamp": "2026-02-04T14:30:22.123456+00:00",
  "duration_seconds": 45.67,
  "status": "completed"
}
```

The `runtime` block and the extra `model` fields are additive — older results
stay readable, and `fps` / `images_per_second` keep their original meaning.

Export:

```bash
curl -o results.csv  http://localhost:8000/api/results/export/csv
curl -o results.json http://localhost:8000/api/results/export/json
```

## Tests

```bash
make test        # CPU-only suite, no hardware needed
make test-cov    # with coverage (writes coverage.xml)
```

Hardware tests are excluded by default and skip cleanly when the runtime or a
model is missing — they are never satisfied by a mocked device:

```bash
make test-hardware      # pytest -m hardware
```

## Docker

```bash
make docker-up          # build + start via compose
make docker-logs
make docker-down
```

Or directly:

```bash
docker build -t edge-bench:local .
docker run --rm -p 8000:8000 -v "$PWD/data:/app/data" edge-bench:local
```

Details:

- Multi-stage build; dependencies come from `poetry.lock`, so the image is
  reproducible.
- Runs as a non-root user. Compose builds the image with `APP_UID`/`APP_GID`
  matching the host user so the bind-mounted `./data` stays writable.
- `HEALTHCHECK` polls `/api/health`; compose reports the container as healthy.
- Dev tooling (ruff, mypy, pytest) is not in the runtime image.
- Graceful shutdown: uvicorn handles `SIGTERM` and drains connections.
- Port is configurable: `EDGEBENCH_PORT=18200 docker compose up -d`.

There is also an optional benchmark image for hosts without a usable Python:

```bash
docker build --target bench -t edge-bench:bench .
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/results:/app/results" \
    edge-bench:bench --runs 30
```

Useful checks:

```bash
make docker-smoke       # build, run, verify reachable, clean up
make docker-config      # validate compose file
```

## CI and quality checks

Everything CI runs is a Make target, so the pipeline is provider-agnostic and
reproducible locally:

```bash
make lint            # ruff check
make format-check    # ruff format --check
make typecheck       # mypy
make test-cov        # pytest + coverage
make build           # import smoke
make ci              # all of the above
```

`make ci` is CPU-only: no GPU, no Edge TPU, no model weights, no dataset. If it
passes locally it passes in CI.

Hardware validation is a separate, opt-in level (`make test-hardware`,
`make benchmark-smoke`) intended for a self-hosted runner.

See [`docs/CI_READINESS.md`](docs/CI_READINESS.md) for the full breakdown, and
[`.github/workflows/ci.yml`](.github/workflows/ci.yml) for a reference
implementation.

## Development

```bash
make dev             # uvicorn with autoreload on 127.0.0.1:8000
make format          # apply ruff fixes + formatting
make check           # fast gate: lint + tests
make clean-pyc       # drop caches
```

Conventions:

- Ruff is the single source of truth for lint and formatting; its configuration
  lives in `pyproject.toml`. Single quotes, 88 columns.
- `agent/` is deployed flat onto the Pi and therefore uses top-level imports
  (`from metrics import ...`). It is excluded from mypy for that reason.
- New agent modules must be added to the allowlist in
  `server/api/files.py` and to the download list in the `/install` script,
  otherwise fresh agent installs will not receive them.
- Adding a page means adding a route in `server/routes/ui.py`;
  `tests/test_app_routes.py` fails on templates or API paths with no route.

## Troubleshooting

**Server will not start — "address already in use"**

```bash
ss -tlnp | grep 8000
EDGEBENCH_PORT=8001 make run
```

**Web UI returns 500 on every page** — usually a stale database after a schema
change. Back it up and let the app recreate it:

```bash
mv data/edgebench.db data/edgebench.db.bak
make run
```

**Device shows offline right after registration**

```bash
curl http://<PI_IP>:8001/health              # agent reachable?
ssh pi@<PI_IP> 'sudo systemctl status edgebench-agent'
```

Check that the IP registered on the server matches the Pi's current address and
that port 8001 is not firewalled.

**Agent will not start on the Pi**

```bash
ssh pi@<PI_IP> 'sudo journalctl -u edgebench-agent -n 50'
```

Most often a missing TFLite runtime. See `agent/tflite_backend.py` for the
resolution order, then install the matching package.

**Edge TPU not detected**

```bash
lsusb | grep -i "google\|global unichip"     # device present?
dpkg -l | grep libedgetpu                    # runtime installed?
sudo apt install libedgetpu1-std
```

Re-plug the Coral after installing the runtime, and use a USB 3.0 port. The
model must be Edge TPU compiled (`*_edgetpu.tflite`).

**"No TFLite runtime found" when running a benchmark locally**

```bash
make install-hardware
```

**Benchmark results flagged with a frequency-drop warning** — expected under the
`powersave` governor. For real measurements:

```bash
sudo cpufreq-set -g performance      # or write to scaling_governor
```

**Charts do not render** — Chart.js is vendored; confirm
`server/static/js/vendor/chart.umd.min.js` is present and served (it is
excluded from no ignore file, but a partial checkout can miss it).

## Known limitations

- **`c6_mobilenet_v2_int8.tflite` was replaced on 2026-08-13.** The original
  file carried an `int8` name but computed in float32 — its first operator was
  a `DEQUANTIZE` and 139 of its 234 tensors were float, i.e. weight-only /
  dynamic-range quantization rather than the full-integer scheme every other
  `_int8_` model here uses. Two measured consequences: it returned a different
  result on every freshly built interpreter (reproduced on `ai-edge-litert`
  2.1.6/x86_64 and `tflite_runtime` 2.14/aarch64, so it was the file and not
  the environment), and its features barely tracked the fp32 reference.

  The canonical name now holds a model re-exported by
  `scripts/export_int8_tflite.py` with the same calibration source and count
  as the C6 bundle (512 images from `/data/plantvillage/color`, seed 42). The
  original is kept at
  `data/models/archive/c6_mobilenet_v2_int8_hybrid_broken.tflite`.

  Source provenance, stated precisely: the export read
  `plantdiag-edge/artifacts/c6_mobilenet_v2_fp32.onnx` at
  `sha256:7dc2651008fe8bd7…`, which is **not** the hash recorded in the C6
  manifest (`1e448d9e…`) — that file was regenerated on 2026-08-13 while this
  work was in progress. The two were compared and are the same model: 169
  nodes, 104 initializers, all 104 weight tensors byte-identical, and
  byte-identical outputs for the same input. The hashes differ only in ONNX
  serialization metadata. The manifest-matching copy survives at
  `plantdiag-edge/artifacts_pq/c6_mobilenet_v2_fp32.onnx`.

  | | original | replacement |
  |---|---|---|
  | sha256 (16) | `1ffd174380c64787` | `4c69b637abe17482` |
  | cosine similarity to fp32 ONNX | 0.501 (min 0.438) | **0.991** (min 0.979) |
  | distinct outputs per 8 fresh interpreters | 8 | **1** |
  | float32 compute ops | all | **0** |
  | latency, x86 CPU | 7.78 ms | **4.45 ms** |
  | latency, Raspberry Pi 4 CPU | 137.90 ms | **56.19 ms** |
  | size | 2.29 MB | 2.72 MB |

  `results/2026-06-06_163338_c6_cpu/` records the original hash and is
  therefore a measurement of the broken model. Results produced through the
  **ONNX** path are unaffected — see below.

- **Uniform noise saturates quantized classifiers.** The synthetic input
  spans the full int8 range, which is far outside the natural image
  distribution. `mobilenetv2_int8_ptq_sbert.tflite` returns byte-identical
  logits for every seed as a result. Latency and memory stay valid — the cost
  of a convolution does not depend on the data — but for such models the
  output signature carries little information, so cross-device agreement on
  it is weak evidence.

- **Synthetic input is not real data.** The benchmark feeds seeded random
  tensors of the model's own dtype. That is adequate for latency and memory,
  but says nothing about accuracy, and the output signature only proves that
  two devices agree with each other — not that either is correct.
- **WebSocket channel is unauthenticated.** `/ws/experiments/{id}` does not
  require `X-Agent-Secret`; anyone on the network can subscribe to live
  metrics.
- **Shared secret is visible in browser JS.** `agent_secret` is injected into
  each page so `fetch()` can send it. That is obfuscation, not protection —
  rely on network-level controls. Real per-session auth is backlog.
- **`/api/health` is intentionally unauthenticated**, so probes work without
  the secret. It exposes version and queue depth only.
- **HTTPS in the install script depends on the proxy.** The scheme comes from
  `request.url.scheme`, which only reads `https` when a TLS proxy sets
  `X-Forwarded-Proto` and `EDGEBENCH_PROXY_TRUST` names it.
- **Results from before the RSS rename show `--` for memory.** The
  `memory_mb_*` → `process_rss_mb_*` change is a breaking schema change; older
  rows display `--` until re-run.
- **Batch cooldown is bounded.** `--cooldown-temp` waits at most 60 s for the
  device to cool, then proceeds anyway.
- **Power consumption is not measured.** Never implemented; listed below.
- **TFLite only.** No ONNX Runtime, no TensorRT, and no GPU backend — there is
  no CUDA code anywhere in this project. Adding one means a new backend in
  `agent/tflite_backend.py` plus new result-schema fields.
- **Fixed run count.** `--runs` is fixed (default 100); there is no
  run-until-CV-stabilises mode.
- **`dependencies.html` is orphaned.** `/dependencies` redirects to `/settings`,
  which absorbed that UI. The template is kept but unrouted.
- **`app.routes` no longer lists mounted routers (FastAPI ≥ 0.137).** An
  included router now appears as one opaque `_IncludedRouter` entry instead of
  being flattened, so code that introspects `app.routes` sees nothing for it.
  Routing itself is unaffected. Enumerate routes via `app.openapi()['paths']`
  instead — `tests/test_app_routes.py` does, and additionally asserts real
  responses through `TestClient`.

## Future work

- [ ] NVIDIA Jetson support
- [ ] Full WebSocket authentication
- [ ] Per-browser session auth (replace shared-secret JS injection)
- [ ] Power consumption estimation (INA219 / USB power meter)
- [ ] ONNX Runtime backend
- [ ] Adaptive run count (stop when CV stabilises)

## License

MIT License
