# CI Readiness

How to wire edge-bench into a CI system. Every check is exposed as a Make
target, so the pipeline definition stays provider-agnostic — GitHub Actions,
Gitea Actions and GitLab CI all just call `make`.

A reference GitHub Actions implementation lives in
[`.github/workflows/ci.yml`](../.github/workflows/ci.yml).

---

## 1. Two independent levels

| Level | Needs | Runs on | Command |
|---|---|---|---|
| **1 — Quality gate** | CPU only. No GPU, no Edge TPU, no model weights, no dataset. | Any hosted runner | `make ci` |
| **2 — Hardware validation** | A real TFLite runtime and a `.tflite` file on disk. Optionally a Coral Edge TPU. | Self-hosted runner | `make test-hardware`, `make benchmark-smoke` |

**`make ci` never depends on level 2.** It must stay runnable on a bare
`ubuntu-latest` container with nothing but Python and Poetry.

---

## 2. Required CI jobs

### `quality` — blocking

```bash
make install       # poetry install --with dev
make lint          # ruff check --no-fix .
make format-check  # ruff format --check .
make typecheck     # mypy (server package)
make test-cov      # pytest with coverage, hardware tests excluded
make build         # import smoke: python -c "import server.main"
```

Or all at once:

```bash
make ci
```

Every target exits non-zero on failure. `make ci` fails on the first failing
step.

### `docker` — blocking

```bash
docker build -t edge-bench:ci .
docker run -d --name edge-bench-ci -p 18000:8000 edge-bench:ci
curl -fsS http://127.0.0.1:18000/api/health
```

Locally the same thing is one target:

```bash
make docker-smoke      # build + run + poll /docs, cleans up after itself
make docker-config     # validate docker-compose.yml
```

The import smoke inside `make build` is not redundant with the Docker job: it
catches import-time breakage in seconds without waiting for an image build.
A wrong `ProxyHeadersMiddleware` import path once reached runtime precisely
because neither check existed.

### `audit` — non-blocking

```bash
poetry export --without-hashes --only main -f requirements.txt -o requirements-audit.txt
pip-audit --requirement requirements-audit.txt
```

Reported, not enforced. An advisory in a transitive dependency should not
block an unrelated merge. It is marked `continue-on-error` so the job goes
yellow instead of hiding a real exit code behind `|| true`.

---

## 3. Pipeline shape

```text
Pull request / push
│
├── quality ────────────────────────────────── blocking
│     install → lint → format-check → typecheck → tests → build
│
├── docker ─────────────────────────────────── blocking
│     build image → run container → /api/health
│
└── audit ──────────────────────────────────── advisory
      pip-audit over runtime deps

main / manual dispatch / scheduled
│
└── hardware  [self-hosted runner]
      install → install-hardware → test-hardware → benchmark-smoke
```

The full scientific benchmark is **not** part of any automatic pipeline. It
takes far too long and its numbers belong in `RESULTS_INDEX.md`, not in CI
logs.

---

## 4. Dependencies

| Need | Where it comes from |
|---|---|
| Python | 3.12 (matches the Docker image; project supports `>=3.11,<3.14`) |
| Poetry | 2.2.1, pinned in the workflow and in the Dockerfile builder stage |
| Runtime deps | `poetry.lock`, `main` group |
| Dev tools | `poetry.lock`, `dev` group: ruff, mypy, pytest, pytest-cov, pre-commit |
| TFLite runtime | **Not locked.** Platform-specific; installed by `make install-hardware` |

`poetry.lock` is tracked in git and must stay in sync with `pyproject.toml` —
`poetry check` fails otherwise. The Docker builder copies both files and runs
`poetry export`, so an out-of-sync lock breaks the image build too.

### Always install from the lock

Dependencies are pinned in `poetry.lock`; CI installs from it via
`make install`. A job that runs bare `pip install fastapi` gets whatever is
current, which is how environments silently drift out of the tested set.

---

## 5. Environment variables

CI needs **none**. Every setting has a working default and the app creates its
own data directories on startup.

| Variable | Default | Needed in CI? |
|---|---|---|
| `EDGEBENCH_HOST` | `0.0.0.0` | no |
| `EDGEBENCH_PORT` | `8000` | no |
| `EDGEBENCH_DATABASE_PATH` | `data/edgebench.db` | no |
| `EDGEBENCH_MODELS_DIR` | `data/models` | no |
| `EDGEBENCH_UPLOAD_DIR` | `data/uploads` | no |
| `EDGEBENCH_SCRIPTS_DIR` | `data/scripts` | no |
| `EDGEBENCH_AGENT_SECRET` | `''` (auth disabled) | no |
| `EDGEBENCH_PROXY_TRUST` | `127.0.0.1` | no |
| `EDGEBENCH_INPUT_SEED` | `42` | hardware job only |

See [`.env.example`](../.env.example) for the full list. No secret is required
for any level-1 job.

---

## 6. Artifacts

| Artifact | Produced by | Keep |
|---|---|---|
| `coverage.xml` | `make test-cov` | per run |
| Docker image | `docker build` | on tags / main |
| `results/smoke/*.json` | `make benchmark-smoke` | per hardware run |

`results/smoke/` is gitignored. Smoke output is validation evidence, never a
scientific result — see [`results/README.md`](../results/README.md).

---

## 7. Caching

| Cache | Key | Effect |
|---|---|---|
| Poetry venv (`.venv`) | `hashFiles('poetry.lock')` | Skips dependency resolution and download |
| Docker layers | `type=gha` | The builder stage only re-runs when `pyproject.toml` / `poetry.lock` change |
| Ruff / mypy caches | not cached | Both are fast enough on this codebase |

Set `virtualenvs-in-project: true` so the venv lands in `.venv` and is
cacheable by path.

---

## 8. Test strategy

```
tests/
├── conftest.py                     # isolated tmp storage per test
├── test_app_routes.py              # real app: routers mounted, templates routed
├── test_devices_api.py             # per-resource API tests
├── test_experiments_api.py
├── test_results_api.py
├── test_files_api.py
├── test_schedules_api.py
├── test_settings_api.py
├── test_dependencies_api.py
├── test_executor_metrics.py        # benchmark math, mocked interpreter
├── test_queue_scheduler_integration.py
├── test_security.py                # auth, path traversal, debug gating
├── test_ws_manager.py
└── test_hardware_benchmark.py      # marked `hardware`, excluded by default
```

Isolation: the `isolated_storage` fixture repoints every settings path at a
per-test `tmp_path` via `monkeypatch`, so tests never touch the developer's
database and can run in parallel.

Marker configuration lives in `pyproject.toml`:

```toml
[tool.pytest.ini_options]
addopts = "-m 'not hardware'"
markers = [
    "hardware: requires a real TFLite runtime and a model file on disk (not run in CI)",
]
```

So plain `pytest` is hardware-free by construction — a contributor cannot
accidentally make CI depend on a Coral device.

Coverage currently sits around **34%** overall. That number is deliberately
not inflated: `agent/benchmark_*.py` are standalone scripts that only execute
on real hardware and report 0%, which drags the total down. The server API
layer, which CI can actually exercise, is far better covered. Coverage is
reported, not gated — a threshold would only encourage tests written for the
metric.

---

## 9. Hardware / GPU CI

### What actually needs hardware

| Check | Command | Requires |
|---|---|---|
| Hardware unit tests | `make test-hardware` | TFLite runtime + any `.tflite` model |
| Pipeline smoke benchmark | `make benchmark-smoke` | same |
| Edge TPU path | `make benchmark-smoke BENCH_BACKEND=edgetpu` | Coral USB device, `libedgetpu1-std`, an `_edgetpu.tflite` model |
| Full scientific benchmark | `agent/benchmark_full.py` | Raspberry Pi 4 (+ optional Coral); **manual only** |

### There is no GPU code path

This is worth stating plainly, because it is easy to assume otherwise from the
project description. edge-bench measures **TFLite inference on ARM CPU and on
the Coral Edge TPU**. It contains no CUDA, no PyTorch, and no TensorRT. An
NVIDIA GPU cannot accelerate anything here, and none of the metrics
(`latency`, `throughput`, `process_rss_mb_max`, thermal warnings) have a GPU
equivalent in the current result schema.

Consequently:

- A GPU runner is **not** a requirement, now or later.
- A general-purpose x86 machine is still useful as a hardware runner: it can
  execute the whole real pipeline via `ai-edge-litert`, which proves model
  loading, inference, timing, memory measurement and serialization work —
  everything except the Edge TPU delegate and Pi-specific thermal telemetry.
- If GPU inference ever becomes in scope, it needs a new backend in
  `agent/tflite_backend.py` plus GPU fields in the result schema
  (`gpu`, `driver`, `cuda`, `precision`, `peak_vram_mb`) before a GPU runner
  makes any sense.

### Markers

```bash
pytest              # hardware tests deselected — this is what CI runs
pytest -m hardware  # hardware tests only
make test-hardware  # same, with -v
```

Hardware tests **skip** with an explicit reason when the runtime or a model is
missing. They are never satisfied by a mock interpreter: a mocked GPU/TPU run
that reports "passed" is worse than no test at all. Mocks are used only in
`test_executor_metrics.py`, where the subject is the statistics, not the
hardware.

### Models and datasets

- Model weights are **not** in git (`data/models/*.tflite` is gitignored).
- A hardware runner must have at least one `.tflite` file in `data/models/`
  or `models/`. `benchmark-smoke` auto-selects the smallest one, or takes
  `BENCH_MODEL=/path/to/model.tflite`.
- No labelled dataset is required: the benchmark feeds seeded synthetic input
  of the model's own input shape (`EDGEBENCH_INPUT_SEED`, default 42). It
  measures latency and memory, not accuracy.

### Environment for the hardware job

| Variable | Purpose |
|---|---|
| `EDGEBENCH_INPUT_SEED` | Fix the synthetic input across runs (default 42) |
| `BENCH_MODEL` | Explicit model path instead of auto-discovery |
| `BENCH_BACKEND` | `cpu` (default) or `edgetpu` |
| `BENCH_RUNS`, `BENCH_WARMUP` | Iteration counts for the smoke run |

### Artifacts to keep

`results/smoke/*.json` — each file records model hash and size, backend,
thread count, warmup/measured counts, input seed, full latency percentiles,
both throughput definitions, peak process RSS, device info, and the runtime
provenance block (`tflite_source`, `tflite_version`, `numpy_version`,
`python_version`, `cpu_governor`).

### Adding a self-hosted runner later

1. Register the runner with labels `self-hosted, edge-bench-hardware`.
2. On the runner: install Poetry, then `make install && make install-hardware`.
   `install-hardware` picks `tflite-runtime` on ARM and `ai-edge-litert` on
   x86_64 automatically.
3. Place at least one `.tflite` model in `data/models/`.
4. For Edge TPU: `sudo apt install libedgetpu1-std`, plug in the Coral, and
   give the runner user access to `/dev/bus/usb`.
5. The `hardware` job in `ci.yml` is already written and gated to
   `workflow_dispatch`. Widen the `if:` condition to add a schedule.

Keep it off pull requests. Benchmark timings on a shared runner are noisy, and
a flaky timing assertion that blocks merges will simply get deleted.

### Docker on a hardware runner

```bash
docker build --target bench -t edge-bench:bench .
docker run --rm -v "$PWD/data:/app/data:ro" -v "$PWD/results:/app/results" \
    edge-bench:bench --runs 30
```

The default image target ships **no** inference runtime — the server never
runs inference itself, the agent does. The `bench` target adds
`ai-edge-litert` for hosts where installing a Python runtime directly is
undesirable. Edge TPU inside a container additionally needs
`--device /dev/bus/usb` and `libedgetpu` on the host. No CUDA base image is
used anywhere, and the CPU container must never gain one.

---

## 10. Local equivalence

The point of routing everything through Make is that a contributor can run the
exact CI gate before pushing:

```bash
make install
make ci
```

If that passes locally, the `quality` job passes in CI. If it does not, the
pipeline definition is drifting from the Make targets and should be fixed —
not worked around in YAML.
