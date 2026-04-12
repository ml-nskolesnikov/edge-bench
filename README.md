# Edge-Bench: Remote ML Benchmarking for Raspberry Pi + Coral Edge TPU

A remote benchmarking system for running ML inference experiments on Raspberry Pi with Google Coral Edge TPU from a host machine.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      HOST MACHINE (Server)                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  FastAPI    │  │  SQLite DB  │  │  Web UI (Jinja2)        │ │
│  │  Backend    │──│  Results    │──│  - Experiments list     │ │
│  │             │  │  Queue      │  │  - Metrics dashboard    │ │
│  └──────┬──────┘  └─────────────┘  └─────────────────────────┘ │
└─────────┼───────────────────────────────────────────────────────┘
          │ HTTP
          ▼
┌─────────────────────────────────────────────────────────────────┐
│                    RASPBERRY PI (Agent)                         │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────────┐ │
│  │  Lightweight│  │  Metrics    │  │  Script Executor        │ │
│  │  HTTP Agent │──│  Collector  │──│  - TFLite inference     │ │
│  │  (uvicorn)  │  │  (psutil)   │  │  - Benchmark scripts    │ │
│  └─────────────┘  └─────────────┘  └─────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │  Google Coral Edge TPU (USB, optional)                      ││
│  └─────────────────────────────────────────────────────────────┘│
└─────────────────────────────────────────────────────────────────┘
```

## Quick Start

### 1. On the HOST machine

**Option A: Poetry (recommended)**
```bash
cd edge-bench
poetry install
python -m server.main
```

**Option B: pip**
```bash
cd edge-bench
pip install -r requirements/server.txt
python -m server.main
```

The server starts at `http://localhost:8000`

### 2. On Raspberry Pi (one command)

```bash
# Install agent
curl -sSL http://<HOST_IP>:8000/install | bash

# Uninstall agent
curl -sSL http://<HOST_IP>:8000/uninstall | bash
```

Or manually:
```bash
# Copy agent directory to RPi
scp -r agent/ pi@raspberrypi:~/edge-bench-agent/

# On RPi
cd ~/edge-bench-agent
chmod +x install.sh
./install.sh
```

### 3. Register the device

After the agent is running on the RPi, register it on the server via Web UI or API:
```bash
curl -X POST http://localhost:8000/api/devices \
  -H "Content-Type: application/json" \
  -d '{"name": "rpi-coral-1", "ip": "192.168.x.x", "port": 8001}'
```

## Web UI

Open `http://localhost:8000` in your browser:

- **/** — Dashboard (device count, recent experiments)
- **/devices** — Device management
- **/experiments** — Experiment list
- **/new-experiment** — Create new experiment
- **/results** — View and compare results
- **/models** — Model repository
- **/benchmark** — Direct benchmark tools
- **/compare** — Side-by-side model comparison
- **/settings** — Server settings and dependencies

## API Endpoints

### Devices
```
GET    /api/devices                      List all devices
POST   /api/devices                      Register a device
GET    /api/devices/{id}                 Get device by ID
DELETE /api/devices/{id}                 Remove a device
GET    /api/devices/{id}/status          Check live status
POST   /api/devices/{id}/ping            Ping device
GET    /api/devices/{id}/version         Get agent version
POST   /api/devices/{id}/update          Push agent update
GET    /api/devices/{id}/models          List models on device
POST   /api/devices/{id}/upload-model    Upload model to device
POST   /api/devices/{id}/check-deploy    Check deploy status (hash-based)
POST   /api/devices/{id}/benchmark/full  Proxy single-model benchmark
POST   /api/devices/{id}/benchmark/batch Proxy batch benchmark
```

### Experiments
```
GET    /api/experiments                  List experiments (filterable)
POST   /api/experiments                  Create experiment
GET    /api/experiments/{id}             Get experiment details
GET    /api/experiments/{id}/logs        Get experiment logs
POST   /api/experiments/{id}/rerun       Re-queue experiment
DELETE /api/experiments/{id}             Delete experiment
POST   /api/experiments/batch            Create batch experiments
```

### Results
```
GET    /api/results                      All results
GET    /api/results/{experiment_id}      Results for experiment
GET    /api/results/export/csv           Export to CSV
GET    /api/results/export/json          Export to JSON
```

### Files
```
POST   /api/files/upload                 Upload model or script
GET    /api/files                        List uploaded files
GET    /api/files/{id}                   Download file
DELETE /api/files/{id}                   Delete file
GET    /api/files/agent/{filename}       Serve agent source (used by installer)
```

## Metrics Format

```json
{
  "experiment_id": "exp_20260204_143022_a1b2",
  "device": "raspberrypi",
  "model": {
    "name": "mobilenetv2_int8_edgetpu.tflite",
    "hash": "sha256:abc123def456",
    "size_bytes": 3456789,
    "quantization": "int8_edgetpu"
  },
  "params": {
    "backend": "edgetpu",
    "batch_size": 1,
    "num_threads": 4,
    "warmup_runs": 10,
    "benchmark_runs": 100
  },
  "latency": {
    "mean_ms": 12.34,
    "std_ms": 1.23,
    "min_ms": 10.12,
    "max_ms": 18.45,
    "p50_ms": 12.01,
    "p90_ms": 14.56,
    "p95_ms": 15.23,
    "p99_ms": 17.89
  },
  "throughput": {
    "fps": 81.03,
    "images_per_second": 81.03
  },
  "cold_start": {
    "model_load_ms": 234.5,
    "first_inference_ms": 45.6
  },
  "system": {
    "cpu_percent_mean": 45.2,
    "cpu_percent_max": 78.9,
    "memory_mb_mean": 123.4,
    "memory_mb_max": 156.7,
    "cpu_temp_celsius": 52.3,
    "tpu_detected": true
  },
  "device_info": {
    "hostname": "raspberrypi",
    "platform": "Linux-6.12.75-aarch64",
    "python_version": "3.11.2",
    "cpu_count": 4,
    "memory_total_mb": 7819.8,
    "tpu_detected": true,
    "tflite_version": "2.14.0"
  },
  "timestamp": "2026-02-04T14:30:22.123456",
  "duration_seconds": 45.67,
  "status": "completed"
}
```

## Configuration

The server is configured via environment variables (prefix `EDGEBENCH_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGEBENCH_HOST` | `0.0.0.0` | Bind address |
| `EDGEBENCH_PORT` | `8000` | Server port |
| `EDGEBENCH_DEBUG` | `false` | Debug/reload mode |
| `EDGEBENCH_DATABASE_PATH` | `data/edgebench.db` | SQLite database path |
| `EDGEBENCH_MODELS_DIR` | `data/models` | Model storage directory |
| `EDGEBENCH_TASK_TIMEOUT_SECONDS` | `3600` | Max experiment duration |
| `EDGEBENCH_AGENT_TIMEOUT_SECONDS` | `30` | Agent health check timeout |

The agent is configured via environment variables (prefix `EDGEBENCH_`):

| Variable | Default | Description |
|----------|---------|-------------|
| `EDGEBENCH_AGENT_PORT` | `8001` | Agent listen port |
| `EDGEBENCH_SERVER` | `` | Server URL for result sync (e.g. `http://192.168.1.x:8000`) |

## Project Structure

```
edge-bench/
├── README.md
├── pyproject.toml              # Poetry project config
├── requirements/
│   ├── server.txt              # Server dependencies
│   └── agent.txt               # Agent dependencies (RPi)
├── server/                     # Server (HOST machine)
│   ├── main.py                 # Entry point + web UI routes
│   ├── api/
│   │   ├── devices.py          # Device management API
│   │   ├── experiments.py      # Experiment API
│   │   ├── results.py          # Results API
│   │   ├── files.py            # File upload/serve API
│   │   ├── dependencies.py     # RPi dependency tracker
│   │   ├── scripts.py          # Script management API
│   │   └── settings.py         # Server settings API
│   ├── core/
│   │   ├── config.py           # Settings (pydantic-settings)
│   │   ├── models.py           # Pydantic schemas
│   │   └── queue.py            # Async task queue with retries
│   ├── db/
│   │   └── database.py         # SQLite + aiosqlite
│   ├── static/
│   └── templates/              # Jinja2 HTML templates
├── agent/                      # Agent (Raspberry Pi)
│   ├── main.py                 # FastAPI agent entrypoint
│   ├── executor.py             # TFLite benchmark executor
│   ├── metrics.py              # System metrics (psutil)
│   ├── config.py               # Agent settings
│   ├── result_cache.py         # Offline result caching
│   ├── install.sh              # Agent install script
│   └── edgebench-agent.service # systemd unit file
├── scripts/                    # Standalone benchmark scripts
│   ├── benchmark_tflite.py     # TFLite CPU benchmark
│   ├── benchmark_full.py       # Full metrics benchmark
│   ├── benchmark_batch.py      # Batch benchmark runner
│   └── convert_edgetpu.py      # EdgeTPU model compiler
├── data/
│   ├── models/                 # TFLite model files
│   └── scripts/                # Uploaded benchmark scripts
└── models/
    └── .gitkeep
```

## Usage Examples

### Compare models on Edge TPU

```bash
curl -X POST http://localhost:8000/api/experiments/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": [
      "mobilenetv2_int8_edgetpu.tflite",
      "efficientnet_int8_edgetpu.tflite"
    ],
    "backends": ["edgetpu"],
    "device": "dev_xxxxxxxx",
    "params": {"benchmark_runs": 100}
  }'
```

### CPU vs Edge TPU

```bash
curl -X POST http://localhost:8000/api/experiments/batch \
  -H "Content-Type: application/json" \
  -d '{
    "models": ["mobilenetv2_int8.tflite"],
    "backends": ["cpu", "edgetpu"],
    "device": "dev_xxxxxxxx"
  }'
```

### Export results

```bash
# CSV export
curl http://localhost:8000/api/results/export/csv > results.csv

# JSON export
curl http://localhost:8000/api/results/export/json > results.json
```

## Troubleshooting

### Server does not start

**Problem:** `ModuleNotFoundError: No module named 'server'`  
**Fix:** Run from inside the `edge-bench/` directory, not from the parent:
```bash
cd edge-bench
python -m server.main
```

**Problem:** `ModuleNotFoundError: No module named 'pydantic_settings'`  
**Fix:** Install server dependencies:
```bash
pip install -r requirements/server.txt
# or
poetry install
```

### Web UI returns 500 on all pages

**Problem:** Starlette ≥ 1.0 changed `TemplateResponse` signature from `(name, context_dict)` to `(request, name, context_dict)`.  
**Fix:** Already applied — all `TemplateResponse` calls in `server/main.py` use the new signature.

### Device shows as offline after registration

**Problem:** Agent IP in the database is stale (changed after DHCP renewal).  
**Fix:** Delete the device and re-register with the current IP, or update via API:
```bash
curl -X POST http://localhost:8000/api/devices \
  -H "Content-Type: application/json" \
  -d '{"name": "rpi-coral-1", "ip": "<current-ip>", "port": 8001}'
```

### Agent not starting on RPi

```bash
sudo systemctl status edgebench-agent
sudo journalctl -u edgebench-agent -n 50
```

Ensure the virtual environment is activated and requirements are installed:
```bash
cd ~/edge-bench-agent
source venv/bin/activate
pip install -r requirements.txt
python main.py
```

### Edge TPU not detected

```bash
lsusb | grep -i "google\|coral"
python3 -c "from pycoral.utils import edgetpu; print(edgetpu.list_edge_tpus())"
```

Install runtime if needed:
```bash
sudo apt install libedgetpu1-std
pip install pycoral
```

## Security Notes

- Scripts are executed in a subprocess with configurable timeout
- Agent only allows updating specific whitelisted files (`main.py`, `executor.py`, `metrics.py`, `config.py`)
- All actions are logged

## Docker

Run the server with a single command — no local Python install required:

```bash
# Build and start
docker compose up

# Background
docker compose up -d

# View logs
docker compose logs -f

# Stop
docker compose down
```

The SQLite database and models persist in `./data/` and `./models/` on the host.

Environment overrides (optional `.env` file):
```bash
EDGEBENCH_PORT=8000
EDGEBENCH_DATABASE_PATH=/app/data/edgebench.db
```

Build the image manually:
```bash
docker build -t edge-bench .
docker run --rm -p 8000:8000 -v ./data:/app/data edge-bench
```

## Real-time Charts (WebSocket)

While a benchmark is running the experiment detail page (`/experiments/{id}`) shows a live latency-over-time chart powered by Chart.js.

- The browser connects to `ws://localhost:8000/ws/experiments/{id}`
- The agent streams a metric update every ~5% of total runs
- The status badge updates automatically (🟢 running → ✅ done / ❌ failed)
- The page reloads automatically when the run completes

## Baseline Comparison

After any completed experiment, the detail page shows a **vs Baseline** table:

| | Current | Baseline | Delta |
|---|---|---|---|
| Mean latency | 12.34 ms | 13.50 ms | ▼ 1.16 ms (−8.6%) |
| FPS | 81.0 | 74.1 | ▲ 6.9 |

To set a baseline manually click **⭐ Set as Baseline** on the experiment page.

API:
```bash
# Compare with baseline
GET /api/results/{experiment_id}/compare-baseline

# Mark as baseline
POST /api/experiments/{experiment_id}/set-baseline
```

## Multiple Edge TPU Support

When multiple Coral Edge TPU devices are connected (`/dev/apex_0`, `/dev/apex_1`, …), the new experiment form shows a TPU index dropdown.

API — device status now includes `tpu_count`:
```bash
GET /api/devices/{id}/status
# → {"status":"online","device_info":{...,"tpu_count":2},"tpu_count":2}
```

Experiment `params` field:
```json
{"backend": "edgetpu", "tpu_index": 1}
```

## Model Conversion Pipeline

Convert `.pt` / `.onnx` → TFLite INT8 → Edge TPU TFLite from the command line:

```bash
# Full pipeline (requires edgetpu_compiler or RPi SSH)
python scripts/convert_pipeline.py \
  --input model.pt \
  --input-shape 1 224 224 3 \
  --target edgetpu \
  --output-dir converted_models

# Compile on RPi via SSH
python scripts/convert_pipeline.py \
  --input model_int8.tflite \
  --target edgetpu \
  --rpi-host pi@192.168.1.100
```

Via API (async, returns `task_id`):
```bash
POST /api/files/{file_id}/convert
{"target": "edgetpu", "input_shape": [1, 224, 224, 3]}

GET /api/files/{file_id}/convert/status
```

## MLflow Integration

Log benchmark results to a self-hosted MLflow tracking server.

1. Start MLflow server: `mlflow server --host 0.0.0.0 --port 5000`
2. Install mlflow: `pip install mlflow`
3. Open `/settings` → **Integrations** → enter `http://localhost:5000`
4. Click **Сохранить** — all subsequent completed experiments are logged automatically

Logged metrics: `latency_mean_ms`, `latency_p95_ms`, `fps`, `cpu_percent_mean`, `cpu_temp_celsius`  
Logged tags: `model_name`, `backend`, `device`, `tpu_detected`, `quantization`

Environment variable alternative:
```bash
EDGEBENCH_MLFLOW_TRACKING_URI=http://mlflow.internal:5000
EDGEBENCH_MLFLOW_EXPERIMENT_NAME=edge-bench
```

## API Reference (Extended)

### Real-time Streaming
```
WS     /ws/experiments/{id}                WebSocket live metrics
POST   /api/experiments/{id}/metric        Agent → server metric push
```

### Baseline
```
GET    /api/results/{id}/compare-baseline  Compare vs baseline
POST   /api/experiments/{id}/set-baseline  Mark as baseline
```

### Model Conversion
```
POST   /api/files/{id}/convert             Start async conversion
GET    /api/files/{id}/convert/status      Check conversion status
```

### Settings
```
GET    /api/settings                        All settings
PUT    /api/settings                        Update (key-value dict)
```

## Future Work

- [ ] NVIDIA Jetson support
- [ ] Automated nightly benchmark schedules

## License

MIT License
