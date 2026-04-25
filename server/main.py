"""
Edge-Bench Server - Main Entry Point
"""

import asyncio
from contextlib import asynccontextmanager
from datetime import UTC, datetime
import json
from pathlib import Path

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from server.api import (
    dependencies,
    devices,
    experiments,
    files,
    results,
    settings as settings_api,
)
from server.api.schedules import router as schedules_router
from server.core.config import settings
from server.core.queue import task_queue
from server.core.scheduler import restore_schedules, scheduler
from server.core.ws_manager import ws_manager
from server.db.database import get_db, init_db

BASE_DIR = Path(__file__).parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    await init_db()
    asyncio.create_task(task_queue.process_queue())
    scheduler.start()
    await restore_schedules()
    print(f'Edge-Bench Server started on http://{settings.HOST}:{settings.PORT}')
    yield
    # Shutdown
    scheduler.shutdown(wait=False)
    task_queue.stop()


app = FastAPI(
    title='Edge-Bench',
    description='Remote ML Benchmarking for Raspberry Pi + Coral Edge TPU',
    version='1.0.0',
    lifespan=lifespan,
)

# Static files and templates
app.mount('/static', StaticFiles(directory=BASE_DIR / 'static'), name='static')
templates = Jinja2Templates(directory=BASE_DIR / 'templates')

# API routers
app.include_router(devices.router, prefix='/api/devices', tags=['devices'])
app.include_router(experiments.router, prefix='/api/experiments', tags=['experiments'])
app.include_router(results.router, prefix='/api/results', tags=['results'])
app.include_router(files.router, prefix='/api/files', tags=['files'])
app.include_router(
    dependencies.router, prefix='/api/dependencies', tags=['dependencies']
)
app.include_router(settings_api.router, prefix='/api/settings', tags=['settings'])
app.include_router(schedules_router, prefix='/api/schedules', tags=['schedules'])


# WebSocket route for real-time experiment updates
@app.websocket('/ws/experiments/{experiment_id}')
async def websocket_experiment(ws: WebSocket, experiment_id: str):
    """WebSocket endpoint for live benchmark metrics.

    Messages:
      {"type": "status", "status": "running" | "completed" | "failed"}
      {"type": "metric", "latency_ms": 12.3, "fps": 81.0, "run": 45}
      {"type": "done"}
    """
    await ws_manager.connect(experiment_id, ws)
    try:
        async with get_db() as db:
            cursor = await db.execute(
                'SELECT status FROM experiments WHERE id = ?', (experiment_id,)
            )
            row = await cursor.fetchone()
        if row:
            await ws.send_text(json.dumps({'type': 'status', 'status': row['status']}))
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        ws_manager.disconnect(experiment_id, ws)


# Web UI routes
@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard."""
    from apscheduler.triggers.cron import CronTrigger

    async with get_db() as db:
        devices_count = await db.execute('SELECT COUNT(*) FROM devices')
        devices_count = (await devices_count.fetchone())[0]

        experiments_count = await db.execute('SELECT COUNT(*) FROM experiments')
        experiments_count = (await experiments_count.fetchone())[0]

        recent = await db.execute(
            'SELECT * FROM experiments ORDER BY created_at DESC LIMIT 10'
        )
        recent_experiments = await recent.fetchall()

        # Upcoming schedules (next 3 by next fire time)
        cursor = await db.execute(
            """SELECT s.*, d.name as device_name
               FROM schedules s
               LEFT JOIN devices d ON s.device_id = d.id
               WHERE s.enabled = 1"""
        )
        sched_rows = await cursor.fetchall()

    upcoming = []
    for row in sched_rows:
        s = dict(row)
        try:
            trigger = CronTrigger.from_crontab(s['cron'], timezone='UTC')
            nf = trigger.get_next_fire_time(None, datetime.now(UTC))
            s['next_run'] = nf.isoformat() if nf else None
        except Exception:
            s['next_run'] = None
        upcoming.append(s)

    upcoming.sort(key=lambda x: x['next_run'] or '')
    upcoming = upcoming[:3]

    return templates.TemplateResponse(
        request,
        'index.html',
        {
            'devices_count': devices_count,
            'experiments_count': experiments_count,
            'recent_experiments': [dict(r) for r in recent_experiments],
            'upcoming_schedules': upcoming,
        },
    )


@app.get('/schedules', response_class=HTMLResponse)
async def schedules_page(request: Request):
    """Nightly benchmark schedules page."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT s.*, d.name as device_name
               FROM schedules s
               LEFT JOIN devices d ON s.device_id = d.id
               ORDER BY s.created_at DESC"""
        )
        schedule_rows = await cursor.fetchall()

        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

    from apscheduler.triggers.cron import CronTrigger

    def next_run(cron: str) -> str | None:
        try:
            from datetime import UTC, datetime
            trigger = CronTrigger.from_crontab(cron, timezone='UTC')
            nf = trigger.get_next_fire_time(None, datetime.now(UTC))
            return nf.isoformat() if nf else None
        except Exception:
            return None

    def human_cron(cron: str) -> str:
        mapping = {
            '0 2 * * *': 'Every day at 02:00 UTC',
            '0 * * * *': 'Every hour',
            '0 */6 * * *': 'Every 6 hours',
            '0 */12 * * *': 'Every 12 hours',
            '0 0 * * *': 'Every day at 00:00 UTC',
            '0 0 * * 0': 'Every Sunday at 00:00 UTC',
            '*/30 * * * *': 'Every 30 minutes',
            '*/15 * * * *': 'Every 15 minutes',
        }
        return mapping.get(cron, cron)

    schedules = []
    for row in schedule_rows:
        s = dict(row)
        if s.get('params'):
            try:
                s['params'] = json.loads(s['params'])
            except (json.JSONDecodeError, TypeError):
                s['params'] = {}
        else:
            s['params'] = {}
        s['next_run'] = next_run(s['cron'])
        s['cron_human'] = human_cron(s['cron'])
        schedules.append(s)

    return templates.TemplateResponse(
        request,
        'schedules.html',
        {
            'schedules': schedules,
            'devices': [dict(d) for d in device_list],
        },
    )


@app.get('/devices', response_class=HTMLResponse)
async def devices_page(request: Request):
    """Devices management page.

    Renders immediately from DB cache. Live status is checked
    asynchronously on the client side via JS after page load.
    """
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

    devices = [dict(d) for d in device_list]

    return templates.TemplateResponse(
        request,
        'devices.html',
        {
            'devices': devices,
        },
    )


@app.get('/experiments', response_class=HTMLResponse)
async def experiments_page(request: Request):
    """Experiments list page."""
    async with get_db() as db:
        cursor = await db.execute("""
            SELECT e.*, d.name as device_name
            FROM experiments e
            LEFT JOIN devices d ON e.device_id = d.id
            ORDER BY e.created_at DESC
        """)
        experiment_list = await cursor.fetchall()

    # Parse params JSON for template access
    experiments = []
    for exp in experiment_list:
        exp_dict = dict(exp)
        if exp_dict.get('params'):
            try:
                exp_dict['params'] = json.loads(exp_dict['params'])
            except (json.JSONDecodeError, TypeError):
                exp_dict['params'] = {}
        else:
            exp_dict['params'] = {}
        experiments.append(exp_dict)

    return templates.TemplateResponse(
        request,
        'experiments.html',
        {
            'experiments': experiments,
        },
    )


@app.get('/experiments/{exp_id}', response_class=HTMLResponse)
async def experiment_detail(request: Request, exp_id: str):
    """Experiment detail page."""
    import json

    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM experiments WHERE id = ?', (exp_id,))
        experiment = await cursor.fetchone()

        cursor = await db.execute(
            'SELECT * FROM results WHERE experiment_id = ?', (exp_id,)
        )
        result = await cursor.fetchone()

    # Parse metrics JSON
    metrics = None
    if result and result['metrics']:
        try:
            metrics = json.loads(result['metrics'])
        except json.JSONDecodeError:
            metrics = None

    return templates.TemplateResponse(
        request,
        'experiment_detail.html',
        {
            'experiment': dict(experiment) if experiment else None,
            'result': dict(result) if result else None,
            'metrics': metrics,
        },
    )


@app.get('/results', response_class=HTMLResponse)
async def results_page(request: Request):
    """Results comparison page."""
    async with get_db() as db:
        cursor = await db.execute("""
            SELECT r.*, e.name as experiment_name, e.model_name, d.name as device_name
            FROM results r
            JOIN experiments e ON r.experiment_id = e.id
            LEFT JOIN devices d ON e.device_id = d.id
            ORDER BY r.created_at DESC
        """)
        rows = await cursor.fetchall()

    # Parse metrics JSON for each result
    result_list = []
    for row in rows:
        result_dict = dict(row)
        if result_dict.get('metrics'):
            try:
                result_dict['metrics'] = json.loads(result_dict['metrics'])
            except json.JSONDecodeError:
                result_dict['metrics'] = {}
        else:
            result_dict['metrics'] = {}
        result_list.append(result_dict)

    return templates.TemplateResponse(
        request,
        'results.html',
        {
            'results': result_list,
        },
    )


@app.get('/models', response_class=HTMLResponse)
async def models_page(request: Request):
    """Model repository page."""
    async with get_db() as db:
        # Get all models
        cursor = await db.execute(
            "SELECT * FROM files WHERE type = 'model' ORDER BY created_at DESC"
        )
        model_list = await cursor.fetchall()

        # Get all devices
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

    # Add quantization info based on filename
    models_with_info = []
    for model in model_list:
        model_dict = dict(model)
        name_lower = model_dict['name'].lower()

        # Auto-detect quantization from filename
        if 'edgetpu' in name_lower:
            model_dict['quantization'] = 'int8_edgetpu'
        elif 'int8' in name_lower or '_quant' in name_lower:
            model_dict['quantization'] = 'int8'
        elif 'fp16' in name_lower:
            model_dict['quantization'] = 'fp16'
        elif 'fp32' in name_lower:
            model_dict['quantization'] = 'fp32'
        else:
            model_dict['quantization'] = None

        models_with_info.append(model_dict)

    return templates.TemplateResponse(
        request,
        'models.html',
        {
            'models': models_with_info,
            'devices': [dict(d) for d in device_list],
        },
    )


@app.get('/new-experiment', response_class=HTMLResponse)
async def new_experiment_page(request: Request):
    """New experiment form."""
    async with get_db() as db:
        # Get all devices (not just online)
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

        cursor = await db.execute("SELECT * FROM files WHERE type = 'model'")
        model_list = await cursor.fetchall()

    return templates.TemplateResponse(
        request,
        'new_experiment.html',
        {
            'devices': [dict(d) for d in device_list],
            'models': [dict(m) for m in model_list],
        },
    )


@app.get('/benchmark', response_class=HTMLResponse)
async def benchmark_page(request: Request):
    """Benchmark tools page."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

    return templates.TemplateResponse(
        request,
        'benchmark.html',
        {
            'devices': [dict(d) for d in device_list],
        },
    )


@app.get('/settings', response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    from server.core.config import AGENT_VERSION, settings as cfg

    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM dependencies ORDER BY is_required DESC, name'
        )
        dependency_list = await cursor.fetchall()

        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

        # Load saved settings overrides
        cursor = await db.execute('SELECT * FROM settings')
        saved = {row['key']: row['value'] for row in await cursor.fetchall()}

    return templates.TemplateResponse(
        request,
        'settings.html',
        {
            'agent_version': AGENT_VERSION,
            'server_port': cfg.PORT,
            'max_tasks': int(saved.get('max_tasks', cfg.MAX_CONCURRENT_TASKS)),
            'task_timeout': int(saved.get('task_timeout', cfg.TASK_TIMEOUT_SECONDS)),
            'agent_timeout': int(saved.get('agent_timeout', cfg.AGENT_TIMEOUT_SECONDS)),
            'paths': {
                'models': str(cfg.MODELS_DIR),
                'scripts': str(cfg.SCRIPTS_DIR),
                'database': str(cfg.DATABASE_PATH),
                'uploads': str(cfg.UPLOAD_DIR),
            },
            'dependencies': [dict(r) for r in dependency_list],
            'devices': [dict(r) for r in device_list],
            'saved_settings': saved,
        },
    )


@app.get('/dependencies', response_class=HTMLResponse)
async def dependencies_page(request: Request):
    """Dependencies management page (redirect to settings)."""
    from starlette.responses import RedirectResponse

    return RedirectResponse(url='/settings', status_code=302)


@app.get('/compare', response_class=HTMLResponse)
async def compare_page(request: Request):
    """Results comparison page."""
    async with get_db() as db:
        cursor = await db.execute("""
            SELECT e.id, e.name, e.model_name, e.params, e.status,
                   d.name as device_name, r.metrics, r.created_at as result_date
            FROM experiments e
            LEFT JOIN devices d ON e.device_id = d.id
            LEFT JOIN results r ON e.id = r.experiment_id
            WHERE e.status = 'completed'
            ORDER BY r.created_at DESC
        """)
        experiments_list = await cursor.fetchall()

    # Parse metrics
    experiments_data = []
    for exp in experiments_list:
        exp_dict = dict(exp)
        if exp_dict.get('metrics'):
            try:
                exp_dict['metrics'] = json.loads(exp_dict['metrics'])
            except json.JSONDecodeError:
                exp_dict['metrics'] = {}
        else:
            exp_dict['metrics'] = {}
        if exp_dict.get('params'):
            try:
                exp_dict['params'] = json.loads(exp_dict['params'])
            except json.JSONDecodeError:
                exp_dict['params'] = {}
        else:
            exp_dict['params'] = {}
        experiments_data.append(exp_dict)

    return templates.TemplateResponse(
        request,
        'compare.html',
        {
            'experiments': experiments_data,
        },
    )


# Agent installation script
@app.get('/install', response_class=PlainTextResponse)
async def install_script(request: Request):
    """Return agent installation script."""
    # Get the actual server IP from the request
    server_host = request.headers.get('host', '').split(':')[0]
    if not server_host or server_host in ('0.0.0.0', '127.0.0.1', 'localhost'):
        server_host = request.client.host if request.client else 'SERVER_IP'

    script = f"""#!/bin/bash
# Edge-Bench Agent Installer for Raspberry Pi
# Usage: curl -sSL http://<SERVER_IP>:8000/install | bash

set -e

echo "=== Edge-Bench Agent Installer ==="
echo ""

# Check if running on Raspberry Pi
if ! grep -q "Raspberry Pi" /proc/cpuinfo 2>/dev/null && ! grep -q "BCM" /proc/cpuinfo 2>/dev/null; then
    echo "Warning: This doesn't appear to be a Raspberry Pi"
    if [ -t 0 ]; then
        read -p "Continue anyway? [y/N] " -n 1 -r </dev/tty
        echo
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            exit 1
        fi
    else
        echo "Non-interactive mode: continuing anyway..."
    fi
fi

INSTALL_DIR="$HOME/edge-bench-agent"
SERVER_URL="http://{server_host}:{settings.PORT}"

echo "[1/6] Creating installation directory..."
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

echo "[2/6] Downloading agent files..."
curl -sSL "$SERVER_URL/api/files/agent/main.py" -o main.py
curl -sSL "$SERVER_URL/api/files/agent/executor.py" -o executor.py
curl -sSL "$SERVER_URL/api/files/agent/metrics.py" -o metrics.py
curl -sSL "$SERVER_URL/api/files/agent/config.py" -o config.py
curl -sSL "$SERVER_URL/api/files/agent/result_cache.py" -o result_cache.py
curl -sSL "$SERVER_URL/api/files/agent/requirements.txt" -o requirements.txt

echo "[3/6] Creating virtual environment and installing dependencies..."
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "[4/6] Installing TFLite Runtime..."
pip install tflite-runtime 2>/dev/null || echo "Note: Install tflite-runtime manually if needed"

echo "[5/6] Checking for Edge TPU..."
if lsusb | grep -q "Google Inc.\\|Global Unichip"; then
    echo "Edge TPU detected!"
    echo "Installing pycoral..."
    pip install pycoral 2>/dev/null || echo "Note: Install pycoral manually from coral.ai"
else
    echo "Edge TPU not detected (will use CPU mode)"
fi

# Create models directory
mkdir -p ~/models

echo "[6/7] Creating systemd service..."
cat << SVCEOF | sudo tee /etc/systemd/system/edgebench-agent.service > /dev/null
[Unit]
Description=Edge-Bench Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/main.py
Restart=always
RestartSec=10
Environment=EDGEBENCH_SERVER={server_host}:{settings.PORT}

[Install]
WantedBy=multi-user.target
SVCEOF

sudo systemctl daemon-reload
sudo systemctl enable edgebench-agent
sudo systemctl start edgebench-agent

# Wait for agent to start
sleep 3

# Get IP address
IP_ADDR=$(hostname -I | awk '{{print $1}}')
HOSTNAME=$(hostname)

echo "[7/7] Registering device on server..."
# Register this device on the server
curl -s -X POST "$SERVER_URL/api/devices" \\
  -H "Content-Type: application/json" \\
  -d "{{\\"name\\": \\"$HOSTNAME\\", \\"ip\\": \\"$IP_ADDR\\", \\"port\\": 8001, \\"description\\": \\"Auto-registered Raspberry Pi\\"}}" \\
  > /dev/null 2>&1 || echo "Note: Could not auto-register (device may already exist)"

# Update device status
curl -s -X GET "$SERVER_URL/api/devices" 2>/dev/null | grep -q "$IP_ADDR" && echo "Device registered successfully!"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "Server:      {server_host}:{settings.PORT}"
echo "Agent:       http://$IP_ADDR:8001"
echo "Device name: $HOSTNAME"
echo ""
echo "Commands:"
echo "  Check status: sudo systemctl status edgebench-agent"
echo "  View logs:    sudo journalctl -u edgebench-agent -f"
echo "  Restart:      sudo systemctl restart edgebench-agent"
echo ""
echo "Open http://{server_host}:{settings.PORT}/devices to see this device"
"""
    return script


# Agent uninstallation script
@app.get('/uninstall', response_class=PlainTextResponse)
async def uninstall_script(request: Request):
    """Return agent uninstallation script."""
    script = """#!/bin/bash
# Edge-Bench Agent Uninstaller for Raspberry Pi
# Usage: curl -sSL http://<SERVER_IP>:8000/uninstall | bash

echo "=== Edge-Bench Agent Uninstaller ==="
echo ""

INSTALL_DIR="$HOME/edge-bench-agent"

# Stop and disable service
echo "[1/4] Stopping agent service..."
sudo systemctl stop edgebench-agent 2>/dev/null || true
sudo systemctl disable edgebench-agent 2>/dev/null || true

# Remove systemd service file
echo "[2/4] Removing systemd service..."
sudo rm -f /etc/systemd/system/edgebench-agent.service
sudo systemctl daemon-reload

# Remove agent directory
echo "[3/4] Removing agent files..."
if [ -d "$INSTALL_DIR" ]; then
    rm -rf "$INSTALL_DIR"
    echo "Removed $INSTALL_DIR"
else
    echo "Agent directory not found (already removed?)"
fi

# Models directory
echo ""
echo "[4/4] Cleanup options..."
if [ -t 0 ]; then
    # Interactive terminal -- ask user
    read -p "Remove ~/models directory? [y/N] " -n 1 -r </dev/tty
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf ~/models
        echo "Removed ~/models"
    else
        echo "Kept ~/models"
    fi
else
    # Piped (curl | bash) -- keep models by default
    echo "Kept ~/models (run 'rm -rf ~/models' manually to remove)"
fi

echo ""
echo "=== Uninstallation Complete ==="
echo ""
echo "The agent has been removed from this device."
echo "You may also want to remove this device from the server's device list."
"""
    return script


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        'server.main:app',
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
    )
