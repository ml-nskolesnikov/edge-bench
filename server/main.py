"""
Edge-Bench Server - Main Entry Point
"""

import asyncio
from contextlib import asynccontextmanager
import json
import os as _os
from pathlib import Path

from fastapi import Depends, FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import PlainTextResponse
from fastapi.staticfiles import StaticFiles
from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

from server.api import (
    dependencies,
    devices,
    experiments,
    files,
    results,
    settings as settings_api,
)
from server.api.schedules import router as schedules_router
from server.core.auth import require_api_secret
from server.core.config import settings
from server.core.queue import task_queue
from server.core.scheduler import restore_schedules, scheduler
from server.core.ws_manager import ws_manager
from server.db.database import get_db, init_db
from server.routes.ui import router as ui_router

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

# Trust X-Forwarded-Proto / X-Forwarded-For only from a proxy on the same host.
# No reverse proxy is currently deployed, so this has no effect yet.
# When you add nginx/caddy/traefik, set EDGEBENCH_PROXY_TRUST to its IP.
_trusted = _os.environ.get('EDGEBENCH_PROXY_TRUST', '127.0.0.1')
app.add_middleware(ProxyHeadersMiddleware, trusted_hosts=_trusted)

app.mount('/static', StaticFiles(directory=BASE_DIR / 'static'), name='static')

# API routers — all protected by shared-secret when EDGEBENCH_AGENT_SECRET is set
_auth = [Depends(require_api_secret)]
app.include_router(devices.router, prefix='/api/devices', tags=['devices'], dependencies=_auth)
app.include_router(experiments.router, prefix='/api/experiments', tags=['experiments'], dependencies=_auth)
app.include_router(results.router, prefix='/api/results', tags=['results'], dependencies=_auth)
app.include_router(files.router, prefix='/api/files', tags=['files'], dependencies=_auth)
app.include_router(
    dependencies.router, prefix='/api/dependencies', tags=['dependencies'], dependencies=_auth
)
app.include_router(settings_api.router, prefix='/api/settings', tags=['settings'], dependencies=_auth)
app.include_router(schedules_router, prefix='/api/schedules', tags=['schedules'], dependencies=_auth)

# Web UI routes (HTML pages — no auth; served via APIRouter in server/routes/ui.py)
app.include_router(ui_router)


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


# Agent installation script
@app.get('/install', response_class=PlainTextResponse)
async def install_script(request: Request):
    """Return agent installation script."""
    server_host = request.headers.get('host', '').split(':')[0]
    if not server_host or server_host in ('0.0.0.0', '127.0.0.1', 'localhost'):
        server_host = request.client.host if request.client else 'SERVER_IP'

    scheme = request.url.scheme  # preserves https when behind a TLS proxy
    script = f"""#!/bin/bash
# Edge-Bench Agent Installer for Raspberry Pi
# Usage: curl -sSL {scheme}://<SERVER_IP>:8000/install | bash
# NOTE: Use HTTPS in production to protect the secret and prevent MITM.

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
SERVER_URL="{scheme}://{server_host}:{settings.PORT}"

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
Environment=EDGEBENCH_SERVER={scheme}://{server_host}:{settings.PORT}

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
curl -s -X POST "$SERVER_URL/api/devices" \\
  -H "Content-Type: application/json" \\
  -d "{{\\"name\\": \\"$HOSTNAME\\", \\"ip\\": \\"$IP_ADDR\\", \\"port\\": 8001, \\"description\\": \\"Auto-registered Raspberry Pi\\"}}" \\
  > /dev/null 2>&1 || echo "Note: Could not auto-register (device may already exist)"

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
        proxy_headers=True,       # honour X-Forwarded-Proto from upstream TLS proxy
        forwarded_allow_ips='*',  # restrict to proxy IP in internet-facing deploys
    )
