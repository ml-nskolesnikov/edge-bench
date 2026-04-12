#!/bin/bash
# Edge-Bench Agent Installer for Raspberry Pi
# Run this script on your Raspberry Pi

set -e

echo "=========================================="
echo "  Edge-Bench Agent Installer"
echo "=========================================="
echo ""

# Configuration
INSTALL_DIR="${EDGEBENCH_INSTALL_DIR:-$HOME/edge-bench-agent}"
AGENT_PORT="${EDGEBENCH_AGENT_PORT:-8001}"

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Check if running on ARM (Raspberry Pi)
ARCH=$(uname -m)
if [[ "$ARCH" != "arm"* && "$ARCH" != "aarch64" ]]; then
    log_warn "Architecture is $ARCH, not ARM. This is designed for Raspberry Pi."
    read -p "Continue anyway? [y/N] " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | cut -d' ' -f2)
log_info "Python version: $PYTHON_VERSION"

# Create installation directory
log_info "Creating installation directory: $INSTALL_DIR"
mkdir -p "$INSTALL_DIR"
cd "$INSTALL_DIR"

# Create models directory
mkdir -p ~/models

# Copy agent files (if running from source)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -f "$SCRIPT_DIR/main.py" ]; then
    log_info "Copying agent files from source..."
    cp "$SCRIPT_DIR"/*.py "$INSTALL_DIR/"
fi

# Create virtual environment and install Python dependencies
log_info "Creating virtual environment..."
python3 -m venv "$INSTALL_DIR/venv"
source "$INSTALL_DIR/venv/bin/activate"

log_info "Installing Python dependencies..."
pip install --upgrade pip
pip install fastapi uvicorn psutil numpy

# Install TFLite Runtime
log_info "Installing TFLite Runtime..."
pip install tflite-runtime 2>/dev/null || {
    log_warn "Could not install tflite-runtime from PyPI"
    log_info "Trying to install from Google Coral repository..."

    # Add Coral repository
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | \
        sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
    sudo apt-get update

    # Install TFLite Runtime (system-wide as fallback)
    sudo apt-get install -y python3-tflite-runtime || log_warn "TFLite runtime installation failed"
}

# Check for Edge TPU
log_info "Checking for Edge TPU..."
if lsusb 2>/dev/null | grep -qi "google\|global unichip"; then
    log_info "Edge TPU detected!"

    # Install Edge TPU runtime (system library)
    log_info "Installing Edge TPU runtime..."
    sudo apt-get install -y libedgetpu1-std 2>/dev/null || {
        log_warn "Could not install Edge TPU runtime from apt"
        log_info "Please install manually from https://coral.ai/docs/accelerator/get-started/"
    }

    # Install PyCoral in venv
    pip install pycoral 2>/dev/null || {
        log_warn "Could not install pycoral"
    }
else
    log_warn "Edge TPU not detected. Agent will run in CPU-only mode."
    log_info "Connect Edge TPU and re-run installer to enable TPU support."
fi

# Create systemd service
log_info "Creating systemd service..."
SERVICE_FILE="/etc/systemd/system/edgebench-agent.service"

sudo tee "$SERVICE_FILE" > /dev/null << EOF
[Unit]
Description=Edge-Bench Agent
After=network.target

[Service]
Type=simple
User=$USER
WorkingDirectory=$INSTALL_DIR
Environment=EDGEBENCH_AGENT_PORT=$AGENT_PORT
ExecStart=$INSTALL_DIR/venv/bin/python $INSTALL_DIR/main.py
Restart=always
RestartSec=10
StandardOutput=journal
StandardError=journal

[Install]
WantedBy=multi-user.target
EOF

# Reload systemd and enable service
sudo systemctl daemon-reload
sudo systemctl enable edgebench-agent

# Start the service
log_info "Starting Edge-Bench Agent..."
sudo systemctl start edgebench-agent

# Wait for agent to start
sleep 3
if systemctl is-active --quiet edgebench-agent; then
    log_info "Agent is running!"
else
    log_error "Agent failed to start. Check logs with: journalctl -u edgebench-agent"
    exit 1
fi

# Get IP address
IP_ADDR=$(hostname -I | awk '{print $1}')
HOSTNAME=$(hostname)

# Try to register on server if SERVER_URL is set
if [ -n "$EDGEBENCH_SERVER" ]; then
    log_info "Registering device on server..."
    curl -s -X POST "http://$EDGEBENCH_SERVER/api/devices" \
      -H "Content-Type: application/json" \
      -d "{\"name\": \"$HOSTNAME\", \"ip\": \"$IP_ADDR\", \"port\": $AGENT_PORT, \"description\": \"Auto-registered Raspberry Pi\"}" \
      > /dev/null 2>&1 && log_info "Device registered successfully!" || log_warn "Could not auto-register (device may already exist)"
fi

echo ""
echo "=========================================="
echo "  Installation Complete!"
echo "=========================================="
echo ""
echo "Agent running on: http://$IP_ADDR:$AGENT_PORT"
echo "Device name:      $HOSTNAME"
echo ""
echo "Commands:"
echo "  Check status:  sudo systemctl status edgebench-agent"
echo "  View logs:     sudo journalctl -u edgebench-agent -f"
echo "  Restart:       sudo systemctl restart edgebench-agent"
echo "  Stop:          sudo systemctl stop edgebench-agent"
echo ""
if [ -n "$EDGEBENCH_SERVER" ]; then
    echo "Server: http://$EDGEBENCH_SERVER/devices"
fi
echo ""
