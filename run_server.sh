#!/bin/bash
# Start Edge-Bench Server

cd "$(dirname "$0")"

# Create virtual environment if not exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements/server.txt
else
    source venv/bin/activate
fi

echo "Starting Edge-Bench Server..."
python -m server.main
