#!/usr/bin/env python3
"""
Edge-Bench Integration for ECCV 2026

Uploads models to edge-bench server and creates benchmark experiments
for Table T4 generation.

Usage:
    # Upload models and create experiments
    python scripts/9.9_run_edgebench.py --server http://localhost:8000 --device rpi-coral-1

    # Just upload models
    python scripts/9.9_run_edgebench.py --upload-only

    # Check status
    python scripts/9.9_run_edgebench.py --status
"""

import argparse
import json
import sys
import time
from pathlib import Path

try:
    import httpx
except ImportError:
    print('[ERROR] httpx not installed. Run: pip install httpx')
    sys.exit(1)


EXPORT_DIR = Path('export')
EDGETPU_DIR = EXPORT_DIR / 'edgetpu'

ECCV_MODELS = [
    'mobilenetv2_int8_ptq_hybrid',
    'mobilenetv2_int8_ptq_Fuzzy',
    'mobilenetv2_int8_ptq_sbert',
]


def check_server(base_url: str) -> bool:
    """Check if edge-bench server is running."""
    try:
        resp = httpx.get(f'{base_url}/api/devices', timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


def get_devices(base_url: str) -> list:
    """Get list of registered devices."""
    resp = httpx.get(f'{base_url}/api/devices')
    resp.raise_for_status()
    return resp.json()


def upload_model(base_url: str, model_path: Path) -> dict:
    """Upload model to server."""
    with open(model_path, 'rb') as f:
        files = {'file': (model_path.name, f, 'application/octet-stream')}
        data = {'file_type': 'model'}
        resp = httpx.post(f'{base_url}/api/files/upload', files=files, data=data, timeout=60)

    resp.raise_for_status()
    return resp.json()


def create_experiment(
    base_url: str,
    device_id: str,
    model_name: str,
    backend: str,
    runs: int = 100,
) -> dict:
    """Create benchmark experiment."""
    payload = {
        'device_id': device_id,
        'model_name': model_name,
        'params': {
            'backend': backend,
            'benchmark_runs': runs,
            'warmup_runs': 20,
            'num_threads': 4,
        },
    }

    resp = httpx.post(f'{base_url}/api/experiments', json=payload, timeout=30)
    resp.raise_for_status()
    return resp.json()


def wait_for_experiment(base_url: str, experiment_id: str, timeout: int = 300) -> dict:
    """Wait for experiment to complete."""
    start = time.time()

    while time.time() - start < timeout:
        resp = httpx.get(f'{base_url}/api/experiments/{experiment_id}')
        exp = resp.json()

        status = exp.get('status', 'unknown')
        if status in ['completed', 'failed', 'error']:
            return exp

        print(f'  Status: {status}...', end='\r')
        time.sleep(2)

    return {'status': 'timeout'}


def get_results(base_url: str, experiment_id: str = None) -> list:
    """Get benchmark results."""
    url = f'{base_url}/api/results'
    if experiment_id:
        url = f'{base_url}/api/results/{experiment_id}'

    resp = httpx.get(url)
    resp.raise_for_status()
    return resp.json()


def export_t4_csv(base_url: str, output_path: Path):
    """Export results as T4 CSV."""
    resp = httpx.get(f'{base_url}/api/results/export/csv')
    resp.raise_for_status()

    with open(output_path, 'w') as f:
        f.write(resp.text)

    print(f'[OK] T4 exported: {output_path}')


def main():
    parser = argparse.ArgumentParser(description='Edge-Bench Integration')
    parser.add_argument('--server', '-s', default='http://localhost:8000',
                        help='Edge-bench server URL')
    parser.add_argument('--device', '-d', help='Device ID for experiments')
    parser.add_argument('--upload-only', action='store_true',
                        help='Only upload models, do not run experiments')
    parser.add_argument('--status', action='store_true',
                        help='Check server and device status')
    parser.add_argument('--runs', '-r', type=int, default=100,
                        help='Number of benchmark runs')
    parser.add_argument('--export-csv', type=Path,
                        help='Export results to CSV')

    args = parser.parse_args()

    # Check server
    print(f'[INFO] Checking server: {args.server}')
    if not check_server(args.server):
        print('[ERROR] Server not reachable')
        print('Start server: cd edge-bench && make server')
        sys.exit(1)
    print('[OK] Server is running')

    # Status check
    if args.status:
        devices = get_devices(args.server)
        print(f'\nRegistered devices: {len(devices)}')
        for dev in devices:
            status = dev.get('status', 'unknown')
            print(f"  - {dev['name']} ({dev['ip']}:{dev['port']}) [{status}]")
        return

    # Find models
    models_to_upload = []

    for name in ECCV_MODELS:
        # INT8 model
        int8_path = EXPORT_DIR / f'{name}.tflite'
        if int8_path.exists():
            models_to_upload.append(int8_path)

        # EdgeTPU model
        edgetpu_path = EDGETPU_DIR / f'{name}_edgetpu.tflite'
        if edgetpu_path.exists():
            models_to_upload.append(edgetpu_path)

    if not models_to_upload:
        print('[ERROR] No models found to upload')
        print(f'Expected in: {EXPORT_DIR}')
        sys.exit(1)

    print(f'\n[INFO] Found {len(models_to_upload)} model(s) to upload')

    # Upload models
    uploaded = []
    for model_path in models_to_upload:
        print(f'  Uploading: {model_path.name}...', end=' ')
        try:
            result = upload_model(args.server, model_path)
            uploaded.append(result)
            print('[OK]')
        except Exception as e:
            print(f'[FAILED] {e}')

    print(f'\n[OK] Uploaded {len(uploaded)} model(s)')

    if args.upload_only:
        return

    # Run experiments
    if not args.device:
        devices = get_devices(args.server)
        if not devices:
            print('[ERROR] No devices registered. Add device in Web UI first.')
            sys.exit(1)

        # Use first online device
        online = [d for d in devices if d.get('status') == 'online']
        if not online:
            print('[ERROR] No online devices. Check RPi agent.')
            print('Registered devices:')
            for d in devices:
                print(f"  - {d['name']}: {d.get('status', 'unknown')}")
            sys.exit(1)

        args.device = online[0]['id']
        print(f"[INFO] Using device: {online[0]['name']}")

    print(f'\n[INFO] Creating experiments on device: {args.device}')

    experiments = []

    for model_path in models_to_upload:
        model_name = model_path.name

        # Determine backend
        if '_edgetpu' in model_name:
            backends = ['edgetpu']
        else:
            backends = ['cpu']

        for backend in backends:
            print(f'  Creating: {model_name} [{backend}]...', end=' ')
            try:
                exp = create_experiment(
                    args.server,
                    args.device,
                    model_name,
                    backend,
                    args.runs,
                )
                experiments.append(exp)
                print(f"[OK] ID: {exp['id']}")
            except Exception as e:
                print(f'[FAILED] {e}')

    print(f'\n[INFO] Waiting for {len(experiments)} experiment(s)...')

    for exp in experiments:
        print(f"  {exp['model_name']}...", end=' ')
        result = wait_for_experiment(args.server, exp['id'])
        status = result.get('status', 'unknown')
        print(f'[{status.upper()}]')

    # Export CSV if requested
    if args.export_csv:
        export_t4_csv(args.server, args.export_csv)

    print('\n[DONE] All experiments completed')
    print(f'View results: {args.server}/results')


if __name__ == '__main__':
    main()
