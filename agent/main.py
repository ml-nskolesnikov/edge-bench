#!/usr/bin/env python3
"""
Edge-Bench Agent - Lightweight benchmark executor for Raspberry Pi
"""

import asyncio
import base64
from datetime import datetime
import hashlib
import os
import subprocess

from config import AGENT_VERSION, settings
from executor import BenchmarkExecutor
from fastapi import FastAPI, HTTPException
from metrics import SystemMetrics
from result_cache import background_sync_loop, result_cache
import uvicorn

app = FastAPI(
    title='Edge-Bench Agent',
    description='Benchmark executor for Raspberry Pi + Edge TPU',
    version=AGENT_VERSION,
)

executor = BenchmarkExecutor()
system_metrics = SystemMetrics()


@app.on_event('startup')
async def startup_event():
    """Start background cache sync on agent startup."""
    server_url = settings.SERVER_URL
    if server_url:
        asyncio.create_task(background_sync_loop(server_url))
        print(f'[Agent] Cache sync enabled -> {server_url}')
    else:
        print('[Agent] No SERVER_URL set, cache sync disabled')

    unsynced = result_cache.count_unsynced()
    if unsynced > 0:
        print(f'[Agent] {unsynced} cached results awaiting sync')


@app.get('/cache/status')
async def cache_status():
    """Get local result cache status."""
    return {
        'unsynced_count': result_cache.count_unsynced(),
        'server_url': settings.SERVER_URL or None,
        'cache_dir': str(result_cache._cache_dir),
    }


@app.post('/cache/sync')
async def cache_sync_now():
    """Force immediate sync of cached results to server."""
    server_url = settings.SERVER_URL
    if not server_url:
        raise HTTPException(400, 'SERVER_URL not configured')

    stats = await result_cache.sync_to_server(server_url)
    return stats


@app.get('/health')
async def health():
    """Health check endpoint."""
    device_info = system_metrics.get_device_info()
    return {
        'status': 'ok',
        'version': AGENT_VERSION,
        'timestamp': datetime.utcnow().isoformat(),
        'device_info': device_info,
    }


@app.get('/version')
async def version():
    """Get agent version."""
    return {
        'version': AGENT_VERSION,
        'install_dir': settings.INSTALL_DIR,
    }


@app.get('/status')
async def status():
    """Get current agent status."""
    return {
        'version': AGENT_VERSION,
        'running_task': executor.current_task,
        'system': system_metrics.get_current(),
        'tpu_available': system_metrics.check_tpu(),
    }


@app.post('/execute')
async def execute_benchmark(request: dict):
    """Execute a benchmark task."""
    experiment_id = request.get('experiment_id')
    model_path = request.get('model_path')
    params = request.get('params', {})

    if not model_path:
        raise HTTPException(400, 'model_path required')

    # Expand ~ and resolve relative paths
    model_path = os.path.expanduser(model_path)
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise HTTPException(404, f'Model not found: {model_path}')

    try:
        result = await executor.run_benchmark(
            experiment_id=experiment_id,
            model_path=model_path,
            params=params,
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post('/execute/script')
async def execute_script(request: dict):
    """Execute a custom script."""
    script_path = request.get('script_path')
    args = request.get('args', [])
    timeout = request.get('timeout', 600)

    if not script_path:
        raise HTTPException(400, 'script_path required')

    try:
        result = await executor.run_script(
            script_path=script_path,
            args=args,
            timeout=timeout,
        )
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post('/execute/code')
async def execute_code(request: dict):
    """Execute arbitrary shell code (for dependency checks)."""
    code = request.get('code')
    timeout = request.get('timeout', 30)

    if not code:
        raise HTTPException(400, 'code required')

    try:
        result = subprocess.run(
            code,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        return {
            'exit_code': result.returncode,
            'stdout': result.stdout,
            'stderr': result.stderr,
        }
    except subprocess.TimeoutExpired:
        return {
            'exit_code': -1,
            'stdout': '',
            'stderr': f'Command timed out after {timeout}s',
        }
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post('/benchmark/batch')
async def run_batch_benchmark(request: dict):
    """Run batch benchmark on all models in directory."""
    models_dir = request.get('models_dir', os.path.expanduser('~/models'))
    backends = request.get('backends', ['cpu', 'edgetpu'])
    warmup = request.get('warmup', 20)
    runs = request.get('runs', 100)
    threads = request.get('threads', 4)
    output_dir = request.get('output_dir', '/tmp/benchmark_results')

    if executor.current_task:
        raise HTTPException(503, f'Agent busy: {executor.current_task}')

    # Build command
    args = [
        '--models-dir',
        models_dir,
        '--output-dir',
        output_dir,
        '--backends',
        *backends,
        '--warmup',
        str(warmup),
        '--runs',
        str(runs),
        '--threads',
        str(threads),
    ]

    # Run batch benchmark script
    script_path = os.path.join(settings.INSTALL_DIR, 'benchmark_batch.py')

    if not os.path.exists(script_path):
        raise HTTPException(404, 'benchmark_batch.py not found on agent')

    try:
        result = await executor.run_script(script_path, args, timeout=3600)
        return result
    except Exception as e:
        raise HTTPException(500, str(e))


@app.post('/benchmark/full')
async def run_full_benchmark(request: dict):
    """Run full benchmark with detailed metrics on a single model."""
    model_path = request.get('model_path')
    backend = request.get('backend', 'cpu')
    warmup = request.get('warmup', 20)
    runs = request.get('runs', 100)
    threads = request.get('threads', 4)

    if not model_path:
        raise HTTPException(400, 'model_path required')

    # Expand ~ and resolve relative paths
    model_path = os.path.expanduser(model_path)
    model_path = os.path.abspath(model_path)

    if not os.path.exists(model_path):
        raise HTTPException(404, f'Model not found: {model_path}')

    if executor.current_task:
        raise HTTPException(503, f'Agent busy: {executor.current_task}')

    # Build command
    args = [
        '--model',
        model_path,
        '--backend',
        backend,
        '--warmup',
        str(warmup),
        '--runs',
        str(runs),
        '--threads',
        str(threads),
        '--compact',
    ]

    script_path = os.path.join(settings.INSTALL_DIR, 'benchmark_full.py')

    if not os.path.exists(script_path):
        raise HTTPException(404, 'benchmark_full.py not found on agent')

    try:
        result = await executor.run_script(script_path, args, timeout=600)

        # Parse JSON output if successful
        if result.get('status') == 'completed' and result.get('stdout'):
            try:
                import json

                benchmark_result = json.loads(result['stdout'])
                return benchmark_result
            except json.JSONDecodeError:
                pass

        return result
    except Exception as e:
        raise HTTPException(500, str(e))


def _file_sha256(filepath: str) -> str:
    """Calculate SHA256 hash of a file efficiently."""
    h = hashlib.sha256()
    with open(filepath, 'rb') as f:
        for chunk in iter(lambda: f.read(65536), b''):
            h.update(chunk)
    return h.hexdigest()


@app.get('/models')
async def list_models():
    """List available models on this device with SHA256 hashes."""
    models = []
    search_paths = [
        '/home/pi/models',
        os.path.expanduser('~/models'),
        '/tmp/models',
        settings.MODELS_DIR,
    ]

    seen = set()
    for path in search_paths:
        if os.path.exists(path):
            for f in os.listdir(path):
                if f.endswith(('.tflite', '.pb')):
                    full_path = os.path.join(path, f)
                    if full_path not in seen:
                        seen.add(full_path)
                        models.append(
                            {
                                'name': f,
                                'path': full_path,
                                'size_bytes': os.path.getsize(full_path),
                                'hash': _file_sha256(full_path),
                            }
                        )

    return {'models': models}


@app.post('/upload')
async def upload_model(request: dict):
    """Receive model file from server."""
    content = request.get('content')
    filename = request.get('filename')

    if not content or not filename:
        raise HTTPException(400, 'content and filename required')

    dest_dir = settings.MODELS_DIR
    os.makedirs(dest_dir, exist_ok=True)

    dest_path = os.path.join(dest_dir, filename)

    with open(dest_path, 'wb') as f:
        f.write(base64.b64decode(content))

    return {
        'path': dest_path,
        'size_bytes': os.path.getsize(dest_path),
        'filename': filename,
    }


@app.delete('/models/{filename}')
async def delete_model(filename: str):
    """Delete a model file."""
    # Security: only allow deleting from models directory
    dest_path = os.path.join(settings.MODELS_DIR, filename)

    if not os.path.exists(dest_path):
        raise HTTPException(404, 'Model not found')

    os.remove(dest_path)
    return {'status': 'deleted', 'filename': filename}


@app.post('/update')
async def update_agent(request: dict):
    """Update agent files from server."""
    files = request.get('files', {})

    if not files:
        raise HTTPException(400, 'No files provided')

    updated = []
    errors = []

    for filename, content in files.items():
        # Security: only allow specific files
        allowed_files = {'main.py', 'executor.py', 'metrics.py', 'config.py'}
        if filename not in allowed_files:
            errors.append(f'{filename}: not allowed')
            continue

        try:
            dest_path = os.path.join(settings.INSTALL_DIR, filename)
            with open(dest_path, 'w') as f:
                f.write(content)
            updated.append(filename)
        except Exception as e:
            errors.append(f'{filename}: {e}')

    return {
        'updated': updated,
        'errors': errors,
        'restart_required': len(updated) > 0,
    }


@app.post('/restart')
async def restart_agent():
    """Restart the agent service."""
    try:
        # Schedule restart after response
        asyncio.create_task(_delayed_restart())
        return {'status': 'restarting', 'message': 'Agent will restart in 2 seconds'}
    except Exception as e:
        raise HTTPException(500, str(e))


async def _delayed_restart():
    """Restart agent after a short delay."""
    await asyncio.sleep(2)
    subprocess.run(['sudo', 'systemctl', 'restart', 'edgebench-agent'], check=False)


if __name__ == '__main__':
    print(f'Starting Edge-Bench Agent v{AGENT_VERSION} on port {settings.PORT}')
    print(f'TPU available: {system_metrics.check_tpu()}')

    uvicorn.run(
        app,
        host='0.0.0.0',
        port=settings.PORT,
        log_level='info',
    )
