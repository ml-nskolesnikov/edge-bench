"""
Scripts API Endpoints
"""

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
import httpx

from server.db.database import get_db

router = APIRouter()


@router.post('/check-deps')
async def check_dependencies(request: dict):
    """Check dependencies on a device."""
    device_id = request.get('device_id')

    if not device_id:
        raise HTTPException(400, 'device_id required')

    # Get device and dependencies from DB
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

        if not device:
            raise HTTPException(404, 'Device not found')

        # Get enabled dependencies from database
        cursor = await db.execute(
            'SELECT * FROM dependencies WHERE enabled = 1 ORDER BY sort_order, id'
        )
        deps_rows = await cursor.fetchall()

    deps = [dict(row) for row in deps_rows]
    agent_url = f'http://{device["ip"]}:{device["port"]}'

    results = []

    for dep in deps:
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                response = await client.post(
                    f'{agent_url}/execute/code',
                    json={
                        'code': f"""
import subprocess
import sys

try:
    result = subprocess.run(
        {repr(dep['check_cmd'])},
        shell=True,
        capture_output=True,
        text=True,
        timeout=5
    )
    if result.returncode == 0:
        print("OK:" + result.stdout.strip())
    else:
        print("ERROR:" + (result.stderr.strip() or "not found"))
except Exception as e:
    print("ERROR:" + str(e))
""",
                        'timeout': 10,
                    },
                )

                if response.status_code == 200:
                    data = response.json()
                    output = data.get('output', '').strip()

                    if output.startswith('OK:'):
                        version = output[3:].strip()
                        results.append(
                            {
                                'name': dep['name'],
                                'status': 'ok',
                                'version': version
                                if version and version != 'ok'
                                else 'установлен',
                                'critical': dep['critical'],
                            }
                        )
                    else:
                        error = output[6:] if output.startswith('ERROR:') else output
                        results.append(
                            {
                                'name': dep['name'],
                                'status': 'missing',
                                'error': error or 'не найден',
                                'critical': dep['critical'],
                            }
                        )
                else:
                    results.append(
                        {
                            'name': dep['name'],
                            'status': 'error',
                            'error': 'Ошибка агента',
                            'critical': dep['critical'],
                        }
                    )
        except Exception as e:
            results.append(
                {
                    'name': dep['name'],
                    'status': 'error',
                    'error': str(e),
                    'critical': dep['critical'],
                }
            )

    return {'dependencies': results}


@router.post('/run')
async def run_script(request: dict):
    """Run a script file on a device."""
    script_id = request.get('script_id')
    device_id = request.get('device_id')
    args = request.get('args', '')
    timeout = request.get('timeout', 300)

    if not script_id or not device_id:
        raise HTTPException(400, 'script_id and device_id required')

    # Get script file
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM files WHERE id = ?', (script_id,))
        script = await cursor.fetchone()

        if not script:
            raise HTTPException(404, 'Script not found')

        # Get device
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

        if not device:
            raise HTTPException(404, 'Device not found')

    # Read script content
    script_path = Path(script['path'])
    if not script_path.exists():
        raise HTTPException(404, 'Script file not found on disk')

    script_content = script_path.read_text()

    # Send to device
    agent_url = f'http://{device["ip"]}:{device["port"]}'

    # Add timeout buffer for network
    client_timeout = timeout + 30

    try:
        async with httpx.AsyncClient(timeout=client_timeout) as client:
            response = await client.post(
                f'{agent_url}/execute/code',
                json={
                    'code': script_content,
                    'args': args,
                    'timeout': timeout,
                },
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'status': 'error',
                    'error': f'Agent error: {response.text}',
                }
    except httpx.TimeoutException:
        return {
            'status': 'error',
            'error': f'Timeout: скрипт выполняется дольше {timeout}s',
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
        }


@router.post('/execute')
async def execute_code(request: dict):
    """Execute raw Python code on a device."""
    device_id = request.get('device_id')
    code = request.get('code')

    if not device_id or not code:
        raise HTTPException(400, 'device_id and code required')

    # Get device
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

        if not device:
            raise HTTPException(404, 'Device not found')

    # Send to device
    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f'{agent_url}/execute/code',
                json={'code': code},
            )

            if response.status_code == 200:
                return response.json()
            else:
                return {
                    'status': 'error',
                    'error': f'Agent error: {response.text}',
                }
    except httpx.TimeoutException:
        return {
            'status': 'error',
            'error': 'Timeout: код выполняется слишком долго',
        }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
        }


@router.post('/system-info')
async def get_system_info(request: dict):
    """Get system information from a device (CPU, RAM, temperature)."""
    device_id = request.get('device_id')

    if not device_id:
        raise HTTPException(400, 'device_id required')

    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

        if not device:
            raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'

    # Get system metrics from agent
    code = '''
import json
import psutil
import platform
import os

def get_cpu_temp():
    """Get CPU temperature."""
    try:
        # Raspberry Pi
        temp_file = "/sys/class/thermal/thermal_zone0/temp"
        if os.path.exists(temp_file):
            with open(temp_file) as f:
                return round(int(f.read().strip()) / 1000, 1)
    except:
        pass

    try:
        temps = psutil.sensors_temperatures()
        if temps:
            for name, entries in temps.items():
                for entry in entries:
                    if entry.current:
                        return round(entry.current, 1)
    except:
        pass

    return None

def get_disk_usage():
    """Get disk usage."""
    try:
        usage = psutil.disk_usage("/")
        return {
            "total_gb": round(usage.total / (1024**3), 1),
            "used_gb": round(usage.used / (1024**3), 1),
            "free_gb": round(usage.free / (1024**3), 1),
            "percent": usage.percent
        }
    except:
        return None

info = {
    "cpu": {
        "percent": psutil.cpu_percent(interval=0.5),
        "count": psutil.cpu_count(),
        "freq_mhz": round(psutil.cpu_freq().current, 0) if psutil.cpu_freq() else None
    },
    "memory": {
        "total_mb": round(psutil.virtual_memory().total / (1024**2), 0),
        "available_mb": round(psutil.virtual_memory().available / (1024**2), 0),
        "used_mb": round(psutil.virtual_memory().used / (1024**2), 0),
        "percent": psutil.virtual_memory().percent
    },
    "temperature_celsius": get_cpu_temp(),
    "disk": get_disk_usage(),
    "platform": {
        "system": platform.system(),
        "release": platform.release(),
        "machine": platform.machine(),
        "hostname": platform.node(),
        "python": platform.python_version()
    },
    "uptime_hours": round((psutil.boot_time() and (
        __import__("time").time() - psutil.boot_time()
    ) / 3600) or 0, 1)
}

print(json.dumps(info))
'''

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.post(
                f'{agent_url}/execute/code',
                json={'code': code, 'timeout': 10},
            )

            if response.status_code == 200:
                data = response.json()
                output = data.get('output', '').strip()

                if data.get('status') == 'completed' and output:
                    try:
                        return {'status': 'ok', 'info': json.loads(output)}
                    except Exception:
                        return {'status': 'error', 'error': 'Invalid JSON response'}

                return {'status': 'error', 'error': data.get('error', 'Unknown error')}

            return {'status': 'error', 'error': f'Agent error: {response.status_code}'}

    except Exception as e:
        return {'status': 'error', 'error': str(e)}
