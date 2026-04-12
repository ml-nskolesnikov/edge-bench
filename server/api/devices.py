"""
Devices API Endpoints
"""

from datetime import datetime
from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException
import httpx

from server.core.config import settings
from server.core.models import DeviceCreate, DeviceResponse, DeviceStatus
from server.db.database import get_db

router = APIRouter()


@router.get('', response_model=list[DeviceResponse])
async def list_devices():
    """Get all registered devices."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        rows = await cursor.fetchall()

    return [dict(row) for row in rows]


@router.post('', response_model=DeviceResponse)
async def create_device(device: DeviceCreate):
    """Register a new device."""
    device_id = f'dev_{uuid.uuid4().hex[:8]}'

    async with get_db() as db:
        # Check if name already exists
        cursor = await db.execute(
            'SELECT id FROM devices WHERE name = ?', (device.name,)
        )
        if await cursor.fetchone():
            raise HTTPException(400, 'Device name already exists')

        # Check if IP:port already exists
        cursor = await db.execute(
            'SELECT id FROM devices WHERE ip = ? AND port = ?', (device.ip, device.port)
        )
        if await cursor.fetchone():
            raise HTTPException(400, 'Device with this IP:port already exists')

        # Check device status immediately
        status = DeviceStatus.OFFLINE
        device_info = None
        agent_url = f'http://{device.ip}:{device.port}'

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                response = await client.get(f'{agent_url}/health')
                if response.status_code == 200:
                    status = DeviceStatus.ONLINE
                    device_info = response.json().get('device_info')
        except Exception:
            pass

        # Insert device
        await db.execute(
            """INSERT INTO devices (id, name, ip, port, status, description, device_info, last_seen, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                device_id,
                device.name,
                device.ip,
                device.port,
                status.value,
                device.description,
                str(device_info) if device_info else None,
                datetime.utcnow().isoformat()
                if status == DeviceStatus.ONLINE
                else None,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()

        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        row = await cursor.fetchone()

    return dict(row)


@router.get('/{device_id}', response_model=DeviceResponse)
async def get_device(device_id: str):
    """Get device by ID."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'Device not found')

    return dict(row)


@router.delete('/{device_id}')
async def delete_device(device_id: str):
    """Delete a device."""
    async with get_db() as db:
        cursor = await db.execute('SELECT id FROM devices WHERE id = ?', (device_id,))
        if not await cursor.fetchone():
            raise HTTPException(404, 'Device not found')

        await db.execute('DELETE FROM devices WHERE id = ?', (device_id,))
        await db.commit()

    return {'status': 'deleted'}


@router.get('/{device_id}/status')
async def check_device_status(device_id: str):
    """Check device status and update in database."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'
    status = DeviceStatus.OFFLINE
    device_info = None

    try:
        async with httpx.AsyncClient(timeout=settings.AGENT_TIMEOUT_SECONDS) as client:
            response = await client.get(f'{agent_url}/health')

            if response.status_code == 200:
                data = response.json()
                status = DeviceStatus.ONLINE
                device_info = data.get('device_info')
    except Exception:
        status = DeviceStatus.OFFLINE

    # Update database
    async with get_db() as db:
        await db.execute(
            """UPDATE devices
            SET status = ?, last_seen = ?, device_info = ?
            WHERE id = ?""",
            (
                status.value,
                datetime.utcnow().isoformat()
                if status == DeviceStatus.ONLINE
                else None,
                str(device_info) if device_info else None,
                device_id,
            ),
        )
        await db.commit()

    return {
        'status': status,
        'device_info': device_info,
    }


@router.post('/{device_id}/ping')
async def ping_device(device_id: str):
    """Ping device to check connectivity."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            start = datetime.utcnow()
            response = await client.get(f'{agent_url}/health')
            latency = (datetime.utcnow() - start).total_seconds() * 1000

            if response.status_code == 200:
                data = response.json()
                return {
                    'status': 'ok',
                    'latency_ms': round(latency, 2),
                    'agent_version': data.get('version'),
                }
    except Exception as e:
        return {
            'status': 'error',
            'error': str(e),
        }


@router.get('/{device_id}/version')
async def get_device_version(device_id: str):
    """Get agent version on device."""
    from server.core.config import AGENT_VERSION

    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=5) as client:
            response = await client.get(f'{agent_url}/version')
            if response.status_code == 200:
                data = response.json()
                remote_version = data.get('version', 'unknown')
                return {
                    'device_version': remote_version,
                    'server_version': AGENT_VERSION,
                    'update_available': remote_version != AGENT_VERSION,
                }
    except Exception as e:
        raise HTTPException(503, f'Cannot connect to device: {e}')


@router.post('/{device_id}/update')
async def update_device_agent(device_id: str):
    """Update agent on device to current version."""
    from pathlib import Path

    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'
    agent_dir = Path(__file__).parent.parent.parent / 'agent'

    # Read agent source files
    files_to_update = {}
    for filename in ['main.py', 'executor.py', 'metrics.py', 'config.py']:
        file_path = agent_dir / filename
        if file_path.exists():
            files_to_update[filename] = file_path.read_text()

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            # Send update
            response = await client.post(
                f'{agent_url}/update',
                json={'files': files_to_update},
            )

            if response.status_code != 200:
                raise HTTPException(500, f'Update failed: {response.text}')

            result = response.json()

            # Restart agent if files were updated
            if result.get('restart_required'):
                await client.post(f'{agent_url}/restart')

            return {
                'status': 'updated',
                'updated_files': result.get('updated', []),
                'errors': result.get('errors', []),
                'restarted': result.get('restart_required', False),
            }

    except httpx.TimeoutException:
        # Agent might have restarted
        return {
            'status': 'updated',
            'message': 'Agent is restarting',
        }
    except Exception as e:
        raise HTTPException(500, f'Update failed: {e}')


@router.get('/{device_id}/models')
async def list_device_models(device_id: str):
    """List models available on device."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=10) as client:
            response = await client.get(f'{agent_url}/models')
            if response.status_code == 200:
                return response.json()
    except Exception as e:
        raise HTTPException(503, f'Cannot connect to device: {e}')


@router.delete('/{device_id}/models/{filename}')
async def delete_device_model(device_id: str, filename: str):
    """Delete a model from device."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=15) as client:
            response = await client.delete(f'{agent_url}/models/{filename}')
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 404:
                raise HTTPException(404, 'Model not found on device')
            else:
                raise HTTPException(500, f'Delete failed: {response.text}')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f'Cannot connect to device: {e}')


async def _get_device_models_hashes(agent_url: str) -> dict[str, dict]:
    """Fetch models from device and index by hash and name.

    Returns dict: {hash: {name, path, size_bytes, hash}, ...}
    """
    try:
        async with httpx.AsyncClient(timeout=15) as client:
            resp = await client.get(f'{agent_url}/models')
            if resp.status_code == 200:
                data = resp.json()
                by_hash = {}
                for m in data.get('models', []):
                    h = m.get('hash')
                    if h:
                        by_hash[h] = m
                return by_hash
    except Exception:
        pass
    return {}


async def _get_device_and_file(device_id: str, file_id: str):
    """Load device and file records. Raises HTTPException on errors."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()
    if not device:
        raise HTTPException(404, 'Device not found')

    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM files WHERE id = ?', (file_id,))
        file_record = await cursor.fetchone()
    if not file_record:
        raise HTTPException(404, 'File not found')

    file_path = Path(file_record['path'])
    if not file_path.exists():
        raise HTTPException(404, 'File not found on disk')

    return device, file_record, file_path


@router.post('/{device_id}/check-deploy')
async def check_deploy_status(device_id: str, request: dict):
    """Check which models need deployment based on SHA256 hash.

    Request: {file_ids: [id1, id2, ...]}
    Response: {results: [{file_id, name, status, detail}, ...]}
      status: "skip"    - identical hash already on device
              "update"  - same name but different hash (needs re-upload)
              "new"     - model not on device at all
              "error"   - could not check
    """
    file_ids = request.get('file_ids', [])
    if not file_ids:
        raise HTTPException(400, 'file_ids required')

    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()
    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'
    device_models = await _get_device_models_hashes(agent_url)

    # Also index device models by name for name-collision detection
    by_name = {}
    for m in device_models.values():
        by_name[m['name']] = m

    results = []
    for fid in file_ids:
        async with get_db() as db:
            cursor = await db.execute('SELECT * FROM files WHERE id = ?', (fid,))
            frec = await cursor.fetchone()

        if not frec:
            results.append(
                {
                    'file_id': fid,
                    'name': '?',
                    'status': 'error',
                    'detail': 'File not found on server',
                }
            )
            continue

        server_hash = frec['hash']
        server_name = frec['name']

        # Case 1: exact hash match on device (same binary, maybe different name)
        if server_hash in device_models:
            dm = device_models[server_hash]
            if dm['name'] == server_name:
                detail = 'Идентичная модель уже на устройстве'
            else:
                detail = f'Та же модель на устройстве как "{dm["name"]}"'
            results.append(
                {
                    'file_id': fid,
                    'name': server_name,
                    'status': 'skip',
                    'detail': detail,
                    'device_path': dm.get('path'),
                }
            )
            continue

        # Case 2: same filename on device but different hash (updated model)
        if server_name in by_name:
            dm = by_name[server_name]
            results.append(
                {
                    'file_id': fid,
                    'name': server_name,
                    'status': 'update',
                    'detail': 'Имя совпадает, но модель отличается -- будет перезаписана',
                    'device_hash': dm.get('hash', '')[:12],
                    'server_hash': server_hash[:12],
                }
            )
            continue

        # Case 3: model not on device
        results.append(
            {
                'file_id': fid,
                'name': server_name,
                'status': 'new',
                'detail': 'Модель отсутствует на устройстве',
            }
        )

    return {'results': results}


@router.post('/{device_id}/upload-model')
async def upload_model_to_device(device_id: str, request: dict):
    """Upload a model file to device with hash-based duplicate check.

    Request: {file_id: str, force: bool (optional)}
    Response includes 'skipped' flag if model already present.
    """
    import base64

    file_id = request.get('file_id')
    force = request.get('force', False)
    if not file_id:
        raise HTTPException(400, 'file_id required')

    device, file_record, file_path = await _get_device_and_file(device_id, file_id)
    agent_url = f'http://{device["ip"]}:{device["port"]}'
    server_hash = file_record['hash']

    # Pre-deploy hash check (skip if force=True)
    if not force:
        device_models = await _get_device_models_hashes(agent_url)
        if server_hash in device_models:
            dm = device_models[server_hash]
            return {
                'path': dm.get('path'),
                'size_bytes': file_record['size_bytes'],
                'filename': dm['name'],
                'skipped': True,
                'reason': 'hash_match',
                'detail': 'Модель уже на устройстве'
                + (f' как "{dm["name"]}"' if dm['name'] != file_record['name'] else ''),
            }

    # Upload
    content = base64.b64encode(file_path.read_bytes()).decode()

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f'{agent_url}/upload',
                json={
                    'filename': file_record['name'],
                    'content': content,
                },
            )

            if response.status_code != 200:
                raise HTTPException(500, f'Upload failed: {response.text}')

            result = response.json()
            result['skipped'] = False
            return result

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(500, f'Upload failed: {e}')


@router.post('/{device_id}/deploy-by-name')
async def deploy_model_by_name(device_id: str, request: dict):
    """Deploy a model to device by matching filename.

    Finds model on server by name pattern and uploads to device.
    """
    import base64

    model_name = request.get('model_name')
    if not model_name:
        raise HTTPException(400, 'model_name required')

    # Find model on server by name (exact or partial match)
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    # Search files by name
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM files WHERE name = ? AND type = 'model'",
            (model_name,),
        )
        file_record = await cursor.fetchone()

        # Try partial match if exact not found
        if not file_record:
            cursor = await db.execute(
                "SELECT * FROM files WHERE name LIKE ? AND type = 'model'",
                (f'%{model_name}%',),
            )
            file_record = await cursor.fetchone()

    if not file_record:
        raise HTTPException(
            404,
            f'Model "{model_name}" not found on server. '
            f'Upload it first via /models page.',
        )

    file_path = Path(file_record['path'])
    if not file_path.exists():
        raise HTTPException(404, 'Model file not found on disk')

    # Upload to device
    content = base64.b64encode(file_path.read_bytes()).decode()
    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=120) as client:
            response = await client.post(
                f'{agent_url}/upload',
                json={
                    'filename': file_record['name'],
                    'content': content,
                },
            )

            if response.status_code != 200:
                raise HTTPException(500, f'Deploy failed: {response.text}')

            return {
                'status': 'deployed',
                'model_name': file_record['name'],
                'device_path': response.json().get('path'),
                'size_bytes': file_record['size_bytes'],
            }

    except httpx.TimeoutException:
        raise HTTPException(504, 'Deploy timed out')
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(503, f'Cannot connect to device: {e}')


@router.post('/{device_id}/benchmark/full')
async def proxy_benchmark_full(device_id: str, request: dict):
    """Proxy single-model benchmark request to device agent."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=300) as client:
            response = await client.post(
                f'{agent_url}/benchmark/full',
                json=request,
            )
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(504, 'Benchmark timed out (5 min limit)')
    except Exception as e:
        raise HTTPException(503, f'Cannot connect to device: {e}')


@router.post('/{device_id}/benchmark/batch')
async def proxy_benchmark_batch(device_id: str, request: dict):
    """Proxy batch benchmark request to device agent."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()

    if not device:
        raise HTTPException(404, 'Device not found')

    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=600) as client:
            response = await client.post(
                f'{agent_url}/benchmark/batch',
                json=request,
            )
            return response.json()
    except httpx.TimeoutException:
        raise HTTPException(504, 'Batch benchmark timed out (10 min limit)')
    except Exception as e:
        raise HTTPException(503, f'Cannot connect to device: {e}')
