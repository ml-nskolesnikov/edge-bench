"""
Dependencies API Endpoints
Manage and check dependencies on Edge devices
"""

from datetime import datetime
import uuid

from fastapi import APIRouter, HTTPException
import httpx

from server.db.database import get_db

router = APIRouter()


@router.get('')
async def list_dependencies():
    """Get all dependencies."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM dependencies ORDER BY is_required DESC, name'
        )
        rows = await cursor.fetchall()

    return [dict(row) for row in rows]


@router.post('')
async def create_dependency(request: dict):
    """Create a new dependency."""
    name = request.get('name')
    package = request.get('package', name)
    check_command = request.get('check_command')
    install_command = request.get('install_command')

    if not name:
        raise HTTPException(400, 'name is required')

    dep_id = f'dep_{uuid.uuid4().hex[:8]}'

    async with get_db() as db:
        # Check if exists
        cursor = await db.execute(
            'SELECT id FROM dependencies WHERE name = ?',
            (name,),
        )
        if await cursor.fetchone():
            raise HTTPException(400, f'Dependency "{name}" already exists')

        await db.execute(
            """INSERT INTO dependencies
               (id, name, package, version, check_command, install_command,
                is_required, description, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                dep_id,
                name,
                package,
                request.get('version'),
                check_command,
                install_command,
                1 if request.get('is_required', True) else 0,
                request.get('description'),
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()

        cursor = await db.execute('SELECT * FROM dependencies WHERE id = ?', (dep_id,))
        row = await cursor.fetchone()

    return dict(row)


@router.put('/{dep_id}')
async def update_dependency(dep_id: str, request: dict):
    """Update a dependency."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM dependencies WHERE id = ?', (dep_id,))
        if not await cursor.fetchone():
            raise HTTPException(404, 'Dependency not found')

        updates = []
        params = []

        for field in [
            'name',
            'package',
            'version',
            'check_command',
            'install_command',
            'description',
        ]:
            if field in request:
                updates.append(f'{field} = ?')
                params.append(request[field])

        if 'is_required' in request:
            updates.append('is_required = ?')
            params.append(1 if request['is_required'] else 0)

        if updates:
            updates.append('updated_at = ?')
            params.append(datetime.utcnow().isoformat())
            params.append(dep_id)

            await db.execute(
                f'UPDATE dependencies SET {", ".join(updates)} WHERE id = ?',
                params,
            )
            await db.commit()

        cursor = await db.execute('SELECT * FROM dependencies WHERE id = ?', (dep_id,))
        row = await cursor.fetchone()

    return dict(row)


@router.delete('/{dep_id}')
async def delete_dependency(dep_id: str):
    """Delete a dependency."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM dependencies WHERE id = ?', (dep_id,))
        if not await cursor.fetchone():
            raise HTTPException(404, 'Dependency not found')

        # Also delete device_dependencies records
        await db.execute(
            'DELETE FROM device_dependencies WHERE dependency_id = ?', (dep_id,)
        )
        await db.execute('DELETE FROM dependencies WHERE id = ?', (dep_id,))
        await db.commit()

    return {'status': 'deleted', 'id': dep_id}


@router.get('/device/{device_id}')
async def get_device_dependencies(device_id: str):
    """Get dependency status for a device."""
    async with get_db() as db:
        # Get device
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()
        if not device:
            raise HTTPException(404, 'Device not found')

        # Get all dependencies with device status
        cursor = await db.execute(
            """SELECT d.*, dd.status, dd.installed_version, dd.error_message, dd.checked_at
               FROM dependencies d
               LEFT JOIN device_dependencies dd ON d.id = dd.dependency_id AND dd.device_id = ?
               ORDER BY d.is_required DESC, d.name""",
            (device_id,),
        )
        rows = await cursor.fetchall()

    return {
        'device_id': device_id,
        'device_name': device['name'],
        'dependencies': [dict(row) for row in rows],
    }


@router.post('/device/{device_id}/check')
async def check_device_dependencies(device_id: str):
    """Check all dependencies on a device."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()
        if not device:
            raise HTTPException(404, 'Device not found')

        cursor = await db.execute('SELECT * FROM dependencies')
        dependencies = await cursor.fetchall()

    agent_url = f'http://{device["ip"]}:{device["port"]}'
    results = []

    for dep in dependencies:
        dep_dict = dict(dep)
        check_cmd = dep['check_command']

        if not check_cmd:
            dep_dict['status'] = 'unknown'
            dep_dict['error_message'] = 'No check command'
            results.append(dep_dict)
            continue

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f'{agent_url}/execute/code',
                    json={'code': check_cmd, 'timeout': 10},
                )

                if resp.status_code == 200:
                    data = resp.json()
                    if data.get('exit_code') == 0:
                        dep_dict['status'] = 'installed'
                        dep_dict['installed_version'] = data.get('stdout', '').strip()
                        dep_dict['error_message'] = None
                    else:
                        dep_dict['status'] = 'missing'
                        dep_dict['error_message'] = data.get('stderr', '')
                else:
                    dep_dict['status'] = 'error'
                    dep_dict['error_message'] = f'Agent error: {resp.status_code}'

        except Exception as e:
            dep_dict['status'] = 'error'
            dep_dict['error_message'] = str(e)

        # Save to database
        async with get_db() as db:
            await db.execute(
                """INSERT INTO device_dependencies
                   (id, device_id, dependency_id, status, installed_version, error_message, checked_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(device_id, dependency_id) DO UPDATE SET
                   status = ?, installed_version = ?, error_message = ?, checked_at = ?""",
                (
                    f'dd_{uuid.uuid4().hex[:8]}',
                    device_id,
                    dep['id'],
                    dep_dict['status'],
                    dep_dict.get('installed_version'),
                    dep_dict.get('error_message'),
                    datetime.utcnow().isoformat(),
                    dep_dict['status'],
                    dep_dict.get('installed_version'),
                    dep_dict.get('error_message'),
                    datetime.utcnow().isoformat(),
                ),
            )
            await db.commit()

        results.append(dep_dict)

    # Summary
    installed = sum(1 for r in results if r['status'] == 'installed')
    missing = sum(1 for r in results if r['status'] == 'missing')
    errors = sum(1 for r in results if r['status'] == 'error')

    return {
        'device_id': device_id,
        'summary': {
            'total': len(results),
            'installed': installed,
            'missing': missing,
            'errors': errors,
        },
        'dependencies': results,
    }


@router.post('/device/{device_id}/check/{dep_id}')
async def check_single_dependency(device_id: str, dep_id: str):
    """Check a single dependency on a device."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices WHERE id = ?', (device_id,))
        device = await cursor.fetchone()
        if not device:
            raise HTTPException(404, 'Device not found')

        cursor = await db.execute('SELECT * FROM dependencies WHERE id = ?', (dep_id,))
        dep = await cursor.fetchone()
        if not dep:
            raise HTTPException(404, 'Dependency not found')

    check_cmd = dep['check_command']
    if not check_cmd:
        return {'status': 'unknown', 'error': 'No check command'}

    agent_url = f'http://{device["ip"]}:{device["port"]}'

    try:
        async with httpx.AsyncClient(timeout=30) as client:
            resp = await client.post(
                f'{agent_url}/execute/code',
                json={'code': check_cmd, 'timeout': 10},
            )

            if resp.status_code == 200:
                data = resp.json()
                if data.get('exit_code') == 0:
                    status = 'installed'
                    version = data.get('stdout', '').strip()
                    error = None
                else:
                    status = 'missing'
                    version = None
                    error = data.get('stderr', '')
            else:
                status = 'error'
                version = None
                error = f'Agent error: {resp.status_code}'

    except Exception as e:
        status = 'error'
        version = None
        error = str(e)

    # Save
    async with get_db() as db:
        await db.execute(
            """INSERT INTO device_dependencies
               (id, device_id, dependency_id, status, installed_version, error_message, checked_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(device_id, dependency_id) DO UPDATE SET
               status = ?, installed_version = ?, error_message = ?, checked_at = ?""",
            (
                f'dd_{uuid.uuid4().hex[:8]}',
                device_id,
                dep_id,
                status,
                version,
                error,
                datetime.utcnow().isoformat(),
                status,
                version,
                error,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()

    return {
        'dependency_id': dep_id,
        'name': dep['name'],
        'status': status,
        'installed_version': version,
        'error_message': error,
    }
