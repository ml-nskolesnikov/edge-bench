"""
Files API Endpoints
"""

from datetime import datetime
import hashlib
from pathlib import Path
import uuid

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, PlainTextResponse

from server.core.config import settings
from server.core.models import FileResponse as FileResponseModel, FileType
from server.db.database import get_db

router = APIRouter()

# Agent source files (served for installation)
AGENT_DIR = Path(__file__).parent.parent.parent / 'agent'


@router.get('', response_model=list[FileResponseModel])
async def list_files(file_type: FileType = None):
    """List all uploaded files."""
    query = 'SELECT * FROM files'
    params = []

    if file_type:
        query += ' WHERE type = ?'
        params.append(file_type.value)

    query += ' ORDER BY created_at DESC'

    async with get_db() as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

    return [dict(row) for row in rows]


@router.post('/upload')
async def upload_file(
    file: UploadFile = File(...),
    file_type: FileType = FileType.MODEL,
):
    """Upload a model or script file."""
    # Read content first to calculate hash
    content = await file.read()
    file_hash = hashlib.sha256(content).hexdigest()

    # Check for duplicate by hash
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT id, name, path FROM files WHERE hash = ?', (file_hash,)
        )
        existing = await cursor.fetchone()

        if existing:
            return {
                'id': existing['id'],
                'name': existing['name'],
                'path': existing['path'],
                'type': file_type.value,
                'size_bytes': len(content),
                'hash': file_hash,
                'duplicate': True,
                'message': f'File already exists as "{existing["name"]}"',
            }

        # Check for same name (different content)
        cursor = await db.execute(
            'SELECT id FROM files WHERE name = ? AND type = ?',
            (file.filename, file_type.value),
        )
        same_name = await cursor.fetchone()

    # Determine destination directory
    if file_type == FileType.MODEL:
        dest_dir = settings.MODELS_DIR
    elif file_type == FileType.SCRIPT:
        dest_dir = settings.SCRIPTS_DIR
    else:
        dest_dir = settings.UPLOAD_DIR

    dest_dir.mkdir(parents=True, exist_ok=True)

    # Generate unique filename
    file_id = f'file_{uuid.uuid4().hex[:8]}'
    file_path = dest_dir / file.filename

    # If same name exists with different content, add suffix
    if same_name:
        base = file_path.stem
        suffix = file_path.suffix
        counter = 1
        while file_path.exists():
            file_path = dest_dir / f'{base}_{counter}{suffix}'
            counter += 1

    # Save file to disk
    with open(file_path, 'wb') as f:
        f.write(content)

    # Save to database
    async with get_db() as db:
        await db.execute(
            """INSERT INTO files (id, name, type, path, size_bytes, hash, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                file_id,
                file_path.name,
                file_type.value,
                str(file_path),
                len(content),
                file_hash,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()

    return {
        'id': file_id,
        'name': file_path.name,
        'type': file_type,
        'size_bytes': len(content),
        'hash': file_hash,
        'duplicate': False,
    }


@router.get('/check-duplicate')
async def check_duplicate(file_hash: str):
    """Check if a file with given hash already exists."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT id, name, path, type FROM files WHERE hash = ?', (file_hash,)
        )
        existing = await cursor.fetchone()

    if existing:
        return {
            'exists': True,
            'file': dict(existing),
        }

    return {'exists': False}


@router.get('/download/{file_id}')
async def download_file(file_id: str):
    """Download a file by ID."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM files WHERE id = ?', (file_id,))
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'File not found')

    file_path = Path(row['path'])
    if not file_path.exists():
        raise HTTPException(404, 'File not found on disk')

    return FileResponse(
        file_path,
        filename=row['name'],
        media_type='application/octet-stream',
    )


@router.get('/{file_id}/download')
async def download_file_alt(file_id: str):
    """Download a file by ID (alternative URL pattern)."""
    return await download_file(file_id)


@router.delete('/{file_id}')
async def delete_file(file_id: str):
    """Delete a file."""
    async with get_db() as db:
        cursor = await db.execute('SELECT path FROM files WHERE id = ?', (file_id,))
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(404, 'File not found')

        # Delete from disk
        file_path = Path(row['path'])
        if file_path.exists():
            file_path.unlink()

        # Delete from database
        await db.execute('DELETE FROM files WHERE id = ?', (file_id,))
        await db.commit()

    return {'status': 'deleted'}


# ---- Model conversion endpoints ----

# In-memory conversion task registry (task_id -> asyncio.Task)
_conversion_tasks: dict[str, object] = {}


@router.post('/{file_id}/convert')
async def start_conversion(file_id: str, request: dict):
    """Start async model conversion to Edge TPU TFLite.

    Request: {"target": "edgetpu"|"tflite"|"onnx", "input_shape": [1,224,224,3]}
    Response: {"task_id": "...", "status": "pending"}
    """
    import asyncio
    import uuid as _uuid

    target = request.get('target', 'edgetpu')
    input_shape = request.get('input_shape', [1, 224, 224, 3])
    rpi_host = request.get('rpi_host')

    if target not in ('edgetpu', 'tflite', 'onnx'):
        raise HTTPException(400, 'target must be edgetpu, tflite, or onnx')

    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM files WHERE id = ?', (file_id,))
        file_record = await cursor.fetchone()

    if not file_record:
        raise HTTPException(404, 'File not found')

    file_path = Path(file_record['path'])
    if not file_path.exists():
        raise HTTPException(404, 'File not found on disk')

    task_id = f'conv_{_uuid.uuid4().hex[:8]}'

    async with get_db() as db:
        await db.execute(
            """INSERT INTO convert_tasks (id, file_id, target, status, input_shape, created_at)
               VALUES (?, ?, ?, 'pending', ?, ?)""",
            (task_id, file_id, target, str(input_shape), datetime.utcnow().isoformat()),
        )
        await db.commit()

    # Run conversion in background
    asyncio.create_task(
        _run_conversion(task_id, file_id, file_path, input_shape, target, rpi_host)
    )

    return {'task_id': task_id, 'status': 'pending'}


@router.get('/{file_id}/convert/status')
async def get_conversion_status(file_id: str):
    """Get status of the latest conversion task for this file."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT * FROM convert_tasks WHERE file_id = ?
               ORDER BY created_at DESC LIMIT 1""",
            (file_id,),
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'No conversion task found for this file')

    return dict(row)


async def _run_conversion(
    task_id: str,
    file_id: str,
    file_path: Path,
    input_shape: list[int],
    target: str,
    rpi_host: str | None,
) -> None:
    """Background coroutine: run pipeline and update convert_tasks table."""
    import asyncio
    import hashlib as _hashlib
    import uuid as _uuid

    async def _set_status(status: str, error: str | None = None):
        async with get_db() as db:
            await db.execute(
                """UPDATE convert_tasks SET status = ?, error_message = ?,
                   completed_at = ? WHERE id = ?""",
                (status, error, datetime.utcnow().isoformat(), task_id),
            )
            await db.commit()

    await _set_status('running')

    try:
        # Run blocking pipeline in thread pool
        loop = asyncio.get_event_loop()
        scripts_dir = Path(__file__).parent.parent.parent / 'scripts'

        def _pipeline():
            import sys

            sys.path.insert(0, str(scripts_dir))
            from convert_pipeline import run_pipeline

            output_dir = file_path.parent / 'converted'
            return run_pipeline(
                input_path=file_path,
                output_dir=output_dir,
                input_shape=input_shape,
                target=target,
                rpi_host=rpi_host,
            )

        result = await loop.run_in_executor(None, _pipeline)
        output_path = Path(result['final_output'])

        # Register the converted file in the DB
        content = output_path.read_bytes()
        file_hash = _hashlib.sha256(content).hexdigest()
        out_file_id = f'file_{_uuid.uuid4().hex[:8]}'

        async with get_db() as db:
            await db.execute(
                """INSERT OR IGNORE INTO files (id, name, type, path, size_bytes, hash, created_at)
                   VALUES (?, ?, 'model', ?, ?, ?, ?)""",
                (
                    out_file_id,
                    output_path.name,
                    str(output_path),
                    len(content),
                    file_hash,
                    datetime.utcnow().isoformat(),
                ),
            )
            await db.execute(
                'UPDATE convert_tasks SET output_file_id = ? WHERE id = ?',
                (out_file_id, task_id),
            )
            await db.commit()

        await _set_status('completed')

    except Exception as e:
        await _set_status('failed', str(e))


# Agent installation files
@router.get('/agent/{filename}', response_class=PlainTextResponse)
async def get_agent_file(filename: str):
    """Serve agent source files for installation."""
    # Security: only allow specific files
    allowed_files = {
        'main.py',
        'executor.py',
        'metrics.py',
        'config.py',
        'result_cache.py',
        'requirements.txt',
        'install.sh',
        'benchmark_full.py',
        'benchmark_batch.py',
        'benchmark_tflite.py',
        'benchmark_eccv_models.py',
    }

    if filename not in allowed_files:
        raise HTTPException(404, 'File not found')

    file_path = AGENT_DIR / filename

    if not file_path.exists():
        raise HTTPException(404, 'File not found')

    return file_path.read_text()
