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
