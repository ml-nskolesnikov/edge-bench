"""
Settings API Endpoints
"""

from fastapi import APIRouter

from server.db.database import get_db

router = APIRouter()


@router.get('')
async def get_settings():
    """Get all settings."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM settings')
        rows = await cursor.fetchall()

    return {row['key']: row['value'] for row in rows}


@router.put('')
async def update_settings(request: dict):
    """Update settings."""
    async with get_db() as db:
        for key, value in request.items():
            await db.execute(
                """INSERT INTO settings (key, value) VALUES (?, ?)
                ON CONFLICT(key) DO UPDATE SET value = ?""",
                (key, str(value), str(value)),
            )
        await db.commit()

    return {'status': 'updated'}


@router.get('/{key}')
async def get_setting(key: str):
    """Get a specific setting."""
    async with get_db() as db:
        cursor = await db.execute('SELECT value FROM settings WHERE key = ?', (key,))
        row = await cursor.fetchone()

    if row:
        return {'key': key, 'value': row['value']}
    return {'key': key, 'value': None}


@router.put('/{key}')
async def set_setting(key: str, request: dict):
    """Set a specific setting."""
    value = request.get('value', '')

    async with get_db() as db:
        await db.execute(
            """INSERT INTO settings (key, value) VALUES (?, ?)
            ON CONFLICT(key) DO UPDATE SET value = ?""",
            (key, str(value), str(value)),
        )
        await db.commit()

    return {'key': key, 'value': value}
