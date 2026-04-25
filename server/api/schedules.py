"""
Schedules API Endpoints — nightly benchmark automation.
"""

from datetime import UTC, datetime
import json
import uuid

from apscheduler.triggers.cron import CronTrigger
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from server.core.scheduler import run_scheduled_experiment, scheduler
from server.db.database import get_db

router = APIRouter()


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class ScheduleCreate(BaseModel):
    name: str = Field(..., min_length=1, max_length=200)
    device_id: str
    model_name: str = Field(..., min_length=1)
    backend: str = 'edgetpu'
    cron: str = Field(..., description='Cron expression, e.g. "0 2 * * *"')
    params: dict | None = None


class ScheduleUpdate(BaseModel):
    name: str | None = None
    cron: str | None = None
    enabled: bool | None = None
    params: dict | None = None
    backend: str | None = None
    model_name: str | None = None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _validate_cron(cron: str) -> None:
    """Raise HTTPException if the cron string is invalid."""
    try:
        CronTrigger.from_crontab(cron, timezone='UTC')
    except Exception as exc:
        raise HTTPException(422, f'Invalid cron expression: {exc}')


def _schedule_row_to_dict(row) -> dict:
    d = dict(row)
    if d.get('params'):
        try:
            d['params'] = json.loads(d['params'])
        except (json.JSONDecodeError, TypeError):
            d['params'] = {}
    else:
        d['params'] = {}
    return d


def _next_run(cron: str) -> str | None:
    """Return ISO timestamp of the next fire time for a cron expression."""
    try:
        trigger = CronTrigger.from_crontab(cron, timezone='UTC')
        next_fire = trigger.get_next_fire_time(None, datetime.now(UTC))
        return next_fire.isoformat() if next_fire else None
    except Exception:
        return None


def _human_cron(cron: str) -> str:
    """Return a human-readable description for common cron patterns."""
    mapping = {
        '0 2 * * *': 'Every day at 02:00 UTC',
        '0 * * * *': 'Every hour',
        '0 */6 * * *': 'Every 6 hours',
        '0 */12 * * *': 'Every 12 hours',
        '0 0 * * *': 'Every day at 00:00 UTC',
        '0 0 * * 0': 'Every Sunday at 00:00 UTC',
        '*/30 * * * *': 'Every 30 minutes',
        '*/15 * * * *': 'Every 15 minutes',
    }
    return mapping.get(cron, cron)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@router.get('')
async def list_schedules():
    """List all schedules."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT s.*, d.name as device_name
               FROM schedules s
               LEFT JOIN devices d ON s.device_id = d.id
               ORDER BY s.created_at DESC"""
        )
        rows = await cursor.fetchall()

    result = []
    for row in rows:
        d = _schedule_row_to_dict(row)
        d['next_run'] = _next_run(d['cron'])
        d['cron_human'] = _human_cron(d['cron'])
        result.append(d)
    return result


@router.post('', status_code=201)
async def create_schedule(body: ScheduleCreate):
    """Create a new schedule."""
    _validate_cron(body.cron)

    # Verify device exists (allow non-existent devices for offline setup)
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT id FROM devices WHERE id = ?', (body.device_id,)
        )
        if not await cursor.fetchone():
            raise HTTPException(404, 'Device not found')

    schedule_id = f'sched_{uuid.uuid4().hex[:12]}'
    now = datetime.now(UTC).isoformat()

    async with get_db() as db:
        await db.execute(
            """INSERT INTO schedules
               (id, name, device_id, model_name, backend, cron, enabled, params, created_at)
               VALUES (?, ?, ?, ?, ?, ?, 1, ?, ?)""",
            (
                schedule_id,
                body.name,
                body.device_id,
                body.model_name,
                body.backend,
                body.cron,
                json.dumps(body.params) if body.params else None,
                now,
            ),
        )
        await db.commit()

        cursor = await db.execute('SELECT * FROM schedules WHERE id = ?', (schedule_id,))
        row = await cursor.fetchone()

    # Register in APScheduler
    scheduler.add_job(
        run_scheduled_experiment,
        CronTrigger.from_crontab(body.cron, timezone='UTC'),
        args=[schedule_id],
        id=schedule_id,
        replace_existing=True,
    )

    d = _schedule_row_to_dict(row)
    d['next_run'] = _next_run(body.cron)
    d['cron_human'] = _human_cron(body.cron)
    return d


@router.get('/{schedule_id}')
async def get_schedule(schedule_id: str):
    """Get schedule details."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT s.*, d.name as device_name
               FROM schedules s
               LEFT JOIN devices d ON s.device_id = d.id
               WHERE s.id = ?""",
            (schedule_id,),
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'Schedule not found')

    d = _schedule_row_to_dict(row)
    d['next_run'] = _next_run(d['cron'])
    d['cron_human'] = _human_cron(d['cron'])
    return d


@router.patch('/{schedule_id}')
async def update_schedule(schedule_id: str, body: ScheduleUpdate):
    """Update schedule (cron, enabled flag, params, etc.)."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM schedules WHERE id = ?', (schedule_id,)
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'Schedule not found')

    updates: dict = {}
    if body.name is not None:
        updates['name'] = body.name
    if body.cron is not None:
        _validate_cron(body.cron)
        updates['cron'] = body.cron
    if body.enabled is not None:
        updates['enabled'] = 1 if body.enabled else 0
    if body.params is not None:
        updates['params'] = json.dumps(body.params)
    if body.backend is not None:
        updates['backend'] = body.backend
    if body.model_name is not None:
        updates['model_name'] = body.model_name

    if updates:
        set_clause = ', '.join(f'{k} = ?' for k in updates)
        async with get_db() as db:
            await db.execute(
                f'UPDATE schedules SET {set_clause} WHERE id = ?',
                [*updates.values(), schedule_id],
            )
            await db.commit()

    # Refresh row
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM schedules WHERE id = ?', (schedule_id,))
        row = await cursor.fetchone()

    schedule = dict(row)

    # Sync APScheduler
    if schedule['enabled']:
        scheduler.add_job(
            run_scheduled_experiment,
            CronTrigger.from_crontab(schedule['cron'], timezone='UTC'),
            args=[schedule_id],
            id=schedule_id,
            replace_existing=True,
        )
    else:
        try:
            scheduler.remove_job(schedule_id)
        except Exception:
            pass

    d = _schedule_row_to_dict(row)
    d['next_run'] = _next_run(schedule['cron'])
    d['cron_human'] = _human_cron(schedule['cron'])
    return d


@router.delete('/{schedule_id}')
async def delete_schedule(schedule_id: str):
    """Delete a schedule."""
    async with get_db() as db:
        cursor = await db.execute(
            'DELETE FROM schedules WHERE id = ?', (schedule_id,)
        )
        if cursor.rowcount == 0:
            raise HTTPException(404, 'Schedule not found')
        await db.commit()

    # Remove from APScheduler
    try:
        scheduler.remove_job(schedule_id)
    except Exception:
        pass

    return {'status': 'deleted'}


@router.post('/{schedule_id}/run-now')
async def run_now(schedule_id: str):
    """Trigger a scheduled benchmark immediately (for testing)."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT id FROM schedules WHERE id = ?', (schedule_id,)
        )
        if not await cursor.fetchone():
            raise HTTPException(404, 'Schedule not found')

    import asyncio

    asyncio.create_task(run_scheduled_experiment(schedule_id))
    return {'status': 'triggered', 'schedule_id': schedule_id}


@router.get('/{schedule_id}/history')
async def get_history(schedule_id: str, limit: int = 20):
    """Return the last N experiments launched by this schedule."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT id FROM schedules WHERE id = ?', (schedule_id,)
        )
        if not await cursor.fetchone():
            raise HTTPException(404, 'Schedule not found')

        cursor = await db.execute(
            """SELECT e.id, e.name, e.status, e.created_at, e.completed_at,
                      e.error_message, r.metrics
               FROM experiments e
               LEFT JOIN results r ON e.id = r.experiment_id
               WHERE e.name LIKE '[Scheduled] %'
                 AND e.name LIKE ?
               ORDER BY e.created_at DESC
               LIMIT ?""",
            ('[Scheduled] %',  limit),
        )
        rows = await cursor.fetchall()

    history = []
    for row in rows:
        d = dict(row)
        if d.get('metrics'):
            try:
                d['metrics'] = json.loads(d['metrics'])
            except (json.JSONDecodeError, TypeError):
                d['metrics'] = {}
        history.append(d)
    return history
