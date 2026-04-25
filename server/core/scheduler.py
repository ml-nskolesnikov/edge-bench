"""
Nightly benchmark scheduler.

Uses APScheduler (AsyncIOScheduler) embedded in the FastAPI lifespan.
Schedules are stored in SQLite and survive server restarts.
"""

from datetime import UTC, datetime
import json
import logging
import uuid

from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.cron import CronTrigger

from server.db.database import get_db

logger = logging.getLogger(__name__)

scheduler = AsyncIOScheduler(timezone='UTC')


async def run_scheduled_experiment(schedule_id: str) -> None:
    """Execute the experiment defined by a schedule entry."""
    from server.core.models import ExperimentStatus
    from server.core.queue import task_queue

    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM schedules WHERE id = ?', (schedule_id,)
        )
        row = await cursor.fetchone()

    if not row:
        logger.warning(f'[Scheduler] Schedule {schedule_id} not found, skipping')
        return

    schedule = dict(row)

    if not schedule['enabled']:
        logger.info(f'[Scheduler] Schedule {schedule_id} is disabled, skipping')
        return

    # Resolve device
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM devices WHERE id = ?', (schedule['device_id'],)
        )
        device = await cursor.fetchone()

    if not device:
        logger.error(
            f'[Scheduler] Device {schedule["device_id"]} not found for schedule {schedule_id}'
        )
        return

    # Build experiment params
    extra_params: dict = {}
    if schedule.get('params'):
        try:
            extra_params = json.loads(schedule['params'])
        except (json.JSONDecodeError, TypeError):
            extra_params = {}

    params = {
        'backend': schedule.get('backend', 'cpu'),
        'num_threads': extra_params.get('num_threads', 4),
        'warmup_runs': extra_params.get('warmup_runs', 10),
        'benchmark_runs': extra_params.get('benchmark_runs', 100),
        'batch_size': extra_params.get('batch_size', 1),
        'timeout_seconds': extra_params.get('timeout_seconds', 600),
        'tpu_index': extra_params.get('tpu_index', 0),
    }

    model_name = schedule['model_name']
    experiment_id = (
        f'exp_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:4]}'
    )

    async with get_db() as db:
        await db.execute(
            """INSERT INTO experiments
            (id, name, device_id, model_name, model_path, script_path, params, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                f'[Scheduled] {schedule["name"]}',
                schedule['device_id'],
                model_name,
                model_name,
                'benchmark_tflite.py',
                json.dumps(params),
                ExperimentStatus.QUEUED.value,
                datetime.now(UTC).isoformat(),
            ),
        )

        # Update schedule's last_run_at and last_exp_id
        await db.execute(
            """UPDATE schedules
               SET last_run_at = ?, last_exp_id = ?
               WHERE id = ?""",
            (datetime.now(UTC).isoformat(), experiment_id, schedule_id),
        )
        await db.commit()

    await task_queue.add_task(experiment_id)
    logger.info(
        f'[Scheduler] Schedule {schedule_id} launched experiment {experiment_id}'
    )


async def restore_schedules() -> int:
    """Re-register all enabled schedules from DB into APScheduler after server restart."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM schedules WHERE enabled=1')
        rows = await cursor.fetchall()

    count = 0
    for row in rows:
        try:
            scheduler.add_job(
                run_scheduled_experiment,
                CronTrigger.from_crontab(row['cron'], timezone='UTC'),
                args=[row['id']],
                id=row['id'],
                replace_existing=True,
            )
            count += 1
            logger.info(f'[Scheduler] Restored schedule {row["id"]} ({row["name"]})')
        except Exception as e:
            logger.error(f'[Scheduler] Failed to restore schedule {row["id"]}: {e}')

    if count:
        logger.info(f'[Scheduler] Restored {count} schedules from database')
    return count
