"""
Experiments API Endpoints
"""

from datetime import datetime
import json
from pathlib import Path
import uuid

from fastapi import APIRouter, HTTPException, Query

from server.core.models import (
    ExperimentBatchCreate,
    ExperimentCreate,
    ExperimentResponse,
    ExperimentStatus,
)
from server.core.queue import task_queue
from server.db.database import get_db

router = APIRouter()


@router.get('', response_model=list[ExperimentResponse])
async def list_experiments(
    status: ExperimentStatus | None = None,
    device_id: str | None = None,
    limit: int = Query(100, le=1000),
    offset: int = 0,
):
    """Get all experiments with optional filtering."""
    query = 'SELECT * FROM experiments WHERE 1=1'
    params = []

    if status:
        query += ' AND status = ?'
        params.append(status.value)

    if device_id:
        query += ' AND device_id = ?'
        params.append(device_id)

    query += ' ORDER BY created_at DESC LIMIT ? OFFSET ?'
    params.extend([limit, offset])

    async with get_db() as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

    result = []
    for row in rows:
        exp = dict(row)
        exp['params'] = json.loads(exp['params'])
        result.append(exp)

    return result


@router.post('', response_model=ExperimentResponse)
async def create_experiment(experiment: ExperimentCreate):
    """Create a new experiment and add to queue."""
    experiment_id = (
        f'exp_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:4]}'
    )

    # Verify device exists
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT id FROM devices WHERE id = ?', (experiment.device_id,)
        )
        if not await cursor.fetchone():
            raise HTTPException(404, 'Device not found')

    # Extract model name from path
    model_name = Path(experiment.model_path).name

    async with get_db() as db:
        await db.execute(
            """INSERT INTO experiments
            (id, name, device_id, model_name, model_path, script_path, params, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                experiment.name,
                experiment.device_id,
                model_name,
                experiment.model_path,
                experiment.script_path,
                json.dumps(experiment.params.model_dump()),
                ExperimentStatus.QUEUED.value,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()

        cursor = await db.execute(
            'SELECT * FROM experiments WHERE id = ?', (experiment_id,)
        )
        row = await cursor.fetchone()

    # Add to queue
    await task_queue.add_task(experiment_id)

    exp = dict(row)
    exp['params'] = json.loads(exp['params'])
    return exp


@router.post('/batch')
async def create_batch_experiments(batch: ExperimentBatchCreate):
    """Create multiple experiments at once."""
    # Find device by name or id
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT id FROM devices WHERE id = ? OR name = ?',
            (batch.device, batch.device),
        )
        device = await cursor.fetchone()

        if not device:
            raise HTTPException(404, 'Device not found')

        device_id = device['id']

    created_experiments = []

    for model in batch.models:
        for backend in batch.backends:
            experiment_id = f'exp_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:4]}'
            model_name = Path(model).name

            params = batch.params.model_dump()
            params['backend'] = backend.value

            async with get_db() as db:
                await db.execute(
                    """INSERT INTO experiments
                    (id, name, device_id, model_name, model_path, script_path, params, status, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        experiment_id,
                        f'{model_name}_{backend.value}',
                        device_id,
                        model_name,
                        model,
                        'benchmark_tflite.py',
                        json.dumps(params),
                        ExperimentStatus.QUEUED.value,
                        datetime.utcnow().isoformat(),
                    ),
                )
                await db.commit()

            await task_queue.add_task(experiment_id)
            created_experiments.append(experiment_id)

    return {
        'created': len(created_experiments),
        'experiment_ids': created_experiments,
    }


# Queue management endpoints (must be before /{experiment_id})
@router.get('/queue/status')
async def get_queue_status():
    """Get current queue status."""
    status = task_queue.get_queue_status()

    # Add pending experiments count from DB
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT status, COUNT(*) as count FROM experiments
               WHERE status IN (?, ?, ?)
               GROUP BY status""",
            (
                ExperimentStatus.QUEUED.value,
                ExperimentStatus.RUNNING.value,
                ExperimentStatus.FAILED.value,
            ),
        )
        rows = await cursor.fetchall()

    status['db_counts'] = {row['status']: row['count'] for row in rows}
    return status


@router.post('/retry-all-failed')
async def retry_all_failed():
    """Retry all failed experiments."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT id FROM experiments
               WHERE status = ?
               ORDER BY created_at ASC""",
            (ExperimentStatus.FAILED.value,),
        )
        failed = await cursor.fetchall()

        if not failed:
            return {'retried': 0, 'message': 'No failed experiments to retry'}

        # Reset all to queued
        await db.execute(
            """UPDATE experiments
               SET status = ?, error_message = NULL, completed_at = NULL
               WHERE status = ?""",
            (ExperimentStatus.QUEUED.value, ExperimentStatus.FAILED.value),
        )
        await db.commit()

    # Add all to queue
    count = 0
    for row in failed:
        await task_queue.add_task(row['id'])
        count += 1

    return {'retried': count, 'message': f'{count} experiments added to queue'}


@router.post('/batch-delete')
async def batch_delete_experiments(request: dict):
    """Delete multiple experiments and their results at once.

    Request: {ids: [exp_id_1, exp_id_2, ...]}
    Optional: {ids: [...], filter_status: "failed"} to delete all with given status.
    """
    ids = request.get('ids', [])
    filter_status = request.get('filter_status')

    if not ids and not filter_status:
        raise HTTPException(400, 'ids or filter_status required')

    async with get_db() as db:
        if filter_status and not ids:
            # Delete all with given status
            cursor = await db.execute(
                'SELECT id FROM experiments WHERE status = ?',
                (filter_status,),
            )
            rows = await cursor.fetchall()
            ids = [row['id'] for row in rows]

        if not ids:
            return {'deleted': 0}

        placeholders = ','.join(['?' for _ in ids])

        # Delete results first
        await db.execute(
            f'DELETE FROM results WHERE experiment_id IN ({placeholders})',
            ids,
        )

        # Delete experiments
        cursor = await db.execute(
            f'DELETE FROM experiments WHERE id IN ({placeholders})',
            ids,
        )

        deleted = cursor.rowcount
        await db.commit()

    return {'deleted': deleted, 'ids': ids}


# Experiment-specific endpoints
@router.get('/{experiment_id}', response_model=ExperimentResponse)
async def get_experiment(experiment_id: str):
    """Get experiment details."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM experiments WHERE id = ?', (experiment_id,)
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'Experiment not found')

    exp = dict(row)
    exp['params'] = json.loads(exp['params'])
    return exp


@router.get('/{experiment_id}/logs')
async def get_experiment_logs(experiment_id: str):
    """Get experiment logs."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT logs, error_message FROM experiments WHERE id = ?', (experiment_id,)
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'Experiment not found')

    return {
        'logs': row['logs'] or '',
        'error': row['error_message'],
    }


@router.post('/{experiment_id}/rerun')
async def rerun_experiment(experiment_id: str):
    """Rerun an experiment with the same parameters."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM experiments WHERE id = ?', (experiment_id,)
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'Experiment not found')

    # Create new experiment with same params
    new_id = f'exp_{datetime.utcnow().strftime("%Y%m%d_%H%M%S")}_{uuid.uuid4().hex[:4]}'

    async with get_db() as db:
        await db.execute(
            """INSERT INTO experiments
            (id, name, device_id, model_name, model_path, script_path, params, status, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                new_id,
                f'{row["name"]}_rerun',
                row['device_id'],
                row['model_name'],
                row['model_path'],
                row['script_path'],
                row['params'],
                ExperimentStatus.QUEUED.value,
                datetime.utcnow().isoformat(),
            ),
        )
        await db.commit()

    await task_queue.add_task(new_id)

    return {'new_experiment_id': new_id}


@router.post('/{experiment_id}/cancel')
async def cancel_experiment(experiment_id: str):
    """Cancel a queued experiment."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT status FROM experiments WHERE id = ?', (experiment_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(404, 'Experiment not found')

        if row['status'] not in (
            ExperimentStatus.QUEUED.value,
            ExperimentStatus.RUNNING.value,
        ):
            raise HTTPException(400, 'Cannot cancel completed experiment')

        await db.execute(
            'UPDATE experiments SET status = ? WHERE id = ?',
            (ExperimentStatus.CANCELLED.value, experiment_id),
        )
        await db.commit()

    return {'status': 'cancelled'}


@router.post('/{experiment_id}/retry')
async def retry_experiment(experiment_id: str):
    """Retry a failed experiment."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT status FROM experiments WHERE id = ?', (experiment_id,)
        )
        row = await cursor.fetchone()

        if not row:
            raise HTTPException(404, 'Experiment not found')

        if row['status'] not in (ExperimentStatus.FAILED.value,):
            raise HTTPException(400, 'Can only retry failed experiments')

        # Reset to queued
        await db.execute(
            """UPDATE experiments
               SET status = ?, error_message = NULL, completed_at = NULL
               WHERE id = ?""",
            (ExperimentStatus.QUEUED.value, experiment_id),
        )
        await db.commit()

    # Add to queue
    await task_queue.add_task(experiment_id, priority=True)

    return {'status': 'queued', 'message': 'Experiment added to queue for retry'}


@router.post('/{experiment_id}/reassign')
async def reassign_experiment(experiment_id: str, request: dict):
    """Reassign a failed experiment to a different device."""
    new_device_id = request.get('device_id')
    if not new_device_id:
        raise HTTPException(400, 'device_id is required')

    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM experiments WHERE id = ?', (experiment_id,)
        )
        exp = await cursor.fetchone()
        if not exp:
            raise HTTPException(404, 'Experiment not found')

        if exp['status'] != ExperimentStatus.FAILED.value:
            raise HTTPException(400, 'Can only reassign failed experiments')

        # Verify the new device exists
        cursor = await db.execute(
            'SELECT id, name FROM devices WHERE id = ?', (new_device_id,)
        )
        device = await cursor.fetchone()
        if not device:
            raise HTTPException(404, 'Device not found')

        await db.execute(
            'UPDATE experiments SET device_id = ? WHERE id = ?',
            (new_device_id, experiment_id),
        )
        await db.commit()

    return {
        'status': 'reassigned',
        'experiment_id': experiment_id,
        'new_device_id': new_device_id,
        'new_device_name': device['name'],
    }


@router.delete('/{experiment_id}')
async def delete_experiment(experiment_id: str):
    """Delete an experiment and its results."""
    async with get_db() as db:
        # Delete results first
        await db.execute(
            'DELETE FROM results WHERE experiment_id = ?', (experiment_id,)
        )

        # Delete experiment
        cursor = await db.execute(
            'DELETE FROM experiments WHERE id = ?', (experiment_id,)
        )

        if cursor.rowcount == 0:
            raise HTTPException(404, 'Experiment not found')

        await db.commit()

    return {'status': 'deleted'}
