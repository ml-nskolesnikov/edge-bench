"""
Results API Endpoints
"""

import csv
from datetime import UTC, datetime
import io
import json

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from server.core.models import ExperimentStatus
from server.db.database import get_db

router = APIRouter()


@router.get('')
async def list_results(
    model: str | None = None,
    backend: str | None = None,
    device_id: str | None = None,
    limit: int = Query(100, le=1000),
):
    """Get all results with optional filtering."""
    query = """
        SELECT r.*, e.name as experiment_name, e.model_name, e.model_path,
               e.params, d.name as device_name
        FROM results r
        JOIN experiments e ON r.experiment_id = e.id
        LEFT JOIN devices d ON e.device_id = d.id
        WHERE 1=1
    """
    params = []

    if model:
        query += ' AND e.model_name LIKE ?'
        params.append(f'%{model}%')

    if device_id:
        query += ' AND e.device_id = ?'
        params.append(device_id)

    query += ' ORDER BY r.created_at DESC LIMIT ?'
    params.append(limit)

    async with get_db() as db:
        cursor = await db.execute(query, params)
        rows = await cursor.fetchall()

    results = []
    for row in rows:
        result = dict(row)
        result['metrics'] = json.loads(result['metrics'])
        result['params'] = json.loads(result['params'])

        # Filter by backend if specified
        if backend and result['params'].get('backend') != backend:
            continue

        results.append(result)

    return results


@router.get('/{experiment_id}')
async def get_result(experiment_id: str):
    """Get result for specific experiment."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT r.*, e.name as experiment_name, e.model_name,
                      e.params, d.name as device_name
               FROM results r
               JOIN experiments e ON r.experiment_id = e.id
               LEFT JOIN devices d ON e.device_id = d.id
               WHERE r.experiment_id = ?""",
            (experiment_id,),
        )
        row = await cursor.fetchone()

    if not row:
        raise HTTPException(404, 'Result not found')

    result = dict(row)
    result['metrics'] = json.loads(result['metrics'])
    result['params'] = json.loads(result['params'])
    return result


@router.get('/export/csv')
async def export_csv(
    model: str | None = None,
    backend: str | None = None,
):
    """Export results to CSV."""
    query = """
        SELECT r.*, e.name as experiment_name, e.model_name, e.model_path,
               e.params, d.name as device_name
        FROM results r
        JOIN experiments e ON r.experiment_id = e.id
        LEFT JOIN devices d ON e.device_id = d.id
        ORDER BY r.created_at DESC
    """

    async with get_db() as db:
        cursor = await db.execute(query)
        rows = await cursor.fetchall()

    # Create CSV
    output = io.StringIO()
    writer = csv.writer(output)

    # Header
    header = [
        'experiment_id',
        'experiment_name',
        'model_name',
        'device_name',
        'backend',
        'batch_size',
        'num_threads',
        'benchmark_runs',
        'latency_mean_ms',
        'latency_std_ms',
        'latency_p50_ms',
        'latency_p90_ms',
        'latency_p95_ms',
        'latency_p99_ms',
        'throughput_fps',
        'cold_start_model_load_ms',
        'cold_start_first_inference_ms',
        'cpu_percent_mean',
        'cpu_percent_max',
        'memory_mb_mean',
        'cpu_temp_celsius',
        'tpu_detected',
        'timestamp',
    ]
    writer.writerow(header)

    # Data rows
    for row in rows:
        metrics = json.loads(row['metrics'])
        params = json.loads(row['params'])

        # Filter
        if model and model not in row['model_name']:
            continue
        if backend and params.get('backend') != backend:
            continue

        latency = metrics.get('latency', {})
        throughput = metrics.get('throughput', {})
        cold_start = metrics.get('cold_start', {})
        system = metrics.get('system', {})

        writer.writerow(
            [
                row['experiment_id'],
                row['experiment_name'],
                row['model_name'],
                row['device_name'],
                params.get('backend', 'cpu'),
                params.get('batch_size', 1),
                params.get('num_threads', 4),
                params.get('benchmark_runs', 100),
                latency.get('mean_ms', ''),
                latency.get('std_ms', ''),
                latency.get('p50_ms', ''),
                latency.get('p90_ms', ''),
                latency.get('p95_ms', ''),
                latency.get('p99_ms', ''),
                throughput.get('fps', ''),
                cold_start.get('model_load_ms', ''),
                cold_start.get('first_inference_ms', ''),
                system.get('cpu_percent_mean', ''),
                system.get('cpu_percent_max', ''),
                system.get('memory_mb_mean', ''),
                system.get('cpu_temp_celsius', ''),
                system.get('tpu_detected', False),
                row['created_at'],
            ]
        )

    output.seek(0)

    return StreamingResponse(
        iter([output.getvalue()]),
        media_type='text/csv',
        headers={
            'Content-Disposition': f'attachment; filename=edgebench_results_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.csv'
        },
    )


@router.get('/export/json')
async def export_json(
    model: str | None = None,
    backend: str | None = None,
):
    """Export results to JSON."""
    results = await list_results(model=model, backend=backend, limit=10000)

    return StreamingResponse(
        iter([json.dumps(results, indent=2, default=str)]),
        media_type='application/json',
        headers={
            'Content-Disposition': f'attachment; filename=edgebench_results_{datetime.now(UTC).strftime("%Y%m%d_%H%M%S")}.json'
        },
    )


@router.get('/compare')
async def compare_results(
    experiment_ids: str = Query(..., description='Comma-separated experiment IDs'),
):
    """Compare results from multiple experiments."""
    ids = [id.strip() for id in experiment_ids.split(',')]

    async with get_db() as db:
        placeholders = ','.join(['?' for _ in ids])
        cursor = await db.execute(
            f"""SELECT r.*, e.name as experiment_name, e.model_name, e.params
                FROM results r
                JOIN experiments e ON r.experiment_id = e.id
                WHERE r.experiment_id IN ({placeholders})""",
            ids,
        )
        rows = await cursor.fetchall()

    comparisons = []
    for row in rows:
        metrics = json.loads(row['metrics'])
        params = json.loads(row['params'])

        comparisons.append(
            {
                'experiment_id': row['experiment_id'],
                'experiment_name': row['experiment_name'],
                'model_name': row['model_name'],
                'backend': params.get('backend', 'cpu'),
                'latency_mean_ms': metrics.get('latency', {}).get('mean_ms'),
                'latency_p95_ms': metrics.get('latency', {}).get('p95_ms'),
                'throughput_fps': metrics.get('throughput', {}).get('fps'),
            }
        )

    # Sort by latency
    comparisons.sort(key=lambda x: x.get('latency_mean_ms', float('inf')))

    return {
        'comparisons': comparisons,
        'best_latency': comparisons[0] if comparisons else None,
        'best_throughput': max(comparisons, key=lambda x: x.get('throughput_fps', 0))
        if comparisons
        else None,
    }


@router.get('/{experiment_id}/compare-baseline')
async def compare_baseline(experiment_id: str):
    """Compare current experiment result against the baseline for the same model+backend+device.

    Returns:
      {"current": {...}, "baseline": {...}, "delta": {"mean_ms": +1.2, "fps": -3.1, "pct": "+3.2%"}}
    """
    # Load current experiment and result
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT e.*, r.metrics FROM experiments e
               LEFT JOIN results r ON e.id = r.experiment_id
               WHERE e.id = ?""",
            (experiment_id,),
        )
        current_row = await cursor.fetchone()

    if not current_row:
        raise HTTPException(404, 'Experiment not found')

    current = dict(current_row)
    current_params = json.loads(current.get('params') or '{}')
    current_metrics = json.loads(current.get('metrics') or '{}')

    model_name = current['model_name']
    backend = current_params.get('backend', 'cpu')
    device_id = current['device_id']

    # Find baseline: explicitly marked, or most recent completed experiment
    # for the same model+backend+device (excluding the current one)
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT e.*, r.metrics FROM experiments e
               LEFT JOIN results r ON e.id = r.experiment_id
               WHERE e.model_name = ?
                 AND e.device_id = ?
                 AND e.status = 'completed'
                 AND e.id != ?
                 AND COALESCE(json_extract(e.params, '$.backend'), 'cpu') = ?
                 AND r.metrics IS NOT NULL
               ORDER BY e.is_baseline DESC, e.completed_at DESC
               LIMIT 1""",
            (model_name, device_id, experiment_id, backend),
        )
        baseline_row = await cursor.fetchone()

    if not baseline_row:
        return {
            'current': _summarize(current_metrics, current),
            'baseline': None,
            'delta': None,
            'message': 'No baseline found for this model/device combination',
        }

    baseline = dict(baseline_row)
    baseline_metrics = json.loads(baseline.get('metrics') or '{}')

    cur_latency = current_metrics.get('latency', {}).get('mean_ms')
    base_latency = baseline_metrics.get('latency', {}).get('mean_ms')
    cur_fps = current_metrics.get('throughput', {}).get('fps')
    base_fps = baseline_metrics.get('throughput', {}).get('fps')

    delta = {}
    if cur_latency is not None and base_latency is not None and base_latency != 0:
        delta['mean_ms'] = round(cur_latency - base_latency, 3)
        pct = (cur_latency - base_latency) / base_latency * 100
        delta['pct'] = f'{pct:+.1f}%'
    if cur_fps is not None and base_fps is not None:
        delta['fps'] = round(cur_fps - base_fps, 2)

    return {
        'current': _summarize(current_metrics, current),
        'baseline': _summarize(baseline_metrics, baseline),
        'delta': delta,
    }


def _summarize(metrics: dict, exp: dict) -> dict:
    """Build a compact summary of an experiment result."""
    latency = metrics.get('latency', {})
    throughput = metrics.get('throughput', {})
    return {
        'experiment_id': exp['id'],
        'completed_at': exp.get('completed_at'),
        'is_baseline': bool(exp.get('is_baseline', 0)),
        'mean_ms': latency.get('mean_ms'),
        'std_ms': latency.get('std_ms'),
        'p95_ms': latency.get('p95_ms'),
        'fps': throughput.get('fps'),
    }


@router.post('/report')
async def report_result(request: dict):
    """Accept a result pushed directly from an agent.

    Used when the agent has cached results locally (e.g. server was down
    during the original benchmark) and syncs them later.
    """
    experiment_id = request.get('experiment_id')
    result = request.get('result')

    if not experiment_id or not result:
        raise HTTPException(400, 'experiment_id and result required')

    async with get_db() as db:
        # Check if result already exists
        cursor = await db.execute(
            'SELECT id FROM results WHERE experiment_id = ?',
            (experiment_id,),
        )
        existing = await cursor.fetchone()

        if existing:
            # 409 Conflict — agent will remove from its cache
            raise HTTPException(409, 'Result already exists for this experiment')

        # Verify experiment exists
        cursor = await db.execute(
            'SELECT id, status FROM experiments WHERE id = ?',
            (experiment_id,),
        )
        experiment = await cursor.fetchone()

        if not experiment:
            raise HTTPException(404, 'Experiment not found')

        # Save the result
        await db.execute(
            """INSERT INTO results (id, experiment_id, metrics, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                f'res_{experiment_id}',
                experiment_id,
                json.dumps(result),
                datetime.now(UTC).isoformat(),
            ),
        )

        # Update experiment status to completed
        await db.execute(
            """UPDATE experiments
               SET status = ?, completed_at = ?, logs = ?, error_message = NULL
               WHERE id = ?""",
            (
                ExperimentStatus.COMPLETED.value,
                datetime.now(UTC).isoformat(),
                result.get('logs', ''),
                experiment_id,
            ),
        )
        await db.commit()

    print(f'[Report] Received cached result for experiment {experiment_id}')

    return {
        'status': 'accepted',
        'experiment_id': experiment_id,
    }
