"""Web UI route handlers (HTML pages)."""

from datetime import UTC, datetime
import json
from pathlib import Path

from apscheduler.triggers.cron import CronTrigger
from fastapi import APIRouter, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from starlette.responses import RedirectResponse

from server.core.config import AGENT_VERSION, settings
from server.db.database import get_db

BASE_DIR = Path(__file__).parent.parent  # server/
templates = Jinja2Templates(directory=BASE_DIR / 'templates')
templates.env.globals['agent_secret'] = settings.AGENT_SECRET

router = APIRouter()


def _parse_json_column(value: str | None, *, default=None):
    """Decode a JSON DB column. Returns default if null, empty, or unparseable."""
    if not value:
        return default
    try:
        return json.loads(value)
    except (json.JSONDecodeError, TypeError):
        return default


def _summarize_results(rows) -> dict:
    """Aggregate headline benchmark numbers for the dashboard cards.

    Returns zeroed/None fields when there are no results yet so the template
    can render an honest empty state instead of a broken placeholder.
    """
    latencies: list[float] = []
    best: dict | None = None

    for row in rows:
        metrics = _parse_json_column(row['metrics'], default={}) or {}
        latency = metrics.get('latency') or {}
        mean_ms = latency.get('mean_ms')
        if not isinstance(mean_ms, int | float) or mean_ms <= 0:
            continue
        latencies.append(float(mean_ms))

        throughput = metrics.get('throughput') or {}
        params = _parse_json_column(row['params'], default={}) or {}
        candidate = {
            'model_name': row['model_name'],
            'latency_ms': float(mean_ms),
            'fps': throughput.get('fps_from_mean') or throughput.get('fps'),
            'backend': params.get('backend') or 'cpu',
        }
        if best is None or candidate['latency_ms'] < best['latency_ms']:
            best = candidate

    if not latencies:
        return {'count': 0, 'avg_latency_ms': None, 'best': None}

    return {
        'count': len(latencies),
        'avg_latency_ms': round(sum(latencies) / len(latencies), 2),
        'best': best,
    }


@router.get('/', response_class=HTMLResponse)
async def index(request: Request):
    """Main dashboard."""
    async with get_db() as db:
        devices_count = await db.execute('SELECT COUNT(*) FROM devices')
        devices_count = (await devices_count.fetchone())[0]

        experiments_count = await db.execute('SELECT COUNT(*) FROM experiments')
        experiments_count = (await experiments_count.fetchone())[0]

        recent = await db.execute(
            'SELECT * FROM experiments ORDER BY created_at DESC LIMIT 10'
        )
        recent_experiments = await recent.fetchall()

        cursor = await db.execute(
            """SELECT s.*, d.name as device_name
               FROM schedules s
               LEFT JOIN devices d ON s.device_id = d.id
               WHERE s.enabled = 1"""
        )
        sched_rows = await cursor.fetchall()

        cursor = await db.execute(
            """SELECT r.metrics, e.model_name, e.params
               FROM results r
               JOIN experiments e ON r.experiment_id = e.id
               ORDER BY r.created_at DESC LIMIT 200"""
        )
        metric_rows = await cursor.fetchall()

    summary = _summarize_results(metric_rows)

    upcoming = []
    for row in sched_rows:
        s = dict(row)
        try:
            trigger = CronTrigger.from_crontab(s['cron'], timezone='UTC')
            nf = trigger.get_next_fire_time(None, datetime.now(UTC))
            s['next_run'] = nf.isoformat() if nf else None
        except Exception:
            s['next_run'] = None
        upcoming.append(s)

    upcoming.sort(key=lambda x: x['next_run'] or '')
    upcoming = upcoming[:3]

    return templates.TemplateResponse(
        request,
        'index.html',
        {
            'devices_count': devices_count,
            'experiments_count': experiments_count,
            'recent_experiments': [dict(r) for r in recent_experiments],
            'upcoming_schedules': upcoming,
            'summary': summary,
        },
    )


@router.get('/schedules', response_class=HTMLResponse)
async def schedules_page(request: Request):
    """Nightly benchmark schedules page."""
    async with get_db() as db:
        cursor = await db.execute(
            """SELECT s.*, d.name as device_name
               FROM schedules s
               LEFT JOIN devices d ON s.device_id = d.id
               ORDER BY s.created_at DESC"""
        )
        schedule_rows = await cursor.fetchall()

        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

    def next_run(cron: str) -> str | None:
        try:
            trigger = CronTrigger.from_crontab(cron, timezone='UTC')
            nf = trigger.get_next_fire_time(None, datetime.now(UTC))
            return nf.isoformat() if nf else None
        except Exception:
            return None

    def human_cron(cron: str) -> str:
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

    schedules = []
    for row in schedule_rows:
        s = dict(row)
        s['params'] = _parse_json_column(s.get('params'), default={})
        s['next_run'] = next_run(s['cron'])
        s['cron_human'] = human_cron(s['cron'])
        schedules.append(s)

    return templates.TemplateResponse(
        request,
        'schedules.html',
        {
            'schedules': schedules,
            'devices': [dict(d) for d in device_list],
        },
    )


@router.get('/devices', response_class=HTMLResponse)
async def devices_page(request: Request):
    """Devices management page.

    Renders immediately from DB cache. Live status is checked
    asynchronously on the client side via JS after page load.
    """
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

    return templates.TemplateResponse(
        request,
        'devices.html',
        {'devices': [dict(d) for d in device_list]},
    )


@router.get('/experiments', response_class=HTMLResponse)
async def experiments_page(request: Request):
    """Experiments list page."""
    async with get_db() as db:
        cursor = await db.execute("""
            SELECT e.*, d.name as device_name
            FROM experiments e
            LEFT JOIN devices d ON e.device_id = d.id
            ORDER BY e.created_at DESC
        """)
        experiment_list = await cursor.fetchall()

    experiments = []
    for exp in experiment_list:
        exp_dict = dict(exp)
        exp_dict['params'] = _parse_json_column(exp_dict.get('params'), default={})
        experiments.append(exp_dict)

    return templates.TemplateResponse(
        request,
        'experiments.html',
        {'experiments': experiments},
    )


@router.get('/experiments/{exp_id}', response_class=HTMLResponse)
async def experiment_detail(request: Request, exp_id: str):
    """Experiment detail page."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM experiments WHERE id = ?', (exp_id,))
        experiment = await cursor.fetchone()

        cursor = await db.execute(
            'SELECT * FROM results WHERE experiment_id = ?', (exp_id,)
        )
        result = await cursor.fetchone()

    metrics = _parse_json_column(result['metrics'] if result else None)

    return templates.TemplateResponse(
        request,
        'experiment_detail.html',
        {
            'experiment': dict(experiment) if experiment else None,
            'result': dict(result) if result else None,
            'metrics': metrics,
        },
    )


@router.get('/results', response_class=HTMLResponse)
async def results_page(request: Request):
    """Results comparison page."""
    async with get_db() as db:
        cursor = await db.execute("""
            SELECT r.*, e.name as experiment_name, e.model_name, d.name as device_name
            FROM results r
            JOIN experiments e ON r.experiment_id = e.id
            LEFT JOIN devices d ON e.device_id = d.id
            ORDER BY r.created_at DESC
        """)
        rows = await cursor.fetchall()

    result_list = []
    for row in rows:
        result_dict = dict(row)
        metrics = _parse_json_column(result_dict.get('metrics'), default={}) or {}
        result_dict['metrics'] = metrics

        latency = metrics.get('latency') or {}
        throughput = metrics.get('throughput') or {}
        params = metrics.get('params') or {}
        system = metrics.get('system') or {}
        model = metrics.get('model') or {}

        # Flattened view so the template stays free of nested lookups.
        result_dict['view'] = {
            'backend': params.get('backend') or '',
            'mean_ms': latency.get('mean_ms'),
            'std_ms': latency.get('std_ms'),
            'p95_ms': latency.get('p95_ms'),
            'fps': throughput.get('fps_from_mean') or throughput.get('fps'),
            'rss_mb': system.get('process_rss_mb_max'),
            'size_mb': round(model['size_bytes'] / 1_048_576, 2)
            if model.get('size_bytes')
            else None,
            'runs': params.get('benchmark_runs'),
            'warnings': metrics.get('warnings') or [],
        }
        result_list.append(result_dict)

    measured = [r['view']['mean_ms'] for r in result_list if r['view']['mean_ms']]
    fps_values = [r['view']['fps'] for r in result_list if r['view']['fps']]
    summary = {
        'count': len(result_list),
        'measured': len(measured),
        'best_ms': min(measured) if measured else None,
        'worst_ms': max(measured) if measured else None,
        'best_fps': max(fps_values) if fps_values else None,
        'flagged': sum(1 for r in result_list if r['view']['warnings']),
    }

    return templates.TemplateResponse(
        request,
        'results.html',
        {'results': result_list, 'summary': summary},
    )


@router.get('/models', response_class=HTMLResponse)
async def models_page(request: Request):
    """Model repository page."""
    async with get_db() as db:
        cursor = await db.execute(
            "SELECT * FROM files WHERE type = 'model' ORDER BY created_at DESC"
        )
        model_list = await cursor.fetchall()

        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

    models_with_info = []
    for model in model_list:
        model_dict = dict(model)
        name_lower = model_dict['name'].lower()
        if 'edgetpu' in name_lower:
            model_dict['quantization'] = 'int8_edgetpu'
        elif 'int8' in name_lower or '_quant' in name_lower:
            model_dict['quantization'] = 'int8'
        elif 'fp16' in name_lower:
            model_dict['quantization'] = 'fp16'
        elif 'fp32' in name_lower:
            model_dict['quantization'] = 'fp32'
        else:
            model_dict['quantization'] = None
        models_with_info.append(model_dict)

    return templates.TemplateResponse(
        request,
        'models.html',
        {
            'models': models_with_info,
            'devices': [dict(d) for d in device_list],
        },
    )


@router.get('/new-experiment', response_class=HTMLResponse)
async def new_experiment_page(request: Request):
    """New experiment form."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

        cursor = await db.execute("SELECT * FROM files WHERE type = 'model'")
        model_list = await cursor.fetchall()

    return templates.TemplateResponse(
        request,
        'new_experiment.html',
        {
            'devices': [dict(d) for d in device_list],
            'models': [dict(m) for m in model_list],
        },
    )


@router.get('/benchmark', response_class=HTMLResponse)
async def benchmark_page(request: Request):
    """Benchmark tools page."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

    return templates.TemplateResponse(
        request,
        'benchmark.html',
        {'devices': [dict(d) for d in device_list]},
    )


@router.get('/scripts', response_class=HTMLResponse)
async def scripts_page(request: Request):
    """Remote script execution and device inspection page."""
    async with get_db() as db:
        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

        cursor = await db.execute(
            "SELECT * FROM files WHERE type = 'script' ORDER BY created_at DESC"
        )
        script_list = await cursor.fetchall()

    return templates.TemplateResponse(
        request,
        'scripts.html',
        {
            'devices': [dict(d) for d in device_list],
            'scripts': [dict(s) for s in script_list],
        },
    )


@router.get('/settings', response_class=HTMLResponse)
async def settings_page(request: Request):
    """Settings page."""
    async with get_db() as db:
        cursor = await db.execute(
            'SELECT * FROM dependencies ORDER BY is_required DESC, name'
        )
        dependency_list = await cursor.fetchall()

        cursor = await db.execute('SELECT * FROM devices ORDER BY name')
        device_list = await cursor.fetchall()

        cursor = await db.execute('SELECT * FROM settings')
        saved = {row['key']: row['value'] for row in await cursor.fetchall()}

    return templates.TemplateResponse(
        request,
        'settings.html',
        {
            'agent_version': AGENT_VERSION,
            'server_port': settings.PORT,
            'max_tasks': int(saved.get('max_tasks', settings.MAX_CONCURRENT_TASKS)),
            'task_timeout': int(
                saved.get('task_timeout', settings.TASK_TIMEOUT_SECONDS)
            ),
            'agent_timeout': int(
                saved.get('agent_timeout', settings.AGENT_TIMEOUT_SECONDS)
            ),
            'paths': {
                'models': str(settings.MODELS_DIR),
                'scripts': str(settings.SCRIPTS_DIR),
                'database': str(settings.DATABASE_PATH),
                'uploads': str(settings.UPLOAD_DIR),
            },
            'dependencies': [dict(r) for r in dependency_list],
            'devices': [dict(r) for r in device_list],
            'saved_settings': saved,
        },
    )


@router.get('/dependencies', response_class=HTMLResponse)
async def dependencies_page(request: Request):
    """Dependencies management page (redirect to settings)."""
    return RedirectResponse(url='/settings', status_code=302)


@router.get('/compare', response_class=HTMLResponse)
async def compare_page(request: Request):
    """Results comparison page."""
    async with get_db() as db:
        cursor = await db.execute("""
            SELECT e.id, e.name, e.model_name, e.params, e.status,
                   d.name as device_name, r.metrics, r.created_at as result_date
            FROM experiments e
            LEFT JOIN devices d ON e.device_id = d.id
            LEFT JOIN results r ON e.id = r.experiment_id
            WHERE e.status = 'completed'
            ORDER BY r.created_at DESC
        """)
        experiments_list = await cursor.fetchall()

    experiments_data = []
    for exp in experiments_list:
        exp_dict = dict(exp)
        exp_dict['metrics'] = _parse_json_column(exp_dict.get('metrics'), default={})
        exp_dict['params'] = _parse_json_column(exp_dict.get('params'), default={})
        experiments_data.append(exp_dict)

    return templates.TemplateResponse(
        request,
        'compare.html',
        {'experiments': experiments_data},
    )
