"""PlantDiagEdge pipeline results: quality and device cost of configurations, side by side.

Two kinds of JSON reports are uploaded and kept as files under UPLOAD_DIR/pipelines
(no database table — the reports are self-describing and stay readable as they are):

  quality  plantdiag-edge scripts/bench.py `report.json`: macro-recall, accuracy and
           AUROC of the confidence, mean ± std over seeds, for one configuration card;
  device   agent/benchmark_pipeline.py `pipeline` output: per-stage latency, RSS and
           model size of one configuration on one device.

/pipelines joins them by configuration (C1…C6) — the latest report of each kind — and
marks the Pareto-optimal configurations: no other one is at least as accurate and at
least as fast with one of the two strictly better.
"""

from datetime import UTC, datetime
import json
import re
import uuid

from fastapi import APIRouter, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse

from server.core.config import settings
from server.routes.ui import templates

api_router = APIRouter()
page_router = APIRouter()

MAX_REPORT_BYTES = 2 * 1024 * 1024
_ID = re.compile(r'^[0-9a-f]{32}$')


def _store():
    path = settings.UPLOAD_DIR / 'pipelines'
    path.mkdir(parents=True, exist_ok=True)
    return path


def classify(doc: dict) -> str | None:
    """'quality', 'device' or None for a report this page does not understand."""
    if doc.get('kind') == 'plantdiag-bench' and isinstance(doc.get('summary'), dict):
        return 'quality'
    if doc.get('mode') == 'pipeline' and isinstance(doc.get('latency'), dict):
        return 'device'
    return None


def _stat(summary: dict, key: str) -> dict | None:
    v = summary.get(key)
    return (
        {'mean': v['mean'], 'std': v['std']}
        if isinstance(v, dict) and 'mean' in v
        else None
    )


def normalize(doc: dict, report_id: str, uploaded: str) -> dict:
    """Flat view of one report for the page and the API."""
    kind = classify(doc)
    if kind == 'quality':
        cfg = doc.get('config') or {}
        summary = doc['summary']
        n_seeds = len(doc.get('seeds') or [])
        return {
            'id': report_id,
            'kind': kind,
            'config': str(cfg.get('name', '')).upper(),
            'label': doc.get('name', ''),
            'uploaded': uploaded,
            'macro_recall': _stat(summary, 'macro_recall'),
            'accuracy': _stat(summary, 'accuracy'),
            'auroc_confidence': _stat(summary, 'auroc_confidence'),
            'n_seeds': n_seeds,
            'dataset': doc.get('dataset', ''),
        }
    lat = doc['latency']
    cfg = doc.get('config') or {}
    bundle = doc.get('bundle') or {}
    device = doc.get('device') or {}
    return {
        'id': report_id,
        'kind': kind,
        'config': str(cfg.get('key', '')).upper(),
        'label': device.get('hostname') or device.get('model') or '',
        'uploaded': uploaded,
        'e2e_p50_ms': (lat.get('end_to_end') or {}).get('p50_ms'),
        'e2e_p95_ms': (lat.get('end_to_end') or {}).get('p95_ms'),
        'stages_p50_ms': {
            k: v.get('p50_ms')
            for k, v in lat.items()
            if k != 'end_to_end' and isinstance(v, dict)
        },
        'rss_mb': doc.get('process_rss_mb'),
        'model_size_mb': bundle.get('model_size_mb'),
        'runs': (doc.get('params') or {}).get('benchmark_runs'),
    }


def load_reports() -> list[dict]:
    out = []
    for path in sorted(_store().glob('*.json')):
        try:
            doc = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if classify(doc):
            meta = doc.get('_uploaded', '')
            out.append(normalize(doc, path.stem, meta))
    return sorted(out, key=lambda r: r['uploaded'], reverse=True)


def pareto(points: list[dict]) -> set[str]:
    """Configs not dominated in (lower latency, higher macro-recall)."""
    front = set()
    for p in points:
        dominated = any(
            q['latency'] <= p['latency']
            and q['quality'] >= p['quality']
            and (q['latency'] < p['latency'] or q['quality'] > p['quality'])
            for q in points
            if q is not p
        )
        if not dominated:
            front.add(p['config'])
    return front


def combined(reports: list[dict]) -> list[dict]:
    """Latest quality and latest device report per configuration, with the Pareto flag."""
    rows: dict[str, dict] = {}
    for r in reports:  # newest first: keep the first of each kind
        row = rows.setdefault(
            r['config'], {'config': r['config'], 'quality': None, 'device': None}
        )
        if row[r['kind']] is None:
            row[r['kind']] = r
    points = [
        {
            'config': c,
            'latency': row['device']['e2e_p50_ms'],
            'quality': row['quality']['macro_recall']['mean'],
        }
        for c, row in rows.items()
        if row['quality']
        and row['device']
        and row['quality'].get('macro_recall')
        and row['device'].get('e2e_p50_ms') is not None
    ]
    front = pareto(points)
    for c, row in rows.items():
        row['pareto'] = c in front
        row['plotted'] = any(p['config'] == c for p in points)
    return [rows[c] for c in sorted(rows)]


# ── API (/api/pipelines, behind the shared secret) ─────────────────────────────


@api_router.post('/upload')
async def upload_report(file: UploadFile):
    data = await file.read(MAX_REPORT_BYTES + 1)
    if len(data) > MAX_REPORT_BYTES:
        raise HTTPException(status_code=413, detail='report larger than 2 MB')
    try:
        doc = json.loads(data)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=422, detail=f'not JSON: {exc}') from exc
    if not isinstance(doc, dict) or classify(doc) is None:
        raise HTTPException(
            status_code=422,
            detail='unknown report: expected bench.py report.json or benchmark_pipeline.py pipeline output',
        )
    report_id = uuid.uuid4().hex
    doc['_uploaded'] = datetime.now(UTC).isoformat()
    (_store() / f'{report_id}.json').write_text(json.dumps(doc))
    return normalize(doc, report_id, doc['_uploaded'])


@api_router.get('')
async def list_reports():
    return load_reports()


@api_router.delete('/{report_id}')
async def delete_report(report_id: str):
    path = _store() / f'{report_id}.json'
    if not _ID.match(report_id) or not path.exists():
        raise HTTPException(status_code=404, detail='report not found')
    path.unlink()
    return {'status': 'deleted', 'id': report_id}


# ── page ───────────────────────────────────────────────────────────────────────


@page_router.get('/pipelines', response_class=HTMLResponse)
async def pipelines_page(request: Request):
    reports = load_reports()
    rows = combined(reports)
    chart = [
        {
            'config': r['config'],
            'x': r['device']['e2e_p50_ms'],
            'y': r['quality']['macro_recall']['mean'],
            'size_mb': r['device'].get('model_size_mb'),
            'pareto': r['pareto'],
        }
        for r in rows
        if r['plotted']
    ]
    return templates.TemplateResponse(
        request,
        'pipelines.html',
        {'reports': reports, 'rows': rows, 'chart': chart},
    )
