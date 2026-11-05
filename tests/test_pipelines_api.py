"""/api/pipelines and /pipelines: PlantDiagEdge quality + device reports and the Pareto view."""

import json

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

from server.routes.pipelines import api_router, page_router, pareto


def _quality(config: str, recall: float) -> dict:
    stat = {'mean': recall, 'std': 0.01, 'per_seed': {'42': recall}}
    return {
        'kind': 'plantdiag-bench',
        'name': config.lower(),
        'config': {'name': config, 'backbone': 'mobilenet_v2'},
        'seeds': [42, 0, 1, 2, 3],
        'summary': {'macro_recall': stat, 'accuracy': stat, 'auroc_confidence': stat},
    }


def _device(config: str, p50: float) -> dict:
    return {
        'mode': 'pipeline',
        'config': {'key': config.lower()},
        'device': {'hostname': 'raspberrypi'},
        'bundle': {'model_size_mb': 2.6},
        'latency': {
            'end_to_end': {'p50_ms': p50, 'p95_ms': p50 * 1.2},
            'inference': {'p50_ms': p50 / 2},
        },
        'process_rss_mb': 186.0,
    }


@pytest.fixture
def client(isolated_storage):
    app = FastAPI()
    app.include_router(api_router, prefix='/api/pipelines')
    app.include_router(page_router)
    return TestClient(app)


def _upload(client, doc: dict):
    return client.post(
        '/api/pipelines/upload',
        files={'file': ('r.json', json.dumps(doc), 'application/json')},
    )


def test_reports_are_joined_per_config_with_a_pareto_front(client):
    for doc in (
        _quality('C4', 0.23),
        _device('C4', 288.0),
        _quality('C6', 0.17),
        _device('C6', 104.0),
    ):
        assert _upload(client, doc).status_code == 200
    # C5 is slower and less accurate than C6, so C6 dominates it.
    assert _upload(client, _quality('C5', 0.15)).status_code == 200
    assert _upload(client, _device('C5', 120.0)).status_code == 200

    listed = client.get('/api/pipelines').json()
    assert {(r['config'], r['kind']) for r in listed} >= {
        ('C4', 'quality'),
        ('C6', 'device'),
    }

    page = client.get('/pipelines')
    assert page.status_code == 200
    html = page.text
    assert 'paretoChart' in html and '0.2300' in html and '104.0 ms' in html
    chart = json.loads(html.split('const CHART = ')[1].split(';\n')[0])
    front = {p['config'] for p in chart if p['pareto']}
    assert front == {'C4', 'C6'}, chart


def test_unknown_or_broken_reports_are_rejected(client):
    assert _upload(client, {'mode': 'model', 'latency': {}}).status_code == 422
    bad = client.post(
        '/api/pipelines/upload',
        files={'file': ('r.json', b'{not json', 'application/json')},
    )
    assert bad.status_code == 422
    assert client.get('/api/pipelines').json() == []


def test_delete_checks_the_id(client):
    rid = _upload(client, _device('C6', 104.0)).json()['id']
    assert client.delete('/api/pipelines/' + '0' * 32).status_code == 404
    assert client.delete('/api/pipelines/..%2Fsecret').status_code == 404
    assert client.delete(f'/api/pipelines/{rid}').status_code == 200
    assert client.get('/api/pipelines').json() == []


def test_pareto_keeps_ties_and_drops_dominated():
    pts = [
        {'config': 'A', 'latency': 100, 'quality': 0.2},
        {'config': 'B', 'latency': 100, 'quality': 0.2},
        {'config': 'C', 'latency': 50, 'quality': 0.1},
        {'config': 'D', 'latency': 120, 'quality': 0.2},
    ]
    assert pareto(pts) == {'A', 'B', 'C'}
