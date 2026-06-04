"""Security path tests.

Covers:
  1. Path traversal in file upload → 400
  2. Server API: missing X-Agent-Secret → 403
  3. Server API: correct X-Agent-Secret → request reaches handler
  4. Agent /execute/code without EDGEBENCH_DEBUG → 403
"""

import io
from pathlib import Path
import sys

from fastapi import Depends, FastAPI
from fastapi.testclient import TestClient
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
AGENT_DIR = ROOT_DIR / 'agent'


# ── 1. Path traversal ─────────────────────────────────────────────────────────

def test_path_traversal_upload_rejected(client):
    """Filename with ../ path components must return 400 before writing any file."""
    resp = client.post(
        '/api/files/upload',
        data={'file_type': 'model'},
        files={'file': ('../../etc/crontab', io.BytesIO(b'fake'), 'application/octet-stream')},
    )
    assert resp.status_code == 400, (
        f'Expected 400 for traversal attempt, got {resp.status_code}: {resp.text}'
    )
    assert 'traversal' in resp.json().get('detail', '').lower()


def test_normal_filename_upload_succeeds(client):
    """A clean filename must pass the traversal check and be accepted."""
    resp = client.post(
        '/api/files/upload',
        data={'file_type': 'model'},
        files={'file': ('valid_model.tflite', io.BytesIO(b'fake model'), 'application/octet-stream')},
    )
    # 200 = accepted (possibly a duplicate hash if run twice, still not 400/403)
    assert resp.status_code == 200, f'Expected 200 for clean filename, got {resp.status_code}'


# ── 2 & 3. Server shared-secret auth ──────────────────────────────────────────

@pytest.fixture()
def auth_client(isolated_storage):
    """Devices endpoint protected by shared-secret, backed by isolated test DB."""
    from server.api import devices as dev_api
    from server.core.auth import require_api_secret
    from server.core.config import settings as srv_settings

    srv_settings.AGENT_SECRET = 'test-secret-p2'
    app = FastAPI()
    app.include_router(
        dev_api.router,
        prefix='/api/devices',
        dependencies=[Depends(require_api_secret)],
    )
    with TestClient(app) as tc:
        yield tc
    srv_settings.AGENT_SECRET = ''


def test_missing_secret_returns_403(auth_client):
    """Request without X-Agent-Secret must be rejected with 403."""
    resp = auth_client.get('/api/devices')
    assert resp.status_code == 403


def test_wrong_secret_returns_403(auth_client):
    """Request with incorrect X-Agent-Secret must be rejected with 403."""
    resp = auth_client.get('/api/devices', headers={'X-Agent-Secret': 'wrong-secret'})
    assert resp.status_code == 403


def test_correct_secret_passes_auth(auth_client):
    """Correct X-Agent-Secret must pass auth (response is 200, not 403)."""
    resp = auth_client.get('/api/devices', headers={'X-Agent-Secret': 'test-secret-p2'})
    assert resp.status_code != 403, 'Expected auth to pass but got 403'
    assert resp.status_code == 200


# ── 4. Agent /execute/code without DEBUG ──────────────────────────────────────

def test_execute_code_without_debug_returns_403():
    """/execute/code must return 403 when EDGEBENCH_DEBUG is not set."""
    if str(AGENT_DIR) not in sys.path:
        sys.path.insert(0, str(AGENT_DIR))

    import config as agent_config

    original_debug = agent_config.settings.DEBUG
    agent_config.settings.DEBUG = False
    try:
        import main as agent_main  # agent/main.py

        with TestClient(agent_main.app) as tc:
            resp = tc.post('/execute/code', json={'code': 'echo hello'})

        assert resp.status_code == 403, (
            f'Expected 403 when DEBUG=False, got {resp.status_code}: {resp.text}'
        )
    finally:
        agent_config.settings.DEBUG = original_debug
