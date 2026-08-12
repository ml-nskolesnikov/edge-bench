"""Guards against template <-> route drift in the real application.

The API tests build their own FastAPI app in conftest, so a router that is
missing from `server.main` still passes them. These tests import the real
application object instead.

Routes are enumerated from the OpenAPI schema rather than from `app.routes`:
since FastAPI 0.137 an included router appears as a single opaque
`_IncludedRouter` entry instead of being flattened into `app.routes`, so
walking that attribute reports nothing for mounted routers even though
requests route correctly. The schema reflects what is actually served.
"""

from pathlib import Path
import re

from fastapi.testclient import TestClient
import pytest

from server.main import app

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / 'server' / 'templates'

# Matches /api/... occurrences in templates, stopping at a Jinja/JS boundary.
API_PATH_RE = re.compile(r'/api/[a-z0-9_/-]+')


def _served_paths() -> set[str]:
    """Every path the application actually serves, per its OpenAPI schema."""
    return set(app.openapi()['paths'])


def _path_prefixes() -> set[str]:
    """Served paths plus their `{param}`-stripped prefixes."""
    prefixes = set()
    for path in _served_paths():
        prefixes.add(path)
        head = path.split('{', 1)[0].rstrip('/')
        if head:
            prefixes.add(head)
    return prefixes


def test_every_api_router_is_mounted():
    """All API modules must be reachable through the real app, not just conftest."""
    expected_prefixes = {
        '/api/devices',
        '/api/experiments',
        '/api/results',
        '/api/files',
        '/api/dependencies',
        '/api/settings',
        '/api/schedules',
        '/api/scripts',
    }
    served = _served_paths()
    for prefix in expected_prefixes:
        assert any(p.startswith(prefix) for p in served), (
            f'No route registered under {prefix} in server.main.app'
        )


def test_mounted_routes_actually_respond(isolated_storage):
    """A registered route must resolve — not 404 — when a request is made.

    Complements the schema check above: the schema proves registration, this
    proves routing. Together they survive FastAPI internals changing shape.

    Depends on `isolated_storage`: these pages read the database, and a clean
    checkout has none. Without the fixture the test only passes on a machine
    that happens to have a populated dev database.
    """
    client = TestClient(app)
    for path in ('/api/health', '/', '/results', '/scripts'):
        response = client.get(path)
        assert response.status_code != 404, f'{path} is not routed'
        assert response.status_code < 500, (
            f'{path} returned {response.status_code} on an empty database'
        )


@pytest.mark.parametrize(
    'path,method',
    [
        ('/api/scripts/system-info', 'post'),
        ('/api/scripts/check-deps', 'post'),
        ('/api/scripts/run', 'post'),
        ('/api/scripts/execute', 'post'),
    ],
)
def test_scripts_endpoints_are_routed(path: str, method: str):
    """Regression guard: the scripts router was absent from the real app.

    A missing router yields 404 with detail "Not Found"; a mounted one
    validates the body and answers 400/404 with a domain message instead.
    """
    client = TestClient(app)
    response = getattr(client, method)(path, json={})
    detail = ''
    if response.headers.get('content-type', '').startswith('application/json'):
        body = response.json()
        detail = body.get('detail', '') if isinstance(body, dict) else ''
    assert not (response.status_code == 404 and detail == 'Not Found'), (
        f'{method.upper()} {path} is not routed'
    )


def test_every_ui_template_has_a_route():
    """Each template extending base.html must be rendered by some UI route."""
    rendered = set()
    ui_source = (ROOT / 'server' / 'routes' / 'ui.py').read_text()
    for match in re.finditer(r"'([a-z_]+\.html)'", ui_source):
        rendered.add(match.group(1))

    orphans = set()
    for template in TEMPLATES_DIR.glob('*.html'):
        if template.name == 'base.html':
            continue
        if template.name not in rendered:
            orphans.add(template.name)

    # dependencies.html is intentionally retired: /dependencies redirects to
    # /settings, which absorbed the dependency management UI.
    assert orphans <= {'dependencies.html'}, (
        f'Templates with no route: {sorted(orphans)}'
    )


def test_api_paths_used_by_templates_exist():
    """Every /api/... path hard-coded in a template must resolve to a route."""
    prefixes = _path_prefixes()
    missing = set()

    for template in TEMPLATES_DIR.glob('*.html'):
        for raw in API_PATH_RE.findall(template.read_text()):
            path = raw.rstrip('/')
            if not path or path.endswith('/api'):
                continue
            # A template path matches if any served path starts with it
            # (covers query strings, trailing IDs and sub-resources).
            if not any(p.startswith(path) or path.startswith(p) for p in prefixes):
                missing.add(f'{template.name}: {path}')

    assert not missing, f'Templates reference unrouted API paths: {sorted(missing)}'
