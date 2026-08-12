"""Guards against template <-> route drift in the real application.

The API tests build their own FastAPI app in conftest, so a router that is
missing from `server.main` still passes them. These tests import the real
application object instead.
"""

from pathlib import Path
import re

from server.main import app

ROOT = Path(__file__).resolve().parents[1]
TEMPLATES_DIR = ROOT / 'server' / 'templates'

# Matches /api/... occurrences in templates, stopping at a Jinja/JS boundary.
API_PATH_RE = re.compile(r'/api/[a-z0-9_/-]+')


def _registered_paths() -> set[str]:
    return {route.path for route in app.routes if hasattr(route, 'path')}


def _path_prefixes() -> set[str]:
    """Registered paths with their `{param}` segments stripped to a prefix."""
    prefixes = set()
    for path in _registered_paths():
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
    registered = _registered_paths()
    for prefix in expected_prefixes:
        assert any(p.startswith(prefix) for p in registered), (
            f'No route registered under {prefix} in server.main.app'
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
            # A template path matches if any registered path starts with it
            # (covers query strings, trailing IDs and sub-resources).
            if not any(p.startswith(path) or path.startswith(p) for p in prefixes):
                missing.add(f'{template.name}: {path}')

    assert not missing, f'Templates reference unrouted API paths: {sorted(missing)}'
