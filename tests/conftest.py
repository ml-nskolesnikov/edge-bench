import asyncio
from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient
import pytest

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from server.api import (
    dependencies,
    devices,
    experiments,
    files,
    results,
    schedules,
    settings as settings_api,
)
from server.core.config import settings
from server.db.database import init_db


@pytest.fixture
def isolated_storage(tmp_path: Path):
    settings.DATABASE_PATH = tmp_path / 'edgebench_test.db'
    settings.UPLOAD_DIR = tmp_path / 'uploads'
    settings.MODELS_DIR = tmp_path / 'models'
    settings.SCRIPTS_DIR = tmp_path / 'scripts'

    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    settings.SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    asyncio.run(init_db())
    return tmp_path


@pytest.fixture
def api_app(isolated_storage: Path):
    app = FastAPI()
    app.include_router(devices.router, prefix='/api/devices')
    app.include_router(experiments.router, prefix='/api/experiments')
    app.include_router(results.router, prefix='/api/results')
    app.include_router(files.router, prefix='/api/files')
    app.include_router(dependencies.router, prefix='/api/dependencies')
    app.include_router(settings_api.router, prefix='/api/settings')
    app.include_router(schedules.router, prefix='/api/schedules')
    return app


@pytest.fixture
def client(api_app: FastAPI):
    with TestClient(api_app) as test_client:
        yield test_client
