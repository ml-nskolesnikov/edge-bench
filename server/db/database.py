"""
Database Connection and Management
"""

from contextlib import asynccontextmanager

import aiosqlite

from server.core.config import settings

SCHEMA = """
-- Devices table
CREATE TABLE IF NOT EXISTS devices (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    ip TEXT NOT NULL,
    port INTEGER DEFAULT 8001,
    status TEXT DEFAULT 'offline',
    description TEXT,
    device_info TEXT,
    last_seen TEXT,
    created_at TEXT NOT NULL
);

-- Experiments table
CREATE TABLE IF NOT EXISTS experiments (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    device_id TEXT NOT NULL,
    model_name TEXT NOT NULL,
    model_path TEXT NOT NULL,
    script_path TEXT DEFAULT 'benchmark_tflite.py',
    params TEXT NOT NULL,
    status TEXT DEFAULT 'queued',
    logs TEXT,
    error_message TEXT,
    created_at TEXT NOT NULL,
    started_at TEXT,
    completed_at TEXT,
    FOREIGN KEY (device_id) REFERENCES devices(id)
);

-- Results table
CREATE TABLE IF NOT EXISTS results (
    id TEXT PRIMARY KEY,
    experiment_id TEXT NOT NULL UNIQUE,
    metrics TEXT NOT NULL,
    created_at TEXT NOT NULL,
    FOREIGN KEY (experiment_id) REFERENCES experiments(id)
);

-- Files table
CREATE TABLE IF NOT EXISTS files (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    path TEXT NOT NULL,
    size_bytes INTEGER NOT NULL,
    hash TEXT NOT NULL,
    created_at TEXT NOT NULL
);

-- Dependencies table (for tracking required packages on RPi)
CREATE TABLE IF NOT EXISTS dependencies (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL UNIQUE,
    package TEXT NOT NULL,
    version TEXT,
    check_command TEXT,
    install_command TEXT,
    is_required INTEGER DEFAULT 1,
    description TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT
);

-- Device dependencies status
CREATE TABLE IF NOT EXISTS device_dependencies (
    id TEXT PRIMARY KEY,
    device_id TEXT NOT NULL,
    dependency_id TEXT NOT NULL,
    status TEXT DEFAULT 'unknown',
    installed_version TEXT,
    error_message TEXT,
    checked_at TEXT,
    FOREIGN KEY (device_id) REFERENCES devices(id),
    FOREIGN KEY (dependency_id) REFERENCES dependencies(id),
    UNIQUE(device_id, dependency_id)
);

-- Settings table
CREATE TABLE IF NOT EXISTS settings (
    key TEXT PRIMARY KEY,
    value TEXT
);

-- Indexes
CREATE INDEX IF NOT EXISTS idx_experiments_device ON experiments(device_id);
CREATE INDEX IF NOT EXISTS idx_experiments_status ON experiments(status);
CREATE INDEX IF NOT EXISTS idx_results_experiment ON results(experiment_id);
CREATE INDEX IF NOT EXISTS idx_device_deps_device ON device_dependencies(device_id);
CREATE INDEX IF NOT EXISTS idx_files_hash ON files(hash);
"""


# Default dependencies for Edge-Bench
DEFAULT_DEPENDENCIES = [
    {
        'id': 'dep_numpy',
        'name': 'NumPy',
        'package': 'numpy',
        'check_command': 'python3 -c "import numpy; print(numpy.__version__)"',
        'description': 'Numerical computing library',
    },
    {
        'id': 'dep_tflite',
        'name': 'TFLite Runtime',
        'package': 'tflite-runtime',
        'check_command': 'python3 -c "import tflite_runtime; print(tflite_runtime.__version__)"',
        'description': 'TensorFlow Lite inference runtime',
    },
    {
        'id': 'dep_pycoral',
        'name': 'PyCoral',
        'package': 'pycoral',
        'check_command': 'python3 -c "from pycoral.utils import edgetpu; print(edgetpu.get_runtime_version())"',
        'description': 'Coral Edge TPU Python API',
    },
    {
        'id': 'dep_pillow',
        'name': 'Pillow',
        'package': 'pillow',
        'check_command': 'python3 -c "import PIL; print(PIL.__version__)"',
        'description': 'Image processing library',
    },
    {
        'id': 'dep_psutil',
        'name': 'psutil',
        'package': 'psutil',
        'check_command': 'python3 -c "import psutil; print(psutil.__version__)"',
        'description': 'System monitoring utilities',
    },
]


async def init_db():
    """Initialize database with schema and default data."""
    from datetime import datetime

    settings.DATABASE_PATH.parent.mkdir(parents=True, exist_ok=True)

    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        await db.executescript(SCHEMA)

        # Migration: add install_command column if missing
        cursor = await db.execute('PRAGMA table_info(dependencies)')
        cols = {row[1] for row in await cursor.fetchall()}
        if 'install_command' not in cols:
            await db.execute('ALTER TABLE dependencies ADD COLUMN install_command TEXT')
            await db.commit()

        # Add default dependencies if not exist
        for dep in DEFAULT_DEPENDENCIES:
            cursor = await db.execute(
                'SELECT id FROM dependencies WHERE id = ?', (dep['id'],)
            )
            if not await cursor.fetchone():
                await db.execute(
                    """INSERT INTO dependencies
                       (id, name, package, check_command, description, is_required, created_at)
                       VALUES (?, ?, ?, ?, ?, 1, ?)""",
                    (
                        dep['id'],
                        dep['name'],
                        dep['package'],
                        dep['check_command'],
                        dep['description'],
                        datetime.utcnow().isoformat(),
                    ),
                )

        await db.commit()


@asynccontextmanager
async def get_db():
    """Get database connection."""
    async with aiosqlite.connect(settings.DATABASE_PATH) as db:
        db.row_factory = aiosqlite.Row
        yield db
