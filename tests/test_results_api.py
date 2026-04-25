from datetime import UTC, datetime
import json

from server.db.database import get_db


async def _insert_device_and_experiment(experiment_id: str):
    now = datetime.now(UTC).isoformat()
    params = {
        'backend': 'cpu',
        'batch_size': 1,
        'num_threads': 4,
        'warmup_runs': 10,
        'benchmark_runs': 100,
        'timeout_seconds': 600,
        'tpu_index': 0,
    }
    async with get_db() as db:
        await db.execute(
            """INSERT INTO devices
               (id, name, ip, port, status, description, last_seen, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ('dev_result', 'result-device', '127.0.0.1', 8001, 'online', None, now, now),
        )
        await db.execute(
            """INSERT INTO experiments
               (id, name, device_id, model_name, model_path, script_path, params, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                'result exp',
                'dev_result',
                'model_a.tflite',
                'models/model_a.tflite',
                'benchmark_tflite.py',
                json.dumps(params),
                'queued',
                now,
            ),
        )
        await db.commit()


async def _insert_completed_result_experiment(
    experiment_id: str,
    *,
    backend: str,
    mean_ms: float,
    fps: float,
    created_at: str,
    completed_at: str,
    is_baseline: int = 0,
):
    params = {
        'backend': backend,
        'batch_size': 1,
        'num_threads': 4,
        'warmup_runs': 10,
        'benchmark_runs': 100,
        'timeout_seconds': 600,
        'tpu_index': 0,
    }
    metrics = {
        'latency': {'mean_ms': mean_ms, 'p95_ms': mean_ms + 0.8, 'std_ms': 0.2},
        'throughput': {'fps': fps},
    }
    async with get_db() as db:
        await db.execute(
            """INSERT INTO experiments
               (id, name, device_id, model_name, model_path, script_path, params, status,
                created_at, completed_at, is_baseline)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                f'exp {experiment_id}',
                'dev_result',
                'model_a.tflite',
                'models/model_a.tflite',
                'benchmark_tflite.py',
                json.dumps(params),
                'completed',
                created_at,
                completed_at,
                is_baseline,
            ),
        )
        await db.execute(
            """INSERT INTO results
               (id, experiment_id, metrics, created_at)
               VALUES (?, ?, ?, ?)""",
            (
                f'res_{experiment_id}',
                experiment_id,
                json.dumps(metrics),
                completed_at,
            ),
        )
        await db.commit()


def test_report_and_get_result(client):
    experiment_id = 'exp_report_1'
    import asyncio

    asyncio.run(_insert_device_and_experiment(experiment_id))

    payload = {
        'experiment_id': experiment_id,
        'result': {
            'latency': {'mean_ms': 7.3, 'p95_ms': 8.1, 'std_ms': 0.3},
            'throughput': {'fps': 136.9},
            'system': {'cpu_percent_mean': 51.0, 'memory_mb_mean': 420.0},
            'logs': 'ok',
        },
    }
    accepted = client.post('/api/results/report', json=payload)
    assert accepted.status_code == 200
    assert accepted.json()['status'] == 'accepted'

    result = client.get(f'/api/results/{experiment_id}')
    assert result.status_code == 200
    body = result.json()
    assert body['experiment_id'] == experiment_id
    assert body['metrics']['throughput']['fps'] == 136.9


def test_duplicate_report_returns_409(client):
    experiment_id = 'exp_report_2'
    import asyncio

    asyncio.run(_insert_device_and_experiment(experiment_id))

    payload = {
        'experiment_id': experiment_id,
        'result': {'latency': {'mean_ms': 5.0}, 'throughput': {'fps': 200.0}},
    }
    first = client.post('/api/results/report', json=payload)
    assert first.status_code == 200

    second = client.post('/api/results/report', json=payload)
    assert second.status_code == 409


def test_compare_baseline_uses_same_backend_even_if_newer_exists(client):
    import asyncio

    # Insert only the device (helper also inserts one queued experiment we can ignore)
    asyncio.run(_insert_device_and_experiment('exp_seed_backend'))
    now = datetime.now(UTC).isoformat()

    # Current CPU experiment
    asyncio.run(
        _insert_completed_result_experiment(
            'exp_current_cpu',
            backend='cpu',
            mean_ms=9.0,
            fps=110.0,
            created_at=now,
            completed_at='2026-01-01T00:00:03+00:00',
        )
    )
    # Newer EdgeTPU experiment should NOT be selected as CPU baseline
    asyncio.run(
        _insert_completed_result_experiment(
            'exp_newer_edgetpu',
            backend='edgetpu',
            mean_ms=3.0,
            fps=320.0,
            created_at=now,
            completed_at='2026-01-01T00:00:02+00:00',
            is_baseline=1,
        )
    )
    # Older CPU baseline should be selected
    asyncio.run(
        _insert_completed_result_experiment(
            'exp_baseline_cpu',
            backend='cpu',
            mean_ms=8.0,
            fps=120.0,
            created_at=now,
            completed_at='2026-01-01T00:00:01+00:00',
            is_baseline=1,
        )
    )

    resp = client.get('/api/results/exp_current_cpu/compare-baseline')
    assert resp.status_code == 200
    body = resp.json()
    assert body['baseline']['experiment_id'] == 'exp_baseline_cpu'
    assert body['delta']['mean_ms'] == 1.0
    assert body['delta']['fps'] == -10.0


def test_set_baseline_only_within_same_backend(client):
    import asyncio

    asyncio.run(_insert_device_and_experiment('exp_seed_set_baseline'))
    now = datetime.now(UTC).isoformat()

    asyncio.run(
        _insert_completed_result_experiment(
            'exp_cpu_old',
            backend='cpu',
            mean_ms=10.0,
            fps=100.0,
            created_at=now,
            completed_at='2026-01-01T00:00:01+00:00',
            is_baseline=1,
        )
    )
    asyncio.run(
        _insert_completed_result_experiment(
            'exp_cpu_new',
            backend='cpu',
            mean_ms=9.5,
            fps=105.0,
            created_at=now,
            completed_at='2026-01-01T00:00:02+00:00',
        )
    )
    asyncio.run(
        _insert_completed_result_experiment(
            'exp_tpu_existing',
            backend='edgetpu',
            mean_ms=4.0,
            fps=280.0,
            created_at=now,
            completed_at='2026-01-01T00:00:03+00:00',
            is_baseline=1,
        )
    )

    set_resp = client.post('/api/experiments/exp_cpu_new/set-baseline')
    assert set_resp.status_code == 200

    async def _assert_flags():
        async with get_db() as db:
            cursor = await db.execute(
                'SELECT id, is_baseline FROM experiments WHERE id IN (?, ?, ?)',
                ('exp_cpu_old', 'exp_cpu_new', 'exp_tpu_existing'),
            )
            rows = {row['id']: row['is_baseline'] for row in await cursor.fetchall()}
        assert rows['exp_cpu_old'] == 0
        assert rows['exp_cpu_new'] == 1
        assert rows['exp_tpu_existing'] == 1

    asyncio.run(_assert_flags())
