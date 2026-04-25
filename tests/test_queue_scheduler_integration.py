import asyncio
from datetime import UTC, datetime
import json

from server.core.queue import task_queue
from server.core.scheduler import run_scheduled_experiment
from server.db.database import get_db


async def _seed_queue_fixture(experiment_id: str):
    now = datetime.now(UTC).isoformat()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO devices
               (id, name, ip, port, status, description, last_seen, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ('dev_queue', 'queue-device', '127.0.0.1', 8001, 'online', None, now, now),
        )
        await db.execute(
            """INSERT INTO experiments
               (id, name, device_id, model_name, model_path, script_path, params, status, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                experiment_id,
                'queue-exp',
                'dev_queue',
                'model_q.tflite',
                'models/model_q.tflite',
                'benchmark_tflite.py',
                json.dumps({'backend': 'cpu', 'benchmark_runs': 2}),
                'queued',
                now,
            ),
        )
        await db.commit()


def test_task_queue_execute_experiment_success(monkeypatch):
    experiment_id = 'exp_queue_success'
    asyncio.run(_seed_queue_fixture(experiment_id))

    async def fake_health(agent_url: str):
        return True, ''

    async def fake_run(agent_url: str, experiment: dict, stream_callback_url: str | None = None):
        return {
            'latency': {'mean_ms': 6.1, 'p95_ms': 7.0, 'std_ms': 0.2},
            'throughput': {'fps': 160.0},
            'system': {'cpu_percent_mean': 40.0, 'memory_mb_mean': 320.0},
            'logs': 'done',
        }

    async def fake_log(experiment_id: str, result: dict):
        return None

    monkeypatch.setattr(task_queue, '_check_device_health', fake_health)
    monkeypatch.setattr(task_queue, '_run_on_agent', fake_run)
    monkeypatch.setattr(task_queue, '_log_to_integrations', fake_log)

    ok = asyncio.run(task_queue._execute_experiment(experiment_id))
    assert ok is True

    async def _assert_saved():
        async with get_db() as db:
            cursor = await db.execute(
                'SELECT status FROM experiments WHERE id = ?',
                (experiment_id,),
            )
            exp = await cursor.fetchone()
            cursor = await db.execute(
                'SELECT metrics FROM results WHERE experiment_id = ?',
                (experiment_id,),
            )
            res = await cursor.fetchone()
        assert exp['status'] == 'completed'
        assert json.loads(res['metrics'])['throughput']['fps'] == 160.0

    asyncio.run(_assert_saved())


async def _seed_scheduler_fixture(schedule_id: str):
    now = datetime.now(UTC).isoformat()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO devices
               (id, name, ip, port, status, description, last_seen, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            ('dev_sched_integration', 'sched-device', '127.0.0.1', 8001, 'online', None, now, now),
        )
        await db.execute(
            """INSERT INTO schedules
               (id, name, device_id, model_name, backend, cron, enabled, params, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                schedule_id,
                'nightly-integration',
                'dev_sched_integration',
                'model_sched.tflite',
                'cpu',
                '0 2 * * *',
                1,
                json.dumps({'benchmark_runs': 3}),
                now,
            ),
        )
        await db.commit()


def test_scheduler_run_creates_experiment_and_queues(monkeypatch):
    schedule_id = 'sched_it_1'
    asyncio.run(_seed_scheduler_fixture(schedule_id))

    queued_ids = []

    async def fake_add_task(experiment_id: str, priority: bool = False):
        queued_ids.append(experiment_id)

    monkeypatch.setattr('server.core.queue.task_queue.add_task', fake_add_task)

    asyncio.run(run_scheduled_experiment(schedule_id))

    async def _assert_state():
        async with get_db() as db:
            cursor = await db.execute(
                'SELECT last_exp_id FROM schedules WHERE id = ?',
                (schedule_id,),
            )
            row = await cursor.fetchone()
            assert row['last_exp_id'] is not None

            cursor = await db.execute(
                'SELECT status, name FROM experiments WHERE id = ?',
                (row['last_exp_id'],),
            )
            exp = await cursor.fetchone()
            assert exp['status'] == 'queued'
            assert exp['name'].startswith('[Scheduled] ')

    asyncio.run(_assert_state())
    assert len(queued_ids) == 1
