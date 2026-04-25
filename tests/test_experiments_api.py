from datetime import UTC, datetime

from server.db.database import get_db


async def _insert_device(device_id: str):
    now = datetime.now(UTC).isoformat()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO devices
               (id, name, ip, port, status, description, last_seen, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (device_id, 'bench-device', '127.0.0.1', 8001, 'online', None, now, now),
        )
        await db.commit()


def test_create_experiment_queues_task(client, monkeypatch):
    device_id = 'dev_test_exp'
    import asyncio

    asyncio.run(_insert_device(device_id))

    queued = []

    async def fake_add_task(experiment_id: str, priority: bool = False):
        queued.append((experiment_id, priority))

    monkeypatch.setattr('server.api.experiments.task_queue.add_task', fake_add_task)

    payload = {
        'name': 'cpu benchmark',
        'device_id': device_id,
        'model_path': 'models/model_int8.tflite',
        'script_path': 'benchmark_tflite.py',
        'params': {
            'backend': 'cpu',
            'batch_size': 1,
            'num_threads': 2,
            'warmup_runs': 2,
            'benchmark_runs': 3,
            'timeout_seconds': 60,
            'tpu_index': 0,
        },
        'description': 'manual test',
    }
    resp = client.post('/api/experiments', json=payload)
    assert resp.status_code == 200
    body = resp.json()
    assert body['status'] == 'queued'
    assert body['model_name'] == 'model_int8.tflite'
    assert len(queued) == 1
    assert queued[0][0] == body['id']

    get_one = client.get(f'/api/experiments/{body["id"]}')
    assert get_one.status_code == 200
    assert get_one.json()['id'] == body['id']


def test_cancel_experiment(client, monkeypatch):
    device_id = 'dev_cancel_exp'
    import asyncio

    asyncio.run(_insert_device(device_id))

    async def no_op_add_task(experiment_id: str, priority: bool = False):
        return None

    monkeypatch.setattr('server.api.experiments.task_queue.add_task', no_op_add_task)

    create = client.post(
        '/api/experiments',
        json={
            'name': 'cancel me',
            'device_id': device_id,
            'model_path': 'models/model_int8.tflite',
            'params': {'backend': 'cpu'},
        },
    )
    exp_id = create.json()['id']

    cancel = client.post(f'/api/experiments/{exp_id}/cancel')
    assert cancel.status_code == 200
    assert cancel.json()['status'] == 'cancelled'
