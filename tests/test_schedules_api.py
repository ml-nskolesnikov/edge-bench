from datetime import UTC, datetime

from server.db.database import get_db


async def _insert_device(device_id: str):
    now = datetime.now(UTC).isoformat()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO devices
               (id, name, ip, port, status, description, last_seen, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (device_id, 'schedule-device', '127.0.0.1', 8001, 'online', None, now, now),
        )
        await db.commit()


def test_create_and_update_schedule(client, monkeypatch):
    device_id = 'dev_sched_1'
    import asyncio

    asyncio.run(_insert_device(device_id))

    jobs = {}

    def fake_add_job(func, trigger, args, id, replace_existing):
        jobs[id] = {'args': args, 'replace_existing': replace_existing}

    def fake_remove_job(schedule_id):
        jobs.pop(schedule_id, None)

    monkeypatch.setattr('server.api.schedules.scheduler.add_job', fake_add_job)
    monkeypatch.setattr('server.api.schedules.scheduler.remove_job', fake_remove_job)

    create = client.post(
        '/api/schedules',
        json={
            'name': 'night bench',
            'device_id': device_id,
            'model_name': 'model_a.tflite',
            'backend': 'cpu',
            'cron': '0 2 * * *',
            'params': {'benchmark_runs': 5},
        },
    )
    assert create.status_code == 201
    created = create.json()
    assert created['name'] == 'night bench'
    assert created['cron_human'] == 'Every day at 02:00 UTC'
    assert created['id'] in jobs

    updated = client.patch(
        f'/api/schedules/{created["id"]}',
        json={'enabled': False},
    )
    assert updated.status_code == 200
    assert updated.json()['enabled'] == 0
    assert created['id'] not in jobs


def test_invalid_cron_rejected(client):
    bad = client.post(
        '/api/schedules',
        json={
            'name': 'bad cron',
            'device_id': 'does-not-matter',
            'model_name': 'm.tflite',
            'backend': 'cpu',
            'cron': 'bad cron',
        },
    )
    assert bad.status_code == 422
