from datetime import UTC, datetime

from server.db.database import get_db


class _OfflineClient:
    def __init__(self, *args, **kwargs):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        return False

    async def get(self, url: str):
        raise RuntimeError('offline')


async def _seed_device(device_id: str, name: str):
    now = datetime.now(UTC).isoformat()
    async with get_db() as db:
        await db.execute(
            """INSERT INTO devices
               (id, name, ip, port, status, description, last_seen, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (device_id, name, '127.0.0.1', 8001, 'offline', None, None, now),
        )
        await db.commit()


def test_create_and_list_devices(client, monkeypatch):
    monkeypatch.setattr('server.api.devices.httpx.AsyncClient', _OfflineClient)

    payload = {
        'name': 'rpi-test-1',
        'ip': '10.10.10.10',
        'port': 8001,
        'description': 'manual backend test',
    }
    created = client.post('/api/devices', json=payload)

    assert created.status_code == 200
    body = created.json()
    assert body['name'] == 'rpi-test-1'
    assert body['status'] == 'offline'

    listed = client.get('/api/devices')
    assert listed.status_code == 200
    assert len(listed.json()) == 1


def test_duplicate_device_name_returns_400(client, monkeypatch):
    monkeypatch.setattr('server.api.devices.httpx.AsyncClient', _OfflineClient)

    payload = {'name': 'dup-device', 'ip': '1.1.1.1', 'port': 8001}
    first = client.post('/api/devices', json=payload)
    assert first.status_code == 200

    second = client.post('/api/devices', json=payload)
    assert second.status_code == 400
    assert second.json()['detail'] == 'Device name already exists'
