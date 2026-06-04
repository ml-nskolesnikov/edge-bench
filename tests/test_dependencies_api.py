def test_check_deps_endpoint_correct_columns(client, monkeypatch):
    """Regression: old code used nonexistent columns (enabled, sort_order, check_cmd, critical).
    This test verifies the endpoint reaches the agent call without a 500 from a bad SQL query.
    The monkeypatched httpx returns 403 (agent not reachable), not 500 (bad SQL).
    """
    import asyncio
    from datetime import UTC, datetime

    from server.db.database import get_db

    async def _insert_device():
        now = datetime.now(UTC).isoformat()
        async with get_db() as db:
            await db.execute(
                'INSERT OR IGNORE INTO devices (id, name, ip, port, status, created_at) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                ('dev_dep_test', 'dep-device', '127.0.0.1', 9999, 'offline', now),
            )
            await db.commit()

    asyncio.run(_insert_device())

    # /check-deps is in scripts router; the client fixture routes /api/* already
    resp = client.post('/api/scripts/check-deps', json={'device_id': 'dev_dep_test'})
    # 200 (empty result) or 500 only if SQL blows up — not 500 means columns are correct
    assert resp.status_code == 200, f'Expected 200, got {resp.status_code}: {resp.text}'
    body = resp.json()
    assert 'dependencies' in body
    # Each result must use is_required, not the old 'critical' key
    for dep in body['dependencies']:
        assert 'is_required' in dep
        assert 'critical' not in dep


def test_list_dependencies_has_defaults(client):
    resp = client.get('/api/dependencies')
    assert resp.status_code == 200
    deps = resp.json()
    assert isinstance(deps, list)
    assert len(deps) >= 1


def test_create_update_delete_dependency(client):
    created = client.post(
        '/api/dependencies',
        json={
            'name': 'Custom Tool',
            'package': 'custom-tool',
            'check_command': 'python3 -c "print(1)"',
            'is_required': False,
        },
    )
    assert created.status_code == 200
    dep = created.json()
    dep_id = dep['id']
    assert dep['name'] == 'Custom Tool'

    updated = client.put(
        f'/api/dependencies/{dep_id}',
        json={'version': '1.2.3', 'is_required': True},
    )
    assert updated.status_code == 200
    assert updated.json()['version'] == '1.2.3'
    assert updated.json()['is_required'] == 1

    deleted = client.delete(f'/api/dependencies/{dep_id}')
    assert deleted.status_code == 200
    assert deleted.json()['status'] == 'deleted'
