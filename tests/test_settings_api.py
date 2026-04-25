def test_settings_roundtrip(client):
    initial = client.get('/api/settings')
    assert initial.status_code == 200
    assert isinstance(initial.json(), dict)

    updated = client.put('/api/settings', json={'max_tasks': 3, 'task_timeout': 120})
    assert updated.status_code == 200
    assert updated.json()['status'] == 'updated'

    one = client.get('/api/settings/max_tasks')
    assert one.status_code == 200
    assert one.json()['value'] == '3'


def test_set_single_setting(client):
    put_one = client.put('/api/settings/agent_timeout', json={'value': 45})
    assert put_one.status_code == 200
    assert put_one.json()['value'] == 45

    get_one = client.get('/api/settings/agent_timeout')
    assert get_one.status_code == 200
    assert get_one.json()['value'] == '45'
