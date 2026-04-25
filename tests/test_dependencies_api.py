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
