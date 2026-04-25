def test_upload_and_delete_file(client):
    content = b'test-model-content'
    uploaded = client.post(
        '/api/files/upload',
        files={'file': ('model_a.tflite', content, 'application/octet-stream')},
    )
    assert uploaded.status_code == 200
    body = uploaded.json()
    file_id = body['id']
    assert body['duplicate'] is False
    assert body['name'] == 'model_a.tflite'

    listed = client.get('/api/files')
    assert listed.status_code == 200
    assert len(listed.json()) == 1

    deleted = client.delete(f'/api/files/{file_id}')
    assert deleted.status_code == 200
    assert deleted.json()['status'] == 'deleted'


def test_upload_duplicate_by_hash(client):
    content = b'same-bytes'
    first = client.post(
        '/api/files/upload',
        files={'file': ('dup_1.tflite', content, 'application/octet-stream')},
    )
    assert first.status_code == 200
    assert first.json()['duplicate'] is False

    second = client.post(
        '/api/files/upload',
        files={'file': ('dup_2.tflite', content, 'application/octet-stream')},
    )
    assert second.status_code == 200
    assert second.json()['duplicate'] is True
