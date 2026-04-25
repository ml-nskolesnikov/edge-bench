import asyncio
import json

from server.core.ws_manager import WebSocketManager


class FakeWebSocket:
    def __init__(self, fail_send: bool = False):
        self.accepted = False
        self.messages = []
        self.fail_send = fail_send

    async def accept(self):
        self.accepted = True

    async def send_text(self, message: str):
        if self.fail_send:
            raise RuntimeError('send failed')
        self.messages.append(message)


def test_connect_broadcast_disconnect():
    manager = WebSocketManager()
    ws = FakeWebSocket()

    asyncio.run(manager.connect('exp_1', ws))
    assert ws.accepted is True
    assert manager.has_clients('exp_1') is True

    asyncio.run(manager.broadcast('exp_1', {'type': 'status', 'status': 'running'}))
    assert len(ws.messages) == 1
    payload = json.loads(ws.messages[0])
    assert payload['status'] == 'running'

    manager.disconnect('exp_1', ws)
    assert manager.has_clients('exp_1') is False


def test_broadcast_removes_dead_socket():
    manager = WebSocketManager()
    ws = FakeWebSocket(fail_send=True)

    asyncio.run(manager.connect('exp_2', ws))
    asyncio.run(manager.broadcast('exp_2', {'type': 'metric', 'run': 1}))

    assert manager.has_clients('exp_2') is False
