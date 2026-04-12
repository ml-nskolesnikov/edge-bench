"""
WebSocket Connection Manager for Real-time Experiment Updates
"""

import json
from typing import Any

from fastapi import WebSocket


class WebSocketManager:
    """Manages WebSocket connections per experiment."""

    def __init__(self):
        # experiment_id -> list of active websockets
        self._connections: dict[str, list[WebSocket]] = {}

    async def connect(self, experiment_id: str, ws: WebSocket):
        """Accept and register a new WebSocket connection."""
        await ws.accept()
        self._connections.setdefault(experiment_id, []).append(ws)

    def disconnect(self, experiment_id: str, ws: WebSocket):
        """Remove a WebSocket connection."""
        conns = self._connections.get(experiment_id, [])
        if ws in conns:
            conns.remove(ws)
        if not conns:
            self._connections.pop(experiment_id, None)

    async def broadcast(self, experiment_id: str, message: dict[str, Any]):
        """Send a JSON message to all clients watching this experiment."""
        conns = list(self._connections.get(experiment_id, []))
        dead = []
        for ws in conns:
            try:
                await ws.send_text(json.dumps(message))
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(experiment_id, ws)

    def has_clients(self, experiment_id: str) -> bool:
        """Return True if any client is watching this experiment."""
        return bool(self._connections.get(experiment_id))


# Global singleton
ws_manager = WebSocketManager()
