"""
WebSocket connection manager.

Each connected client subscribes to events for a specific user (their
connectors only). The sync service calls broadcast() whenever something
changes; the manager fans it out to all subscribed sockets.
"""
from __future__ import annotations

import asyncio
import json
import logging
from collections import defaultdict
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)


class ConnectionManager:
    def __init__(self):
        self._connections: dict[int, set[WebSocket]] = defaultdict(set)
        self._lock = asyncio.Lock()

    async def connect(self, user_id: int, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections[user_id].add(ws)
        logger.info("WS connected: user=%s total=%d", user_id, len(self._connections[user_id]))

    async def disconnect(self, user_id: int, ws: WebSocket) -> None:
        async with self._lock:
            self._connections.get(user_id, set()).discard(ws)
        logger.info("WS disconnected: user=%s", user_id)

    async def send_to_user(self, user_id: int, message: dict[str, Any]) -> None:
        payload = json.dumps(message, default=str)
        dead: list[WebSocket] = []
        async with self._lock:
            sockets = list(self._connections.get(user_id, set()))
        for ws in sockets:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        if dead:
            async with self._lock:
                for ws in dead:
                    self._connections.get(user_id, set()).discard(ws)

    async def broadcast(self, message: dict[str, Any]) -> None:
        async with self._lock:
            user_ids = list(self._connections.keys())
        for uid in user_ids:
            await self.send_to_user(uid, message)


manager = ConnectionManager()
