from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager


class A2ATaskExecutionCoordinator:
    """Serializes initial execution binding and cancellation for one local task."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}
        self._initialized: dict[str, asyncio.Event] = {}
        self._guard = asyncio.Lock()

    @asynccontextmanager
    async def hold(self, task_id: str) -> AsyncIterator[None]:
        async with self._guard:
            lock = self._locks.setdefault(task_id, asyncio.Lock())
        async with lock:
            yield

    async def mark_initialized(self, task_id: str) -> None:
        async with self._guard:
            event = self._initialized.setdefault(task_id, asyncio.Event())
            event.set()

    async def wait_initialized(self, task_id: str) -> None:
        async with self._guard:
            event = self._initialized.setdefault(task_id, asyncio.Event())
        await event.wait()
