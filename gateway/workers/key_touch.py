"""Batched `api_keys.last_used_at` updates (spec §6.1: never on the hot path).

Authn calls `KeyToucher.touch(key_id)`, which only adds to an in-memory set. Every
few seconds the set is swapped out and flushed in a single UPDATE. Losing the
pending set on a crash just makes `last_used_at` a few seconds stale.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

from sqlalchemy import func, update

from gateway.db.models import ApiKey
from gateway.db.session import Database

log = logging.getLogger(__name__)


class KeyToucher:
    def __init__(self, db: Database, interval_s: float = 5.0) -> None:
        self._db = db
        self._interval = interval_s
        self._pending: set[uuid.UUID] = set()

    def touch(self, key_id: uuid.UUID) -> None:
        self._pending.add(key_id)

    async def flush(self) -> int:
        if not self._pending:
            return 0
        batch, self._pending = self._pending, set()
        # Keys from many tenants in one statement → system role.
        async with self._db.system_session() as s:
            await s.execute(update(ApiKey).where(ApiKey.id.in_(batch)).values(last_used_at=func.now()))
        return len(batch)

    async def run(self) -> None:
        while True:
            await asyncio.sleep(self._interval)
            try:
                await self.flush()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("last_used_at flush failed")
