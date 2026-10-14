"""Usage batch writer: Redis Stream `sys:usage_events` → Postgres `usage_events` (spec §11).

The meter stage `XADD`s one entry per request, keeping Postgres off the hot path.
This worker:
1. Reads up to `batch_size` entries via a consumer group, blocking for at most
   `flush_interval_s`. Inserts therefore happen about once per second, or sooner
   under load.
2. Inserts the whole batch in one statement, as the system role. A batch spans
   tenants, and the rows carry their own `tenant_id`.
3. `XACK`s and `XDEL`s the entries only after the insert commits.

Failures leave entries pending. On startup, and every loop, the worker reclaims
entries idle for more than 30 s with `XAUTOCLAIM`, so a crash or a Postgres outage
loses nothing. Inserts are idempotent, so redelivery is safe: the `(ts, id)` primary
key comes from the event, and the insert uses `ON CONFLICT DO NOTHING`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import socket
import uuid
from datetime import datetime
from decimal import Decimal
from typing import Any

from redis.asyncio import Redis
from redis.exceptions import ResponseError
from sqlalchemy.dialects.postgresql import insert

from gateway.db.models import UsageEvent
from gateway.db.session import Database

log = logging.getLogger(__name__)

RECLAIM_IDLE_MS = 30_000


def _s(v: Any) -> str:
    return v.decode() if isinstance(v, bytes) else str(v)


def event_to_row(fields: dict[Any, Any]) -> dict[str, Any]:
    f = {_s(k): _s(v) for k, v in fields.items()}
    return {
        "id": uuid.UUID(f["id"]),
        "ts": datetime.fromisoformat(f["ts"]),
        "tenant_id": uuid.UUID(f["tenant_id"]),
        "app_id": uuid.UUID(f["app_id"]),
        "api_key_id": uuid.UUID(f["api_key_id"]),
        "request_id": f["request_id"],
        "requested_model": f.get("requested_model") or None,
        "routed_model": f.get("routed_model") or None,
        "route_reason": f.get("route_reason") or None,
        "escalations": int(f.get("escalations", 0)),
        "cache_status": f.get("cache_status", "miss"),
        "input_tokens": int(f.get("input_tokens", 0)),
        "output_tokens": int(f.get("output_tokens", 0)),
        "actual_cost_usd": Decimal(f["actual_cost_usd"]),
        "baseline_cost_usd": Decimal(f["baseline_cost_usd"]),
        "pii_entities": json.loads(f["pii_entities"]) if f.get("pii_entities") else None,
        "latency_ms": int(f["latency_ms"]) if f.get("latency_ms") else None,
        "gateway_ms": int(f["gateway_ms"]) if f.get("gateway_ms") else None,
        "status_code": int(f["status_code"]),
        "attempts": json.loads(f["attempts"]) if f.get("attempts") else None,
    }


class UsageWriter:
    def __init__(
        self, redis: Redis, db: Database, *, stream: str, group: str, batch_size: int = 500, flush_interval_s: float = 1.0
    ) -> None:
        self._redis = redis
        self._db = db
        self._stream = stream
        self._group = group
        self._consumer = f"{socket.gethostname()}-{uuid.uuid4().hex[:6]}"
        self._batch = batch_size
        self._block_ms = int(flush_interval_s * 1000)

    async def ensure_group(self) -> None:
        try:
            await self._redis.xgroup_create(self._stream, self._group, id="0", mkstream=True)
        except ResponseError as exc:
            if "BUSYGROUP" not in str(exc):
                raise

    async def _write(self, entries: list[tuple[Any, dict[Any, Any]]]) -> int:
        if not entries:
            return 0
        rows, ids, bad = [], [], []
        for entry_id, fields in entries:
            try:
                rows.append(event_to_row(fields))
                ids.append(entry_id)
            except (KeyError, ValueError):
                bad.append(entry_id)  # poison message: drop rather than block the stream
        if rows:
            async with self._db.system_session() as s:
                await s.execute(insert(UsageEvent).values(rows).on_conflict_do_nothing(index_elements=["ts", "id"]))
        done = ids + bad
        if bad:
            log.error("dropped malformed usage events", extra={"count": len(bad)})
        await self._redis.xack(self._stream, self._group, *done)
        await self._redis.xdel(self._stream, *done)
        return len(rows)

    async def run_once(self) -> int:
        """Reclaim stale pending entries, then read and write one batch. Returns rows written."""
        written = 0
        _, claimed, *_ = await self._redis.xautoclaim(
            self._stream, self._group, self._consumer, min_idle_time=RECLAIM_IDLE_MS, start_id="0-0", count=self._batch
        )
        written += await self._write(claimed)
        resp = await self._redis.xreadgroup(
            self._group, self._consumer, {self._stream: ">"}, count=self._batch, block=self._block_ms
        )
        for _stream, entries in resp or []:
            written += await self._write(entries)
        return written

    async def run(self) -> None:
        await self.ensure_group()
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("usage writer iteration failed; will retry")
                await asyncio.sleep(1.0)
