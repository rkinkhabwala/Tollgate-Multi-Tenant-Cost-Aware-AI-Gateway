"""Periodic maintenance: future `usage_events` partitions and the `usage_daily` rollup.

Every `interval_s` (default 5 minutes, spec §5):
* ensures monthly partitions exist for this month and the next two, via the
  SECURITY DEFINER `ensure_usage_partition()`, so rows never pile up in DEFAULT;
* runs `REFRESH MATERIALIZED VIEW CONCURRENTLY usage_daily` via
  `refresh_usage_daily()`.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import UTC, date, datetime

from sqlalchemy import text

from gateway.db.session import Database

log = logging.getLogger(__name__)


def _month(d: date, offset: int) -> date:
    y, m = divmod(d.month - 1 + offset, 12)
    return date(d.year + y, m + 1, 1)


class RollupWorker:
    def __init__(self, db: Database, interval_s: float = 300.0) -> None:
        self._db = db
        self._interval = interval_s

    async def run_once(self) -> None:
        today = datetime.now(UTC).date()
        async with self._db.system_session() as s:
            for offset in range(3):
                await s.execute(text("SELECT ensure_usage_partition(:m)"), {"m": _month(today, offset)})
            await s.execute(text("SELECT refresh_usage_daily()"))

    async def run(self) -> None:
        while True:
            try:
                await self.run_once()
            except asyncio.CancelledError:
                raise
            except Exception:
                log.exception("rollup iteration failed")
            await asyncio.sleep(self._interval)
