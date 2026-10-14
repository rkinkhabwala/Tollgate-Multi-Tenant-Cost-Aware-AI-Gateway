"""Database engines and RLS-aware session helpers.

Two runtime engines, one per Postgres role (see `docker/postgres/01-roles.sh`):

* `tenant_session(tenant_id)` - connects as `tollgate_app` (NOBYPASSRLS) and sets
  `app.tenant_id` for the transaction with `set_config(..., is_local => true)`, so the
  setting can never leak to the next user of a pooled connection. Every RLS policy
  compares against it. This is the default for anything tenant-scoped.
* `system_session()` - connects as `tollgate_system` (BYPASSRLS). Reserved for the
  few places that must run before a tenant is known or across tenants: API-key
  lookup by prefix, admin login, the usage-event batch writer, platform-admin reads.
"""

from __future__ import annotations

import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession, async_sessionmaker, create_async_engine

from gateway.settings import get_settings


class Database:
    """Holds both engines; created once in the app lifespan."""

    def __init__(self, app_url: str, system_url: str, pool_size: int = 10) -> None:
        self.app_engine: AsyncEngine = create_async_engine(app_url, pool_size=pool_size, pool_pre_ping=True)
        self.system_engine: AsyncEngine = create_async_engine(system_url, pool_size=pool_size // 2 or 1, pool_pre_ping=True)
        self._app_sessions = async_sessionmaker(self.app_engine, expire_on_commit=False)
        self._system_sessions = async_sessionmaker(self.system_engine, expire_on_commit=False)

    @asynccontextmanager
    async def tenant_session(self, tenant_id: uuid.UUID | str) -> AsyncIterator[AsyncSession]:
        """A transaction as `tollgate_app` with `app.tenant_id` set (RLS enforced).

        Commits on success, rolls back on exception.
        """
        async with self._app_sessions() as session, session.begin():
            await session.execute(
                text("SELECT set_config('app.tenant_id', :tid, true)"), {"tid": str(uuid.UUID(str(tenant_id)))}
            )
            yield session

    @asynccontextmanager
    async def system_session(self) -> AsyncIterator[AsyncSession]:
        """A transaction as `tollgate_system` (bypasses RLS). Use sparingly."""
        async with self._system_sessions() as session, session.begin():
            yield session

    async def dispose(self) -> None:
        await self.app_engine.dispose()
        await self.system_engine.dispose()


def create_database() -> Database:
    s = get_settings()
    return Database(s.database_url_app, s.database_url_system)
