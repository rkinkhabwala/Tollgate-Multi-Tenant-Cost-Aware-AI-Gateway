"""Shared test fixtures.

Unit tests need nothing running. Tests marked `stack` (isolation + integration) need
`docker compose up` and FAIL (never skip) when it is unreachable: the isolation suite
must never be silently skipped (spec §17).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal

import pytest
from sqlalchemy import text

from gateway.db.session import Database
from gateway.settings import get_settings

BASE_URL = os.environ.get("TOLLGATE_TEST_BASE_URL", "http://localhost:8000")
ADMIN_EMAIL = os.environ.get("TOLLGATE_BOOTSTRAP_ADMIN_EMAIL", "admin@tollgate.local")
ADMIN_PASSWORD = os.environ.get("TOLLGATE_BOOTSTRAP_ADMIN_PASSWORD", "admin-dev-password")


@pytest.fixture(scope="session")
async def db() -> AsyncIterator[Database]:
    s = get_settings()
    database = Database(s.database_url_app, s.database_url_system, pool_size=4)
    try:
        async with database.system_session() as session:
            await session.execute(text("SELECT 1"))
    except Exception as exc:  # pragma: no cover - environment problem
        pytest.fail(f"Postgres unreachable ({type(exc).__name__}). Run `docker compose up -d` first.", pytrace=False)
    yield database
    await database.dispose()


async def make_tenant_with_app(db: Database, label: str) -> dict[str, uuid.UUID]:
    """Insert a throwaway tenant + app + key row directly (system role)."""
    suffix = uuid.uuid4().hex[:8]
    async with db.system_session() as s:
        tid = (await s.execute(text("INSERT INTO tenants (name) VALUES (:n) RETURNING id"), {"n": f"test-{label}-{suffix}"})).scalar_one()
        aid = (
            await s.execute(
                text("INSERT INTO apps (tenant_id, name, baseline_model) VALUES (:t, :n, 'anthropic/claude-sonnet-5-5') RETURNING id"),
                {"t": tid, "n": f"app-{suffix}"},
            )
        ).scalar_one()
        kid = (
            await s.execute(
                text(
                    "INSERT INTO api_keys (tenant_id, app_id, prefix, secret_hash, scopes) "
                    "VALUES (:t, :a, :p, :h, ARRAY['chat:write']) RETURNING id"
                ),
                {"t": tid, "a": aid, "p": f"x{suffix[:7]}", "h": os.urandom(32)},
            )
        ).scalar_one()
        await s.execute(
            text(
                "INSERT INTO usage_events (tenant_id, app_id, api_key_id, request_id, cache_status, "
                "actual_cost_usd, baseline_cost_usd, status_code) VALUES (:t, :a, :k, :r, 'miss', 0, :b, 200)"
            ),
            {"t": tid, "a": aid, "k": kid, "r": f"req-{suffix}", "b": Decimal("0.01")},
        )
    return {"tenant_id": tid, "app_id": aid, "key_id": kid}
