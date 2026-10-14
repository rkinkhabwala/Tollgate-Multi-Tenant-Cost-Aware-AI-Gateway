"""Shared test fixtures.

Unit tests need nothing running. Tests marked `stack` (isolation + integration) need
`docker compose up` and FAIL (never skip) when it is unreachable: the isolation suite
must never be silently skipped (spec §17).
"""

from __future__ import annotations

import os
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass
from decimal import Decimal

import httpx
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


@pytest.fixture(scope="session")
async def gateway() -> AsyncIterator[httpx.AsyncClient]:
    """HTTP client for the running gateway (docker compose)."""
    async with httpx.AsyncClient(base_url=BASE_URL, timeout=httpx.Timeout(300, connect=3)) as client:
        try:
            (await client.get("/healthz")).raise_for_status()
        except Exception as exc:  # pragma: no cover - environment problem
            pytest.fail(f"Gateway unreachable at {BASE_URL} ({type(exc).__name__}). Run `docker compose up -d`.", pytrace=False)
        yield client


async def login(client: httpx.AsyncClient, email: str, password: str) -> dict[str, str]:
    r = await client.post("/admin/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
async def platform_admin(gateway: httpx.AsyncClient) -> dict[str, str]:
    return await login(gateway, ADMIN_EMAIL, ADMIN_PASSWORD)


@dataclass
class Provisioned:
    tenant_id: str
    app_id: str
    app2_id: str
    key: str
    key_id: str
    key2: str  # key for the tenant's second app
    admin_headers: dict[str, str]  # tenant_admin of this tenant


async def provision_tenant(gateway: httpx.AsyncClient, platform_admin: dict[str, str], label: str) -> Provisioned:
    """Create tenant → tenant_admin → 2 apps → 1 key each, all through the admin API."""
    suffix = uuid.uuid4().hex[:8]
    r = await gateway.post("/admin/tenants", json={"name": f"{label}-{suffix}"}, headers=platform_admin)
    assert r.status_code == 201, r.text
    tid = r.json()["id"]
    email, pw = f"admin-{suffix}@{label}.test", "tenant-password-123"
    r = await gateway.post(f"/admin/tenants/{tid}/users", json={"email": email, "password": pw, "role": "tenant_admin"}, headers=platform_admin)
    assert r.status_code == 201, r.text
    headers = await login(gateway, email, pw)
    apps, keys = [], []
    for name in ("app-one", "app-two"):
        r = await gateway.post(
            f"/admin/tenants/{tid}/apps", json={"name": name, "baseline_model": "anthropic/claude-sonnet-5-5"}, headers=headers
        )
        assert r.status_code == 201, r.text
        apps.append(r.json()["id"])
        r = await gateway.post(f"/admin/apps/{apps[-1]}/keys", json={}, headers=headers)
        assert r.status_code == 201, r.text
        keys.append(r.json())
    return Provisioned(tid, apps[0], apps[1], keys[0]["key"], keys[0]["id"], keys[1]["key"], headers)


async def insert_usage(db: Database, *, tenant_id: str, app_id: str, key_id: str, n: int, baseline: str) -> None:
    """Write synthetic ledger rows directly (system role) — fast, deterministic usage data."""
    async with db.system_session() as s:
        for i in range(n):
            await s.execute(
                text(
                    "INSERT INTO usage_events (tenant_id, app_id, api_key_id, request_id, routed_model, cache_status, "
                    "input_tokens, output_tokens, actual_cost_usd, baseline_cost_usd, status_code) "
                    "VALUES (:t, :a, :k, :r, 'ollama/qwen2.5:7b', 'miss', 100, 10, 0, :b, 200)"
                ),
                {"t": tenant_id, "a": app_id, "k": key_id, "r": f"synthetic-{uuid.uuid4().hex}", "b": Decimal(baseline)},
            )


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
