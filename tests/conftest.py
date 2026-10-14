"""Shared test fixtures.

Unit tests need nothing running. Tests that use the stack fixtures (isolation,
integration, leak) need Postgres and Redis from `docker compose up -d`, plus a native
Ollama for the real-model tests. They FAIL, never skip, when it is unreachable:
the isolation suite must never be silently skipped (spec §17).

**Test data never touches dev data.** Each pytest session:
* creates a throwaway database `tollgate_test_<rand>` on the compose Postgres,
  migrates it with Alembic, and drops it (`WITH (FORCE)`) at the end;
* uses Redis DB 15 (dev uses DB 0), flushed before and after;
* runs its own gateway: a real uvicorn server, in-process on a free port, configured
  for the test DB and Redis and logging to a per-session file that the leak test
  scans;
* checks at teardown that row counts in the dev database did not change.

Fixtures that provision tenants purge them again (`TenantFactory`), so even within a
session, modules don't see each other's leftovers.
"""

from __future__ import annotations

import asyncio
import os
import socket
import tempfile
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from decimal import Decimal
from pathlib import Path
from urllib.parse import urlsplit

import asyncpg
import httpx
import pytest
from alembic import command
from alembic.config import Config as AlembicConfig
from redis.asyncio import Redis
from sqlalchemy import text
from sqlalchemy.engine import make_url

from gateway.db.session import Database
from gateway.redaction.engine import RedactionEngine
from gateway.redis_keys import auth_cache_key
from gateway.settings import REPO_ROOT, Settings, get_settings

ADMIN_EMAIL = "admin@tollgate.local"
ADMIN_PASSWORD = "admin-test-password"
TEST_REDIS_DB = 15
DEV_GUARD_TABLES = ("tenants", "apps", "api_keys", "usage_events", "audit_log", "admin_users")


def _with_db(url: str, name: str) -> str:
    return make_url(url).set(database=name).render_as_string(hide_password=False)


def _asyncpg_dsn(url: str) -> str:
    return make_url(url).set(drivername="postgresql").render_as_string(hide_password=False)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


async def _dev_counts(dev: Settings) -> dict[str, int]:
    conn = await asyncpg.connect(_asyncpg_dsn(dev.database_url_owner))
    try:
        return {t: await conn.fetchval(f"SELECT count(*) FROM {t}") for t in DEV_GUARD_TABLES}
    finally:
        await conn.close()


# --- redaction engine (no infrastructure needed) ------------------------------------------
@pytest.fixture(scope="session")
def engine() -> RedactionEngine:
    """Presidio + spaCy en_core_web_lg, loaded once per session (≈ 1–7 s)."""
    eng = RedactionEngine(workers=2)
    yield eng
    eng.close()


# --- session infrastructure ------------------------------------------------------
@pytest.fixture(scope="session")
async def test_settings() -> AsyncIterator[Settings]:
    """Create + migrate a throwaway database; yield Settings pointing at it; drop it."""
    dev = get_settings()
    name = f"tollgate_test_{uuid.uuid4().hex[:10]}"
    try:
        dev_before = await _dev_counts(dev)
        admin = await asyncpg.connect(_asyncpg_dsn(dev.database_url_owner))
    except Exception as exc:  # pragma: no cover - environment problem
        pytest.fail(f"Postgres unreachable ({type(exc).__name__}). Run `docker compose up -d` first.", pytrace=False)
    try:
        await admin.execute(f'CREATE DATABASE "{name}"')
        await admin.execute(f'GRANT CONNECT ON DATABASE "{name}" TO tollgate_app, tollgate_system')
    finally:
        await admin.close()

    cfg = AlembicConfig()  # no ini file → alembic won't reconfigure logging
    cfg.set_main_option("script_location", str(REPO_ROOT / "gateway" / "db" / "migrations"))
    cfg.set_main_option("sqlalchemy.url", _with_db(dev.database_url_owner, name).replace("%", "%%"))
    await asyncio.to_thread(command.upgrade, cfg, "head")
    # Schema-level grants live in the initdb script for the dev DB; repeat them here.
    conn = await asyncpg.connect(_asyncpg_dsn(_with_db(dev.database_url_owner, name)))
    try:
        await conn.execute("GRANT USAGE ON SCHEMA public TO tollgate_app, tollgate_system")
    finally:
        await conn.close()

    log_file = Path(tempfile.gettempdir()) / f"{name}.log"
    settings = Settings(
        database_url_owner=_with_db(dev.database_url_owner, name),
        database_url_app=_with_db(dev.database_url_app, name),
        database_url_system=_with_db(dev.database_url_system, name),
        redis_url=urlsplit(dev.redis_url)._replace(path=f"/{TEST_REDIS_DB}").geturl(),
        bootstrap_admin_email=ADMIN_EMAIL,
        bootstrap_admin_password=ADMIN_PASSWORD,
        log_file=log_file,
        usage_flush_interval_s=0.2,
        admin_rpm_per_ip=10_000,
    )
    redis = Redis.from_url(settings.redis_url)
    await redis.flushdb()
    try:
        yield settings
    finally:
        await redis.flushdb()
        await redis.aclose()
        admin = await asyncpg.connect(_asyncpg_dsn(dev.database_url_owner))
        try:
            await admin.execute(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)')
        finally:
            await admin.close()
        assert await _dev_counts(dev) == dev_before, "test run modified the dev database!"


@pytest.fixture(scope="session")
async def db(test_settings: Settings) -> AsyncIterator[Database]:
    database = Database(test_settings.database_url_app, test_settings.database_url_system, pool_size=4)
    yield database
    await database.dispose()


@pytest.fixture(scope="session")
async def redis(test_settings: Settings) -> AsyncIterator[Redis]:
    client = Redis.from_url(test_settings.redis_url)
    yield client
    await client.aclose()


@pytest.fixture(scope="session")
async def gateway_url(test_settings: Settings) -> AsyncIterator[str]:
    """Run a real gateway (uvicorn, real HTTP) against the test DB/Redis."""
    import uvicorn

    from gateway.main import create_app

    port = _free_port()
    server = uvicorn.Server(
        uvicorn.Config(create_app(test_settings), host="127.0.0.1", port=port, log_config=None, lifespan="on")
    )
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]  # pytest owns signals
    task = asyncio.create_task(server.serve())
    for _ in range(300):
        if server.started or task.done():
            break
        await asyncio.sleep(0.05)
    if not server.started:
        pytest.fail("in-process gateway failed to start (see log output)", pytrace=False)
    yield f"http://127.0.0.1:{port}"
    server.should_exit = True
    await task


@pytest.fixture(scope="session")
async def gateway(gateway_url: str) -> AsyncIterator[httpx.AsyncClient]:
    async with httpx.AsyncClient(base_url=gateway_url, timeout=httpx.Timeout(300, connect=3)) as client:
        (await client.get("/healthz")).raise_for_status()
        yield client


async def login(client: httpx.AsyncClient, email: str, password: str) -> dict[str, str]:
    r = await client.post("/admin/auth/login", json={"email": email, "password": password})
    assert r.status_code == 200, r.text
    return {"authorization": f"Bearer {r.json()['access_token']}"}


@pytest.fixture(scope="session")
async def platform_admin(gateway: httpx.AsyncClient) -> dict[str, str]:
    return await login(gateway, ADMIN_EMAIL, ADMIN_PASSWORD)


# --- tenant provisioning + cleanup ----------------------------------------------------
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


async def purge_tenant(db: Database, redis: Redis, tenant_id: str | uuid.UUID) -> None:
    """Delete every row and Redis key belonging to a tenant (system role, FK order)."""
    tid = str(tenant_id)
    async with db.system_session() as s:
        prefixes = list((await s.execute(text("SELECT prefix FROM api_keys WHERE tenant_id = :t"), {"t": tid})).scalars())
        for stmt in (
            "DELETE FROM usage_events WHERE tenant_id = :t",
            "DELETE FROM audit_log WHERE tenant_id = :t",
            "DELETE FROM admin_refresh_tokens WHERE user_id IN (SELECT id FROM admin_users WHERE tenant_id = :t)",
            "DELETE FROM admin_users WHERE tenant_id = :t",
            "DELETE FROM api_keys WHERE tenant_id = :t",
            "DELETE FROM app_policies WHERE tenant_id = :t",
            "DELETE FROM budgets WHERE tenant_id = :t",
            "DELETE FROM apps WHERE tenant_id = :t",
            "DELETE FROM tenants WHERE id = :t",
        ):
            await s.execute(text(stmt), {"t": tid})
    keys = [k async for k in redis.scan_iter(match=f"t:{tid}:*", count=1000)]
    keys += [auth_cache_key(p) for p in prefixes]
    if keys:
        await redis.delete(*keys)


@dataclass
class TenantFactory:
    """Provisions tenants for a module and purges them when the module finishes."""

    gateway: httpx.AsyncClient
    platform_admin: dict[str, str]
    created: list[str] = field(default_factory=list)

    async def __call__(self, label: str) -> Provisioned:
        p = await provision_tenant(self.gateway, self.platform_admin, label)
        self.created.append(p.tenant_id)
        return p


@pytest.fixture(scope="module")
async def tenant_factory(
    gateway: httpx.AsyncClient, platform_admin: dict[str, str], db: Database, redis: Redis
) -> AsyncIterator[TenantFactory]:
    factory = TenantFactory(gateway, platform_admin)
    yield factory
    for tid in factory.created:
        await purge_tenant(db, redis, tid)


# --- direct-DB helpers --------------------------------------------------------------
async def insert_usage(db: Database, *, tenant_id: str, app_id: str, key_id: str, n: int, baseline: str) -> None:
    """Write synthetic ledger rows directly (system role) — fast, deterministic usage data."""
    async with db.system_session() as s:
        for _ in range(n):
            await s.execute(
                text(
                    "INSERT INTO usage_events (tenant_id, app_id, api_key_id, request_id, routed_model, cache_status, "
                    "input_tokens, output_tokens, actual_cost_usd, baseline_cost_usd, status_code) "
                    "VALUES (:t, :a, :k, :r, 'ollama/qwen2.5:7b', 'miss', 100, 10, 0, :b, 200)"
                ),
                {"t": tenant_id, "a": app_id, "k": key_id, "r": f"synthetic-{uuid.uuid4().hex}", "b": Decimal(baseline)},
            )


async def make_tenant_with_app(db: Database, label: str) -> dict[str, uuid.UUID]:
    """Insert a throwaway tenant + app + key + usage row directly (system role)."""
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

