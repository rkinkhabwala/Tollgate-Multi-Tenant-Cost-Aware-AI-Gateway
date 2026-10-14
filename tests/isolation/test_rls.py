"""Isolation suite: Postgres RLS must confine `tollgate_app` sessions to one tenant.

These tests talk to the real database as the runtime roles. They prove that even
hand-written SQL that "forgets" a tenant filter cannot read or write another tenant's
rows (spec §4, §15, §17).
"""

import uuid

import pytest
from sqlalchemy import text
from sqlalchemy.exc import DBAPIError

from gateway.db.session import Database
from tests.conftest import make_tenant_with_app

pytestmark = pytest.mark.stack

TABLES = ["tenants", "apps", "api_keys", "usage_events"]


@pytest.fixture(scope="module")
async def two_tenants(db: Database) -> tuple[dict, dict]:
    return await make_tenant_with_app(db, "rls-a"), await make_tenant_with_app(db, "rls-b")


@pytest.mark.parametrize("table", TABLES)
async def test_unfiltered_select_sees_only_own_tenant(db: Database, two_tenants, table: str) -> None:
    a, b = two_tenants
    col = "id" if table == "tenants" else "tenant_id"
    async with db.tenant_session(a["tenant_id"]) as s:
        seen = set((await s.execute(text(f"SELECT DISTINCT {col} FROM {table}"))).scalars())
    assert seen == {a["tenant_id"]}
    assert b["tenant_id"] not in seen


async def test_explicit_cross_tenant_filter_returns_nothing(db: Database, two_tenants) -> None:
    a, b = two_tenants
    async with db.tenant_session(a["tenant_id"]) as s:
        rows = (await s.execute(text("SELECT * FROM usage_events WHERE tenant_id = :b"), {"b": b["tenant_id"]})).all()
        keys = (await s.execute(text("SELECT * FROM api_keys WHERE id = :k"), {"k": b["key_id"]})).all()
    assert rows == [] and keys == []


async def test_cannot_insert_into_other_tenant(db: Database, two_tenants) -> None:
    a, b = two_tenants
    with pytest.raises(DBAPIError, match="row-level security"):
        async with db.tenant_session(a["tenant_id"]) as s:
            await s.execute(
                text("INSERT INTO apps (tenant_id, name, baseline_model) VALUES (:b, 'evil', 'x')"),
                {"b": b["tenant_id"]},
            )


async def test_cannot_update_other_tenant_rows(db: Database, two_tenants) -> None:
    a, b = two_tenants
    async with db.tenant_session(a["tenant_id"]) as s:
        result = await s.execute(text("UPDATE api_keys SET revoked_at = now() WHERE id = :k"), {"k": b["key_id"]})
    assert result.rowcount == 0


async def test_no_tenant_set_sees_nothing(db: Database, two_tenants) -> None:
    async with db._app_sessions() as s, s.begin():  # deliberately bypass tenant_session
        for table in TABLES:
            assert (await s.execute(text(f"SELECT count(*) FROM {table}"))).scalar_one() == 0


async def test_cannot_reset_tenant_to_bypass(db: Database, two_tenants) -> None:
    """Setting app.tenant_id to garbage must not widen visibility."""
    a, _ = two_tenants
    async with db.tenant_session(a["tenant_id"]) as s:
        await s.execute(text("SELECT set_config('app.tenant_id', '', true)"))
        assert (await s.execute(text("SELECT count(*) FROM apps"))).scalar_one() == 0


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT * FROM usage_daily",  # matview: no RLS, so app role has no grant
        "SELECT * FROM usage_events_default",  # partitions: no direct access
        "SELECT * FROM admin_refresh_tokens",
        "SELECT refresh_usage_daily()",
        "SET ROLE tollgate_system",
    ],
)
async def test_app_role_cannot_reach_around_rls(db: Database, sql: str) -> None:
    with pytest.raises(DBAPIError, match="permission denied"):
        async with db.tenant_session(uuid.uuid4()) as s:
            await s.execute(text(sql))


async def test_tenant_setting_is_transaction_local(db: Database, two_tenants) -> None:
    """A pooled connection must not carry the previous tenant into the next transaction."""
    a, _ = two_tenants
    async with db.tenant_session(a["tenant_id"]):
        pass
    async with db._app_sessions() as s, s.begin():
        assert (await s.execute(text("SELECT current_setting('app.tenant_id', true)"))).scalar_one() in (None, "")
