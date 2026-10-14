"""Control plane against the live gateway: key lifecycle, revocation latency, audit, lockout."""

import time
import uuid

import httpx
import pytest
from sqlalchemy import text

from gateway.db.session import Database
from tests.conftest import Provisioned, TenantFactory, login

pytestmark = pytest.mark.stack


@pytest.fixture(scope="module")
async def tenant(tenant_factory: TenantFactory) -> Provisioned:
    return await tenant_factory("admin-it")


def bearer(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


async def test_revoked_key_fails_within_one_second(gateway: httpx.AsyncClient, tenant: Provisioned) -> None:
    r = await gateway.post(f"/admin/apps/{tenant.app_id}/keys", json={}, headers=tenant.admin_headers)
    key, key_id = r.json()["key"], r.json()["id"]
    # Warm the 60 s AuthContext cache so revocation has to beat it.
    for _ in range(3):
        assert (await gateway.get("/v1/models", headers=bearer(key))).status_code == 200

    t0 = time.perf_counter()
    assert (await gateway.delete(f"/admin/keys/{key_id}", headers=tenant.admin_headers)).status_code == 200
    while True:
        status = (await gateway.get("/v1/models", headers=bearer(key))).status_code
        elapsed = time.perf_counter() - t0
        if status == 401 or elapsed > 1.0:
            break
    assert status == 401, f"revoked key still accepted after {elapsed:.3f}s"
    assert elapsed < 1.0


async def test_rotation_both_keys_valid_until_old_revoked(gateway: httpx.AsyncClient, tenant: Provisioned) -> None:
    old = (await gateway.post(f"/admin/apps/{tenant.app_id}/keys", json={}, headers=tenant.admin_headers)).json()
    new = (await gateway.post(f"/admin/apps/{tenant.app_id}/keys", json={}, headers=tenant.admin_headers)).json()
    for k in (old, new):
        assert (await gateway.get("/v1/models", headers=bearer(k["key"]))).status_code == 200
    await gateway.delete(f"/admin/keys/{old['id']}", headers=tenant.admin_headers)
    assert (await gateway.get("/v1/models", headers=bearer(old["key"]))).status_code == 401
    assert (await gateway.get("/v1/models", headers=bearer(new["key"]))).status_code == 200


async def test_key_shown_once_and_listing_shows_prefix(gateway: httpx.AsyncClient, tenant: Provisioned) -> None:
    created = (await gateway.post(f"/admin/apps/{tenant.app_id}/keys", json={"scopes": ["usage:read"]}, headers=tenant.admin_headers)).json()
    assert created["key"].startswith(f"tg_live_{created['prefix']}_")
    listed = (await gateway.get(f"/admin/apps/{tenant.app_id}/keys", headers=tenant.admin_headers)).json()["data"]
    row = next(k for k in listed if k["id"] == created["id"])
    assert "key" not in row and row["prefix"] == created["prefix"] and row["scopes"] == ["usage:read"]


async def test_scope_enforced_on_data_plane(gateway: httpx.AsyncClient, tenant: Provisioned) -> None:
    usage_only = (await gateway.post(f"/admin/apps/{tenant.app_id}/keys", json={"scopes": ["usage:read"]}, headers=tenant.admin_headers)).json()
    r = await gateway.post(
        "/v1/chat/completions", json={"model": "auto", "messages": [{"role": "user", "content": "hi"}]}, headers=bearer(usage_only["key"])
    )
    assert r.status_code == 403 and r.json()["error"]["code"] == "permission_denied"


async def test_ip_allowlist_enforced(gateway: httpx.AsyncClient, tenant: Provisioned) -> None:
    k = (await gateway.post(f"/admin/apps/{tenant.app_id}/keys", json={"ip_allowlist": ["203.0.113.0/24"]}, headers=tenant.admin_headers)).json()
    r = await gateway.get("/v1/models", headers=bearer(k["key"]))
    assert r.status_code == 403


async def test_suspending_app_blocks_keys_immediately(gateway: httpx.AsyncClient, tenant_factory: TenantFactory) -> None:
    t = await tenant_factory("suspend")
    assert (await gateway.get("/v1/models", headers=bearer(t.key))).status_code == 200
    r = await gateway.patch(f"/admin/tenants/{t.tenant_id}/apps/{t.app_id}", json={"status": "suspended"}, headers=t.admin_headers)
    assert r.status_code == 200 and r.json()["status"] == "suspended"
    assert (await gateway.get("/v1/models", headers=bearer(t.key))).status_code == 403


async def test_mutations_are_audited(gateway: httpx.AsyncClient, tenant: Provisioned, db: Database) -> None:
    async with db.tenant_session(tenant.tenant_id) as s:
        actions = set((await s.execute(text("SELECT action FROM audit_log"))).scalars())
    assert {"tenant.create", "user.create", "app.create", "key.create"} <= actions


async def test_login_lockout_after_repeated_failures(gateway: httpx.AsyncClient, platform_admin: dict, tenant: Provisioned) -> None:
    email = f"lock-{uuid.uuid4().hex[:6]}@lock.test"
    r = await gateway.post(
        f"/admin/tenants/{tenant.tenant_id}/users",
        json={"email": email, "password": "right-password-123", "role": "tenant_viewer"},
        headers=platform_admin,
    )
    assert r.status_code == 201
    for _ in range(5):
        r = await gateway.post("/admin/auth/login", json={"email": email, "password": "wrong-password"})
        assert r.status_code == 401
    r = await gateway.post("/admin/auth/login", json={"email": email, "password": "right-password-123"})
    assert r.status_code == 423 and r.json()["error"]["code"] == "account_locked"


async def test_refresh_token_rotates(gateway: httpx.AsyncClient) -> None:
    from tests.conftest import ADMIN_EMAIL, ADMIN_PASSWORD

    tokens = (await gateway.post("/admin/auth/login", json={"email": ADMIN_EMAIL, "password": ADMIN_PASSWORD})).json()
    r1 = await gateway.post("/admin/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r1.status_code == 200
    r2 = await gateway.post("/admin/auth/refresh", json={"refresh_token": tokens["refresh_token"]})
    assert r2.status_code == 401  # old refresh token is single-use
    assert (await gateway.get("/admin/tenants", headers={"authorization": f"Bearer {r1.json()['access_token']}"})).status_code == 200


async def test_viewer_cannot_mutate(gateway: httpx.AsyncClient, platform_admin: dict, tenant: Provisioned) -> None:
    email, pw = f"viewer-{uuid.uuid4().hex[:6]}@v.test", "viewer-password-123"
    await gateway.post(f"/admin/tenants/{tenant.tenant_id}/users", json={"email": email, "password": pw, "role": "tenant_viewer"}, headers=platform_admin)
    viewer = await login(gateway, email, pw)
    assert (await gateway.get(f"/admin/tenants/{tenant.tenant_id}/apps", headers=viewer)).status_code == 200
    assert (await gateway.post(f"/admin/apps/{tenant.app_id}/keys", json={}, headers=viewer)).status_code == 403
