"""Isolation suite (API level): tenant A can never read B's data or use B's keys or apps.

Runs against the live gateway. Two tenants are provisioned through the admin API,
and synthetic ledger rows are written for both. Every cross-tenant access path is
then probed: data plane, control plane, and Redis key naming.
"""

import uuid

import httpx
import pytest
from redis.asyncio import Redis

from gateway.db.session import Database
from gateway.settings import get_settings
from tests.conftest import Provisioned, insert_usage, provision_tenant

pytestmark = pytest.mark.stack


@pytest.fixture(scope="module")
async def tenants(gateway: httpx.AsyncClient, platform_admin: dict, db: Database) -> tuple[Provisioned, Provisioned]:
    a = await provision_tenant(gateway, platform_admin, "iso-a")
    b = await provision_tenant(gateway, platform_admin, "iso-b")
    await insert_usage(db, tenant_id=a.tenant_id, app_id=a.app_id, key_id=a.key_id, n=3, baseline="0.010000")
    await insert_usage(db, tenant_id=b.tenant_id, app_id=b.app_id, key_id=b.key_id, n=5, baseline="0.020000")
    return a, b


def bearer(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


# --- data plane -----------------------------------------------------------------
async def test_each_key_sees_only_its_own_usage(gateway: httpx.AsyncClient, tenants) -> None:
    a, b = tenants
    ua = (await gateway.get("/v1/usage", headers=bearer(a.key))).json()
    ub = (await gateway.get("/v1/usage", headers=bearer(b.key))).json()
    assert ua["app_id"] == a.app_id and ub["app_id"] == b.app_id
    assert ua["totals"]["requests"] == 3 and ub["totals"]["requests"] == 5
    assert ua["totals"]["baseline_cost_usd"] == pytest.approx(0.03)
    assert ub["totals"]["baseline_cost_usd"] == pytest.approx(0.10)


async def test_sibling_app_in_same_tenant_is_also_isolated(gateway: httpx.AsyncClient, tenants) -> None:
    a, _ = tenants
    u = (await gateway.get("/v1/usage", headers=bearer(a.key2))).json()
    assert u["app_id"] == a.app2_id and u["totals"]["requests"] == 0


async def test_query_params_cannot_widen_scope(gateway: httpx.AsyncClient, tenants) -> None:
    a, b = tenants
    r = await gateway.get("/v1/usage", params={"app_id": b.app_id, "tenant_id": b.tenant_id}, headers=bearer(a.key))
    assert r.json()["app_id"] == a.app_id and r.json()["totals"]["requests"] == 3


async def test_forged_key_mixing_a_prefix_and_b_secret_fails(gateway: httpx.AsyncClient, tenants) -> None:
    a, b = tenants
    a_prefix, b_secret = a.key.split("_")[2], b.key.split("_")[3]
    forged = f"tg_{a.key.split('_')[1]}_{a_prefix}_{b_secret}"
    r = await gateway.get("/v1/models", headers=bearer(forged))
    assert r.status_code == 401 and r.json()["error"]["code"] == "invalid_api_key"


# --- control plane --------------------------------------------------------------
async def test_tenant_admin_cannot_list_or_create_apps_in_other_tenant(gateway: httpx.AsyncClient, tenants) -> None:
    a, b = tenants
    assert (await gateway.get(f"/admin/tenants/{b.tenant_id}/apps", headers=a.admin_headers)).status_code == 403
    r = await gateway.post(
        f"/admin/tenants/{b.tenant_id}/apps", json={"name": "evil", "baseline_model": "anthropic/claude-sonnet-5-5"}, headers=a.admin_headers
    )
    assert r.status_code == 403
    r = await gateway.patch(f"/admin/tenants/{b.tenant_id}/apps/{b.app_id}", json={"status": "suspended"}, headers=a.admin_headers)
    assert r.status_code == 403


async def test_tenant_admin_cannot_patch_other_tenants_app_via_own_tenant_path(gateway: httpx.AsyncClient, tenants) -> None:
    a, b = tenants
    r = await gateway.patch(f"/admin/tenants/{a.tenant_id}/apps/{b.app_id}", json={"status": "suspended"}, headers=a.admin_headers)
    assert r.status_code == 404  # RLS: B's app is invisible inside A's session


async def test_tenant_admin_cannot_touch_other_tenants_keys(gateway: httpx.AsyncClient, tenants) -> None:
    a, b = tenants
    assert (await gateway.get(f"/admin/apps/{b.app_id}/keys", headers=a.admin_headers)).status_code == 404
    assert (await gateway.post(f"/admin/apps/{b.app_id}/keys", json={}, headers=a.admin_headers)).status_code == 404
    assert (await gateway.delete(f"/admin/keys/{b.key_id}", headers=a.admin_headers)).status_code == 404
    # ...and B's key still works afterwards.
    assert (await gateway.get("/v1/models", headers=bearer(b.key))).status_code == 200


async def test_tenant_admin_cannot_use_platform_endpoints(gateway: httpx.AsyncClient, tenants) -> None:
    a, _ = tenants
    assert (await gateway.get("/admin/tenants", headers=a.admin_headers)).status_code == 403
    assert (await gateway.post("/admin/tenants", json={"name": "x"}, headers=a.admin_headers)).status_code == 403


async def test_api_key_is_not_an_admin_credential(gateway: httpx.AsyncClient, tenants) -> None:
    a, _ = tenants
    assert (await gateway.get(f"/admin/tenants/{a.tenant_id}/apps", headers=bearer(a.key))).status_code == 401


async def test_listing_keys_never_returns_secrets(gateway: httpx.AsyncClient, tenants) -> None:
    a, _ = tenants
    r = await gateway.get(f"/admin/apps/{a.app_id}/keys", headers=a.admin_headers)
    assert r.status_code == 200
    assert a.key.split("_")[3] not in r.text and "secret_hash" not in r.text


# --- Redis key naming -------------------------------------------------------------
async def test_tenant_redis_keys_are_prefixed(gateway: httpx.AsyncClient, tenants) -> None:
    a, _ = tenants
    await gateway.get("/v1/models", headers=bearer(a.key))  # touches rate-limit + policy keys
    redis = Redis.from_url(get_settings().redis_url)
    try:
        keys = [k.decode() async for k in redis.scan_iter(match="*", count=1000)]
    finally:
        await redis.aclose()
    for k in keys:
        assert k.startswith("t:") or k.startswith("sys:"), f"unscoped redis key: {k}"
        if k.startswith("t:"):
            tid, aid = k.split(":")[1], k.split(":")[3]
            uuid.UUID(tid), uuid.UUID(aid)  # well-formed tenant/app prefix
    assert any(k.startswith(f"t:{a.tenant_id}:a:{a.app_id}:") for k in keys)
