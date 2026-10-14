"""Policy endpoints, /admin/redaction/test, and redaction on the live gateway."""

import asyncio
import json

import httpx
import pytest
from sqlalchemy import text

from gateway.db.session import Database
from tests.conftest import Provisioned, TenantFactory

pytestmark = pytest.mark.stack

EMAIL = "rowan.castellanos@fabrikam.example"


@pytest.fixture(scope="module")
async def pair(tenant_factory: TenantFactory) -> tuple[Provisioned, Provisioned]:
    return await tenant_factory("pol-a"), await tenant_factory("pol-b")


def bearer(key: str) -> dict[str, str]:
    return {"authorization": f"Bearer {key}"}


async def test_get_default_policy(gateway: httpx.AsyncClient, pair) -> None:
    a, _ = pair
    r = await gateway.get(f"/admin/apps/{a.app_id}/policy", headers=a.admin_headers)
    assert r.status_code == 200
    body = r.json()
    assert body["version"] == 1 and body["policy"]["redaction"]["entities"]["SECRET"] == "block"


@pytest.mark.parametrize(
    "bad",
    [
        {"limits": {"rpm": 10, "rmp": 5}},  # unknown nested key
        {"surprise": True},  # unknown top-level key
        {"routing": {"min_tier": "large", "max_tier": "small"}},
        {"redaction": {"entities": {"EMAIL_ADDRESS": "shred"}}},
        {"redaction": {"custom_patterns": [{"name": "X", "regex": "("}]}},
    ],
)
async def test_put_rejects_invalid_policy(gateway: httpx.AsyncClient, pair, bad: dict) -> None:
    a, _ = pair
    r = await gateway.put(f"/admin/apps/{a.app_id}/policy", json=bad, headers=a.admin_headers)
    assert r.status_code == 400 and r.json()["error"]["code"] == "invalid_request"


async def test_put_bumps_version_audits_and_applies_immediately(gateway: httpx.AsyncClient, pair, db: Database) -> None:
    a, _ = pair
    # Warm caches with the v1 policy (redaction on).
    r = await gateway.post(
        "/admin/redaction/test", json={"app_id": a.app_id, "text": f"mail {EMAIL}"}, headers=a.admin_headers
    )
    assert r.json()["counts"] == {"EMAIL_ADDRESS": 1}
    assert (await gateway.get("/v1/models", headers=bearer(a.key))).status_code == 200  # AuthContext cached

    policy = (await gateway.get(f"/admin/apps/{a.app_id}/policy", headers=a.admin_headers)).json()["policy"]
    policy["redaction"]["entities"]["EMAIL_ADDRESS"] = "allow"
    policy["limits"]["rpm"] = 7
    r = await gateway.put(f"/admin/apps/{a.app_id}/policy", json=policy, headers=a.admin_headers)
    assert r.status_code == 200 and r.json()["version"] == 2

    # Immediately effective: the data plane sees the new rpm limit on the very next call...
    r = await gateway.get("/v1/models", headers=bearer(a.key))
    assert r.headers["x-ratelimit-limit-requests"] == "7"
    # ...and the dry run sees EMAIL as allowed.
    r = await gateway.post("/admin/redaction/test", json={"app_id": a.app_id, "text": f"mail {EMAIL}"}, headers=a.admin_headers)
    assert r.json()["counts"] == {}

    async with db.tenant_session(a.tenant_id) as s:
        diff = (await s.execute(text("SELECT diff FROM audit_log WHERE action = 'policy.update' ORDER BY id DESC LIMIT 1"))).scalar_one()
    assert diff["version"] == 2 and set(diff["changed"]) == {"redaction", "limits"}


async def test_policy_and_redaction_test_are_tenant_scoped(gateway: httpx.AsyncClient, pair) -> None:
    a, b = pair
    assert (await gateway.get(f"/admin/apps/{b.app_id}/policy", headers=a.admin_headers)).status_code == 404
    assert (await gateway.put(f"/admin/apps/{b.app_id}/policy", json={}, headers=a.admin_headers)).status_code == 404
    r = await gateway.post("/admin/redaction/test", json={"app_id": b.app_id, "text": "x"}, headers=a.admin_headers)
    assert r.status_code == 404


async def test_redaction_test_output(gateway: httpx.AsyncClient, pair) -> None:
    _, b = pair
    sample = f"Contact {EMAIL} or 415-555-0123. SSN 536-22-8187. Key AK\x49AIOSFODNN7EXAMPLE."
    r = await gateway.post("/admin/redaction/test", json={"app_id": b.app_id, "text": sample}, headers=b.admin_headers)
    body = r.json()
    assert r.status_code == 200 and body["blocked"] is True
    assert body["counts"] == {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1, "US_SSN": 1, "SECRET": 1}
    assert body["redacted_text"] == "Contact <<EMAIL_1>> or <<PHONE_1>>. SSN <<SSN>>. Key <<SECRET>>."
    entity_json = json.dumps(body["entities"])
    assert EMAIL not in entity_json and "AKIA" not in entity_json  # spans only, never values


async def test_blocked_secret_on_data_plane(gateway: httpx.AsyncClient, pair) -> None:
    _, b = pair
    r = await gateway.post(
        "/v1/chat/completions",
        json={"model": "auto", "messages": [{"role": "user", "content": "my key is sk\x2dproj-abcdefghijklmnopqrstuvwx1234"}]},
        headers=bearer(b.key),
    )
    assert r.status_code == 400 and r.json()["error"]["code"] == "sensitive_data_blocked"
    assert "sk-proj" not in r.text


@pytest.mark.ollama
async def test_pii_counts_recorded_on_usage_event(gateway: httpx.AsyncClient, pair, db: Database) -> None:
    _, b = pair
    r = await gateway.post(
        "/v1/chat/completions",
        json={"model": "auto", "max_tokens": 3, "messages": [{"role": "user", "content": f"Say ok. My email is {EMAIL}."}]},
        headers=bearer(b.key),
    )
    assert r.status_code == 200 and r.headers["x-tollgate-redacted"] == "1"
    rid = r.headers["x-tollgate-request-id"]
    for _ in range(40):
        async with db.system_session() as s:
            row = (await s.execute(text("SELECT pii_entities FROM usage_events WHERE request_id = :r"), {"r": rid})).scalar_one_or_none()
        if row is not None:
            break
        await asyncio.sleep(0.1)
    assert row == {"EMAIL_ADDRESS": 1}
