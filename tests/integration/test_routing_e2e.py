"""Phase 3 end-to-end: routing, availability fallback, circuit breaker, quality escalation.

A second in-process gateway is started for this module (same test DB, Redis and
log file as the session gateway), pointed at:
* `FlakyOllama`, a proxy to the real Ollama that can take models "down" mid-run;
* `FakeAnthropic`, a scripted Messages API with a fake key.

What the tests show:
* fallback: with `qwen2.5:1.5b` stopped, clients see no errors, and the ledger's
  `attempts` + `route_reason` show the fallback to `qwen2.5:7b`;
* breaker: after 5 failures the down model is no longer even tried;
* streaming fallback happens before the first byte;
* escalation: invalid JSON from Haiku (small) → one escalation to Sonnet (medium);
  both attempts billed at their own prices, and `escalations = 1`;
* `/admin/routing/test` explains a decision without echoing text, tenant-scoped.
"""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import httpx
import pytest
from pydantic import SecretStr
from sqlalchemy import text

from gateway.db.session import Database
from gateway.settings import Settings
from tests.conftest import Provisioned, TenantFactory
from tests.fake_servers import FakeAnthropic, FlakyOllama, ServerHandle

pytestmark = [pytest.mark.stack, pytest.mark.ollama]

LOCAL_ONLY = {"profile": "local-only", "max_tier": "large"}


@pytest.fixture(scope="module")
async def fakes() -> AsyncIterator[tuple[FlakyOllama, FakeAnthropic, ServerHandle, ServerHandle]]:
    # Load the local models first so cold-start latency doesn't hit the short timeout.
    async with httpx.AsyncClient(base_url="http://localhost:11434", timeout=300) as ollama:
        for model in ("qwen2.5:1.5b", "qwen2.5:7b"):
            (await ollama.post("/api/generate", json={"model": model, "prompt": "", "keep_alive": "15m"})).raise_for_status()
    flaky = FlakyOllama()
    anth = FakeAnthropic({"claude-haiku-5-5": "Sure! name: Ada, age: 36", "claude-sonnet-5-5": '{"name": "Ada", "age": 36}'})
    h1, h2 = await ServerHandle(flaky.app).start(), await ServerHandle(anth.app).start()
    yield flaky, anth, h1, h2
    await h1.stop()
    await h2.stop()


@pytest.fixture(scope="module")
async def gw(fakes, test_settings: Settings) -> AsyncIterator[httpx.AsyncClient]:
    import uvicorn

    from gateway.main import create_app
    from tests.conftest import _free_port

    _, _, h1, h2 = fakes
    settings = test_settings.model_copy(
        update={
            "ollama_base_url": h1.url,
            "ollama_timeout_s": 10.0,
            "anthropic_api_key": SecretStr("test-anthropic-key"),  # model_copy skips validation
            "anthropic_base_url": h2.url,
        }
    )
    port = _free_port()
    server = uvicorn.Server(uvicorn.Config(create_app(settings), host="127.0.0.1", port=port, log_config=None, lifespan="on", timeout_graceful_shutdown=2))
    server.install_signal_handlers = lambda: None  # type: ignore[method-assign]
    task = asyncio.create_task(server.serve())
    while not server.started:
        await asyncio.sleep(0.05)
    async with httpx.AsyncClient(base_url=f"http://127.0.0.1:{port}", timeout=120) as client:
        yield client
    server.should_exit = True
    await task


async def set_policy(client: httpx.AsyncClient, t: Provisioned, routing: dict) -> None:
    pol = (await client.get(f"/admin/apps/{t.app_id}/policy", headers=t.admin_headers)).json()["policy"]
    pol["routing"].update(routing)
    r = await client.put(f"/admin/apps/{t.app_id}/policy", json=pol, headers=t.admin_headers)
    assert r.status_code == 200, r.text


@pytest.fixture(scope="module")
async def local_app(gw: httpx.AsyncClient, tenant_factory: TenantFactory) -> Provisioned:
    t = await tenant_factory("route-local")
    await set_policy(gw, t, LOCAL_ONLY)
    return t


@pytest.fixture(scope="module")
async def hosted_app(gw: httpx.AsyncClient, tenant_factory: TenantFactory) -> Provisioned:
    return await tenant_factory("route-hosted")  # default profile: hosted


async def ledger(db: Database, request_id: str) -> dict:
    for _ in range(60):
        async with db.system_session() as s:
            row = (await s.execute(text("SELECT routed_model, route_reason, escalations, attempts, actual_cost_usd, baseline_cost_usd, status_code FROM usage_events WHERE request_id = :r"), {"r": request_id})).mappings().first()
        if row:
            return dict(row)
        await asyncio.sleep(0.1)
    raise AssertionError("usage event not written")


def chat(t: Provisioned, content: str, **kw) -> tuple[dict, dict]:
    return {"authorization": f"Bearer {t.key}"}, {"model": kw.pop("model", "auto"), "max_tokens": 8, "temperature": 0, "messages": [{"role": "user", "content": content}], **kw}


async def test_fallback_when_provider_stops_mid_run(gw: httpx.AsyncClient, fakes, local_app: Provisioned, db: Database) -> None:
    flaky = fakes[0]
    h, body = chat(local_app, "Say hi.", model="tollgate/small")  # local-only small = qwen2.5:1.5b
    r = await gw.post("/v1/chat/completions", headers=h, json=body)
    assert r.status_code == 200 and r.headers["x-tollgate-model"] == "ollama/qwen2.5:1.5b"

    flaky.down = {"qwen2.5:1.5b": "error"}  # provider stops mid-run
    r = await gw.post("/v1/chat/completions", headers=h, json=body)
    assert r.status_code == 200  # no client-visible error
    assert r.headers["x-tollgate-model"] == "ollama/qwen2.5:7b"
    assert "fallback:ollama/qwen2.5:1.5b(server_error)>ollama/qwen2.5:7b" in r.headers["x-tollgate-route-reason"]
    row = await ledger(db, r.headers["x-tollgate-request-id"])
    assert row["routed_model"] == "ollama/qwen2.5:7b" and row["status_code"] == 200
    assert [(a["model"], a["kind"], a["status"], a["error"]) for a in row["attempts"]] == [
        ("ollama/qwen2.5:1.5b", "primary", 502, "server_error"),
        ("ollama/qwen2.5:7b", "fallback", 200, None),
    ]


async def test_streaming_fallback_before_first_byte(gw: httpx.AsyncClient, fakes, local_app: Provisioned) -> None:
    fakes[0].down = {"qwen2.5:1.5b": "error"}
    h, body = chat(local_app, "Say hi.", model="tollgate/small", stream=True)
    async with gw.stream("POST", "/v1/chat/completions", headers=h, json=body) as r:
        assert r.status_code == 200 and r.headers["x-tollgate-model"] == "ollama/qwen2.5:7b"
        lines = [line async for line in r.aiter_lines() if line.startswith("data: ")]
    assert lines[-1] == "data: [DONE]" and not any('"error"' in line for line in lines)


async def test_timeout_falls_back(gw: httpx.AsyncClient, fakes, local_app: Provisioned) -> None:
    fakes[0].down = {"qwen2.5:1.5b": "timeout"}  # hangs past the gateway's 10 s timeout
    h, body = chat(local_app, "Say hi.", model="tollgate/small")
    r = await gw.post("/v1/chat/completions", headers=h, json=body)
    assert r.status_code == 200
    assert r.headers["x-tollgate-model"] == "ollama/qwen2.5:7b"
    assert "fallback:ollama/qwen2.5:1.5b(timeout)>ollama/qwen2.5:7b" in r.headers["x-tollgate-route-reason"]


async def test_breaker_opens_and_skips_dead_model(gw: httpx.AsyncClient, fakes, local_app: Provisioned, db: Database) -> None:
    flaky = fakes[0]
    flaky.down = {"qwen2.5:1.5b": "error"}
    h, body = chat(local_app, "Say hi.", model="tollgate/small")
    for _ in range(5):
        assert (await gw.post("/v1/chat/completions", headers=h, json=body)).status_code == 200
    hits = flaky.hits.get("qwen2.5:1.5b", 0)
    r = await gw.post("/v1/chat/completions", headers=h, json=body)
    assert r.status_code == 200 and flaky.hits.get("qwen2.5:1.5b", 0) == hits  # not even tried
    row = await ledger(db, r.headers["x-tollgate-request-id"])
    assert [a["model"] for a in row["attempts"]] == ["ollama/qwen2.5:7b"]
    ready = (await gw.get("/readyz")).json()
    assert ready["circuits"]["ollama/qwen2.5:1.5b"] == "open"
    flaky.down = {}


async def test_invalid_json_escalates_once_and_bills_both(gw: httpx.AsyncClient, hosted_app: Provisioned, db: Database) -> None:
    h, body = chat(hosted_app, "Extract name and age as JSON: Ada is 36.", model="tollgate/small", response_format={"type": "json_object"}, max_tokens=100)
    r = await gw.post("/v1/chat/completions", headers=h, json=body)
    assert r.status_code == 200
    assert json.loads(r.json()["choices"][0]["message"]["content"]) == {"name": "Ada", "age": 36}
    assert r.headers["x-tollgate-model"] == "anthropic/claude-sonnet-5-5"
    assert "escalate:invalid_json>medium" in r.headers["x-tollgate-route-reason"]
    row = await ledger(db, r.headers["x-tollgate-request-id"])
    assert row["escalations"] == 1
    attempts = row["attempts"]
    assert [(a["model"], a["kind"]) for a in attempts] == [("anthropic/claude-haiku-5-5", "primary"), ("anthropic/claude-sonnet-5-5", "escalation")]
    # 120 in / 30 out per fake call: haiku 0.000027, sonnet 0.000540 — both billed.
    assert [a["cost_usd"] for a in attempts] == ["0.000027", "0.000540"]
    assert float(row["actual_cost_usd"]) == pytest.approx(0.000567)


async def test_routing_test_endpoint(gw: httpx.AsyncClient, local_app: Provisioned, hosted_app: Provisioned) -> None:
    body = {"app_id": local_app.app_id, "messages": [{"role": "user", "content": "Write a Python function that parses ISO dates; email jane@corp.example the result.\n```python\ndef f(): pass\n```"}]}
    r = await gw.post("/admin/routing/test", json=body, headers=local_app.admin_headers)
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["classification"]["task_type"] == "code" and out["redaction"] == {"EMAIL_ADDRESS": 1}
    assert out["model"].startswith("ollama/") and out["fallback_order"]
    assert "jane@corp.example" not in r.text and "parses ISO dates" not in r.text
    # tenant-scoped: another tenant's admin can't probe this app
    assert (await gw.post("/admin/routing/test", json=body, headers=hosted_app.admin_headers)).status_code == 404
