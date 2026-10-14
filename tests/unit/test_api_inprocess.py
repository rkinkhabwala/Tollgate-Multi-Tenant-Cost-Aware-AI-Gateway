"""HTTP-level tests of the data plane with the official `openai` SDK, fully in-process.

The FastAPI app runs on an ASGI transport, with fakes in place of Postgres, Redis and
Ollama. This checks wire compatibility (JSON + SSE), the `x-tollgate-*` headers and
OpenAI-shaped errors without the compose stack. The same flows run against the real
stack in tests/integration.
"""

from __future__ import annotations

import httpx
import openai
import pytest
from openai import AsyncOpenAI

from gateway.api.deps import Services
from gateway.main import create_app
from gateway.pipeline.authn import AuthNStage
from gateway.pipeline.authz import AuthZStage
from gateway.pipeline.base import Pipeline
from gateway.pipeline.invoke import InvokeStage
from gateway.pipeline.meter import MeterStage
from gateway.pipeline.ratelimit import RateLimiter, RateLimitStage
from gateway.pipeline.route import StaticRouteStage
from gateway.policy.schema import AppPolicy
from gateway.providers.registry import ProviderRegistry
from gateway.settings import Settings
from tests.unit.fakes import FakeAdapter, FakePolicyStore, FakeResolver, make_catalog, make_key_record, make_redis, new_key

STREAM = "sys:usage_events"


@pytest.fixture
def harness():
    key, prefix = new_key()
    redis = make_redis()
    catalog = make_catalog()
    adapter = FakeAdapter(text="Hello from the fake model", usage=(25, 6))
    registry = ProviderRegistry({"ollama": adapter})
    limiter = RateLimiter(redis)
    policy = AppPolicy.model_validate({"limits": {"rpm": 5}})
    authn = AuthNStage(redis, FakeResolver({prefix: make_key_record(key, prefix)}))
    authz = AuthZStage(FakePolicyStore(policy), catalog)  # type: ignore[arg-type]
    rl = RateLimitStage(limiter)
    settings = Settings(max_body_bytes=4096)
    app = create_app(settings)
    app.state.services = Services(
        settings=settings,
        db=None,  # type: ignore[arg-type]
        redis=redis,
        catalog=catalog,
        registry=registry,
        policies=None,  # type: ignore[arg-type]
        default_policy=policy,
        chat_pipeline=Pipeline(
            [authn, authz, rl, StaticRouteStage(catalog, registry, "ollama/qwen2.5:7b"), InvokeStage(catalog, registry)],
            [MeterStage(redis, catalog, limiter, STREAM)],
        ),
        auth_pipeline=Pipeline([authn, authz, rl]),
        admin_auth=None,  # type: ignore[arg-type]
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://tollgate.test")
    client = AsyncOpenAI(base_url="http://tollgate.test/v1", api_key=key, http_client=http, max_retries=0)
    return client, http, key, redis


async def test_sdk_non_streaming_with_headers(harness) -> None:
    client, _, _, redis = harness
    raw = await client.chat.completions.with_raw_response.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    completion = raw.parse()
    assert completion.choices[0].message.content == "Hello from the fake model"
    assert completion.model == "ollama/qwen2.5:7b"
    assert completion.usage.prompt_tokens == 25
    h = raw.headers
    assert h["x-tollgate-model"] == "ollama/qwen2.5:7b"
    assert h["x-tollgate-route-reason"] == "static:default"
    assert h["x-tollgate-cache"] == "miss"
    assert h["x-tollgate-request-id"].startswith("req_")
    assert h["x-tollgate-redacted"] == "0"
    assert float(h["x-tollgate-cost-usd"]) == 0.0
    assert float(h["x-tollgate-saved-usd"]) > 0  # baseline (sonnet) minus local ($0)
    assert len(await redis.xrange(STREAM)) == 1  # usage event emitted


async def test_sdk_streaming_with_include_usage(harness) -> None:
    client, _, _, redis = harness
    stream = await client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}], stream=True, stream_options={"include_usage": True}
    )
    text, usage, finish = "", None, None
    async for chunk in stream:
        if chunk.usage:
            usage = chunk.usage
        for choice in chunk.choices:
            text += choice.delta.content or ""
            finish = choice.finish_reason or finish
    assert text.strip() == "Hello from the fake model"
    assert finish == "stop"
    assert usage is not None and usage.completion_tokens == 6
    assert len(await redis.xrange(STREAM)) == 1  # metered exactly once


async def test_include_usage_chunk_carries_tollgate_extension(harness) -> None:
    client, *_ = harness
    stream = await client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": "hi"}], stream=True, stream_options={"include_usage": True}
    )
    final = [c async for c in stream][-1]
    assert final.choices == [] and final.usage is not None
    ext = final.model_extra["tollgate"]
    assert ext["model"] == "ollama/qwen2.5:7b" and ext["cache"] == "miss" and ext["redacted"] == 0
    assert ext["cost_usd"] == 0.0 and ext["saved_usd"] > 0  # real numbers, not "pending"


async def test_stream_without_include_usage_has_no_usage_chunk(harness) -> None:
    _, http, key, _ = harness
    r = await http.post(
        "/v1/chat/completions",
        headers={"authorization": f"Bearer {key}"},
        json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert '"usage"' not in r.text and '"tollgate"' not in r.text


async def test_sse_wire_format_ends_with_done(harness) -> None:
    _, http, key, _ = harness
    r = await http.post(
        "/v1/chat/completions",
        headers={"authorization": f"Bearer {key}"},
        json={"model": "auto", "stream": True, "messages": [{"role": "user", "content": "hi"}]},
    )
    assert r.headers["content-type"].startswith("text/event-stream")
    assert r.headers["x-tollgate-cost-usd"] == "pending"
    events = [line for line in r.text.split("\n\n") if line]
    assert all(e.startswith("data: ") for e in events)
    assert events[-1] == "data: [DONE]"


async def test_models_endpoint(harness) -> None:
    client, *_ = harness
    ids = [m.id async for m in client.models.list()]
    assert "auto" in ids and "ollama/qwen2.5:7b" in ids and "tollgate/small" in ids


async def test_bad_key_is_openai_auth_error(harness) -> None:
    _, http, _, _ = harness
    bad = AsyncOpenAI(base_url="http://tollgate.test/v1", api_key="tg_live_aaaaaaaa_" + "b" * 32, http_client=http, max_retries=0)
    with pytest.raises(openai.AuthenticationError) as e:
        await bad.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    assert e.value.body["code"] == "invalid_api_key"


async def test_disallowed_model_is_permission_error(harness) -> None:
    client, *_ = harness
    with pytest.raises(openai.PermissionDeniedError):
        await client.chat.completions.create(model="anthropic/claude-opus-5-5", messages=[{"role": "user", "content": "hi"}])


async def test_rate_limit_is_openai_429(harness) -> None:
    client, *_ = harness
    for _ in range(5):
        await client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    with pytest.raises(openai.RateLimitError) as e:
        await client.chat.completions.create(model="auto", messages=[{"role": "user", "content": "hi"}])
    assert e.value.response.headers["retry-after"]


async def test_validation_error_does_not_echo_prompt(harness) -> None:
    _, http, key, _ = harness
    r = await http.post(
        "/v1/chat/completions",
        headers={"authorization": f"Bearer {key}"},
        json={"model": "auto", "messages": [{"role": "wizard", "content": "SECRET-PROMPT-123"}]},
    )
    assert r.status_code == 400
    assert r.json()["error"]["code"] == "invalid_request"
    assert "SECRET-PROMPT-123" not in r.text


async def test_body_size_limit(harness) -> None:
    _, http, key, _ = harness
    r = await http.post(
        "/v1/chat/completions",
        headers={"authorization": f"Bearer {key}"},
        json={"model": "auto", "messages": [{"role": "user", "content": "x" * 10_000}]},
    )
    assert r.status_code == 413 and r.json()["error"]["code"] == "request_too_large"


async def test_usage_requires_scope(harness) -> None:
    """The fake key has usage:read; a chat-only key must be refused with 403."""
    _, http, _, redis = harness
    key, prefix = new_key()
    app = http._transport.app  # type: ignore[attr-defined]
    resolver = app.state.services.auth_pipeline.stages[0]._resolver
    resolver.records[prefix] = make_key_record(key, prefix, scopes=["chat:write"])
    r = await http.get("/v1/usage", headers={"authorization": f"Bearer {key}"})
    assert r.status_code == 403 and r.json()["error"]["code"] == "permission_denied"
