"""Static route stage and invoke stage (non-streaming, streaming, failures)."""

import pytest

from gateway.errors import GatewayError
from gateway.pipeline.invoke import InvokeStage
from gateway.pipeline.route import StaticRouteStage
from gateway.policy.schema import AppPolicy
from gateway.providers.base import ProviderError
from gateway.providers.registry import ProviderRegistry
from tests.unit.fakes import FakeAdapter, auth_ctx, chat_request, make_catalog, make_ctx

DEFAULT = "ollama/qwen2.5:7b"


def route_stage(registry: ProviderRegistry | None = None) -> StaticRouteStage:
    return StaticRouteStage(make_catalog(), registry or ProviderRegistry({"ollama": FakeAdapter()}), DEFAULT)


async def test_auto_routes_to_default() -> None:
    ctx = await route_stage().process(make_ctx(chat_request(), policy=AppPolicy()))
    assert ctx.route is not None
    assert (ctx.route.model_id, ctx.route.tier, ctx.route.reason) == (DEFAULT, "local", "static:default")


async def test_pin_ignored_when_pinning_disabled() -> None:
    ctx = await route_stage().process(make_ctx(chat_request(model="anthropic/claude-haiku-5-5"), policy=AppPolicy()))
    assert ctx.route.model_id == DEFAULT
    assert ctx.route.reason == "static:default(pinning-disabled)"


async def test_pin_honoured_when_allowed_and_provider_configured() -> None:
    p = AppPolicy.model_validate({"routing": {"allow_pinning": True}})
    reg = ProviderRegistry({"ollama": FakeAdapter(), "anthropic": FakeAdapter()})
    ctx = await route_stage(reg).process(make_ctx(chat_request(model="anthropic/claude-haiku-5-5"), policy=p))
    assert (ctx.route.model_id, ctx.route.reason) == ("anthropic/claude-haiku-5-5", "pinned")


async def test_pin_falls_back_when_provider_not_configured() -> None:
    p = AppPolicy.model_validate({"routing": {"allow_pinning": True}})
    ctx = await route_stage().process(make_ctx(chat_request(model="anthropic/claude-haiku-5-5"), policy=p))
    assert ctx.route.model_id == DEFAULT and "unavailable" in ctx.route.reason


async def _routed(req, adapter: FakeAdapter):
    reg = ProviderRegistry({"ollama": adapter})
    ctx = await route_stage(reg).process(make_ctx(req, auth=auth_ctx(), policy=AppPolicy()))
    return InvokeStage(make_catalog(), reg), ctx


async def test_invoke_non_streaming_records_attempt() -> None:
    adapter = FakeAdapter(text="pong", usage=(20, 4))
    stage, ctx = await _routed(chat_request("ping"), adapter)
    ctx = await stage.process(ctx)
    assert adapter.calls[0][0] == "qwen2.5:7b"  # provider-native model name upstream
    assert ctx.response is not None and ctx.response.model == DEFAULT
    assert ctx.response.choices[0].message.content == "pong"
    a = ctx.attempts[0]
    assert (a.model_id, a.input_tokens, a.output_tokens, a.status_code) == (DEFAULT, 20, 4, 200)
    assert ctx.working is not None and ctx.working is not ctx.original


async def test_invoke_streaming_relays_and_captures_usage() -> None:
    adapter = FakeAdapter(text="a b c", usage=(7, 3))
    stage, ctx = await _routed(chat_request("ping", stream=True), adapter)
    ctx = await stage.process(ctx)
    assert ctx.response is None and ctx.stream is not None
    chunks = [c async for c in ctx.stream]
    assert all(c.model == DEFAULT for c in chunks)
    assert "".join(c.choices[0].delta.get("content", "") for c in chunks if c.choices).strip() == "a b c"
    assert (ctx.attempts[0].input_tokens, ctx.attempts[0].output_tokens) == (7, 3)


@pytest.mark.parametrize("stream", [False, True])
async def test_provider_failure_becomes_gateway_error_before_first_byte(stream: bool) -> None:
    adapter = FakeAdapter(error=ProviderError("upstream unreachable", status=503))
    stage, ctx = await _routed(chat_request("ping", stream=stream), adapter)
    with pytest.raises(GatewayError) as e:
        await stage.process(ctx)
    assert e.value.status == 503 and e.value.code == "model_unavailable"
    assert ctx.attempts[0].status_code == 503 and ctx.status_code == 503
