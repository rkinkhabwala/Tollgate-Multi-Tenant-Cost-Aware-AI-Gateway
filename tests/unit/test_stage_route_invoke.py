"""Router decisions, invoke fallback + circuit breaker, validate + escalation (spec §8.2, §8.4)."""

from __future__ import annotations

import json

import pytest

from gateway.errors import GatewayError
from gateway.pipeline.context import Classification
from gateway.pipeline.invoke import InvokeStage
from gateway.pipeline.route import RouteStage
from gateway.pipeline.validate import ValidateStage, failure_reason
from gateway.policy.schema import AppPolicy
from gateway.providers.base import ChatMessage, ChatRequest, ChatResponse, Choice, ProviderError, Usage
from gateway.providers.circuit_breaker import CircuitBreaker
from gateway.providers.registry import ProviderRegistry
from gateway.routing.table import DEFAULT_TABLE, target_tier
from tests.unit.fakes import FakeAdapter, auth_ctx, chat_request, make_catalog, make_ctx, make_router

LOCAL_ONLY = AppPolicy.model_validate({"routing": {"profile": "local-only", "max_tier": "large"}})


def cls(task: str, cx: str, conf: float = 0.9) -> Classification:
    return Classification(task, cx, conf)


class ModelAdapter(FakeAdapter):
    """FakeAdapter that can fail or answer differently per provider model."""

    def __init__(self, fail: dict[str, ProviderError] | None = None, answers: dict[str, str] | None = None) -> None:
        super().__init__()
        self.fail = fail or {}
        self.answers = answers or {}

    async def chat(self, model: str, req: ChatRequest) -> ChatResponse:
        self.calls.append((model, req))
        if model in self.fail:
            raise self.fail[model]
        text = self.answers.get(model, f"answer from {model}")
        return ChatResponse(model=model, choices=[Choice(message=ChatMessage(role="assistant", content=text), finish_reason="stop")], usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    async def stream(self, model: str, req: ChatRequest):  # type: ignore[no-untyped-def,override]
        if model in self.fail:
            raise self.fail[model]
        async for c in super().stream(model, req):
            yield c


# --- table ------------------------------------------------------------------------
def test_default_table_matches_spec() -> None:
    assert DEFAULT_TABLE["extraction"] == {"low": "local", "medium": "local", "high": "small"}
    assert DEFAULT_TABLE["qa_with_context"] == {"low": "local", "medium": "small", "high": "medium"}
    assert DEFAULT_TABLE["code"] == {"low": "small", "medium": "medium", "high": "large"}
    assert DEFAULT_TABLE["creative_long"]["high"] == "medium"


def test_bump_overrides_and_clamps() -> None:
    p = AppPolicy.model_validate({"routing": {"max_tier": "large", "table_overrides": {"code": {"high": "medium"}}}})
    assert target_tier("code", "high", 0.9, p.routing) == ("medium", [])
    assert target_tier("code", "high", 0.4, p.routing) == ("large", ["bump:low-confidence"])
    assert target_tier("code", "high", 0.9, AppPolicy().routing) == ("medium", ["clamp:max=medium"])  # default max_tier
    floor = AppPolicy.model_validate({"routing": {"min_tier": "small"}})
    assert target_tier("chitchat", "low", 0.9, floor.routing) == ("small", ["clamp:min=small"])
    assert target_tier("code", "high", 0.9, AppPolicy().routing, budget_pressure=True)[0] == "small"


# --- router ---------------------------------------------------------------------------
def router_for(*providers: str, breaker: CircuitBreaker | None = None):  # type: ignore[no-untyped-def]
    return make_router(ProviderRegistry({p: FakeAdapter() for p in providers}), breaker)


def test_local_only_routes_by_classification() -> None:
    r = router_for("ollama")
    req = chat_request("hi")
    assert r.plan(req, LOCAL_ONLY, cls("chitchat", "low")).candidates[0].id == "ollama/qwen2.5:1.5b"
    plan = r.plan(req, LOCAL_ONLY, cls("code", "medium"))
    assert (plan.tier, [m.id for m in plan.candidates]) == ("medium", ["ollama/qwen2.5:7b", "ollama/qwen2.5:14b"])
    plan = r.plan(req, LOCAL_ONLY, cls("reasoning_math", "high"))
    assert plan.candidates[0].id == "ollama/qwen2.5:14b" and plan.reason == "classifier:reasoning_math/high@0.90;>large"


def test_hosted_without_keys_downgrades_to_local() -> None:
    plan = router_for("ollama").plan(chat_request("x"), AppPolicy(), cls("code", "medium"))
    assert plan.candidates[0].id == "ollama/qwen2.5:7b"
    assert "downgrade:no-eligible-medium" in plan.reason and plan.rejected["anthropic/claude-sonnet-5-5"] == "provider-not-configured"


def test_hosted_with_keys_uses_hosted_tiers_and_respects_max_tier() -> None:
    r = router_for("ollama", "anthropic")
    plan = r.plan(chat_request("x"), AppPolicy(), cls("code", "high"))  # table: large, clamp max=medium
    assert plan.candidates[0].id == "anthropic/claude-sonnet-5-5" and "clamp:max=medium" in plan.reason
    assert [m.id for m in plan.candidates] == ["anthropic/claude-sonnet-5-5"]  # nothing above max_tier


def test_alias_and_pinning() -> None:
    r = router_for("ollama", "anthropic")
    assert r.plan(chat_request("x", model="tollgate/small"), AppPolicy(), None).candidates[0].id == "anthropic/claude-haiku-5-5"
    pin = AppPolicy.model_validate({"routing": {"allow_pinning": True}})
    plan = r.plan(chat_request("x", model="anthropic/claude-sonnet-5-5"), pin, None)
    assert (plan.reason, plan.candidates[0].id) == ("pinned", "anthropic/claude-sonnet-5-5")
    plan = r.plan(chat_request("x", model="anthropic/claude-sonnet-5-5"), AppPolicy(), cls("chitchat", "low"))
    assert plan.candidates[0].id == "ollama/qwen2.5:7b" and "pin-ignored:pinning-disabled" in plan.reason


def test_hard_constraints_context_window_and_tools() -> None:
    r = router_for("ollama", "anthropic")
    long_req = chat_request("word " * 34_000)  # ~34k Qwen tokens + 1024 out > 32k window
    plan = r.plan(long_req, AppPolicy(), cls("summarization", "low"))
    assert plan.rejected["ollama/qwen2.5:7b"] == "context-window"
    assert plan.candidates[0].id == "anthropic/claude-haiku-5-5" and "up:no-eligible-local" in plan.reason


def test_open_circuit_excludes_model() -> None:
    cb = CircuitBreaker()
    for _ in range(5):
        cb.record_failure("ollama/qwen2.5:1.5b")
    plan = router_for("ollama", breaker=cb).plan(chat_request("hi"), LOCAL_ONLY, cls("chitchat", "low"))
    assert plan.candidates[0].id == "ollama/qwen2.5:7b" and plan.rejected["ollama/qwen2.5:1.5b"] == "circuit-open"


def test_nothing_eligible_is_503() -> None:
    with pytest.raises(GatewayError) as e:
        router_for().plan(chat_request("hi"), AppPolicy(), cls("chitchat", "low"))
    assert e.value.status == 503


# --- invoke: fallback + breaker ---------------------------------------------------------
async def routed(adapter: ModelAdapter, req: ChatRequest, policy: AppPolicy = LOCAL_ONLY, c: Classification | None = None, breaker: CircuitBreaker | None = None):  # type: ignore[no-untyped-def]
    breaker = breaker or CircuitBreaker()
    reg = ProviderRegistry({"ollama": adapter})
    router = make_router(reg, breaker)
    ctx = make_ctx(req, auth=auth_ctx(), policy=policy, classification=c or cls("chitchat", "low"))
    ctx = await RouteStage(router).process(ctx)
    return InvokeStage(make_catalog(), reg, breaker), router, ctx


@pytest.mark.parametrize("stream", [False, True])
async def test_retryable_failure_falls_back_and_is_recorded(stream: bool) -> None:
    adapter = ModelAdapter(fail={"qwen2.5:1.5b": ProviderError("upstream timed out", status=504, code="timeout")})
    invoke, _, ctx = await routed(adapter, chat_request("hi", stream=stream))
    ctx = await invoke.process(ctx)
    if stream:
        [c async for c in ctx.stream]
    assert ctx.route.model_id == "ollama/qwen2.5:7b"
    assert "fallback:ollama/qwen2.5:1.5b(timeout)>ollama/qwen2.5:7b" in ctx.route.reason
    assert [(a.model_id, a.kind, a.status_code, a.error) for a in ctx.attempts] == [
        ("ollama/qwen2.5:1.5b", "primary", 504, "timeout"),
        ("ollama/qwen2.5:7b", "fallback", 200, None),
    ]


async def test_non_retryable_failure_is_returned_without_fallback() -> None:
    adapter = ModelAdapter(fail={"qwen2.5:1.5b": ProviderError("upstream rejected the request", status=400, retryable=False, code="bad_request")})
    invoke, _, ctx = await routed(adapter, chat_request("hi"))
    with pytest.raises(GatewayError) as e:
        await invoke.process(ctx)
    assert e.value.status == 400 and len(ctx.attempts) == 1


async def test_breaker_opens_after_five_failures_then_skips_model() -> None:
    breaker = CircuitBreaker()
    adapter = ModelAdapter(fail={"qwen2.5:1.5b": ProviderError("down", status=503, code="unreachable")})
    for _ in range(5):
        invoke, _, ctx = await routed(adapter, chat_request("hi"), breaker=breaker)
        await invoke.process(ctx)
    assert breaker.state("ollama/qwen2.5:1.5b") == "open"
    calls_before = sum(1 for m, _ in adapter.calls if m == "qwen2.5:1.5b")
    invoke, _, ctx = await routed(adapter, chat_request("hi"), breaker=breaker)
    await invoke.process(ctx)
    assert sum(1 for m, _ in adapter.calls if m == "qwen2.5:1.5b") == calls_before  # skipped, not even tried
    assert ctx.route.model_id == "ollama/qwen2.5:7b" and len(ctx.attempts) == 1


async def test_all_candidates_fail_returns_last_error() -> None:
    err = ProviderError("upstream unreachable", status=503, code="unreachable")
    adapter = ModelAdapter(fail={m: err for m in ("qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b")})
    invoke, _, ctx = await routed(adapter, chat_request("hi"))
    with pytest.raises(GatewayError) as e:
        await invoke.process(ctx)
    assert e.value.status == 503 and len(ctx.attempts) == 3


# --- validate + escalation --------------------------------------------------------------------------
JSON_REQ = dict(response_format={"type": "json_object"})


async def test_invalid_json_escalates_once_and_bills_both() -> None:
    adapter = ModelAdapter(answers={"qwen2.5:1.5b": "Sure! Here you go: {name: x", "qwen2.5:7b": '{"name": "x"}'})
    invoke, router, ctx = await routed(adapter, chat_request("give json", **JSON_REQ), c=cls("extraction", "low"))
    ctx = await invoke.process(ctx)
    ctx = await ValidateStage(invoke, router).process(ctx)
    assert ctx.escalations == 1 and ctx.route.model_id == "ollama/qwen2.5:7b"
    assert ctx.response.choices[0].message.content == '{"name": "x"}'
    assert [(a.model_id, a.kind) for a in ctx.attempts] == [("ollama/qwen2.5:1.5b", "primary"), ("ollama/qwen2.5:7b", "escalation")]
    assert "escalate:invalid_json>medium" in ctx.route.reason


async def test_escalation_limited_by_max_escalations() -> None:
    adapter = ModelAdapter(answers={m: "not json" for m in ("qwen2.5:1.5b", "qwen2.5:7b", "qwen2.5:14b")})
    invoke, router, ctx = await routed(adapter, chat_request("json", **JSON_REQ), c=cls("extraction", "low"))
    ctx = await invoke.process(ctx)
    ctx = await ValidateStage(invoke, router).process(ctx)
    assert ctx.escalations == 1 and len(ctx.attempts) == 2  # default max_escalations = 1


async def test_valid_answer_does_not_escalate_and_streams_skip_validation() -> None:
    adapter = ModelAdapter()
    invoke, router, ctx = await routed(adapter, chat_request("hi"))
    ctx = await ValidateStage(invoke, router).process(await invoke.process(ctx))
    assert ctx.escalations == 0
    invoke, router, ctx = await routed(ModelAdapter(answers={"qwen2.5:1.5b": ""}), chat_request("hi", stream=True))
    ctx = await ValidateStage(invoke, router).process(await invoke.process(ctx))
    assert ctx.escalations == 0


def _resp(text: str = "ok", finish: str = "stop", completion: int = 5, calls: list | None = None) -> ChatResponse:
    return ChatResponse(model="m", choices=[Choice(message=ChatMessage(role="assistant", content=text, tool_calls=calls), finish_reason=finish)], usage=Usage(completion_tokens=completion))


TOOL = {"type": "function", "function": {"name": "f", "parameters": {"type": "object", "properties": {"x": {"type": "integer"}}, "required": ["x"]}}}


@pytest.mark.parametrize(
    ("req_kw", "resp", "why"),
    [
        ({}, _resp(""), "empty"),
        ({}, _resp("I'm sorry, but I can't help with that."), "refusal"),
        ({"max_tokens": 100}, _resp("partial", finish="length", completion=40), "truncated"),
        ({"max_tokens": 40}, _resp("partial", finish="length", completion=40), None),  # hit the client's own cap
        ({"response_format": {"type": "json_schema", "json_schema": {"schema": {"type": "object", "required": ["a"]}}}}, _resp('{"b": 1}'), "invalid_json"),
        ({"tools": [TOOL]}, _resp("", calls=[{"id": "1", "type": "function", "function": {"name": "f", "arguments": '{"x": "nope"}'}}]), "invalid_tool_args"),
        ({"tools": [TOOL]}, _resp("", calls=[{"id": "1", "type": "function", "function": {"name": "g", "arguments": "{}"}}]), "invalid_tool_args"),
        ({"tools": [TOOL]}, _resp("", calls=[{"id": "1", "type": "function", "function": {"name": "f", "arguments": json.dumps({"x": 3})}}]), None),
    ],
)
def test_failure_reasons(req_kw: dict, resp: ChatResponse, why: str | None) -> None:
    assert failure_reason(chat_request("q", **req_kw), resp) == why
