"""End-to-end redaction through the HTTP API with the OpenAI SDK, deterministically.

An `EchoAdapter` plays the model: it records exactly what it received and answers
by repeating the last user message. Because that message contains placeholders, the
tests can prove three things:
* upstream never sees raw PII (only placeholders);
* the client gets its real values back, non-streaming and streaming, even when the
  adapter streams in 3-char chunks so every placeholder is split across chunks;
* `x-tollgate-redacted`, the usage event's `pii_entities`, and the `include_usage`
  extension all carry counts only, never values.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator

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
from gateway.pipeline.redact import RedactStage
from gateway.pipeline.rehydrate import RehydrateStage
from gateway.pipeline.route import StaticRouteStage
from gateway.policy.schema import AppPolicy
from gateway.providers.base import ChatChunk, ChatMessage, ChatRequest, ChatResponse, Choice, ChunkChoice, Usage
from gateway.providers.registry import ProviderRegistry
from gateway.redaction.engine import RedactionEngine
from gateway.settings import Settings
from tests.unit.fakes import FakePolicyStore, FakeResolver, make_catalog, make_key_record, make_redis, new_key

STREAM = "sys:usage_events"
EMAIL, PHONE, NAME = "dana.whitfield@northwind.example", "415-555-0188", "Dana Whitfield"


class EchoAdapter:
    def __init__(self) -> None:
        self.seen: list[ChatRequest] = []

    def _reply(self, req: ChatRequest) -> str:
        self.seen.append(req)
        last_user = [m for m in req.messages if m.role == "user"][-1]
        return f"You said: {last_user.text()}"

    async def chat(self, model: str, req: ChatRequest) -> ChatResponse:
        text = self._reply(req)
        return ChatResponse(model=model, choices=[Choice(message=ChatMessage(role="assistant", content=text), finish_reason="stop")], usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    async def stream(self, model: str, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        text = self._reply(req)
        now = int(time.time())
        for i in range(0, len(text), 3):  # 3-char chunks split every placeholder
            yield ChatChunk(id="c", created=now, model=model, choices=[ChunkChoice(delta={"content": text[i : i + 3]})])
        yield ChatChunk(id="c", created=now, model=model, choices=[ChunkChoice(delta={}, finish_reason="stop")], usage=Usage(prompt_tokens=10, completion_tokens=5, total_tokens=15))

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return []

    def count_tokens(self, model: str, req: ChatRequest) -> int:
        return 1


@pytest.fixture
def harness(engine: RedactionEngine):
    key, prefix = new_key()
    redis = make_redis()
    catalog = make_catalog()
    echo = EchoAdapter()
    registry = ProviderRegistry({"ollama": echo})
    limiter = RateLimiter(redis)
    policy = AppPolicy()
    authn = AuthNStage(redis, FakeResolver({prefix: make_key_record(key, prefix)}))
    authz = AuthZStage(FakePolicyStore(policy), catalog)  # type: ignore[arg-type]
    settings = Settings()
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
            [
                authn,
                authz,
                RateLimitStage(limiter),
                RedactStage(engine),
                StaticRouteStage(catalog, registry, "ollama/qwen2.5:7b"),
                InvokeStage(catalog, registry),
                RehydrateStage(),
            ],
            [MeterStage(redis, catalog, limiter, STREAM)],
        ),
        auth_pipeline=Pipeline([authn, authz]),
        admin_auth=None,  # type: ignore[arg-type]
        redaction=engine,
    )
    http = httpx.AsyncClient(transport=httpx.ASGITransport(app=app), base_url="http://t")
    client = AsyncOpenAI(base_url="http://t/v1", api_key=key, http_client=http, max_retries=0)
    return client, echo, redis


PROMPT = f"Please email {EMAIL} or call {PHONE}; ask for {NAME}."


async def test_non_streaming_upstream_sees_placeholders_client_sees_values(harness) -> None:
    client, echo, redis = harness
    raw = await client.chat.completions.with_raw_response.create(model="auto", messages=[{"role": "user", "content": PROMPT}])
    text = raw.parse().choices[0].message.content

    upstream = echo.seen[-1].model_dump_json()
    for value in (EMAIL, PHONE, NAME):
        assert value not in upstream  # never left the gateway
        assert value in text  # re-hydrated for the caller
    assert "<<EMAIL_1>>" in upstream and "<<PHONE_1>>" in upstream and "<<PERSON_1>>" in upstream
    assert echo.seen[-1].messages[0].role == "system" and "placeholder" in echo.seen[-1].messages[0].text()
    assert "<<" not in text
    assert raw.headers["x-tollgate-redacted"] == "3"

    (_, fields), = await redis.xrange(STREAM)
    pii = json.loads(fields[b"pii_entities"])
    assert pii == {"EMAIL_ADDRESS": 1, "PHONE_NUMBER": 1, "PERSON": 1}
    assert all(v.encode() not in b"".join(fields.values()) for v in (EMAIL, PHONE, NAME))


async def test_streaming_rehydrates_placeholders_split_across_chunks(harness) -> None:
    client, echo, _ = harness
    stream = await client.chat.completions.create(
        model="auto", messages=[{"role": "user", "content": PROMPT}], stream=True, stream_options={"include_usage": True}
    )
    text, ext = "", None
    async for chunk in stream:
        text += "".join(c.delta.content or "" for c in chunk.choices)
        ext = (chunk.model_extra or {}).get("tollgate") or ext
    assert text == f"You said: {PROMPT}"  # exact: nothing lost, nothing duplicated
    assert ext["redacted"] == 3
    assert EMAIL not in echo.seen[-1].model_dump_json()


async def test_history_and_system_are_redacted_too(harness) -> None:
    client, echo, _ = harness
    await client.chat.completions.create(
        model="auto",
        messages=[
            {"role": "system", "content": f"The account owner is {NAME}."},
            {"role": "user", "content": f"My email is {EMAIL}"},
            {"role": "assistant", "content": f"Noted, {EMAIL}."},
            {"role": "user", "content": "Thanks"},
        ],
    )
    upstream = echo.seen[-1].model_dump_json()
    assert EMAIL not in upstream and NAME not in upstream
    assert upstream.count("<<EMAIL_1>>") == 2  # same value, same placeholder


async def test_secret_is_blocked_before_upstream(harness) -> None:
    client, echo, _ = harness
    with pytest.raises(openai.BadRequestError) as e:
        await client.chat.completions.create(
            model="auto", messages=[{"role": "user", "content": "deploy with AK\x49AIOSFODNN7EXAMPLE please"}]
        )
    assert e.value.body["code"] == "sensitive_data_blocked"
    assert "AKIA" not in json.dumps(e.value.body)
    assert echo.seen == []  # nothing was sent upstream
