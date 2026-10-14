"""In-memory fakes so pipeline stages can be unit-tested without Postgres or Ollama.

Redis is faked with `fakeredis` (including Lua, for the rate-limit scripts).
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from decimal import Decimal

import fakeredis

from gateway.catalog import CatalogModel, ModelCatalog
from gateway.pipeline.authn import KeyRecord
from gateway.pipeline.context import AuthContext, RequestContext
from gateway.policy.schema import AppPolicy
from gateway.providers.base import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    ProviderError,
    Usage,
)
from gateway.security.api_keys import generate_api_key, hash_api_key

TENANT = uuid.UUID("11111111-1111-1111-1111-111111111111")
APP = uuid.UUID("22222222-2222-2222-2222-222222222222")
KEY_ID = uuid.UUID("33333333-3333-3333-3333-333333333333")


def make_redis() -> fakeredis.aioredis.FakeRedis:
    return fakeredis.aioredis.FakeRedis()


def make_catalog() -> ModelCatalog:
    return ModelCatalog(
        [
            CatalogModel("ollama/qwen2.5:7b", "ollama", "qwen2.5:7b", "local", Decimal(0), Decimal(0), 32768, True, True),
            CatalogModel("anthropic/claude-haiku-5-5", "anthropic", "claude-haiku-5-5", "small", Decimal(1), Decimal(5), 200000, True, True),
            CatalogModel("anthropic/claude-sonnet-5-5", "anthropic", "claude-sonnet-5-5", "medium", Decimal(3), Decimal(15), 200000, True, True),
        ]
    )


def make_key_record(full_key: str, prefix: str, **overrides: object) -> KeyRecord:
    base = dict(
        key_id=str(KEY_ID),
        tenant_id=str(TENANT),
        app_id=str(APP),
        prefix=prefix,
        secret_hash_hex=hash_api_key(full_key).hex(),
        scopes=["chat:write", "usage:read"],
        ip_allowlist=[],
        expires_at=None,
        revoked_at=None,
        tenant_status="active",
        app_status="active",
        baseline_model="anthropic/claude-sonnet-5-5",
        policy_version=1,
    )
    base.update(overrides)
    return KeyRecord(**base)  # type: ignore[arg-type]


class FakeResolver:
    def __init__(self, records: dict[str, KeyRecord] | None = None) -> None:
        self.records = records or {}
        self.calls = 0

    async def by_prefix(self, prefix: str) -> KeyRecord | None:
        self.calls += 1
        return self.records.get(prefix)


def new_key() -> tuple[str, str]:
    return generate_api_key("live")


class FakePolicyStore:
    def __init__(self, policy: AppPolicy | None = None) -> None:
        self.policy = policy or AppPolicy()

    async def get(self, tenant_id: uuid.UUID, app_id: uuid.UUID, version: int) -> AppPolicy:
        return self.policy


def auth_ctx(**overrides: object) -> AuthContext:
    base = dict(
        tenant_id=TENANT,
        app_id=APP,
        key_id=KEY_ID,
        key_prefix="abcd1234",
        scopes=frozenset({"chat:write", "usage:read"}),
        policy_version=1,
        baseline_model="anthropic/claude-sonnet-5-5",
    )
    base.update(overrides)
    return AuthContext(**base)  # type: ignore[arg-type]


def chat_request(content: str = "hello", **kw: object) -> ChatRequest:
    return ChatRequest(model=kw.pop("model", "auto"), messages=[ChatMessage(role="user", content=content)], **kw)  # type: ignore[arg-type]


def make_ctx(req: ChatRequest | None = None, *, scope: str | None = "chat:write", **kw: object) -> RequestContext:
    ctx = RequestContext(request_id="req_test", endpoint="chat", required_scope=scope, client_ip="10.0.0.5")
    ctx.original = req
    for k, v in kw.items():
        setattr(ctx, k, v)
    return ctx


class FakeAdapter:
    """Scriptable ProviderAdapter."""

    def __init__(self, text: str = "hi there", usage: tuple[int, int] = (12, 3), error: ProviderError | None = None) -> None:
        self.text = text
        self.usage = usage
        self.error = error
        self.calls: list[tuple[str, ChatRequest]] = []

    async def chat(self, model: str, req: ChatRequest) -> ChatResponse:
        self.calls.append((model, req))
        if self.error:
            raise self.error
        p, c = self.usage
        return ChatResponse(
            model=model,
            choices=[Choice(message=ChatMessage(role="assistant", content=self.text), finish_reason="stop")],
            usage=Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c),
        )

    async def stream(self, model: str, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        self.calls.append((model, req))
        if self.error:
            raise self.error
        now = int(time.time())
        yield ChatChunk(id="c1", created=now, model=model, choices=[ChunkChoice(delta={"role": "assistant"})])
        for word in self.text.split(" "):
            yield ChatChunk(id="c1", created=now, model=model, choices=[ChunkChoice(delta={"content": word + " "})])
        p, c = self.usage
        yield ChatChunk(
            id="c1",
            created=now,
            model=model,
            choices=[ChunkChoice(delta={}, finish_reason="stop")],
            usage=Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c),
        )

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        return [[0.0] for _ in texts]

    def count_tokens(self, model: str, req: ChatRequest) -> int:
        return 1
