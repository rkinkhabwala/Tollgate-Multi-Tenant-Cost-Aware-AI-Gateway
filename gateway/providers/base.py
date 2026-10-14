"""OpenAI-shaped wire models and the `ProviderAdapter` protocol (spec §13).

The gateway speaks OpenAI's Chat Completions format on both sides: clients send a
`ChatRequest`, adapters return `ChatResponse` / stream `ChatChunk`s. Each adapter
translates to and from its provider's native API.

The request models allow extra fields (`extra="allow"`) so that newer OpenAI SDK
parameters don't break clients. Unknown fields are not forwarded upstream.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from typing import Any, Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")

    role: Literal["system", "developer", "user", "assistant", "tool", "function"]
    content: str | list[dict[str, Any]] | None = None
    name: str | None = None
    tool_calls: list[dict[str, Any]] | None = None
    tool_call_id: str | None = None

    def text(self) -> str:
        """Concatenated text content (text parts only)."""
        if self.content is None:
            return ""
        if isinstance(self.content, str):
            return self.content
        return "".join(p.get("text", "") for p in self.content if p.get("type") == "text")


class StreamOptions(BaseModel):
    model_config = ConfigDict(extra="allow")
    include_usage: bool = False


class ChatRequest(BaseModel):
    """`POST /v1/chat/completions` body. Never logged (it carries raw prompts)."""

    model_config = ConfigDict(extra="allow")

    model: str = "auto"
    messages: list[ChatMessage] = Field(min_length=1)
    temperature: float | None = Field(None, ge=0, le=2)
    top_p: float | None = Field(None, ge=0, le=1)
    max_tokens: int | None = Field(None, ge=1)
    max_completion_tokens: int | None = Field(None, ge=1)
    stream: bool = False
    stream_options: StreamOptions | None = None
    stop: str | list[str] | None = None
    seed: int | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    tools: list[dict[str, Any]] | None = None
    tool_choice: str | dict[str, Any] | None = None
    response_format: dict[str, Any] | None = None
    n: int | None = None
    user: str | None = None

    @field_validator("n")
    @classmethod
    def _single_choice(cls, v: int | None) -> int | None:
        if v not in (None, 1):
            raise ValueError("only n=1 is supported")
        return v

    def output_limit(self) -> int | None:
        return self.max_completion_tokens or self.max_tokens


class Usage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str | None = None


class ChatResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex}")
    object: Literal["chat.completion"] = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage = Field(default_factory=Usage)


class ChunkChoice(BaseModel):
    index: int = 0
    delta: dict[str, Any] = Field(default_factory=dict)
    finish_reason: str | None = None


class ChatChunk(BaseModel):
    id: str
    object: Literal["chat.completion.chunk"] = "chat.completion.chunk"
    created: int
    model: str
    choices: list[ChunkChoice] = Field(default_factory=list)
    usage: Usage | None = None
    # Tollgate extension on the final `include_usage` chunk: routed model, cache
    # status, cost/savings, redaction count. OpenAI SDKs keep unknown fields, so
    # clients can read it via `chunk.model_extra["tollgate"]`.
    tollgate: dict[str, Any] | None = None


class ProviderError(Exception):
    """Upstream failure. `message` is safe for clients/logs (no prompt content).

    `retryable` failures (timeouts, connection errors, 429, 5xx, overload) trigger
    availability fallback to the next candidate model and count against that model's
    circuit breaker. Non-retryable ones (400, auth, unknown model) don't.
    `code` is a short machine string recorded on the ledger attempt.
    """

    def __init__(self, message: str, *, status: int = 502, retryable: bool = True, code: str = "upstream_error") -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        self.retryable = retryable
        self.code = code


class ProviderAdapter(Protocol):
    """Translate OpenAI-shaped requests to a provider and back (spec §13)."""

    async def chat(self, model: str, req: ChatRequest) -> ChatResponse: ...
    def stream(self, model: str, req: ChatRequest) -> AsyncIterator[ChatChunk]: ...
    async def embed(self, model: str, texts: list[str]) -> list[list[float]]: ...
    def count_tokens(self, model: str, req: ChatRequest) -> int: ...


def estimate_tokens(req: ChatRequest) -> int:
    """Cheap, provider-agnostic input-token estimate (~4 chars/token + per-message overhead).

    Used before the call for limits and TPM reservation. Actual counts reported by the
    provider replace it for billing.
    """
    chars = sum(len(m.text()) + len(m.role) for m in req.messages)
    if req.tools:
        chars += sum(len(str(t)) for t in req.tools)
    return chars // 4 + 4 * len(req.messages) + 3
