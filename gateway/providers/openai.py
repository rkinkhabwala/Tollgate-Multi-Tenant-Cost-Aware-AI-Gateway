"""OpenAI adapter: the gateway already speaks this wire format, so this is a pass-through.

Not in the model catalog yet (spec §8.1 lists `openai/<small-model>` placeholders);
registered only when `TOLLGATE_OPENAI_API_KEY` is set. It is tested with respx mocks.

* The request is forwarded with only the standard Chat Completions fields; Tollgate
  extras and unknown fields are dropped. For streams, `stream_options.include_usage`
  is forced on so the final chunk carries token counts for billing.
* The response and stream chunks are parsed back into Tollgate's models.
* HTTP errors map to `ProviderError` with the same codes as the other adapters.
"""

from __future__ import annotations

import json
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.providers.base import ChatChunk, ChatRequest, ChatResponse, ProviderError
from gateway.providers.tokens import conservative_count

_FORWARD = (
    "messages", "temperature", "top_p", "max_tokens", "max_completion_tokens", "stop", "seed",
    "presence_penalty", "frequency_penalty", "tools", "tool_choice", "response_format", "user",
)


def _error_for(status: int) -> ProviderError:
    if status == 429:
        return ProviderError("upstream rate limited", status=429, code="rate_limited")
    if status >= 500:
        return ProviderError(f"upstream error (HTTP {status})", status=502, code="server_error")
    if status in (401, 403):
        return ProviderError("provider authentication failed", status=502, retryable=False, code="auth")
    if status == 404:
        return ProviderError("model is not available at the provider", status=503, retryable=False, code="not_found")
    return ProviderError("upstream rejected the request", status=400, retryable=False, code="bad_request")


class OpenAIAdapter:
    def __init__(self, api_key: str, base_url: str = "https://api.openai.com", timeout_s: float = 300.0, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_s, connect=5.0), headers={"authorization": f"Bearer {api_key}"}
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    @staticmethod
    def _body(model: str, req: ChatRequest, stream: bool) -> dict[str, Any]:
        data = req.model_dump(mode="json", exclude_none=True, include=set(_FORWARD))
        body: dict[str, Any] = {"model": model, **data, "stream": stream}
        if stream:
            body["stream_options"] = {"include_usage": True}
        return body

    async def chat(self, model: str, req: ChatRequest) -> ChatResponse:
        try:
            resp = await self._client.post("/v1/chat/completions", json=self._body(model, req, False))
        except httpx.TimeoutException as exc:
            raise ProviderError("upstream timed out", status=504, code="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("upstream unreachable", status=503, code="unreachable") from exc
        if resp.status_code != 200:
            raise _error_for(resp.status_code)
        return ChatResponse.model_validate(resp.json())

    async def stream(self, model: str, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        try:
            async with self._client.stream("POST", "/v1/chat/completions", json=self._body(model, req, True)) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise _error_for(resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    payload = line[5:].strip()
                    if payload == "[DONE]":
                        return
                    yield ChatChunk.model_validate(json.loads(payload))
        except httpx.TimeoutException as exc:
            raise ProviderError("upstream timed out", status=504, code="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("upstream unreachable", status=503, code="unreachable") from exc

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        resp = await self._client.post("/v1/embeddings", json={"model": model, "input": texts})
        if resp.status_code != 200:
            raise _error_for(resp.status_code)
        return [d["embedding"] for d in resp.json()["data"]]

    def count_tokens(self, model: str, req: ChatRequest) -> int:
        return conservative_count(req)
