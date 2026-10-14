"""Ollama adapter using the native API (`/api/chat`, `/api/embed`).

The native API is used instead of Ollama's OpenAI-compatible shim because it reports
exact token counts (`prompt_eval_count`, `eval_count`) on the final frame, in both
streaming and non-streaming mode. Billing and TPM reconciliation use those counts.

Translation:
* OpenAI `messages` → Ollama `messages`. Content parts are flattened to text, and
  assistant tool-call arguments are converted from a JSON string to an object.
* `temperature/top_p/max_tokens/stop/seed/penalties` → Ollama `options`.
* `response_format: json_object | json_schema` → Ollama `format`.
* Ollama `done_reason` → OpenAI `finish_reason` (`stop`, `length`, `tool_calls`).
* Streaming: NDJSON frames → `ChatChunk`s; the final frame carries usage.
"""

from __future__ import annotations

import json
import time
import uuid
from collections.abc import AsyncIterator
from typing import Any

import httpx

from gateway.providers.base import (
    ChatChunk,
    ChatMessage,
    ChatRequest,
    ChatResponse,
    Choice,
    ChunkChoice,
    ProviderError,
    Usage,
    estimate_tokens,
)


def _to_ollama_messages(req: ChatRequest) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for m in req.messages:
        role = "system" if m.role == "developer" else m.role
        msg: dict[str, Any] = {"role": role, "content": m.text()}
        if m.tool_calls:
            calls = []
            for tc in m.tool_calls:
                fn = tc.get("function", {})
                args = fn.get("arguments", {})
                if isinstance(args, str):
                    try:
                        args = json.loads(args) if args else {}
                    except json.JSONDecodeError:
                        args = {}
                calls.append({"function": {"name": fn.get("name", ""), "arguments": args}})
            msg["tool_calls"] = calls
        if m.role == "tool" and m.name:
            msg["tool_name"] = m.name
        out.append(msg)
    return out


def _options(req: ChatRequest) -> dict[str, Any]:
    opts: dict[str, Any] = {}
    if req.temperature is not None:
        opts["temperature"] = req.temperature
    if req.top_p is not None:
        opts["top_p"] = req.top_p
    if (limit := req.output_limit()) is not None:
        opts["num_predict"] = limit
    if req.stop is not None:
        opts["stop"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if req.seed is not None:
        opts["seed"] = req.seed
    if req.presence_penalty is not None:
        opts["presence_penalty"] = req.presence_penalty
    if req.frequency_penalty is not None:
        opts["frequency_penalty"] = req.frequency_penalty
    return opts


def _format(req: ChatRequest) -> Any:
    rf = req.response_format or {}
    if rf.get("type") == "json_object":
        return "json"
    if rf.get("type") == "json_schema":
        return rf.get("json_schema", {}).get("schema") or "json"
    return None


def _openai_tool_calls(calls: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not calls:
        return None
    return [
        {
            "id": f"call_{uuid.uuid4().hex[:24]}",
            "type": "function",
            "function": {
                "name": c.get("function", {}).get("name", ""),
                "arguments": json.dumps(c.get("function", {}).get("arguments", {})),
            },
        }
        for c in calls
    ]


def _finish_reason(frame: dict[str, Any], has_tool_calls: bool) -> str:
    if has_tool_calls:
        return "tool_calls"
    return "length" if frame.get("done_reason") == "length" else "stop"


def _usage(frame: dict[str, Any]) -> Usage:
    p, c = int(frame.get("prompt_eval_count") or 0), int(frame.get("eval_count") or 0)
    return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


class OllamaAdapter:
    """`ProviderAdapter` for a local or remote Ollama server."""

    def __init__(self, base_url: str, timeout_s: float = 300.0, client: httpx.AsyncClient | None = None) -> None:
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"), timeout=httpx.Timeout(timeout_s, connect=5.0)
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    def _payload(self, model: str, req: ChatRequest, stream: bool) -> dict[str, Any]:
        payload: dict[str, Any] = {"model": model, "messages": _to_ollama_messages(req), "stream": stream}
        if opts := _options(req):
            payload["options"] = opts
        if (fmt := _format(req)) is not None:
            payload["format"] = fmt
        if req.tools:
            payload["tools"] = req.tools
        return payload

    @staticmethod
    def _raise_for(resp_status: int, body: bytes, model: str) -> None:
        # The body may echo request content, so it never goes into the message.
        if resp_status == 404:
            raise ProviderError(f"model '{model}' is not available on the Ollama server", status=503, retryable=False, code="not_found")
        if resp_status == 400:
            raise ProviderError("upstream rejected the request", status=400, retryable=False, code="bad_request")
        raise ProviderError(f"upstream error (HTTP {resp_status})", status=502, code="server_error")

    async def chat(self, model: str, req: ChatRequest) -> ChatResponse:
        try:
            resp = await self._client.post("/api/chat", json=self._payload(model, req, stream=False))
        except httpx.TimeoutException as exc:
            raise ProviderError("upstream timed out", status=504, code="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("upstream unreachable", status=503, code="unreachable") from exc
        if resp.status_code != 200:
            self._raise_for(resp.status_code, resp.content, model)
        frame = resp.json()
        msg = frame.get("message", {})
        tool_calls = _openai_tool_calls(msg.get("tool_calls"))
        return ChatResponse(
            model=model,
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content=msg.get("content", ""), tool_calls=tool_calls),
                    finish_reason=_finish_reason(frame, bool(tool_calls)),
                )
            ],
            usage=_usage(frame),
        )

    async def stream(self, model: str, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        chunk_id = f"chatcmpl-{uuid.uuid4().hex}"
        created = int(time.time())
        first = True
        saw_tool_calls = False
        try:
            async with self._client.stream("POST", "/api/chat", json=self._payload(model, req, stream=True)) as resp:
                if resp.status_code != 200:
                    self._raise_for(resp.status_code, await resp.aread(), model)
                async for line in resp.aiter_lines():
                    if not line.strip():
                        continue
                    frame = json.loads(line)
                    if "error" in frame:
                        raise ProviderError("upstream error mid-stream", status=502, code="stream_error")
                    msg = frame.get("message", {})
                    delta: dict[str, Any] = {}
                    if first:
                        delta["role"] = "assistant"
                        first = False
                    if msg.get("content"):
                        delta["content"] = msg["content"]
                    tool_calls = _openai_tool_calls(msg.get("tool_calls"))
                    if tool_calls:
                        saw_tool_calls = True
                        delta["tool_calls"] = [{"index": i, **tc} for i, tc in enumerate(tool_calls)]
                    if frame.get("done"):
                        if delta:
                            yield ChatChunk(id=chunk_id, created=created, model=model, choices=[ChunkChoice(delta=delta)])
                        yield ChatChunk(
                            id=chunk_id,
                            created=created,
                            model=model,
                            choices=[ChunkChoice(delta={}, finish_reason=_finish_reason(frame, saw_tool_calls))],
                            usage=_usage(frame),
                        )
                        return
                    if delta:
                        yield ChatChunk(id=chunk_id, created=created, model=model, choices=[ChunkChoice(delta=delta)])
        except httpx.TimeoutException as exc:
            raise ProviderError("upstream timed out", status=504, code="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("upstream unreachable", status=503, code="unreachable") from exc

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        try:
            resp = await self._client.post("/api/embed", json={"model": model, "input": texts})
        except httpx.HTTPError as exc:
            raise ProviderError("upstream unreachable", status=503, code="unreachable") from exc
        if resp.status_code != 200:
            self._raise_for(resp.status_code, resp.content, model)
        return resp.json()["embeddings"]

    def count_tokens(self, model: str, req: ChatRequest) -> int:
        return estimate_tokens(req)
