"""Anthropic adapter: OpenAI Chat Completions shape ⇄ Claude Messages API (spec §8.1).

Raw `httpx`, matching the Ollama adapter: the gateway is a wire-format translator,
and tests mock the HTTP layer with respx fixtures. (The official `anthropic` SDK 1.x
is built on `httpx2`, which respx cannot intercept.) Registered only when
`TOLLGATE_ANTHROPIC_API_KEY` is set; the key comes from env, never from the DB.

Request translation (OpenAI → Anthropic):
* `system` and `developer` messages → top-level `system` (joined in order);
* user content parts → `text` / `image` blocks (data: URLs become base64 sources);
* assistant `tool_calls` → `tool_use` blocks, with arguments JSON-parsed into `input`;
* `role: tool` → a user turn with `tool_result` blocks (consecutive results merged);
* `tools` → `{name, description, input_schema}`;
* `tool_choice` `none`/`auto` map directly. `required` or a named function becomes
  `auto` plus a system instruction, because forced tool choice returns 400 on
  Claude Opus 5.5 / Sonnet 5.5;
* `stop` → `stop_sequences`. `max_tokens` is required by the API; when the client
  omits it, the adapter default is used;
* `temperature` / `top_p` are **not** forwarded: current 5.5 models reject
  non-default sampling parameters;
* `response_format: json_schema` → `output_config.format`. The schema is adapted to
  Anthropic's supported subset: `additionalProperties: false` is added and
  numeric/length/array-size constraints are dropped. Tollgate's validate stage still
  checks the *original* schema. `json_object` (no schema) becomes a system
  instruction.

Response translation (Anthropic → OpenAI):
* `text` blocks → `message.content`, `tool_use` blocks → `tool_calls`; `thinking`
  blocks are dropped;
* `stop_reason`: end_turn/stop_sequence/pause_turn → stop, max_tokens → length,
  tool_use → tool_calls, refusal → content_filter;
* usage: input (+ cache read/creation) and output tokens.

Streaming: Anthropic SSE events (`message_start`, `content_block_start/delta/stop`,
`message_delta`, `message_stop`, `ping`, `error`) become OpenAI `chat.completion.chunk`s.
`text_delta` → content, `input_json_delta` → tool-call argument fragments,
thinking deltas are dropped, and the final chunk carries `finish_reason` and usage.
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
)
from gateway.providers.tokens import conservative_count

API_VERSION = "2023-06-01"
DEFAULT_MAX_TOKENS = 4096
FINISH = {"end_turn": "stop", "stop_sequence": "stop", "pause_turn": "stop", "max_tokens": "length", "tool_use": "tool_calls", "refusal": "content_filter"}
_UNSUPPORTED_SCHEMA_KEYS = {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minLength", "maxLength", "pattern", "minItems", "maxItems", "uniqueItems"}
JSON_ONLY = "Respond with a single valid JSON object and nothing else."


def adapt_schema(schema: Any) -> Any:
    """Map a JSON schema to Anthropic's structured-output subset (see module doc)."""
    if isinstance(schema, list):
        return [adapt_schema(s) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out = {k: adapt_schema(v) for k, v in schema.items() if k not in _UNSUPPORTED_SCHEMA_KEYS}
    if out.get("type") == "object" or "properties" in out:
        out["additionalProperties"] = False
    return out


def _image_block(part: dict[str, Any]) -> dict[str, Any] | None:
    url = (part.get("image_url") or {}).get("url", "")
    if url.startswith("data:") and ";base64," in url:
        media, data = url[5:].split(";base64,", 1)
        return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}
    if url:
        return {"type": "image", "source": {"type": "url", "url": url}}
    return None


def _user_content(m: ChatMessage) -> str | list[dict[str, Any]]:
    if isinstance(m.content, list):
        blocks: list[dict[str, Any]] = []
        for part in m.content:
            if part.get("type") == "text":
                blocks.append({"type": "text", "text": part.get("text", "")})
            elif part.get("type") == "image_url" and (img := _image_block(part)):
                blocks.append(img)
        return blocks
    return m.content or ""


def to_anthropic(model: str, req: ChatRequest, default_max_tokens: int = DEFAULT_MAX_TOKENS) -> dict[str, Any]:
    system: list[str] = []
    messages: list[dict[str, Any]] = []
    for m in req.messages:
        if m.role in ("system", "developer"):
            system.append(m.text())
        elif m.role == "user":
            messages.append({"role": "user", "content": _user_content(m)})
        elif m.role == "assistant":
            blocks: list[dict[str, Any]] = []
            if m.text():
                blocks.append({"type": "text", "text": m.text()})
            for tc in m.tool_calls or []:
                fn = tc.get("function", {})
                try:
                    args = json.loads(fn.get("arguments") or "{}")
                except json.JSONDecodeError:
                    args = {}
                blocks.append({"type": "tool_use", "id": tc.get("id") or f"toolu_{uuid.uuid4().hex[:20]}", "name": fn.get("name", ""), "input": args})
            messages.append({"role": "assistant", "content": blocks or ""})
        elif m.role in ("tool", "function"):
            result = {"type": "tool_result", "tool_use_id": m.tool_call_id or "", "content": m.text()}
            if messages and messages[-1]["role"] == "user" and isinstance(messages[-1]["content"], list) and all(
                b.get("type") == "tool_result" for b in messages[-1]["content"]
            ):
                messages[-1]["content"].append(result)  # merge consecutive tool results
            else:
                messages.append({"role": "user", "content": [result]})

    body: dict[str, Any] = {"model": model, "max_tokens": req.output_limit() or default_max_tokens, "messages": messages}
    if req.stop is not None:
        body["stop_sequences"] = [req.stop] if isinstance(req.stop, str) else req.stop
    if req.tools:
        body["tools"] = [
            {
                "name": t.get("function", {}).get("name", ""),
                "description": t.get("function", {}).get("description", ""),
                "input_schema": t.get("function", {}).get("parameters") or {"type": "object", "properties": {}},
            }
            for t in req.tools
            if t.get("type", "function") == "function"
        ]
        choice = req.tool_choice
        if choice == "none":
            body["tool_choice"] = {"type": "none"}
        elif choice == "required":
            body["tool_choice"] = {"type": "auto"}
            system.append("You must call one of the provided tools to respond.")
        elif isinstance(choice, dict) and choice.get("function", {}).get("name"):
            body["tool_choice"] = {"type": "auto"}
            system.append(f"You must call the `{choice['function']['name']}` tool to respond.")
    rf = req.response_format or {}
    if rf.get("type") == "json_schema" and (schema := (rf.get("json_schema") or {}).get("schema")):
        body["output_config"] = {"format": {"type": "json_schema", "schema": adapt_schema(schema)}}
    elif rf.get("type") in ("json_object", "json_schema"):
        system.append(JSON_ONLY)
    if system:
        body["system"] = "\n\n".join(s for s in system if s)
    return body


def _usage(u: dict[str, Any], output: int | None = None) -> Usage:
    p = int(u.get("input_tokens") or 0) + int(u.get("cache_read_input_tokens") or 0) + int(u.get("cache_creation_input_tokens") or 0)
    c = int(output if output is not None else u.get("output_tokens") or 0)
    return Usage(prompt_tokens=p, completion_tokens=c, total_tokens=p + c)


def from_anthropic(model: str, data: dict[str, Any]) -> ChatResponse:
    text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
    calls = [
        {"id": b["id"], "type": "function", "function": {"name": b["name"], "arguments": json.dumps(b.get("input", {}))}}
        for b in data.get("content", [])
        if b.get("type") == "tool_use"
    ]
    return ChatResponse(
        model=model,
        choices=[Choice(message=ChatMessage(role="assistant", content=text, tool_calls=calls or None), finish_reason=FINISH.get(data.get("stop_reason") or "", "stop"))],
        usage=_usage(data.get("usage") or {}),
    )


def _error_for(status: int) -> ProviderError:
    if status == 429:
        return ProviderError("upstream rate limited", status=429, code="rate_limited")
    if status == 529:
        return ProviderError("upstream overloaded", status=503, code="overloaded")
    if status >= 500:
        return ProviderError(f"upstream error (HTTP {status})", status=502, code="server_error")
    if status in (401, 403):
        return ProviderError("provider authentication failed", status=502, retryable=False, code="auth")
    if status == 404:
        return ProviderError("model is not available at the provider", status=503, retryable=False, code="not_found")
    if status == 413:
        return ProviderError("request too large for the provider", status=400, retryable=False, code="too_large")
    return ProviderError("upstream rejected the request", status=400, retryable=False, code="bad_request")


class AnthropicAdapter:
    """`ProviderAdapter` for the Claude Messages API."""

    def __init__(
        self,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout_s: float = 300.0,
        default_max_tokens: int = DEFAULT_MAX_TOKENS,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self._default_max_tokens = default_max_tokens
        self._client = client or httpx.AsyncClient(
            base_url=base_url.rstrip("/"),
            timeout=httpx.Timeout(timeout_s, connect=5.0),
            headers={"x-api-key": api_key, "anthropic-version": API_VERSION, "content-type": "application/json"},
        )

    async def aclose(self) -> None:
        await self._client.aclose()

    async def chat(self, model: str, req: ChatRequest) -> ChatResponse:
        try:
            resp = await self._client.post("/v1/messages", json=to_anthropic(model, req, self._default_max_tokens))
        except httpx.TimeoutException as exc:
            raise ProviderError("upstream timed out", status=504, code="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("upstream unreachable", status=503, code="unreachable") from exc
        if resp.status_code != 200:
            raise _error_for(resp.status_code)
        return from_anthropic(model, resp.json())

    async def stream(self, model: str, req: ChatRequest) -> AsyncIterator[ChatChunk]:
        body = {**to_anthropic(model, req, self._default_max_tokens), "stream": True}
        chunk_id, created = f"chatcmpl-{uuid.uuid4().hex}", int(time.time())
        usage: dict[str, Any] = {}
        output_tokens = 0
        stop_reason: str | None = None
        tool_index: dict[int, int] = {}  # content block index → OpenAI tool_calls index
        first = True

        def chunk(delta: dict[str, Any], finish: str | None = None, final_usage: Usage | None = None) -> ChatChunk:
            nonlocal first
            if first:
                delta = {"role": "assistant", **delta}
                first = False
            return ChatChunk(id=chunk_id, created=created, model=model, choices=[ChunkChoice(delta=delta, finish_reason=finish)], usage=final_usage)

        try:
            async with self._client.stream("POST", "/v1/messages", json=body) as resp:
                if resp.status_code != 200:
                    await resp.aread()
                    raise _error_for(resp.status_code)
                async for line in resp.aiter_lines():
                    if not line.startswith("data:"):
                        continue
                    event = json.loads(line[5:].strip())
                    etype = event.get("type")
                    if etype == "message_start":
                        usage = event.get("message", {}).get("usage") or {}
                    elif etype == "content_block_start":
                        block = event.get("content_block", {})
                        if block.get("type") == "tool_use":
                            n = tool_index[event["index"]] = len(tool_index)
                            yield chunk({"tool_calls": [{"index": n, "id": block.get("id"), "type": "function", "function": {"name": block.get("name", ""), "arguments": ""}}]})
                    elif etype == "content_block_delta":
                        delta = event.get("delta", {})
                        if delta.get("type") == "text_delta" and delta.get("text"):
                            yield chunk({"content": delta["text"]})
                        elif delta.get("type") == "input_json_delta" and event["index"] in tool_index:
                            yield chunk({"tool_calls": [{"index": tool_index[event["index"]], "function": {"arguments": delta.get("partial_json", "")}}]})
                    elif etype == "message_delta":
                        stop_reason = (event.get("delta") or {}).get("stop_reason") or stop_reason
                        output_tokens = int((event.get("usage") or {}).get("output_tokens") or output_tokens)
                    elif etype == "message_stop":
                        yield chunk({}, FINISH.get(stop_reason or "", "stop"), _usage(usage, output_tokens))
                        return
                    elif etype == "error":
                        err = (event.get("error") or {}).get("type", "")
                        raise ProviderError("upstream error mid-stream", status=503 if err == "overloaded_error" else 502, code="stream_error")
        except httpx.TimeoutException as exc:
            raise ProviderError("upstream timed out", status=504, code="timeout") from exc
        except httpx.HTTPError as exc:
            raise ProviderError("upstream unreachable", status=503, code="unreachable") from exc

    async def embed(self, model: str, texts: list[str]) -> list[list[float]]:
        raise ProviderError("Anthropic does not offer embeddings", status=400, retryable=False, code="unsupported")

    def count_tokens(self, model: str, req: ChatRequest) -> int:
        return conservative_count(req)
