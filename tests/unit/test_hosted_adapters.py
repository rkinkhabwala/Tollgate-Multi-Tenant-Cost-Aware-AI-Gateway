"""Anthropic + OpenAI adapters against respx fixtures; circuit breaker state machine."""

from __future__ import annotations

import json

import httpx
import pytest
import respx

from gateway.providers.anthropic import JSON_ONLY, AnthropicAdapter, adapt_schema, to_anthropic
from gateway.providers.base import ChatMessage, ChatRequest, ProviderError
from gateway.providers.circuit_breaker import CircuitBreaker
from gateway.providers.openai import OpenAIAdapter

A = "https://anthropic.test"
O = "https://openai.test"
WEATHER_TOOL = {"type": "function", "function": {"name": "get_weather", "description": "Weather", "parameters": {"type": "object", "properties": {"city": {"type": "string"}}, "required": ["city"]}}}


def sse(events: list[dict]) -> bytes:
    return "".join(f"event: {e['type']}\ndata: {json.dumps(e)}\n\n" for e in events).encode()


# --- request translation --------------------------------------------------------------
def test_request_translation_full_conversation() -> None:
    req = ChatRequest(
        messages=[
            ChatMessage(role="system", content="Be brief."),
            ChatMessage(role="user", content="Weather in Paris and Rome?"),
            ChatMessage(role="assistant", content=None, tool_calls=[
                {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}},
                {"id": "call_2", "type": "function", "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'}},
            ]),
            ChatMessage(role="tool", tool_call_id="call_1", content="18C"),
            ChatMessage(role="tool", tool_call_id="call_2", content="24C"),
        ],
        tools=[WEATHER_TOOL],
        tool_choice="required",
        temperature=0.3,
        top_p=0.9,
        stop="END",
    )
    body = to_anthropic("claude-sonnet-5-5", req)
    assert body["system"].startswith("Be brief.") and "must call" in body["system"]
    assert body["tool_choice"] == {"type": "auto"}  # forced choice is a 400 on 5.5 models
    assert "temperature" not in body and "top_p" not in body  # non-default sampling rejected on 5.5
    assert body["max_tokens"] == 4096 and body["stop_sequences"] == ["END"]
    assert body["tools"][0] == {"name": "get_weather", "description": "Weather", "input_schema": WEATHER_TOOL["function"]["parameters"]}
    user, assistant, results = body["messages"]
    assert user == {"role": "user", "content": "Weather in Paris and Rome?"}
    assert [b["input"] for b in assistant["content"]] == [{"city": "Paris"}, {"city": "Rome"}]
    assert assistant["content"][0]["id"] == "call_1"
    assert results["role"] == "user" and [b["tool_use_id"] for b in results["content"]] == ["call_1", "call_2"]  # merged


def test_json_modes() -> None:
    schema = {"type": "object", "properties": {"n": {"type": "integer", "minimum": 0}, "tags": {"type": "array", "items": {"type": "string"}, "maxItems": 3}}}
    body = to_anthropic("m", ChatRequest(messages=[ChatMessage(role="user", content="x")], response_format={"type": "json_schema", "json_schema": {"name": "s", "schema": schema}}))
    sent = body["output_config"]["format"]
    assert sent["type"] == "json_schema" and sent["schema"]["additionalProperties"] is False
    assert "minimum" not in json.dumps(sent) and "maxItems" not in json.dumps(sent)
    body = to_anthropic("m", ChatRequest(messages=[ChatMessage(role="user", content="x")], response_format={"type": "json_object"}))
    assert "output_config" not in body and JSON_ONLY in body["system"]
    assert adapt_schema({"type": "object", "properties": {"a": {"type": "object", "properties": {}}}})["properties"]["a"]["additionalProperties"] is False


def test_image_parts() -> None:
    req = ChatRequest(messages=[ChatMessage(role="user", content=[{"type": "text", "text": "what?"}, {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAAA"}}])])
    blocks = to_anthropic("m", req)["messages"][0]["content"]
    assert blocks[1] == {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "AAAA"}}


# --- responses ---------------------------------------------------------------------------
@respx.mock
async def test_chat_response_with_tool_use_and_thinking_dropped() -> None:
    route = respx.post(f"{A}/v1/messages").mock(return_value=httpx.Response(200, json={
        "id": "msg_1", "type": "message", "role": "assistant", "stop_reason": "tool_use",
        "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Checking."},
                    {"type": "tool_use", "id": "toolu_9", "name": "get_weather", "input": {"city": "Oslo"}}],
        "usage": {"input_tokens": 40, "cache_read_input_tokens": 10, "output_tokens": 12},
    }))
    resp = await AnthropicAdapter("sk-test", base_url=A).chat("claude-haiku-5-5", ChatRequest(messages=[ChatMessage(role="user", content="Oslo?")], tools=[WEATHER_TOOL]))
    sent = route.calls[0].request
    assert sent.headers["x-api-key"] == "sk-test" and sent.headers["anthropic-version"] == "2023-06-01"
    msg = resp.choices[0].message
    assert msg.content == "Checking." and resp.choices[0].finish_reason == "tool_calls"
    assert msg.tool_calls[0]["id"] == "toolu_9" and json.loads(msg.tool_calls[0]["function"]["arguments"]) == {"city": "Oslo"}
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens) == (50, 12)


@respx.mock
async def test_stream_text_and_tool_call() -> None:
    events = [
        {"type": "message_start", "message": {"id": "msg_1", "usage": {"input_tokens": 25, "output_tokens": 1}}},
        {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
        {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "hmm"}},
        {"type": "content_block_stop", "index": 0},
        {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
        {"type": "ping"},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "Let me "}},
        {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "check."}},
        {"type": "content_block_stop", "index": 1},
        {"type": "content_block_start", "index": 2, "content_block": {"type": "tool_use", "id": "toolu_1", "name": "get_weather", "input": {}}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '{"city": '}},
        {"type": "content_block_delta", "index": 2, "delta": {"type": "input_json_delta", "partial_json": '"Lima"}'}},
        {"type": "content_block_stop", "index": 2},
        {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 17}},
        {"type": "message_stop"},
    ]
    respx.post(f"{A}/v1/messages").mock(return_value=httpx.Response(200, content=sse(events)))
    chunks = [c async for c in AnthropicAdapter("k", base_url=A).stream("claude-sonnet-5-5", ChatRequest(messages=[ChatMessage(role="user", content="Lima?")], stream=True))]
    text = "".join(c.choices[0].delta.get("content", "") for c in chunks)
    args = "".join(tc["function"].get("arguments", "") for c in chunks for tc in c.choices[0].delta.get("tool_calls", []))
    assert text == "Let me check." and json.loads(args) == {"city": "Lima"}
    assert chunks[0].choices[0].delta["role"] == "assistant"
    assert chunks[-1].choices[0].finish_reason == "tool_calls"
    assert (chunks[-1].usage.prompt_tokens, chunks[-1].usage.completion_tokens) == (25, 17)
    assert "hmm" not in json.dumps([c.model_dump() for c in chunks])


@respx.mock
async def test_stream_error_event_is_retryable_overload() -> None:
    events = [{"type": "message_start", "message": {"usage": {"input_tokens": 3}}}, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}]
    respx.post(f"{A}/v1/messages").mock(return_value=httpx.Response(200, content=sse(events)))
    with pytest.raises(ProviderError) as e:
        [c async for c in AnthropicAdapter("k", base_url=A).stream("m", ChatRequest(messages=[ChatMessage(role="user", content="x")], stream=True))]
    assert e.value.status == 503 and e.value.retryable


@respx.mock
@pytest.mark.parametrize(
    ("status", "code", "retryable"),
    [(429, "rate_limited", True), (529, "overloaded", True), (500, "server_error", True), (401, "auth", False), (400, "bad_request", False), (404, "not_found", False)],
)
async def test_anthropic_http_errors(status: int, code: str, retryable: bool) -> None:
    respx.post(f"{A}/v1/messages").mock(return_value=httpx.Response(status, json={"type": "error", "error": {"message": "echo: SECRET PROMPT"}}))
    with pytest.raises(ProviderError) as e:
        await AnthropicAdapter("k", base_url=A).chat("m", ChatRequest(messages=[ChatMessage(role="user", content="SECRET PROMPT")]))
    assert (e.value.code, e.value.retryable) == (code, retryable)
    assert "SECRET" not in e.value.message


@respx.mock
async def test_anthropic_timeout() -> None:
    respx.post(f"{A}/v1/messages").mock(side_effect=httpx.ReadTimeout("slow"))
    with pytest.raises(ProviderError) as e:
        await AnthropicAdapter("k", base_url=A).chat("m", ChatRequest(messages=[ChatMessage(role="user", content="x")]))
    assert e.value.code == "timeout" and e.value.retryable


# --- OpenAI ---------------------------------------------------------------------------------------
@respx.mock
async def test_openai_passthrough_and_stream_usage() -> None:
    chat = respx.post(f"{O}/v1/chat/completions")
    chat.side_effect = [
        httpx.Response(200, json={"id": "c1", "object": "chat.completion", "created": 1, "model": "gpt-x",
                                  "choices": [{"index": 0, "message": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}],
                                  "usage": {"prompt_tokens": 5, "completion_tokens": 1, "total_tokens": 6}}),
        httpx.Response(200, content=(
            'data: {"id":"c2","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[{"index":0,"delta":{"content":"h"}}]}\n\n'
            'data: {"id":"c2","object":"chat.completion.chunk","created":1,"model":"gpt-x","choices":[],"usage":{"prompt_tokens":5,"completion_tokens":1,"total_tokens":6}}\n\n'
            "data: [DONE]\n\n").encode()),
    ]
    adapter = OpenAIAdapter("sk-openai", base_url=O)
    resp = await adapter.chat("gpt-x", ChatRequest(messages=[ChatMessage(role="user", content="hi")], temperature=0.2, extra_field=1))
    sent = json.loads(chat.calls[0].request.content)
    assert sent["model"] == "gpt-x" and sent["temperature"] == 0.2 and "extra_field" not in sent
    assert chat.calls[0].request.headers["authorization"] == "Bearer sk-openai"
    assert resp.choices[0].message.content == "hi" and resp.usage.total_tokens == 6
    chunks = [c async for c in adapter.stream("gpt-x", ChatRequest(messages=[ChatMessage(role="user", content="hi")], stream=True))]
    assert json.loads(chat.calls[1].request.content)["stream_options"] == {"include_usage": True}
    assert chunks[-1].usage.total_tokens == 6


@respx.mock
async def test_openai_errors() -> None:
    respx.post(f"{O}/v1/chat/completions").mock(return_value=httpx.Response(429))
    with pytest.raises(ProviderError) as e:
        await OpenAIAdapter("k", base_url=O).chat("gpt-x", ChatRequest(messages=[ChatMessage(role="user", content="x")]))
    assert e.value.code == "rate_limited" and e.value.retryable


# --- circuit breaker ---------------------------------------------------------------------------
class Clock:
    def __init__(self) -> None:
        self.t = 1000.0

    def __call__(self) -> float:
        return self.t


def test_breaker_opens_after_5_failures_in_30s_and_half_opens_after_60s() -> None:
    clock = Clock()
    cb = CircuitBreaker(clock=clock)
    for _ in range(4):
        cb.record_failure("m")
        clock.t += 5
    assert cb.state("m") == "closed" and cb.allow("m")
    cb.record_failure("m")
    assert cb.state("m") == "open" and not cb.allow("m") and not cb.available("m")
    clock.t += 59
    assert not cb.allow("m")
    clock.t += 1
    assert cb.available("m") and cb.allow("m")  # half-open: one trial
    assert not cb.allow("m")  # second concurrent trial refused
    cb.record_success("m")
    assert cb.state("m") == "closed" and cb.allow("m")


def test_breaker_failures_outside_window_dont_count_and_failed_trial_reopens() -> None:
    clock = Clock()
    cb = CircuitBreaker(clock=clock)
    for _ in range(4):
        cb.record_failure("m")
    clock.t += 31
    cb.record_failure("m")
    assert cb.state("m") == "closed"  # the first four aged out
    for _ in range(4):
        cb.record_failure("m")
    assert cb.state("m") == "open"
    clock.t += 60
    assert cb.allow("m")
    cb.record_failure("m")  # trial failed
    assert cb.state("m") == "open" and not cb.allow("m")
    assert cb.snapshot() == {"m": "open"}
