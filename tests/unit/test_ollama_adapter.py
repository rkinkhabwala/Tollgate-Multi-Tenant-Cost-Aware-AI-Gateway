"""Ollama adapter translation, against respx-mocked HTTP (no Ollama needed)."""

import json

import httpx
import pytest
import respx

from gateway.providers.base import ChatMessage, ChatRequest, ProviderError
from gateway.providers.ollama import OllamaAdapter

BASE = "http://ollama.test"


@pytest.fixture
def adapter() -> OllamaAdapter:
    return OllamaAdapter(BASE)


def req(**kw) -> ChatRequest:
    return ChatRequest(messages=[ChatMessage(role="system", content="be brief"), ChatMessage(role="user", content="hi")], **kw)


@respx.mock
async def test_chat_translates_request_and_response(adapter: OllamaAdapter) -> None:
    route = respx.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "hello!"},
                "done": True,
                "done_reason": "stop",
                "prompt_eval_count": 17,
                "eval_count": 4,
            },
        )
    )
    resp = await adapter.chat("qwen2.5:7b", req(temperature=0.2, max_tokens=8, response_format={"type": "json_object"}))
    sent = json.loads(route.calls[0].request.content)
    assert sent["model"] == "qwen2.5:7b" and sent["stream"] is False
    assert sent["options"] == {"temperature": 0.2, "num_predict": 8}
    assert sent["format"] == "json"
    assert [m["role"] for m in sent["messages"]] == ["system", "user"]
    assert resp.choices[0].message.content == "hello!"
    assert resp.choices[0].finish_reason == "stop"
    assert (resp.usage.prompt_tokens, resp.usage.completion_tokens, resp.usage.total_tokens) == (17, 4, 21)


@respx.mock
async def test_length_finish_and_tool_calls(adapter: OllamaAdapter) -> None:
    respx.post(f"{BASE}/api/chat").mock(
        return_value=httpx.Response(
            200,
            json={
                "message": {"role": "assistant", "content": "", "tool_calls": [{"function": {"name": "get_weather", "arguments": {"city": "Paris"}}}]},
                "done": True,
                "done_reason": "stop",
            },
        )
    )
    resp = await adapter.chat("qwen2.5:7b", req(tools=[{"type": "function", "function": {"name": "get_weather"}}]))
    tc = resp.choices[0].message.tool_calls[0]
    assert resp.choices[0].finish_reason == "tool_calls"
    assert tc["function"]["name"] == "get_weather" and json.loads(tc["function"]["arguments"]) == {"city": "Paris"}


@respx.mock
async def test_stream_ndjson_to_chunks(adapter: OllamaAdapter) -> None:
    frames = [
        {"message": {"role": "assistant", "content": "Hel"}, "done": False},
        {"message": {"role": "assistant", "content": "lo"}, "done": False},
        {"message": {"role": "assistant", "content": ""}, "done": True, "done_reason": "length", "prompt_eval_count": 9, "eval_count": 2},
    ]
    body = "\n".join(json.dumps(f) for f in frames) + "\n"
    respx.post(f"{BASE}/api/chat").mock(return_value=httpx.Response(200, content=body.encode()))
    chunks = [c async for c in adapter.stream("qwen2.5:7b", req(stream=True))]
    assert chunks[0].choices[0].delta == {"role": "assistant", "content": "Hel"}
    assert "".join(c.choices[0].delta.get("content", "") for c in chunks) == "Hello"
    assert chunks[-1].choices[0].finish_reason == "length"
    assert chunks[-1].usage is not None and chunks[-1].usage.total_tokens == 11
    assert len({c.id for c in chunks}) == 1


@respx.mock
@pytest.mark.parametrize(
    ("mock", "status"),
    [
        (httpx.Response(404, json={"error": "model 'qwen2.5:7b' not found"}), 503),
        (httpx.Response(500, text="boom"), 502),
        (httpx.ConnectError("refused"), 503),
        (httpx.ReadTimeout("slow"), 504),
    ],
)
async def test_errors_map_to_provider_error(adapter: OllamaAdapter, mock, status: int) -> None:
    route = respx.post(f"{BASE}/api/chat")
    route.mock(side_effect=mock) if isinstance(mock, Exception) else route.mock(return_value=mock)
    with pytest.raises(ProviderError) as e:
        await adapter.chat("qwen2.5:7b", req())
    assert e.value.status == status
    assert "hi" not in e.value.message and "be brief" not in e.value.message


@respx.mock
async def test_embed(adapter: OllamaAdapter) -> None:
    respx.post(f"{BASE}/api/embed").mock(return_value=httpx.Response(200, json={"embeddings": [[0.1, 0.2]]}))
    assert await adapter.embed("nomic-embed-text", ["x"]) == [[0.1, 0.2]]
