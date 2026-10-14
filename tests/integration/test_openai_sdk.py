"""Phase 1 "done when": apps in two tenants call the real gateway with the official OpenAI
SDK (backed by a real Ollama model), and each sees only its own `/v1/usage`.

These tests are slow on CPU-only Ollama (seconds per call), so prompts and
`max_tokens` are tiny.
"""

import asyncio

import httpx
import pytest
from openai import AsyncOpenAI

from tests.conftest import Provisioned, TenantFactory

pytestmark = [pytest.mark.stack, pytest.mark.ollama]

MSG = [{"role": "user", "content": "Reply with the single word: pong"}]


@pytest.fixture(scope="module")
async def two(tenant_factory: TenantFactory) -> tuple[Provisioned, Provisioned]:
    return await tenant_factory("sdk-a"), await tenant_factory("sdk-b")


@pytest.fixture(scope="module")
def sdk(gateway_url: str):
    def make(key: str) -> AsyncOpenAI:
        return AsyncOpenAI(base_url=f"{gateway_url}/v1", api_key=key, timeout=300, max_retries=0)

    return make


async def usage(gateway: httpx.AsyncClient, key: str) -> dict:
    return (await gateway.get("/v1/usage", headers={"authorization": f"Bearer {key}"})).json()


async def wait_for_requests(gateway: httpx.AsyncClient, key: str, n: int, timeout_s: float = 5.0) -> dict:
    """Usage lands via Redis Stream → batch writer (~1 s)."""
    deadline = asyncio.get_running_loop().time() + timeout_s
    while True:
        u = await usage(gateway, key)
        if u["totals"]["requests"] >= n or asyncio.get_running_loop().time() > deadline:
            return u
        await asyncio.sleep(0.25)


async def test_two_tenants_use_sdk_and_see_only_their_own_usage(gateway: httpx.AsyncClient, two, sdk) -> None:
    a, b = two
    client_a, client_b = sdk(a.key), sdk(b.key)

    # Tenant A: one non-streaming call, with headers.
    raw = await client_a.chat.completions.with_raw_response.create(model="auto", messages=MSG, max_tokens=5, temperature=0)
    completion = raw.parse()
    assert completion.choices[0].message.content
    assert completion.model == "ollama/qwen2.5:7b"
    assert completion.usage.prompt_tokens > 0
    assert raw.headers["x-tollgate-model"] == "ollama/qwen2.5:7b"
    assert raw.headers["x-tollgate-cache"] == "miss"
    assert float(raw.headers["x-tollgate-cost-usd"]) == 0.0
    assert float(raw.headers["x-tollgate-saved-usd"]) > 0

    # Tenant B: two streaming calls.
    for _ in range(2):
        stream = await client_b.chat.completions.create(
            model="auto", messages=MSG, max_tokens=5, temperature=0, stream=True, stream_options={"include_usage": True}
        )
        text, final_usage = "", None
        async for chunk in stream:
            final_usage = chunk.usage or final_usage
            text += "".join(c.delta.content or "" for c in chunk.choices)
        assert text and final_usage is not None and final_usage.completion_tokens > 0

    ua = await wait_for_requests(gateway, a.key, 1)
    ub = await wait_for_requests(gateway, b.key, 2)
    assert (ua["app_id"], ua["totals"]["requests"]) == (a.app_id, 1)
    assert (ub["app_id"], ub["totals"]["requests"]) == (b.app_id, 2)
    assert ua["totals"]["actual_cost_usd"] == 0.0 and ua["totals"]["saved_usd"] > 0
    # A's second app made no calls and sees nothing.
    assert (await usage(gateway, a.key2))["totals"]["requests"] == 0


async def test_models_list_via_sdk(two, sdk) -> None:
    a, _ = two
    ids = [m.id async for m in sdk(a.key).models.list()]
    assert "auto" in ids and "ollama/qwen2.5:7b" in ids
