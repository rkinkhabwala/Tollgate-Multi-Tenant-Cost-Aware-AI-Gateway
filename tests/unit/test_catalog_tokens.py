"""Catalog profiles + long-context pricing; token counting (exact Qwen, conservative otherwise)."""

from decimal import Decimal

from gateway.catalog import ModelCatalog
from gateway.providers.base import ChatMessage, ChatRequest
from gateway.providers.tokens import TokenCounter, conservative_count
from gateway.settings import REPO_ROOT, get_settings

CAT = ModelCatalog.from_yaml(get_settings().models_yaml)
COUNTER = TokenCounter(REPO_ROOT / "config" / "tokenizers")


def test_profiles_loaded() -> None:
    assert CAT.profile("hosted")["small"] == ["anthropic/claude-haiku-5-5"]
    assert CAT.profile("local-only") == {
        "local": ["ollama/qwen2.5:1.5b"],
        "small": ["ollama/qwen2.5:1.5b"],
        "medium": ["ollama/qwen2.5:7b"],
        "large": ["ollama/qwen2.5:14b"],
    }
    assert CAT.escalation_order == ["local", "small", "medium", "large"]
    assert CAT.prices_checked == "2026-10-14"


def test_anthropic_prices_and_haiku_long_context_tier() -> None:
    assert CAT.cost("anthropic/claude-sonnet-5-5", 1_000_000, 1_000_000) == Decimal("12.000000")
    assert CAT.cost("anthropic/claude-opus-5-5", 1_000_000, 0) == Decimal("4.000000")
    assert CAT.cost("anthropic/claude-haiku-5-5", 100_000, 1_000) == Decimal("0.010500")  # at threshold: base
    assert CAT.cost("anthropic/claude-haiku-5-5", 200_000, 1_000) == Decimal("0.102500")  # over: $0.50/$2.50


def test_qwen_count_is_exact_and_conservative_overcounts() -> None:
    req = ChatRequest(messages=[ChatMessage(role="user", content="Hello Maria, the invoice INV-2024-0457 is overdue.")])
    qwen = CAT.get("ollama/qwen2.5:7b")
    exact = COUNTER.count(qwen, req)
    assert exact == 19 + 5 + 3  # tokenizer + framing + primer
    assert COUNTER.count(CAT.get("anthropic/claude-sonnet-5-5"), req) == conservative_count(req) > exact


def test_conservative_is_safe_for_cjk_and_code() -> None:
    cjk = ChatRequest(messages=[ChatMessage(role="user", content="请总结这份报告的主要内容" * 50)])
    assert conservative_count(cjk) >= COUNTER.qwen_count(cjk)
    code = ChatRequest(messages=[ChatMessage(role="user", content="def f(x):\n    return {k: v**2 for k, v in x.items()}\n" * 40)])
    assert conservative_count(code) >= COUNTER.qwen_count(code)


def test_tools_add_tokens() -> None:
    plain = ChatRequest(messages=[ChatMessage(role="user", content="weather?")])
    tools = plain.model_copy(update={"tools": [{"type": "function", "function": {"name": "get_weather", "parameters": {"type": "object"}}}]})
    qwen = CAT.get("ollama/qwen2.5:7b")
    assert COUNTER.count(qwen, tools) > COUNTER.count(qwen, plain) + 100
