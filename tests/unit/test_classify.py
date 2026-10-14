"""Classifier: Layer A heuristics, layer combination, kNN tally, stage behaviour."""

from __future__ import annotations

import pytest

from gateway.pipeline.classify import ClassifyStage, merge_complexity
from gateway.policy.schema import AppPolicy
from gateway.providers.base import ChatMessage, ChatRequest
from gateway.routing.features import extract, vote
from gateway.routing.knn import KnnVote, tally
from tests.unit.fakes import auth_ctx, chat_request, make_ctx

TOOL = {"type": "function", "function": {"name": "search", "parameters": {"type": "object"}}}


@pytest.mark.parametrize(
    ("text", "kw", "task", "hard"),
    [
        ("Book me a table for two", {"tools": [TOOL]}, "agentic_tool_use", True),
        ("Why does this fail?\n```python\nx = 1/0\n```", {}, "code", True),
        ("Traceback (most recent call last):\n  File \"a.py\", line 3\nZeroDivisionError", {}, "code", False),
        ("Summarize this article in 3 bullets: ...", {}, "summarization", False),
        ("Classify the sentiment of: 'great product'", {}, "classification", False),
        ("Rewrite this more formally: hey guys", {}, "rewrite", False),
        ("Write a short story about a dragon", {}, "creative_long", False),
        ("hey, how are you today?", {}, "chitchat", False),
        ("Calculate 15% of 240 and add 3.5 then divide by 2", {}, "reasoning_math", False),
        ("Tell me something", {}, None, False),
    ],
)
def test_layer_a_votes(text: str, kw: dict, task: str | None, hard: bool) -> None:
    v = vote(extract(chat_request(text, **kw)))
    assert v.task_type == task
    if task in ("agentic_tool_use",):
        assert v.hard
    if "```" in text:
        assert v.hard


def test_layer_a_complexity_hint() -> None:
    assert vote(extract(chat_request("hi"))).complexity == "low"
    long = "Please ensure you include at least five sections. You must be thorough and detailed. " * 60
    assert vote(extract(chat_request(long))).complexity == "high"


def test_tool_choice_none_is_not_agentic() -> None:
    assert not extract(chat_request("find flights", tools=[TOOL], tool_choice="none")).tools_requested


def test_knn_tally_weighted_vote() -> None:
    v = tally([("code", "high", 0.9), ("code", "medium", 0.8), ("rewrite", "medium", 0.5)])
    assert v.task_type == "code" and v.complexity == "medium"
    assert v.task_confidence == pytest.approx(1.7 / 2.2, abs=1e-3)
    assert tally([]) is None


def test_merge_complexity() -> None:
    assert merge_complexity("low", "high") == "medium"
    assert merge_complexity("medium", "high") == "medium"
    assert merge_complexity(None, "high") == "high"


class StubKnn:
    def __init__(self, v: KnnVote | None, fail: bool = False) -> None:
        self.v, self.fail, self.seen = v, fail, []

    async def vote(self, text: str) -> KnnVote | None:
        self.seen.append(text)
        if self.fail:
            raise ConnectionError("ollama down")
        return self.v


def kv(task: str, conf: float = 0.8, cx: str = "low") -> KnnVote:
    return KnnVote(task, conf, cx, 0.8, ())


@pytest.mark.parametrize(
    ("text", "kw", "knn", "task", "source"),
    [
        ("Summarize this: the meeting moved to May.", {}, kv("summarization", 0.6), "summarization", "A+B"),
        ("Why does this crash?\n```js\nfoo()\n```", {}, kv("qa_with_context", 0.7), "code", "A(hard)"),
        ("Summarize what you think about tea", {}, kv("chitchat", 0.85), "chitchat", "B"),
        ("tell me more", {}, None, "chitchat", "default"),
    ],
)
async def test_layer_combination(text: str, kw: dict, knn: KnnVote | None, task: str, source: str) -> None:
    c = await ClassifyStage(StubKnn(knn)).classify(chat_request(text, **kw))  # type: ignore[arg-type]
    assert (c.task_type, c.detail["source"]) == (task, source)
    if source == "default":
        assert c.confidence < 0.6  # triggers the confidence bump downstream


async def test_knn_failure_falls_back_to_heuristics() -> None:
    c = await ClassifyStage(StubKnn(None, fail=True)).classify(chat_request("Summarize: x"))  # type: ignore[arg-type]
    assert c.task_type == "summarization" and c.detail["layer_b"] is None


async def test_stage_uses_redacted_last_user_turn_and_skips_aliases() -> None:
    knn = StubKnn(kv("chitchat"))
    stage = ClassifyStage(knn)  # type: ignore[arg-type]
    original = ChatRequest(messages=[ChatMessage(role="user", content="Email jane@corp.example")])
    working = ChatRequest(messages=[ChatMessage(role="system", content="placeholder note"), ChatMessage(role="user", content="Email <<EMAIL_1>>")])
    ctx = make_ctx(original, auth=auth_ctx(), policy=AppPolicy(), working=working)
    ctx = await stage.process(ctx)
    assert knn.seen == ["Email <<EMAIL_1>>"] and ctx.classification is not None  # never the raw value
    ctx = await stage.process(make_ctx(chat_request("x", model="tollgate/small"), auth=auth_ctx(), policy=AppPolicy()))
    assert ctx.classification is None
