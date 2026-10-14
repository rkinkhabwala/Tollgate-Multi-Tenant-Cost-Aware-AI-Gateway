"""Stage 8 — Classify: task_type + complexity + confidence (spec §8.3).

The input is the **redacted** last user turn: `ctx.working`, falling back to the
original only when redaction is disabled for the app. Nothing from this stage is
stored. The kNN query embedding is discarded after the search.

Layers:
* **A (heuristics, <1 ms):** structural features and keyword cues (`routing/features.py`).
* **B (embedding kNN):** k=7 nearest labeled seed prompts in Redis (`routing/knn.py`).
* **C (small LLM, optional):** `qwen2.5:1.5b` answering in strict JSON. It runs only
  when `settings.classifier_layer_c` is on *and* A and B disagree *and* confidence is
  below the app's `min_confidence`. Off by default.

Combining (task type):
* A and B agree → that task, with confidence = max(A, B);
* A has a hard structural signal (tools present, code/stack trace) → A wins, at
  0.9 × its confidence;
* otherwise B wins with its vote share. With no kNN (index not built, embedder
  down), A's guess is used, or `chitchat` at confidence 0.3, which triggers the
  low-confidence bump.

Complexity is the kNN vote, nudged by Layer A's length/constraint hint: when they
differ by two levels (low vs high), the result is medium.

Skipped when the request is pinned with pinning allowed, or uses a tier alias, since
routing doesn't need a classification then.
"""

from __future__ import annotations

import json
import logging
import time

from gateway.pipeline.authz import TIER_ALIASES, is_pinned
from gateway.pipeline.context import Classification, RequestContext
from gateway.policy.schema import Complexity, TaskType
from gateway.providers.base import ChatMessage, ChatRequest, ProviderError
from gateway.providers.registry import ProviderRegistry
from gateway.routing.features import extract, last_user_text, vote
from gateway.routing.knn import KnnClassifier

log = logging.getLogger(__name__)
TASKS: tuple[str, ...] = TaskType.__args__  # type: ignore[attr-defined]
LEVELS: tuple[str, ...] = Complexity.__args__  # type: ignore[attr-defined]

LAYER_C_PROMPT = (
    "Classify the user's message. Reply with JSON only: "
    '{"task_type": one of ' + json.dumps(list(TASKS)) + ', "complexity": "low"|"medium"|"high"}.\n\nMessage:\n'
)


def merge_complexity(knn_cx: str | None, hint: str) -> str:
    if knn_cx is None:
        return hint
    a, b = LEVELS.index(knn_cx), LEVELS.index(hint)
    return "medium" if abs(a - b) == 2 else knn_cx


class ClassifyStage:
    name = "classify"

    def __init__(self, knn: KnnClassifier | None, registry: ProviderRegistry | None = None, layer_c_model: str | None = None) -> None:
        self._knn = knn
        self._registry = registry
        self._layer_c_model = layer_c_model  # provider model name, e.g. "qwen2.5:1.5b"; None = off

    async def classify(self, req: ChatRequest, min_confidence: float = 0.6) -> Classification:
        text = last_user_text(req)
        t0 = time.perf_counter()
        feats = extract(req, text)
        a = vote(feats)
        b = None
        if self._knn is not None:
            try:
                b = await self._knn.vote(text)
            except (ProviderError, OSError, Exception) as exc:  # classifier must never fail the request
                log.warning("kNN layer unavailable", extra={"error": type(exc).__name__})

        if b is None:
            task, conf = (a.task_type, a.confidence) if a.task_type else ("chitchat", 0.3)
            source = "A" if a.task_type else "default"
        elif a.task_type == b.task_type:
            task, conf, source = b.task_type, max(a.confidence, b.task_confidence), "A+B"
        elif a.hard and a.task_type:
            task, conf, source = a.task_type, a.confidence * 0.9, "A(hard)"
        else:
            task, conf, source = b.task_type, b.task_confidence, "B"
        complexity = merge_complexity(b.complexity if b else None, a.complexity)

        layer_c = None
        if self._layer_c_model and b is not None and a.task_type and a.task_type != b.task_type and conf < min_confidence:
            layer_c = await self._layer_c(text)
            if layer_c is not None:
                task, complexity, conf, source = layer_c[0], layer_c[1], max(conf, 0.7), "C"

        detail = {
            "source": source,
            "layer_a": {"task_type": a.task_type, "confidence": round(a.confidence, 3), "complexity_hint": a.complexity, "hard": a.hard},
            "layer_b": None
            if b is None
            else {"task_type": b.task_type, "confidence": round(b.task_confidence, 3), "complexity": b.complexity, "complexity_confidence": round(b.complexity_confidence, 3), "neighbors": [n[:2] for n in b.neighbors]},
            "layer_c": None if layer_c is None else {"task_type": layer_c[0], "complexity": layer_c[1]},
            "features": {k: v for k, v in feats.as_dict().items() if k != "cue_hits"} | {"cues": sorted(feats.cue_hits)},
            "ms": round((time.perf_counter() - t0) * 1000, 2),
        }
        return Classification(task, complexity, round(conf, 3), detail)

    async def _layer_c(self, text: str) -> tuple[str, str] | None:
        adapter = self._registry.get("ollama") if self._registry else None
        if adapter is None or self._layer_c_model is None:
            return None
        req = ChatRequest(
            messages=[ChatMessage(role="user", content=LAYER_C_PROMPT + text[:4000])],
            response_format={"type": "json_object"},
            temperature=0,
            max_tokens=40,
        )
        try:
            resp = await adapter.chat(self._layer_c_model, req)
            data = json.loads(resp.choices[0].message.text())
        except (ProviderError, json.JSONDecodeError, AttributeError):
            return None
        if data.get("task_type") in TASKS and data.get("complexity") in LEVELS:
            return data["task_type"], data["complexity"]
        return None

    async def process(self, ctx: RequestContext) -> RequestContext:
        req, policy = ctx.original, ctx.require_policy()
        if req is None:
            return ctx
        if req.model in TIER_ALIASES or (is_pinned(req.model) and policy.routing.allow_pinning):
            return ctx
        ctx.classification = await self.classify(ctx.working or req, policy.routing.min_confidence)
        return ctx
