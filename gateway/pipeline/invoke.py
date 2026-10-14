"""Stage 10 — Invoke: call the routed model through its provider adapter.

* Non-streaming: `adapter.chat()` → `ctx.response`. The response's `model` field is
  rewritten to the catalog id (e.g. `ollama/qwen2.5:7b`).
* Streaming: `adapter.stream()` is opened and the *first chunk is pre-fetched* before
  the stage returns. Connection errors, unknown models and the like therefore surface
  as normal HTTP errors before any SSE bytes are sent (spec §8.4: fallback only
  before the first byte). `ctx.stream` is a wrapper that re-labels chunks and records
  the final usage frame on the `Attempt`, so `meter` can bill once the stream drains.

Every call is recorded as an `Attempt`, including failures. Fallback across models
and tiers, and circuit breakers, arrive in Phase 3.
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from gateway.catalog import ModelCatalog
from gateway.errors import upstream_error
from gateway.pipeline.context import Attempt, RequestContext
from gateway.pipeline.redact import use_unredacted_for_local
from gateway.providers.base import ChatChunk, ProviderError
from gateway.providers.registry import ProviderRegistry


class InvokeStage:
    name = "invoke"

    def __init__(self, catalog: ModelCatalog, registry: ProviderRegistry) -> None:
        self._catalog = catalog
        self._registry = registry

    async def process(self, ctx: RequestContext) -> RequestContext:
        if ctx.route is None or ctx.original is None:
            raise RuntimeError("invoke requires a route and a request")
        model = self._catalog.get(ctx.route.model_id)
        adapter = self._registry.get(model.provider) if model else None
        if model is None or adapter is None:
            raise upstream_error(503, "The routed model is not available.")
        use_unredacted_for_local(ctx)  # §7.6: local tier + redact_for_local=false
        if ctx.working is None:
            ctx.working = ctx.original.model_copy(deep=True)

        attempt = Attempt(model_id=model.id)
        ctx.attempts.append(attempt)
        t0 = time.perf_counter()
        try:
            if ctx.working.stream:
                iterator = aiter(adapter.stream(model.provider_model, ctx.working))
                first = await anext(iterator, None)
                ctx.stream = self._relay(ctx, attempt, model.id, first, iterator, t0)
            else:
                resp = await adapter.chat(model.provider_model, ctx.working)
                resp.model = model.id
                attempt.input_tokens = resp.usage.prompt_tokens
                attempt.output_tokens = resp.usage.completion_tokens
                attempt.latency_ms = int((time.perf_counter() - t0) * 1000)
                ctx.response = resp
        except ProviderError as exc:
            attempt.latency_ms = int((time.perf_counter() - t0) * 1000)
            attempt.status_code = exc.status
            attempt.error = exc.message
            ctx.status_code = exc.status
            raise upstream_error(exc.status, exc.message) from exc
        return ctx

    @staticmethod
    async def _relay(
        ctx: RequestContext,
        attempt: Attempt,
        model_id: str,
        first: ChatChunk | None,
        rest: AsyncIterator[ChatChunk],
        t0: float,
    ) -> AsyncIterator[ChatChunk]:
        """Yield chunks relabelled with the catalog id; capture usage from the last frame."""

        async def chain() -> AsyncIterator[ChatChunk]:
            if first is not None:
                yield first
            async for c in rest:
                yield c

        try:
            async for chunk in chain():
                chunk.model = model_id
                if chunk.usage is not None:
                    attempt.input_tokens = chunk.usage.prompt_tokens
                    attempt.output_tokens = chunk.usage.completion_tokens
                yield chunk
        except ProviderError as exc:
            attempt.status_code = exc.status
            attempt.error = exc.message
            ctx.status_code = exc.status
            raise
        finally:
            attempt.latency_ms = int((time.perf_counter() - t0) * 1000)
