"""Stage 10 — Invoke: call the routed model, falling back across candidates (spec §8.4).

`ctx.candidates` (from route, or from validate when escalating) is walked in order:
* skip a model whose circuit is open (`CircuitBreaker.allow`);
* call it. A *retryable* failure (timeout, connection error, 429, 5xx, overload) is
  recorded on that model's breaker and as a failed `Attempt` (kind `primary` or
  `fallback`), and the next candidate is tried. A non-retryable failure (400, auth,
  unknown model) goes straight to the client;
* on success, the breaker is reset and `ctx.route` updated to the model that
  actually served, with `;fallback:<from>(<code>)><to>` appended to the reason.

Streaming:
* the first chunk is pre-fetched, so availability fallback happens before any SSE
  byte is sent (spec §8.4: "streams only get availability fallback before the first
  byte");
* after the first byte there is no fallback; a mid-stream error ends the stream;
* `ctx.stream` relabels chunks with the catalog id and captures the final usage frame.

Every attempt is billed by meter, failed ones included. If all candidates fail, the
last error is returned with its status (502/503/504).
"""

from __future__ import annotations

import time
from collections.abc import AsyncIterator

from gateway.catalog import CatalogModel, ModelCatalog
from gateway.errors import upstream_error
from gateway.pipeline.context import Attempt, RequestContext, RouteDecision
from gateway.pipeline.redact import use_unredacted_for_local
from gateway.providers.base import ChatChunk, ProviderError
from gateway.providers.circuit_breaker import CircuitBreaker
from gateway.providers.registry import ProviderRegistry


class InvokeStage:
    name = "invoke"

    def __init__(self, catalog: ModelCatalog, registry: ProviderRegistry, breaker: CircuitBreaker | None = None) -> None:
        self._catalog = catalog
        self._registry = registry
        self._breaker = breaker or CircuitBreaker()

    def _candidates(self, ctx: RequestContext) -> list[CatalogModel]:
        if ctx.candidates:
            return list(ctx.candidates)
        if ctx.route is not None and (m := self._catalog.get(ctx.route.model_id)) is not None:
            return [m]
        return []

    async def process(self, ctx: RequestContext, kind: str = "primary") -> RequestContext:
        if ctx.route is None or ctx.original is None:
            raise RuntimeError("invoke requires a route and a request")
        use_unredacted_for_local(ctx)  # §7.6: local tier + redact_for_local=false
        if ctx.working is None:
            ctx.working = ctx.original.model_copy(deep=True)

        last_error: ProviderError | None = None
        failed: list[str] = []
        for model in self._candidates(ctx):
            adapter = self._registry.get(model.provider)
            if adapter is None or not self._breaker.allow(model.id):
                continue
            attempt = Attempt(model_id=model.id, kind=kind if not failed else "fallback")
            ctx.attempts.append(attempt)
            t0 = time.perf_counter()
            try:
                if ctx.working.stream:
                    iterator = aiter(adapter.stream(model.provider_model, ctx.working))
                    first = await anext(iterator, None)  # errors surface here, before any byte is sent
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
                attempt.status_code, attempt.error = exc.status, exc.code
                ctx.status_code = exc.status
                last_error = exc
                if not exc.retryable:
                    raise upstream_error(exc.status, exc.message) from exc
                self._breaker.record_failure(model.id)
                failed.append(f"{model.id}({exc.code})")
                continue
            self._breaker.record_success(model.id)
            ctx.status_code = 200
            reason = ctx.route.reason
            if failed:
                reason += f";fallback:{','.join(failed)}>{model.id}"
            ctx.route = RouteDecision(model.id, ctx.route.tier, reason)
            return ctx

        if last_error is not None:
            raise upstream_error(last_error.status, last_error.message)
        raise upstream_error(503, "No routed model is currently available.")

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
            attempt.status_code, attempt.error = exc.status, exc.code
            ctx.status_code = exc.status
            raise
        finally:
            attempt.latency_ms = int((time.perf_counter() - t0) * 1000)
