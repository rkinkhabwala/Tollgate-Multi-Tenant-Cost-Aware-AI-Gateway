"""Stage 9 — Route: choose the model to call.

Phase 1 implementation (`StaticRouteStage`): every request goes to one configured
model (`settings.default_model`, Ollama), per spec §16.

The one exception is a pinned model (spec §8.2 step 1). It is honoured when all of
these hold: the policy has `allow_pinning: true`, authz has confirmed the model is in
`allowed_models`, and its provider is configured. Otherwise the request is treated
as `auto`, and the route reason records why.

Phase 3 replaces this with policy-driven routing: hard constraints, the classifier,
the routing table, confidence/budget/tier clamps.
"""

from __future__ import annotations

from gateway.catalog import ModelCatalog
from gateway.errors import upstream_error
from gateway.pipeline.authz import is_pinned
from gateway.pipeline.context import RequestContext, RouteDecision
from gateway.providers.registry import ProviderRegistry


class StaticRouteStage:
    name = "route"

    def __init__(self, catalog: ModelCatalog, registry: ProviderRegistry, default_model: str) -> None:
        self._catalog = catalog
        self._registry = registry
        self._default = default_model

    async def process(self, ctx: RequestContext) -> RequestContext:
        req, policy = ctx.original, ctx.require_policy()
        if req is not None and is_pinned(req.model):
            pinned = self._catalog.get(req.model)
            if policy.routing.allow_pinning and pinned and pinned.provider in self._registry:
                ctx.route = RouteDecision(pinned.id, pinned.tier, "pinned")
                return ctx
            reason = "static:default(pinning-disabled)" if not policy.routing.allow_pinning else "static:default(pinned-unavailable)"
        else:
            reason = "static:default"

        model = self._catalog.get(self._default)
        if model is None or model.provider not in self._registry:
            raise upstream_error(503, "The configured default model is not available.")
        ctx.route = RouteDecision(model.id, model.tier, reason)
        return ctx
