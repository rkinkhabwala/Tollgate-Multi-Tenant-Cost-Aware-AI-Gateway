"""Stage 9 — Route: pick the tier and the ordered candidate models (spec §8.2).

All decisions live in `routing/router.py`, a pure function of the request, the app
policy, the classification, the catalog profile and the circuit-breaker view. This
stage stores the result on the context:
* `ctx.route`: the first candidate, its tier, and the reason string;
* `ctx.candidates`: the fallback order that invoke walks.

Hard constraints are evaluated against the *working* request, i.e. the redacted copy
plus the placeholder instruction, which is what is actually sent.
"""

from __future__ import annotations

from gateway.pipeline.context import RequestContext, RouteDecision
from gateway.routing.router import Router


class RouteStage:
    name = "route"

    def __init__(self, router: Router) -> None:
        self._router = router

    async def process(self, ctx: RequestContext) -> RequestContext:
        req, policy = ctx.working or ctx.original, ctx.require_policy()
        if req is None:
            return ctx
        plan = self._router.plan(req, policy, ctx.classification)
        ctx.candidates = plan.candidates
        first = plan.candidates[0]
        ctx.route = RouteDecision(first.id, plan.tier, plan.reason)
        return ctx
