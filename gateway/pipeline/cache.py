"""Stages 6 and 12 — Exact and semantic cache, tenant/app-scoped (spec §10.1-2).

Phase 1: pass-through placeholder so the pipeline order from spec §3.1 is visible
and the stage can be dropped in without touching the runner. Implemented in Phase 4.
"""

from gateway.pipeline.context import RequestContext


class CacheLookupStage:
    name = "cache"

    async def process(self, ctx: RequestContext) -> RequestContext:
        return ctx
