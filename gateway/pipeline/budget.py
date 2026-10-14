"""Stage 4 — Budget check: soft limit → cheaper tier; hard limit → 402 (spec §3.1, §8.2 step 7).

Phase 1: pass-through placeholder so the pipeline order from spec §3.1 is visible
and the stage can be dropped in without touching the runner. Implemented in Phase 4.
"""

from gateway.pipeline.context import RequestContext


class BudgetStage:
    name = "budget"

    async def process(self, ctx: RequestContext) -> RequestContext:
        return ctx
