"""Stage 7 — Optimize: trim history, cap max_tokens, dedupe RAG chunks (spec §10.3-5).

Phase 1: pass-through placeholder so the pipeline order from spec §3.1 is visible
and the stage can be dropped in without touching the runner. Implemented in Phase 4.
"""

from gateway.pipeline.context import RequestContext


class OptimizeStage:
    name = "optimize"

    async def process(self, ctx: RequestContext) -> RequestContext:
        return ctx
