"""Stage 8 — Classify: task_type + complexity + confidence (spec §8.3).

Phase 1: pass-through placeholder so the pipeline order from spec §3.1 is visible
and the stage can be dropped in without touching the runner. Implemented in Phase 3.
"""

from gateway.pipeline.context import RequestContext


class ClassifyStage:
    name = "classify"

    async def process(self, ctx: RequestContext) -> RequestContext:
        return ctx
