"""Stage 5 — Redact: PII and secrets → placeholders; the vault is kept in the request context only (spec §7).

Phase 1: pass-through placeholder so the pipeline order from spec §3.1 is visible
and the stage can be dropped in without touching the runner. Implemented in Phase 2.
"""

from gateway.pipeline.context import RequestContext


class RedactStage:
    name = "redact"

    async def process(self, ctx: RequestContext) -> RequestContext:
        return ctx
