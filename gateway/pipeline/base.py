"""`Stage` protocol and the `Pipeline` runner (spec §3.1, §13).

A pipeline is two ordered lists of stages:

* `stages` run before the response is returned (authn → ... → invoke).
* `post_stages` run after the response is complete (validate, cache write,
  rehydrate, meter). For non-streaming requests they run inside `run()`. For
  streaming requests the caller drains `ctx.stream` and then calls `finish()`;
  metering has to wait for the provider's final usage frame.

Stages are independent classes so they can be reordered, toggled per app, and
unit-tested in isolation. The runner records per-stage wall time in `ctx.stage_ms`.
That feeds `gateway_ms` now and the per-stage histograms in Phase 5.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from typing import Protocol, runtime_checkable

from gateway.pipeline.context import RequestContext


@runtime_checkable
class Stage(Protocol):
    name: str

    async def process(self, ctx: RequestContext) -> RequestContext: ...


class Pipeline:
    def __init__(self, stages: Sequence[Stage], post_stages: Sequence[Stage] = ()) -> None:
        self.stages = list(stages)
        self.post_stages = list(post_stages)

    @staticmethod
    async def _run_stage(stage: Stage, ctx: RequestContext) -> RequestContext:
        t0 = time.perf_counter()
        try:
            return await stage.process(ctx)
        finally:
            ctx.stage_ms[stage.name] = (time.perf_counter() - t0) * 1000

    async def run(self, ctx: RequestContext) -> RequestContext:
        """Run the pre-response stages; also the post stages unless streaming."""
        for stage in self.stages:
            ctx = await self._run_stage(stage, ctx)
        if ctx.stream is None:
            ctx = await self.finish(ctx)
        return ctx

    async def finish(self, ctx: RequestContext) -> RequestContext:
        """Run post-response stages (called directly by the SSE path after draining)."""
        for stage in self.post_stages:
            ctx = await self._run_stage(stage, ctx)
        return ctx
