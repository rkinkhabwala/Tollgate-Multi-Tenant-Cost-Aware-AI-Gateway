"""Pipeline runner: stage ordering, timings, deferred post-stages for streams."""

from gateway.pipeline.base import Pipeline, Stage
from gateway.pipeline.context import RequestContext
from tests.unit.fakes import make_ctx


class Recorder:
    def __init__(self, name: str, log: list[str], set_stream: bool = False) -> None:
        self.name = name
        self.log = log
        self.set_stream = set_stream

    async def process(self, ctx: RequestContext) -> RequestContext:
        self.log.append(self.name)
        if self.set_stream:

            async def gen():
                yield None

            ctx.stream = gen()
        return ctx


async def test_runs_in_order_and_records_timing() -> None:
    log: list[str] = []
    p = Pipeline([Recorder("a", log), Recorder("b", log)], [Recorder("post", log)])
    ctx = await p.run(make_ctx())
    assert log == ["a", "b", "post"]
    assert set(ctx.stage_ms) == {"a", "b", "post"}


async def test_streaming_defers_post_stages_until_finish() -> None:
    log: list[str] = []
    p = Pipeline([Recorder("invoke", log, set_stream=True)], [Recorder("meter", log)])
    ctx = await p.run(make_ctx())
    assert log == ["invoke"]
    await p.finish(ctx)
    assert log == ["invoke", "meter"]


def test_stages_satisfy_protocol() -> None:
    from gateway.pipeline.budget import BudgetStage
    from gateway.pipeline.rehydrate import RehydrateStage

    assert isinstance(Recorder("x", []), Stage)
    assert isinstance(BudgetStage(), Stage) and isinstance(RehydrateStage(), Stage)


def test_context_repr_hides_sensitive_fields() -> None:
    from tests.unit.fakes import chat_request

    ctx = make_ctx(chat_request("my password is hunter2"), api_key="tg_live_abcdefgh_" + "S" * 32)
    ctx.vault.tokenize("EMAIL_ADDRESS", "a@b.c")
    text = repr(ctx)
    assert "hunter2" not in text and "S" * 32 not in text and "a@b.c" not in text
