"""Meter stage: cost/baseline math, Redis counters, usage event, writer row mapping."""

import json
from decimal import Decimal

from gateway.pipeline.context import Attempt, RouteDecision
from gateway.pipeline.meter import MeterStage, compute_costs
from gateway.pipeline.ratelimit import RateLimiter
from gateway.policy.schema import AppPolicy
from gateway.workers.usage_writer import event_to_row
from tests.unit.fakes import APP, KEY_ID, TENANT, auth_ctx, chat_request, make_catalog, make_ctx, make_redis

STREAM = "sys:usage_events"


def metered_ctx(attempts: list[Attempt]):
    return make_ctx(
        chat_request("secret prompt text"),
        auth=auth_ctx(),
        policy=AppPolicy(),
        route=RouteDecision("ollama/qwen2.5:7b", "local", "static:default"),
        attempts=attempts,
        reserved_tokens=500,
    )


def test_local_model_actual_zero_baseline_priced_at_sonnet() -> None:
    ctx = metered_ctx([Attempt("ollama/qwen2.5:7b", input_tokens=1000, output_tokens=500)])
    actual, baseline = compute_costs(ctx, make_catalog())
    assert actual == Decimal("0.000000")
    # sonnet in fakes: $3 in / $15 out per MTok
    assert baseline == Decimal("0.010500")


def test_all_attempts_billed_baseline_uses_final() -> None:
    ctx = metered_ctx(
        [Attempt("anthropic/claude-haiku-5-5", 1000, 1000), Attempt("anthropic/claude-sonnet-5-5", 1000, 100)]
    )
    actual, baseline = compute_costs(ctx, make_catalog())
    assert actual == Decimal("0.006000") + Decimal("0.004500")
    assert baseline == Decimal("0.004500")


async def test_meter_emits_event_and_counters() -> None:
    redis = make_redis()
    stage = MeterStage(redis, make_catalog(), RateLimiter(redis), STREAM)
    ctx = await stage.process(metered_ctx([Attempt("ollama/qwen2.5:7b", 100, 20, latency_ms=50)]))
    assert ctx.metered and ctx.saved_usd == ctx.baseline_usd > 0

    entries = await redis.xrange(STREAM)
    assert len(entries) == 1
    fields = {k.decode(): v.decode() for k, v in entries[0][1].items()}
    assert fields["tenant_id"] == str(TENANT) and fields["app_id"] == str(APP) and fields["api_key_id"] == str(KEY_ID)
    assert fields["routed_model"] == "ollama/qwen2.5:7b" and fields["cache_status"] == "miss"
    assert (fields["input_tokens"], fields["output_tokens"]) == ("100", "20")
    assert "secret prompt text" not in json.dumps(fields)  # never the prompt

    spend_keys = [k.decode() for k in await redis.keys("*spend*")]
    assert len(spend_keys) == 1 and spend_keys[0].startswith(f"t:{TENANT}:a:{APP}:spend:")


async def test_meter_is_idempotent() -> None:
    redis = make_redis()
    stage = MeterStage(redis, make_catalog(), RateLimiter(redis), STREAM)
    ctx = metered_ctx([Attempt("ollama/qwen2.5:7b", 1, 1)])
    await stage.process(ctx)
    await stage.process(ctx)
    assert len(await redis.xrange(STREAM)) == 1


async def test_writer_row_mapping_roundtrip() -> None:
    redis = make_redis()
    stage = MeterStage(redis, make_catalog(), RateLimiter(redis), STREAM)
    await stage.process(metered_ctx([Attempt("ollama/qwen2.5:7b", 10, 5)]))
    (_, fields), = await redis.xrange(STREAM)
    row = event_to_row(fields)
    assert row["tenant_id"] == TENANT and row["status_code"] == 200
    assert row["baseline_cost_usd"] > row["actual_cost_usd"] == Decimal(0)
    assert row["pii_entities"] == {}
