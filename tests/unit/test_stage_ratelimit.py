"""Rate-limit stage: RPM/TPM token buckets, 429 headers, TPM reconciliation, key scoping."""

import pytest

from gateway.errors import GatewayError
from gateway.pipeline.ratelimit import RateLimiter, RateLimitStage
from gateway.policy.schema import AppPolicy
from tests.unit.fakes import APP, TENANT, auth_ctx, chat_request, make_ctx, make_redis


def policy(rpm: int = 600, tpm: int = 200_000) -> AppPolicy:
    return AppPolicy.model_validate({"limits": {"rpm": rpm, "tpm": tpm}})


def ctx_for(p: AppPolicy, tokens_est: int = 10, max_tokens: int | None = 5, **auth_kw):
    return make_ctx(chat_request(max_tokens=max_tokens), auth=auth_ctx(**auth_kw), policy=p, estimated_input_tokens=tokens_est)


async def test_rpm_exhaustion_returns_429_with_headers() -> None:
    stage = RateLimitStage(RateLimiter(make_redis()))
    p = policy(rpm=3)
    for _ in range(3):
        await stage.process(ctx_for(p))
    with pytest.raises(GatewayError) as e:
        await stage.process(ctx_for(p))
    err = e.value
    assert err.status == 429 and err.code == "rate_limit_exceeded"
    assert int(err.headers["retry-after"]) >= 1
    assert err.headers["x-ratelimit-limit-requests"] == "3"
    assert err.headers["x-ratelimit-remaining-requests"] == "0"


async def test_tpm_exhaustion_and_nothing_charged_on_deny() -> None:
    redis = make_redis()
    limiter = RateLimiter(redis)
    stage = RateLimitStage(limiter)
    p = policy(rpm=100, tpm=100)
    await stage.process(ctx_for(p, tokens_est=50, max_tokens=40))  # charges 90
    with pytest.raises(GatewayError):
        await stage.process(ctx_for(p, tokens_est=50, max_tokens=40))
    rpm_key, _ = limiter.keys(TENANT, APP)
    # The denied request must not have consumed an RPM token (99 left, not 98).
    assert float(await redis.hget(rpm_key, "tokens")) == pytest.approx(99, abs=0.1)


async def test_default_max_tokens_used_for_reservation() -> None:
    stage = RateLimitStage(RateLimiter(make_redis()))
    ctx = await stage.process(ctx_for(policy(), tokens_est=10, max_tokens=None))
    assert ctx.reserved_tokens == 10 + AppPolicy().optimize.default_max_tokens


async def test_reservation_clamped_to_bucket_capacity() -> None:
    stage = RateLimitStage(RateLimiter(make_redis()))
    ctx = await stage.process(ctx_for(policy(tpm=500), tokens_est=10_000, max_tokens=10))
    assert ctx.reserved_tokens == 500


async def test_reconcile_refunds_overestimate() -> None:
    redis = make_redis()
    limiter = RateLimiter(redis)
    p = policy(tpm=1000)
    await RateLimitStage(limiter).process(ctx_for(p, tokens_est=100, max_tokens=500))  # reserve 600
    await limiter.adjust_tokens(TENANT, APP, tpm=1000, delta=150 - 600)  # actually used 150
    _, tpm_key = limiter.keys(TENANT, APP)
    assert float(await redis.hget(tpm_key, "tokens")) == pytest.approx(850, abs=1)


async def test_buckets_are_per_app_and_tenant_prefixed() -> None:
    import uuid

    redis = make_redis()
    stage = RateLimitStage(RateLimiter(redis))
    p = policy(rpm=1)
    await stage.process(ctx_for(p))
    # A different app is unaffected by the first app's exhausted bucket.
    await stage.process(ctx_for(p, app_id=uuid.uuid4()))
    keys = [k.decode() for k in await redis.keys("*")]
    assert keys and all(k.startswith("t:") and ":a:" in k for k in keys)
