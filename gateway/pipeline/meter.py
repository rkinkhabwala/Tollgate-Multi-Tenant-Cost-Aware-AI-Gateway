"""Stage 14 — Meter: compute cost and savings, then emit a usage event (spec §11).

* `actual_cost` = the sum over every attempt of tokens × the attempted model's
  catalog price. Failed and escalated attempts are billed too (spec §8.4).
* `baseline_cost` = the final attempt's tokens priced at the app's `baseline_model`
  (what the premium default would have cost). `saved = baseline − actual`.
* Inline Redis counters keep budgets and dashboards current immediately:
  `t:{tenant}:a:{app}:spend:{YYYYMM}` (USD) and `t:{tenant}:a:{app}:tokens:{minute}`.
* TPM reconciliation: the rate limiter pre-charged an estimate, and the
  actual − estimate difference is charged or refunded here.
* The event is `XADD`-ed to the `sys:usage_events` stream. The batch writer
  (`workers/usage_writer.py`) inserts it into Postgres within about a second.

Metering runs after the response is produced and is best-effort: a Redis hiccup is
logged, but it never turns a successful completion into an error.
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from datetime import UTC, datetime
from decimal import Decimal

from redis.asyncio import Redis

from gateway.catalog import USD_QUANT, ModelCatalog
from gateway.pipeline.context import RequestContext
from gateway.pipeline.ratelimit import RateLimiter
from gateway.redis_keys import app_key

log = logging.getLogger(__name__)

STREAM_MAXLEN = 1_000_000  # approximate cap; the writer normally keeps it near empty


def compute_costs(ctx: RequestContext, catalog: ModelCatalog) -> tuple[Decimal, Decimal]:
    """Return `(actual_usd, baseline_usd)` for the context's attempts."""
    auth = ctx.require_auth()
    actual = sum((catalog.cost(a.model_id, a.input_tokens, a.output_tokens) for a in ctx.attempts), Decimal(0))
    final = ctx.attempts[-1] if ctx.attempts else None
    if final is None:
        return Decimal(0).quantize(USD_QUANT), Decimal(0).quantize(USD_QUANT)
    if catalog.get(auth.baseline_model) is None:
        log.warning("baseline model missing from catalog", extra={"app_id": str(auth.app_id)})
        baseline = actual
    else:
        baseline = catalog.cost(auth.baseline_model, final.input_tokens, final.output_tokens)
    return actual.quantize(USD_QUANT), baseline.quantize(USD_QUANT)


class MeterStage:
    name = "meter"

    def __init__(self, redis: Redis, catalog: ModelCatalog, limiter: RateLimiter, stream: str) -> None:
        self._redis = redis
        self._catalog = catalog
        self._limiter = limiter
        self._stream = stream

    def build_event(self, ctx: RequestContext) -> dict[str, str]:
        auth = ctx.require_auth()
        final = ctx.attempts[-1] if ctx.attempts else None
        model_ms = sum(a.latency_ms for a in ctx.attempts)
        total_ms = int((time.perf_counter() - ctx.started_at) * 1000)
        return {
            "id": str(uuid.uuid4()),
            "ts": datetime.now(UTC).isoformat(),
            "tenant_id": str(auth.tenant_id),
            "app_id": str(auth.app_id),
            "api_key_id": str(auth.key_id),
            "request_id": ctx.request_id,
            "requested_model": ctx.original.model if ctx.original else "",
            "routed_model": ctx.route.model_id if ctx.route else "",
            "route_reason": ctx.route.reason if ctx.route else "",
            "escalations": str(max(0, len(ctx.attempts) - 1)),
            "cache_status": ctx.cache_status,
            "input_tokens": str(final.input_tokens if final else 0),
            "output_tokens": str(final.output_tokens if final else 0),
            "actual_cost_usd": str(ctx.cost_usd),
            "baseline_cost_usd": str(ctx.baseline_usd),
            "pii_entities": json.dumps(ctx.pii_counts),  # counts by type only, never values
            "latency_ms": str(model_ms),
            "gateway_ms": str(max(0, total_ms - model_ms)),
            "status_code": str(ctx.status_code),
        }

    async def process(self, ctx: RequestContext) -> RequestContext:
        if ctx.metered:
            return ctx
        ctx.metered = True
        auth, policy = ctx.require_auth(), ctx.require_policy()
        ctx.cost_usd, ctx.baseline_usd = compute_costs(ctx, self._catalog)
        try:
            actual_tokens = sum(a.input_tokens + a.output_tokens for a in ctx.attempts)
            await self._limiter.adjust_tokens(
                auth.tenant_id, auth.app_id, tpm=policy.limits.tpm, delta=actual_tokens - ctx.reserved_tokens
            )
            now = datetime.now(UTC)
            pipe = self._redis.pipeline(transaction=False)
            spend_key = app_key(auth.tenant_id, auth.app_id, f"spend:{now:%Y%m}")
            pipe.incrbyfloat(spend_key, float(ctx.cost_usd))
            pipe.expire(spend_key, 40 * 86400)
            tok_key = app_key(auth.tenant_id, auth.app_id, f"tokens:{int(now.timestamp()) // 60}")
            pipe.incrby(tok_key, actual_tokens)
            pipe.expire(tok_key, 120)
            pipe.xadd(self._stream, self.build_event(ctx), maxlen=STREAM_MAXLEN, approximate=True)
            await pipe.execute()
        except Exception:
            log.exception("metering failed", extra={"request_id": ctx.request_id})
        return ctx
