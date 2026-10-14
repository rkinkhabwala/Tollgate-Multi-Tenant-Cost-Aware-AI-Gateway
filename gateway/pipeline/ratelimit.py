"""Stage 3 — Rate limit: per-app RPM and TPM token buckets in Redis (spec §3.1, §9 `limits`).

Two buckets per app, checked and charged atomically by a single Lua script:
* `t:{tenant}:a:{app}:rl:rpm` - capacity `limits.rpm`, refills rpm/60 per second.
* `t:{tenant}:a:{app}:rl:tpm` - capacity `limits.tpm`, refills tpm/60 per second.

A request is admitted only if *both* buckets have room; then both are charged. If
either lacks room, nothing is charged.

Tokens aren't known until the provider answers, so the TPM bucket is pre-charged with
`estimated_input_tokens + max_tokens` (or the policy's `default_max_tokens`). The
meter stage then calls `RateLimiter.reconcile()` with the actual count, refunding or
adding the difference.

A 429 carries `Retry-After` plus OpenAI-style `x-ratelimit-*` headers.
"""

from __future__ import annotations

import math
import time
import uuid

from redis.asyncio import Redis

from gateway.errors import rate_limited
from gateway.pipeline.context import RequestContext
from gateway.redis_keys import app_key

# KEYS: rpm bucket, tpm bucket
# ARGV: now_ms, rpm_cap, tpm_cap, rpm_cost, tpm_cost
# Returns: {allowed(0|1), rpm_left, tpm_left, wait_ms}
_ACQUIRE = """
local now = tonumber(ARGV[1])
local function load(key, cap)
  local v = redis.call('HMGET', key, 'tokens', 'ts')
  local tokens, ts = tonumber(v[1]), tonumber(v[2])
  if tokens == nil then return cap end
  return math.min(cap, tokens + math.max(0, now - ts) * cap / 60000.0)
end
local rcap, tcap = tonumber(ARGV[2]), tonumber(ARGV[3])
local rcost, tcost = tonumber(ARGV[4]), tonumber(ARGV[5])
local r, t = load(KEYS[1], rcap), load(KEYS[2], tcap)
local allowed, wait = 0, 0
if r >= rcost and t >= tcost then
  allowed = 1
  r = r - rcost
  t = t - tcost
else
  if r < rcost then wait = math.max(wait, (rcost - r) * 60000.0 / rcap) end
  if t < tcost then wait = math.max(wait, (tcost - t) * 60000.0 / tcap) end
end
redis.call('HSET', KEYS[1], 'tokens', r, 'ts', now)
redis.call('HSET', KEYS[2], 'tokens', t, 'ts', now)
redis.call('PEXPIRE', KEYS[1], 120000)
redis.call('PEXPIRE', KEYS[2], 120000)
return {allowed, tostring(r), tostring(t), tostring(math.ceil(wait))}
"""

# KEYS: tpm bucket; ARGV: now_ms, tpm_cap, delta (positive = charge more, negative = refund)
_ADJUST = """
local now = tonumber(ARGV[1])
local cap, delta = tonumber(ARGV[2]), tonumber(ARGV[3])
local v = redis.call('HMGET', KEYS[1], 'tokens', 'ts')
local tokens, ts = tonumber(v[1]), tonumber(v[2])
if tokens == nil then tokens, ts = cap, now end
tokens = math.min(cap, tokens + math.max(0, now - ts) * cap / 60000.0)
tokens = math.min(cap, tokens - delta)
redis.call('HSET', KEYS[1], 'tokens', tokens, 'ts', now)
redis.call('PEXPIRE', KEYS[1], 120000)
return tostring(tokens)
"""


def _now_ms() -> int:
    return int(time.time() * 1000)


class RateLimiter:
    """Thin wrapper around the two Lua scripts; shared by the ratelimit and meter stages."""

    def __init__(self, redis: Redis) -> None:
        self._redis = redis
        self._acquire = redis.register_script(_ACQUIRE)
        self._adjust = redis.register_script(_ADJUST)

    @staticmethod
    def keys(tenant_id: uuid.UUID, app_id: uuid.UUID) -> tuple[str, str]:
        return app_key(tenant_id, app_id, "rl:rpm"), app_key(tenant_id, app_id, "rl:tpm")

    async def acquire(
        self, tenant_id: uuid.UUID, app_id: uuid.UUID, *, rpm: int, tpm: int, tokens: int
    ) -> tuple[bool, float, float, int]:
        rpm_key, tpm_key = self.keys(tenant_id, app_id)
        allowed, r, t, wait = await self._acquire(keys=[rpm_key, tpm_key], args=[_now_ms(), rpm, tpm, 1, tokens])
        return bool(int(allowed)), float(r), float(t), int(float(wait))

    async def adjust_tokens(self, tenant_id: uuid.UUID, app_id: uuid.UUID, *, tpm: int, delta: int) -> None:
        if delta == 0:
            return
        _, tpm_key = self.keys(tenant_id, app_id)
        await self._adjust(keys=[tpm_key], args=[_now_ms(), tpm, delta])


class RateLimitStage:
    name = "ratelimit"

    def __init__(self, limiter: RateLimiter) -> None:
        self._limiter = limiter

    async def process(self, ctx: RequestContext) -> RequestContext:
        auth, policy = ctx.require_auth(), ctx.require_policy()
        limits = policy.limits
        tokens = 0
        if ctx.original is not None:
            out = ctx.original.output_limit() or policy.optimize.default_max_tokens
            # Never ask for more than the bucket can ever hold, or the request could never pass.
            tokens = min(ctx.estimated_input_tokens + out, limits.tpm)

        allowed, rpm_left, tpm_left, wait_ms = await self._limiter.acquire(
            auth.tenant_id, auth.app_id, rpm=limits.rpm, tpm=limits.tpm, tokens=tokens
        )
        headers = {
            "x-ratelimit-limit-requests": str(limits.rpm),
            "x-ratelimit-remaining-requests": str(max(0, math.floor(rpm_left))),
            "x-ratelimit-limit-tokens": str(limits.tpm),
            "x-ratelimit-remaining-tokens": str(max(0, math.floor(tpm_left))),
        }
        if not allowed:
            headers["retry-after"] = str(max(1, math.ceil(wait_ms / 1000)))
            headers["retry-after-ms"] = str(max(1, wait_ms))
            raise rate_limited("Rate limit reached for this app. Retry after the indicated delay.", headers)
        ctx.reserved_tokens = tokens
        ctx.rate_limit_headers = headers
        return ctx
