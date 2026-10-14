"""Stage 1 — AuthN: resolve `Authorization: Bearer tg_...` to tenant, app, key, scopes.

Flow (spec §6.1):
1. Parse the key format → public `prefix8`. A malformed key gets 401 without any I/O.
2. Look up `sys:auth:{prefix8}` in Redis (60 s TTL). On a miss, load the key row and
   its app/tenant/policy version via the system role, and cache it with `SET NX`. NX
   means a revocation tombstone written concurrently is never overwritten.
3. Constant-time compare `SHA-256(presented key)` with the stored hash.
4. Reject revoked or expired keys (401).
5. Queue `last_used_at` for the batched toucher (never written on the hot path).

Every failure returns the same generic 401, so callers can't distinguish an unknown
prefix from a wrong secret.
"""

from __future__ import annotations

import json
import uuid
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Protocol

from redis.asyncio import Redis
from sqlalchemy import select

from gateway.db.models import ApiKey, App, AppPolicy, Tenant
from gateway.db.session import Database
from gateway.errors import invalid_api_key
from gateway.pipeline.context import AuthContext, RequestContext
from gateway.redis_keys import REVOKED_TOMBSTONE, auth_cache_key
from gateway.security.api_keys import parse_api_key, verify_api_key


@dataclass(frozen=True)
class KeyRecord:
    """Everything authn/authz need about a key, in a cache-friendly shape."""

    key_id: str
    tenant_id: str
    app_id: str
    prefix: str
    secret_hash_hex: str
    scopes: list[str]
    ip_allowlist: list[str]
    expires_at: str | None
    revoked_at: str | None
    tenant_status: str
    app_status: str
    baseline_model: str
    policy_version: int

    def to_json(self) -> str:
        return json.dumps(asdict(self))

    @classmethod
    def from_json(cls, raw: str | bytes) -> KeyRecord:
        return cls(**json.loads(raw))


class KeyResolver(Protocol):
    async def by_prefix(self, prefix: str) -> KeyRecord | None: ...


class DbKeyResolver:
    """Loads a `KeyRecord` from Postgres.

    Uses the system role because the tenant is not known until the key is resolved.
    This is one of the few sanctioned BYPASSRLS paths.
    """

    def __init__(self, db: Database) -> None:
        self._db = db

    async def by_prefix(self, prefix: str) -> KeyRecord | None:
        stmt = (
            select(ApiKey, App.status, App.baseline_model, Tenant.status, AppPolicy.version)
            .join(App, App.id == ApiKey.app_id)
            .join(Tenant, Tenant.id == ApiKey.tenant_id)
            .outerjoin(AppPolicy, AppPolicy.app_id == ApiKey.app_id)
            .where(ApiKey.prefix == prefix)
        )
        async with self._db.system_session() as s:
            row = (await s.execute(stmt)).first()
        if row is None:
            return None
        key, app_status, baseline, tenant_status, version = row
        return KeyRecord(
            key_id=str(key.id),
            tenant_id=str(key.tenant_id),
            app_id=str(key.app_id),
            prefix=key.prefix,
            secret_hash_hex=key.secret_hash.hex(),
            scopes=list(key.scopes),
            ip_allowlist=[str(n) for n in key.ip_allowlist or []],
            expires_at=key.expires_at.isoformat() if key.expires_at else None,
            revoked_at=key.revoked_at.isoformat() if key.revoked_at else None,
            tenant_status=tenant_status,
            app_status=app_status,
            baseline_model=baseline,
            policy_version=version or 1,
        )


class AuthNStage:
    name = "authn"

    def __init__(
        self,
        redis: Redis,
        resolver: KeyResolver,
        cache_ttl_s: int = 60,
        on_key_used: Callable[[uuid.UUID], None] | None = None,
    ) -> None:
        self._redis = redis
        self._resolver = resolver
        self._ttl = cache_ttl_s
        self._on_key_used = on_key_used

    async def _load(self, prefix: str) -> KeyRecord | None:
        cache_key = auth_cache_key(prefix)
        cached = await self._redis.get(cache_key)
        if cached is not None:
            if cached in (REVOKED_TOMBSTONE, REVOKED_TOMBSTONE.encode()):
                return None
            return KeyRecord.from_json(cached)
        record = await self._resolver.by_prefix(prefix)
        if record is not None:
            await self._redis.set(cache_key, record.to_json(), ex=self._ttl, nx=True)
        return record

    async def process(self, ctx: RequestContext) -> RequestContext:
        parsed = parse_api_key(ctx.api_key or "")
        if parsed is None:
            raise invalid_api_key()
        record = await self._load(parsed.prefix)
        if record is None or not verify_api_key(parsed.full_key, bytes.fromhex(record.secret_hash_hex)):
            raise invalid_api_key()
        now = datetime.now(UTC)
        if record.revoked_at is not None:
            raise invalid_api_key("This API key has been revoked.")
        expires_at = datetime.fromisoformat(record.expires_at) if record.expires_at else None
        if expires_at is not None and expires_at <= now:
            raise invalid_api_key("This API key has expired.")

        ctx.auth = AuthContext(
            tenant_id=uuid.UUID(record.tenant_id),
            app_id=uuid.UUID(record.app_id),
            key_id=uuid.UUID(record.key_id),
            key_prefix=record.prefix,
            scopes=frozenset(record.scopes),
            policy_version=record.policy_version,
            baseline_model=record.baseline_model,
            tenant_active=record.tenant_status == "active",
            app_active=record.app_status == "active",
            expires_at=expires_at,
            ip_allowlist=tuple(record.ip_allowlist),
        )
        if self._on_key_used is not None:
            self._on_key_used(ctx.auth.key_id)
        return ctx
