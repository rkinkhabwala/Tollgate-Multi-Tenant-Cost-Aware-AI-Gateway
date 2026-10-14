"""Loading app policies: the default from disk, per-app rows from Postgres via Redis.

Hot path: `PolicyStore.get(auth)` reads `t:{tenant}:a:{app}:policy:v{version}` from
Redis. On a miss it loads the row from `app_policies` under the tenant's RLS session,
validates it and caches it for 60 s. The version is part of the key, so a policy
update (which bumps `version`) never serves a stale cached copy once the auth context
picks up the new version.
"""

from __future__ import annotations

import json
import uuid
from functools import lru_cache
from pathlib import Path

from redis.asyncio import Redis
from sqlalchemy import select

from gateway.db.models import AppPolicy as AppPolicyRow
from gateway.db.session import Database
from gateway.policy.schema import AppPolicy
from gateway.redis_keys import app_key

POLICY_CACHE_TTL_S = 60


@lru_cache
def load_default_policy(path: Path) -> AppPolicy:
    return AppPolicy.model_validate(json.loads(path.read_text()))


class PolicyStore:
    def __init__(self, db: Database, redis: Redis) -> None:
        self._db = db
        self._redis = redis

    async def get(self, tenant_id: uuid.UUID, app_id: uuid.UUID, version: int) -> AppPolicy:
        key = app_key(tenant_id, app_id, f"policy:v{version}")
        cached = await self._redis.get(key)
        if cached is not None:
            return AppPolicy.model_validate_json(cached)
        async with self._db.tenant_session(tenant_id) as s:
            row = (await s.execute(select(AppPolicyRow).where(AppPolicyRow.app_id == app_id))).scalar_one_or_none()
        policy = AppPolicy.model_validate(row.policy) if row else AppPolicy()
        await self._redis.set(key, policy.model_dump_json(), ex=POLICY_CACHE_TTL_S)
        return policy
