"""Redis key naming — the single place keys are built.

Rule (spec §4): every tenant-owned key is prefixed `t:{tenant_id}:a:{app_id}:`.
Use `app_key()` for anything that belongs to a tenant/app.

The only exceptions are `sys:` keys. These are system infrastructure that is read
only by the gateway itself, never served back to a tenant:
* `sys:auth:{prefix8}` - API-key resolver cache. It must be global because the
  tenant is unknown until the key is resolved.
* `sys:usage_events` - the usage Redis Stream that the batch writer consumes.
* `sys:admin_rl:{ip}` - admin-endpoint rate limiting.
"""

from __future__ import annotations

import uuid


def app_prefix(tenant_id: uuid.UUID | str, app_id: uuid.UUID | str) -> str:
    return f"t:{tenant_id}:a:{app_id}:"


def app_key(tenant_id: uuid.UUID | str, app_id: uuid.UUID | str, suffix: str) -> str:
    return f"{app_prefix(tenant_id, app_id)}{suffix}"


REVOKED_TOMBSTONE = "revoked"
AUTH_TOMBSTONE_TTL_S = 300  # comfortably longer than the 60 s auth-cache TTL


def auth_cache_key(prefix8: str) -> str:
    return f"sys:auth:{prefix8}"


def admin_ratelimit_key(client_ip: str) -> str:
    return f"sys:admin_rl:{client_ip}"
