"""Control-plane dependencies: JWT principal, role checks, tenant scoping, rate limiting.

Tenant scoping rule: a `tenant_admin` or `tenant_viewer` may only touch its own
tenant; a `platform_admin` may act on any tenant. Even for platform admins, the
actual reads and writes run in that tenant's RLS session (see `services/admin.py`),
so RLS still guards against a mixed-up id.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Awaitable, Callable

from fastapi import Depends, Request

from gateway.api.deps import bearer_token, client_ip, get_services
from gateway.errors import GatewayError, bad_request, not_found, permission_denied, rate_limited
from gateway.redis_keys import admin_ratelimit_key
from gateway.security.admin_auth import AdminPrincipal, AuthError
from gateway.services import admin as svc_admin


async def admin_rate_limit(request: Request) -> None:
    """Per-IP fixed-window limit on every /admin endpoint (spec §15)."""
    services = get_services(request)
    ip = client_ip(request, services.settings.trust_proxy_headers) or "unknown"
    window = int(time.time()) // 60
    key = f"{admin_ratelimit_key(ip)}:{window}"
    count = await services.redis.incr(key)
    if count == 1:
        await services.redis.expire(key, 61)
    if count > services.settings.admin_rpm_per_ip:
        raise rate_limited("Too many admin requests.", {"retry-after": str(60 - int(time.time()) % 60)})


async def current_principal(request: Request) -> AdminPrincipal:
    token = bearer_token(request)
    if not token:
        raise GatewayError(401, "unauthorized", "Missing bearer token.", type_="invalid_request_error")
    try:
        return get_services(request).admin_auth.verify_access_token(token)
    except AuthError as exc:
        raise GatewayError(401, "unauthorized", exc.reason.capitalize() + ".", type_="invalid_request_error") from exc


def require_roles(*roles: str) -> Callable[..., Awaitable[AdminPrincipal]]:
    async def dep(principal: AdminPrincipal = Depends(current_principal)) -> AdminPrincipal:
        if principal.role not in roles:
            raise permission_denied("Your role does not allow this action.")
        return principal

    return dep


def check_tenant_access(principal: AdminPrincipal, tenant_id: uuid.UUID) -> None:
    if principal.role != "platform_admin" and principal.tenant_id != tenant_id:
        raise permission_denied("You do not have access to this tenant.")


async def tenant_for_app(request: Request, principal: AdminPrincipal, app_id: uuid.UUID) -> uuid.UUID:
    """Which tenant session to use for an /admin/apps/{aid} call."""
    if principal.role == "platform_admin":
        return await svc_admin.tenant_of_app(get_services(request).db, app_id)
    assert principal.tenant_id is not None
    return principal.tenant_id  # RLS makes other tenants' apps 404


async def tenant_for_key(request: Request, principal: AdminPrincipal, key_id: uuid.UUID) -> uuid.UUID:
    if principal.role == "platform_admin":
        return await svc_admin.tenant_of_key(get_services(request).db, key_id)
    assert principal.tenant_id is not None
    return principal.tenant_id


def service_error(exc: Exception) -> GatewayError:
    """Map service-layer exceptions to OpenAI-shaped HTTP errors."""
    if isinstance(exc, svc_admin.NotFound):
        return not_found(str(exc))
    if isinstance(exc, svc_admin.Conflict):
        return GatewayError(409, "conflict", str(exc))
    if isinstance(exc, svc_admin.Invalid):
        return bad_request(str(exc))
    raise exc


ADMIN_ROLES = ("platform_admin", "tenant_admin")
READ_ROLES = ("platform_admin", "tenant_admin", "tenant_viewer")
