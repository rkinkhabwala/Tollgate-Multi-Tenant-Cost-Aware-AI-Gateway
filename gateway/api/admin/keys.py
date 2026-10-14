"""API key management (spec §6.1, §12).

* `POST /admin/apps/{aid}/keys` creates a key and returns the full key **once**.
* `GET /admin/apps/{aid}/keys` lists keys (prefix, scopes, status), never secrets.
* `DELETE /admin/keys/{kid}` revokes a key. Its cached AuthContext is replaced with
  a tombstone in the same request, so the key fails on the very next call.

Rotation (spec §6.1) is: create a new key, deploy it, revoke the old one. Both keys
work in between.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from gateway.api.admin.deps import ADMIN_ROLES, require_roles, service_error, tenant_for_app, tenant_for_key
from gateway.api.deps import get_services
from gateway.db.models import ApiKey
from gateway.security.admin_auth import AdminPrincipal
from gateway.services import admin as svc_admin

router = APIRouter(prefix="/admin", tags=["admin:keys"])


class KeyIn(BaseModel):
    scopes: list[str] | None = None
    ip_allowlist: list[str] | None = Field(None, max_length=100)
    expires_at: datetime | None = None


class KeyOut(BaseModel):
    id: uuid.UUID
    app_id: uuid.UUID
    tenant_id: uuid.UUID
    prefix: str
    scopes: list[str]
    ip_allowlist: list[str] | None
    expires_at: datetime | None
    revoked_at: datetime | None
    last_used_at: datetime | None
    created_at: datetime | None

    @classmethod
    def of(cls, k: ApiKey) -> KeyOut:
        return cls(
            id=k.id,
            app_id=k.app_id,
            tenant_id=k.tenant_id,
            prefix=k.prefix,
            scopes=list(k.scopes),
            ip_allowlist=[str(n) for n in k.ip_allowlist] if k.ip_allowlist else None,
            expires_at=k.expires_at,
            revoked_at=k.revoked_at,
            last_used_at=k.last_used_at,
            created_at=k.created_at,
        )


class CreatedKeyOut(KeyOut):
    key: str = Field(description="The full API key. Shown once; store it securely.")


@router.post("/apps/{app_id}/keys", status_code=201)
async def create_key(
    app_id: uuid.UUID, body: KeyIn, request: Request, principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES))
) -> CreatedKeyOut:
    s = get_services(request)
    try:
        tenant_id = await tenant_for_app(request, principal, app_id)
        issued = await svc_admin.create_key(
            s.db,
            actor=principal.actor,
            tenant_id=tenant_id,
            app_id=app_id,
            scopes=body.scopes,
            ip_allowlist=body.ip_allowlist,
            expires_at=body.expires_at,
        )
    except Exception as exc:
        raise service_error(exc) from exc
    return CreatedKeyOut(**KeyOut.of(issued.row).model_dump(), key=issued.full_key)


@router.get("/apps/{app_id}/keys")
async def list_keys(app_id: uuid.UUID, request: Request, principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES))) -> dict:
    s = get_services(request)
    try:
        tenant_id = await tenant_for_app(request, principal, app_id)
        keys = await svc_admin.list_keys(s.db, tenant_id, app_id)
    except Exception as exc:
        raise service_error(exc) from exc
    return {"object": "list", "data": [KeyOut.of(k) for k in keys]}


@router.delete("/keys/{key_id}")
async def revoke_key(key_id: uuid.UUID, request: Request, principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES))) -> KeyOut:
    s = get_services(request)
    try:
        tenant_id = await tenant_for_key(request, principal, key_id)
        key = await svc_admin.revoke_key(s.db, s.redis, actor=principal.actor, tenant_id=tenant_id, key_id=key_id)
    except Exception as exc:
        raise service_error(exc) from exc
    return KeyOut.of(key)
