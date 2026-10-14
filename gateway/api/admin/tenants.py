"""Tenant management: `POST/GET /admin/tenants` (platform_admin), plus tenant users.

`POST /admin/tenants/{tid}/users` is an addition to spec §12. Without it there is no
way to create the tenant-scoped admin users that §6.3 requires. Platform admins
can add users to any tenant; tenant admins only to their own.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from gateway.api.admin.deps import ADMIN_ROLES, check_tenant_access, require_roles, service_error
from gateway.api.deps import get_services
from gateway.security.admin_auth import AdminPrincipal
from gateway.services import admin as svc_admin

router = APIRouter(prefix="/admin/tenants", tags=["admin:tenants"])


class TenantIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)


class TenantOut(BaseModel):
    id: uuid.UUID
    name: str
    status: str
    created_at: datetime | None = None


class UserIn(BaseModel):
    email: str = Field(min_length=3, max_length=320, pattern=r"^[^@\s]+@[^@\s]+$")
    password: str = Field(min_length=12, max_length=1024)
    role: Literal["tenant_admin", "tenant_viewer"]


class UserOut(BaseModel):
    id: uuid.UUID
    email: str
    role: str
    tenant_id: uuid.UUID | None


@router.post("", status_code=201)
async def create_tenant(
    body: TenantIn, request: Request, principal: AdminPrincipal = Depends(require_roles("platform_admin"))
) -> TenantOut:
    try:
        t = await svc_admin.create_tenant(get_services(request).db, actor=principal.actor, name=body.name)
    except Exception as exc:
        raise service_error(exc) from exc
    return TenantOut(id=t.id, name=t.name, status=t.status or "active", created_at=t.created_at)


@router.get("")
async def list_tenants(request: Request, _: AdminPrincipal = Depends(require_roles("platform_admin"))) -> dict:
    tenants = await svc_admin.list_tenants(get_services(request).db)
    return {"object": "list", "data": [TenantOut(id=t.id, name=t.name, status=t.status, created_at=t.created_at) for t in tenants]}


@router.post("/{tenant_id}/users", status_code=201)
async def create_tenant_user(
    tenant_id: uuid.UUID, body: UserIn, request: Request, principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES))
) -> UserOut:
    check_tenant_access(principal, tenant_id)
    try:
        u = await svc_admin.create_admin_user(
            get_services(request).db, actor=principal.actor, email=body.email, password=body.password, role=body.role, tenant_id=tenant_id
        )
    except Exception as exc:
        raise service_error(exc) from exc
    return UserOut(id=u.id, email=u.email, role=u.role, tenant_id=u.tenant_id)
