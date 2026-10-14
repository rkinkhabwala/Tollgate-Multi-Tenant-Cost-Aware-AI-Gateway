"""App management: `POST/GET /admin/tenants/{tid}/apps`, `PATCH .../apps/{aid}` (spec §12).

Creating an app also creates its default policy (`config/default_policy.json`) and a
monthly budget (spec §12 onboarding step 1). Spec §12 writes the PATCH path as
`/admin/tenants/{tid}/apps`; it is implemented on the item path, which is what a
PATCH needs.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from typing import Literal

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, ConfigDict, Field

from gateway.api.admin.deps import ADMIN_ROLES, READ_ROLES, check_tenant_access, require_roles, service_error
from gateway.api.deps import get_services
from gateway.db.models import App
from gateway.security.admin_auth import AdminPrincipal
from gateway.services import admin as svc_admin

router = APIRouter(prefix="/admin/tenants/{tenant_id}/apps", tags=["admin:apps"])

Environment = Literal["dev", "staging", "prod"]


class AppIn(BaseModel):
    name: str = Field(min_length=1, max_length=200)
    environment: Environment = "prod"
    baseline_model: str
    monthly_budget_usd: Decimal | None = Field(None, ge=0)


class AppPatch(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str | None = Field(None, min_length=1, max_length=200)
    status: Literal["active", "suspended"] | None = None
    baseline_model: str | None = None


class AppOut(BaseModel):
    id: uuid.UUID
    tenant_id: uuid.UUID
    name: str
    environment: str
    baseline_model: str
    status: str
    created_at: datetime | None = None

    @classmethod
    def of(cls, a: App) -> AppOut:
        return cls(
            id=a.id,
            tenant_id=a.tenant_id,
            name=a.name,
            environment=a.environment or "prod",
            baseline_model=a.baseline_model,
            status=a.status or "active",
            created_at=a.created_at,
        )


@router.post("", status_code=201)
async def create_app(
    tenant_id: uuid.UUID, body: AppIn, request: Request, principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES))
) -> AppOut:
    check_tenant_access(principal, tenant_id)
    s = get_services(request)
    try:
        app = await svc_admin.create_app(
            s.db,
            actor=principal.actor,
            tenant_id=tenant_id,
            name=body.name,
            environment=body.environment,
            baseline_model=body.baseline_model,
            catalog=s.catalog,
            default_policy=s.default_policy,
            monthly_budget_usd=body.monthly_budget_usd if body.monthly_budget_usd is not None else Decimal(str(s.settings.default_budget_usd)),
        )
    except Exception as exc:
        raise service_error(exc) from exc
    return AppOut.of(app)


@router.get("")
async def list_apps(tenant_id: uuid.UUID, request: Request, principal: AdminPrincipal = Depends(require_roles(*READ_ROLES))) -> dict:
    check_tenant_access(principal, tenant_id)
    apps = await svc_admin.list_apps(get_services(request).db, tenant_id)
    return {"object": "list", "data": [AppOut.of(a) for a in apps]}


@router.patch("/{app_id}")
async def update_app(
    tenant_id: uuid.UUID,
    app_id: uuid.UUID,
    body: AppPatch,
    request: Request,
    principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES)),
) -> AppOut:
    check_tenant_access(principal, tenant_id)
    s = get_services(request)
    try:
        app = await svc_admin.update_app(
            s.db, s.redis, actor=principal.actor, tenant_id=tenant_id, app_id=app_id, changes=body.model_dump(exclude_none=True), catalog=s.catalog
        )
    except Exception as exc:
        raise service_error(exc) from exc
    return AppOut.of(app)
