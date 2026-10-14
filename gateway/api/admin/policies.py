"""`GET/PUT /admin/apps/{aid}/policy` (spec §9, §12).

PUT replaces the whole policy document. It is validated against the full §9
Pydantic schema, which rejects unknown keys at every level: a typo is a 400, not a
silently ignored setting. Each PUT bumps `version`, writes `policy.update` to the
audit log, and invalidates cached auth contexts and policy documents, so the change
applies to the very next request.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel

from gateway.api.admin.deps import ADMIN_ROLES, READ_ROLES, require_roles, service_error, tenant_for_app
from gateway.api.deps import get_services
from gateway.db.models import AppPolicy as AppPolicyRow
from gateway.policy.schema import AppPolicy
from gateway.security.admin_auth import AdminPrincipal
from gateway.services import admin as svc_admin

router = APIRouter(prefix="/admin/apps/{app_id}/policy", tags=["admin:policies"])


class PolicyOut(BaseModel):
    app_id: uuid.UUID
    version: int
    updated_at: datetime | None
    policy: dict[str, Any]

    @classmethod
    def of(cls, row: AppPolicyRow) -> PolicyOut:
        return cls(app_id=row.app_id, version=row.version, updated_at=row.updated_at, policy=row.policy)


@router.get("")
async def get_policy(app_id: uuid.UUID, request: Request, principal: AdminPrincipal = Depends(require_roles(*READ_ROLES))) -> PolicyOut:
    try:
        tenant_id = await tenant_for_app(request, principal, app_id)
        row = await svc_admin.get_policy(get_services(request).db, tenant_id=tenant_id, app_id=app_id)
    except Exception as exc:
        raise service_error(exc) from exc
    return PolicyOut.of(row)


@router.put("")
async def put_policy(
    app_id: uuid.UUID, body: AppPolicy, request: Request, principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES))
) -> PolicyOut:
    s = get_services(request)
    try:
        tenant_id = await tenant_for_app(request, principal, app_id)
        row = await svc_admin.put_policy(s.db, s.redis, actor=principal.actor, tenant_id=tenant_id, app_id=app_id, policy=body)
    except Exception as exc:
        raise service_error(exc) from exc
    return PolicyOut.of(row)
