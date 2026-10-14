"""Control-plane operations: tenants, admin users, apps, API keys — plus audit logging.

Shared by the `/admin/*` routers and `scripts/seed_demo_tenants.py`, so there is
one implementation of each mutation and every one writes `audit_log` (spec §6.3)
in the same transaction as the change.

Tenant scoping: anything inside a tenant runs in `db.tenant_session(tenant_id)`
(RLS enforced). Only tenant creation/listing and "which tenant owns this
app/key?" lookups for platform admins use the system session.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from ipaddress import ip_network
from typing import Any

from redis.asyncio import Redis
from sqlalchemy import func, select, update
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from gateway.catalog import ModelCatalog
from gateway.db.models import AdminUser, ApiKey, App, AppPolicy, AuditLog, Budget, Tenant
from gateway.db.session import Database
from gateway.policy.schema import AppPolicy as AppPolicyModel
from gateway.redis_keys import AUTH_TOMBSTONE_TTL_S, REVOKED_TOMBSTONE, app_key, auth_cache_key
from gateway.security.api_keys import env_for_app, generate_api_key, hash_api_key
from gateway.security.passwords import hash_password

ALL_SCOPES = frozenset({"chat:write", "embeddings:write", "usage:read"})
DEFAULT_SCOPES = ["chat:write", "embeddings:write", "usage:read"]


class NotFound(Exception):
    """The target does not exist *or* is not visible to the caller's tenant."""


class Conflict(Exception):
    pass


class Invalid(Exception):
    pass


@dataclass(frozen=True)
class IssuedKey:
    row: ApiKey
    full_key: str  # shown exactly once to the caller; never stored or logged


async def audit(
    session: AsyncSession, *, tenant_id: uuid.UUID | None, actor: str, action: str, target: str, diff: dict[str, Any] | None = None
) -> None:
    session.add(AuditLog(tenant_id=tenant_id, actor=actor, action=action, target=target, diff=diff))


# --- tenants ------------------------------------------------------------------
async def create_tenant(db: Database, *, actor: str, name: str) -> Tenant:
    async with db.system_session() as s:
        if (await s.execute(select(Tenant).where(Tenant.name == name))).scalar_one_or_none():
            raise Conflict(f"tenant '{name}' already exists")
        tenant = Tenant(name=name)
        s.add(tenant)
        await s.flush()
        await audit(s, tenant_id=tenant.id, actor=actor, action="tenant.create", target=f"tenant:{tenant.id}", diff={"name": name})
    return tenant


async def list_tenants(db: Database) -> list[Tenant]:
    async with db.system_session() as s:
        return list((await s.execute(select(Tenant).order_by(Tenant.created_at))).scalars())


async def get_tenant_by_name(db: Database, name: str) -> Tenant | None:
    async with db.system_session() as s:
        return (await s.execute(select(Tenant).where(Tenant.name == name))).scalar_one_or_none()


# --- admin users --------------------------------------------------------------
async def create_admin_user(
    db: Database, *, actor: str, email: str, password: str, role: str, tenant_id: uuid.UUID | None
) -> AdminUser:
    if role == "platform_admin":
        if tenant_id is not None:
            raise Invalid("platform_admin users do not belong to a tenant")
        ctx = db.system_session()
    else:
        if role not in ("tenant_admin", "tenant_viewer") or tenant_id is None:
            raise Invalid("tenant users need role tenant_admin|tenant_viewer and a tenant")
        ctx = db.tenant_session(tenant_id)
    user = AdminUser(email=email.lower(), password_hash=hash_password(password), role=role, tenant_id=tenant_id)
    try:
        async with ctx as s:
            s.add(user)
            await s.flush()
            await audit(s, tenant_id=tenant_id, actor=actor, action="user.create", target=f"user:{user.id}", diff={"email": user.email, "role": role})
    except IntegrityError as exc:
        raise Conflict("a user with that email already exists") from exc
    return user


async def ensure_bootstrap_admin(db: Database, email: str, password: str) -> bool:
    """Create the first platform admin if no admin users exist. Returns True if created."""
    async with db.system_session() as s:
        if (await s.execute(select(AdminUser.id).where(AdminUser.role == "platform_admin").limit(1))).first():
            return False
    await create_admin_user(db, actor="system:bootstrap", email=email, password=password, role="platform_admin", tenant_id=None)
    return True


# --- apps ---------------------------------------------------------------------
async def create_app(
    db: Database,
    *,
    actor: str,
    tenant_id: uuid.UUID,
    name: str,
    environment: str,
    baseline_model: str,
    catalog: ModelCatalog,
    default_policy: AppPolicyModel,
    monthly_budget_usd: Decimal,
) -> App:
    if catalog.get(baseline_model) is None:
        raise Invalid(f"baseline_model '{baseline_model}' is not in the model catalog")
    app = App(tenant_id=tenant_id, name=name, environment=environment, baseline_model=baseline_model)
    try:
        async with db.tenant_session(tenant_id) as s:
            if await s.get(Tenant, tenant_id) is None:
                raise NotFound("tenant not found")
            s.add(app)
            await s.flush()
            s.add(AppPolicy(app_id=app.id, tenant_id=tenant_id, policy=default_policy.model_dump(mode="json")))
            s.add(Budget(app_id=app.id, tenant_id=tenant_id, monthly_usd=monthly_budget_usd))
            await audit(
                s,
                tenant_id=tenant_id,
                actor=actor,
                action="app.create",
                target=f"app:{app.id}",
                diff={"name": name, "environment": environment, "baseline_model": baseline_model},
            )
    except IntegrityError as exc:
        raise Conflict(f"app '{name}' ({environment}) already exists in this tenant") from exc
    return app


async def list_apps(db: Database, tenant_id: uuid.UUID) -> list[App]:
    async with db.tenant_session(tenant_id) as s:
        return list((await s.execute(select(App).order_by(App.created_at))).scalars())


async def update_app(
    db: Database, redis: Redis, *, actor: str, tenant_id: uuid.UUID, app_id: uuid.UUID, changes: dict[str, Any], catalog: ModelCatalog
) -> App:
    if "baseline_model" in changes and catalog.get(changes["baseline_model"]) is None:
        raise Invalid("baseline_model is not in the model catalog")
    async with db.tenant_session(tenant_id) as s:
        app = await s.get(App, app_id)
        if app is None:
            raise NotFound("app not found")
        before = {k: getattr(app, k) for k in changes}
        for k, v in changes.items():
            setattr(app, k, v)
        prefixes = list((await s.execute(select(ApiKey.prefix).where(ApiKey.app_id == app_id))).scalars())
        await audit(s, tenant_id=tenant_id, actor=actor, action="app.update", target=f"app:{app_id}", diff={"before": before, "after": changes})
    # Status/baseline live in the cached AuthContext: drop it so the change applies now.
    if prefixes:
        await redis.delete(*[auth_cache_key(p) for p in prefixes])
    return app


# --- API keys -----------------------------------------------------------------
def _validate_scopes(scopes: list[str]) -> list[str]:
    unknown = set(scopes) - ALL_SCOPES
    if unknown or not scopes:
        raise Invalid(f"scopes must be a non-empty subset of {sorted(ALL_SCOPES)}")
    return sorted(set(scopes))


async def create_key(
    db: Database,
    *,
    actor: str,
    tenant_id: uuid.UUID,
    app_id: uuid.UUID,
    scopes: list[str] | None = None,
    ip_allowlist: list[str] | None = None,
    expires_at: datetime | None = None,
) -> IssuedKey:
    scopes = _validate_scopes(scopes or DEFAULT_SCOPES)
    try:
        networks = [ip_network(c, strict=False) for c in ip_allowlist] if ip_allowlist else None
    except ValueError as exc:
        raise Invalid("ip_allowlist entries must be IPs or CIDRs") from exc
    async with db.tenant_session(tenant_id) as s:
        app = await s.get(App, app_id)  # RLS: other tenants' apps are invisible
        if app is None:
            raise NotFound("app not found")
        for _ in range(5):  # prefix collisions are ~impossible; retry anyway
            full_key, prefix = generate_api_key(env_for_app(app.environment))
            row = ApiKey(
                tenant_id=tenant_id,
                app_id=app_id,
                prefix=prefix,
                secret_hash=hash_api_key(full_key),
                scopes=scopes,
                ip_allowlist=networks,
                expires_at=expires_at,
            )
            try:
                async with s.begin_nested():
                    s.add(row)
                    await s.flush()
                break
            except IntegrityError:
                continue
        else:  # pragma: no cover
            raise Conflict("could not allocate a unique key prefix")
        await audit(
            s,
            tenant_id=tenant_id,
            actor=actor,
            action="key.create",
            target=f"key:{row.id}",
            diff={"app_id": str(app_id), "prefix": prefix, "scopes": scopes},
        )
    return IssuedKey(row=row, full_key=full_key)


async def list_keys(db: Database, tenant_id: uuid.UUID, app_id: uuid.UUID) -> list[ApiKey]:
    async with db.tenant_session(tenant_id) as s:
        if await s.get(App, app_id) is None:
            raise NotFound("app not found")
        return list((await s.execute(select(ApiKey).where(ApiKey.app_id == app_id).order_by(ApiKey.created_at))).scalars())


async def revoke_key(db: Database, redis: Redis, *, actor: str, tenant_id: uuid.UUID, key_id: uuid.UUID) -> ApiKey:
    """Revoke a key and evict its cached AuthContext, so it fails on the very next request."""
    async with db.tenant_session(tenant_id) as s:
        key = await s.get(ApiKey, key_id)
        if key is None:
            raise NotFound("key not found")
        if key.revoked_at is None:
            await s.execute(update(ApiKey).where(ApiKey.id == key_id).values(revoked_at=datetime.now(UTC)))
            await s.refresh(key)
            await audit(s, tenant_id=tenant_id, actor=actor, action="key.revoke", target=f"key:{key_id}", diff={"prefix": key.prefix})
    # Replace the cached AuthContext with a tombstone rather than deleting it: the
    # authn stage populates the cache with SET NX, so a request that read the row
    # just before this commit cannot resurrect the key in the cache.
    await redis.set(auth_cache_key(key.prefix), REVOKED_TOMBSTONE, ex=AUTH_TOMBSTONE_TTL_S)
    return key


# --- ownership lookups for platform admins (system role) -----------------------
async def tenant_of_app(db: Database, app_id: uuid.UUID) -> uuid.UUID:
    async with db.system_session() as s:
        tid = (await s.execute(select(App.tenant_id).where(App.id == app_id))).scalar_one_or_none()
    if tid is None:
        raise NotFound("app not found")
    return tid


async def tenant_of_key(db: Database, key_id: uuid.UUID) -> uuid.UUID:
    async with db.system_session() as s:
        tid = (await s.execute(select(ApiKey.tenant_id).where(ApiKey.id == key_id))).scalar_one_or_none()
    if tid is None:
        raise NotFound("key not found")
    return tid


# --- policies (spec §9) ------------------------------------------------------------
async def get_policy(db: Database, *, tenant_id: uuid.UUID, app_id: uuid.UUID) -> AppPolicy:
    async with db.tenant_session(tenant_id) as s:
        row = (await s.execute(select(AppPolicy).where(AppPolicy.app_id == app_id))).scalar_one_or_none()
    if row is None:
        raise NotFound("app not found")
    return row


async def put_policy(
    db: Database, redis: Redis, *, actor: str, tenant_id: uuid.UUID, app_id: uuid.UUID, policy: AppPolicyModel
) -> AppPolicy:
    """Replace an app's policy: bump `version`, audit the change, invalidate caches now.

    Invalidation: the AuthContext cached in `sys:auth:{prefix}` carries the policy
    version, so it is dropped for every *active* key of the app (revocation
    tombstones are kept). Cached policy documents under `t:{tid}:a:{aid}:policy:*`
    are deleted too. The very next request therefore loads the new version.
    """
    new_doc = policy.model_dump(mode="json")
    async with db.tenant_session(tenant_id) as s:
        if await s.get(App, app_id) is None:
            raise NotFound("app not found")
        row = (await s.execute(select(AppPolicy).where(AppPolicy.app_id == app_id).with_for_update())).scalar_one_or_none()
        if row is None:
            row = AppPolicy(app_id=app_id, tenant_id=tenant_id, policy=new_doc, version=1)
            s.add(row)
            before: dict[str, Any] = {}
        else:
            before = dict(row.policy)
            row.policy = new_doc
            row.version += 1
            row.updated_at = func.now()
        await s.flush()
        await s.refresh(row)
        changed = {k: {"before": before.get(k), "after": new_doc.get(k)} for k in new_doc if before.get(k) != new_doc.get(k)}
        await audit(
            s,
            tenant_id=tenant_id,
            actor=actor,
            action="policy.update",
            target=f"app:{app_id}",
            diff={"version": row.version, "changed": changed},
        )
        prefixes = list(
            (await s.execute(select(ApiKey.prefix).where(ApiKey.app_id == app_id, ApiKey.revoked_at.is_(None)))).scalars()
        )
    stale = [auth_cache_key(p) for p in prefixes]
    stale += [k async for k in redis.scan_iter(match=app_key(tenant_id, app_id, "policy:*"), count=500)]
    if stale:
        await redis.delete(*stale)
    return row
