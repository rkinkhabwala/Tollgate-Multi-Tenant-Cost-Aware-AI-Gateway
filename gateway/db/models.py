"""SQLAlchemy ORM mappings for the spec §5 tables (plus admin-user tables for §6.3).

The DDL itself lives in the Alembic migration (raw SQL), because partitioning,
generated columns, RLS policies and the materialized view are clearer in SQL. These
mappings are used for querying and must stay in sync with that migration.

Every tenant-scoped table carries `tenant_id` and is protected by Postgres RLS
(`tenant_id = current_setting('app.tenant_id')`). `models` is the only global,
platform-managed table.
"""

from __future__ import annotations

import uuid
from datetime import datetime
from decimal import Decimal
from ipaddress import IPv4Network, IPv6Network
from typing import Any

from sqlalchemy import BigInteger, Boolean, Computed, DateTime, ForeignKey, Integer, LargeBinary, Numeric, Text, text
from sqlalchemy.dialects.postgresql import ARRAY, CIDR, JSONB, UUID
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    # Fetch server-generated values (ids, timestamps, status defaults) via RETURNING
    # on insert, so objects stay usable after their session closes.
    __mapper_args__ = {"eager_defaults": True}


def _uuid_pk() -> Mapped[uuid.UUID]:
    return mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))


def _created_at() -> Mapped[datetime]:
    return mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class Tenant(Base):
    __tablename__ = "tenants"
    id: Mapped[uuid.UUID] = _uuid_pk()
    name: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    created_at: Mapped[datetime] = _created_at()


class App(Base):
    __tablename__ = "apps"
    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    name: Mapped[str] = mapped_column(Text, nullable=False)
    environment: Mapped[str] = mapped_column(Text, nullable=False, server_default="prod")
    baseline_model: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(Text, nullable=False, server_default="active")
    created_at: Mapped[datetime] = _created_at()


class ApiKey(Base):
    __tablename__ = "api_keys"
    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    app_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("apps.id"), nullable=False)
    prefix: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    secret_hash: Mapped[bytes] = mapped_column(LargeBinary, nullable=False)
    scopes: Mapped[list[str]] = mapped_column(ARRAY(Text), nullable=False)
    ip_allowlist: Mapped[list[IPv4Network | IPv6Network] | None] = mapped_column(ARRAY(CIDR))
    expires_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    last_used_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()


class AppPolicy(Base):
    __tablename__ = "app_policies"
    app_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("apps.id"), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    policy: Mapped[dict[str, Any]] = mapped_column(JSONB, nullable=False)
    version: Mapped[int] = mapped_column(Integer, nullable=False, server_default="1")
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))


class Budget(Base):
    __tablename__ = "budgets"
    app_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("apps.id"), primary_key=True)
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"), nullable=False)
    monthly_usd: Mapped[Decimal] = mapped_column(Numeric(12, 4), nullable=False)
    soft_pct: Mapped[int] = mapped_column(Integer, nullable=False, server_default="80")
    hard_pct: Mapped[int] = mapped_column(Integer, nullable=False, server_default="100")


class Model(Base):
    """Global model catalog (platform-managed, no tenant_id, no RLS)."""

    __tablename__ = "models"
    id: Mapped[str] = mapped_column(Text, primary_key=True)
    provider: Mapped[str] = mapped_column(Text, nullable=False)
    provider_model: Mapped[str] = mapped_column(Text, nullable=False)
    tier: Mapped[str] = mapped_column(Text, nullable=False)
    input_usd_per_mtok: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    output_usd_per_mtok: Mapped[Decimal] = mapped_column(Numeric(10, 4), nullable=False)
    context_window: Mapped[int] = mapped_column(Integer, nullable=False)
    supports_tools: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    supports_json: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="false")
    enabled: Mapped[bool] = mapped_column(Boolean, nullable=False, server_default="true")
    long_context_threshold: Mapped[int | None] = mapped_column(Integer)
    long_input_usd_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))
    long_output_usd_per_mtok: Mapped[Decimal | None] = mapped_column(Numeric(10, 4))


class UsageEvent(Base):
    """Usage ledger, range-partitioned by month on `ts`."""

    __tablename__ = "usage_events"
    id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), primary_key=True, server_default=text("gen_random_uuid()"))
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), primary_key=True, server_default=text("now()"))
    tenant_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    app_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    api_key_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), nullable=False)
    request_id: Mapped[str] = mapped_column(Text, nullable=False)
    requested_model: Mapped[str | None] = mapped_column(Text)
    routed_model: Mapped[str | None] = mapped_column(Text)
    route_reason: Mapped[str | None] = mapped_column(Text)
    escalations: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    cache_status: Mapped[str] = mapped_column(Text, nullable=False)
    input_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    output_tokens: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    actual_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    baseline_cost_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), nullable=False)
    saved_usd: Mapped[Decimal] = mapped_column(Numeric(12, 6), Computed("baseline_cost_usd - actual_cost_usd"))
    pii_entities: Mapped[dict[str, int] | None] = mapped_column(JSONB)
    latency_ms: Mapped[int | None] = mapped_column(Integer)
    gateway_ms: Mapped[int | None] = mapped_column(Integer)
    status_code: Mapped[int] = mapped_column(Integer, nullable=False)
    attempts: Mapped[list[dict[str, Any]] | None] = mapped_column(JSONB)  # per-call, no text


class AuditLog(Base):
    __tablename__ = "audit_log"
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    ts: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False, server_default=text("now()"))
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    actor: Mapped[str] = mapped_column(Text, nullable=False)
    action: Mapped[str] = mapped_column(Text, nullable=False)
    target: Mapped[str] = mapped_column(Text, nullable=False)
    diff: Mapped[dict[str, Any] | None] = mapped_column(JSONB)


class AdminUser(Base):
    """Local control-plane user (spec §6.3). `tenant_id` is NULL for platform admins."""

    __tablename__ = "admin_users"
    id: Mapped[uuid.UUID] = _uuid_pk()
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True), ForeignKey("tenants.id"))
    email: Mapped[str] = mapped_column(Text, unique=True, nullable=False)
    password_hash: Mapped[str] = mapped_column(Text, nullable=False)
    role: Mapped[str] = mapped_column(Text, nullable=False)  # platform_admin | tenant_admin | tenant_viewer
    failed_attempts: Mapped[int] = mapped_column(Integer, nullable=False, server_default="0")
    locked_until: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()


class AdminRefreshToken(Base):
    """Opaque refresh tokens, stored as SHA-256 and rotated on every use."""

    __tablename__ = "admin_refresh_tokens"
    id: Mapped[uuid.UUID] = _uuid_pk()
    user_id: Mapped[uuid.UUID] = mapped_column(UUID(as_uuid=True), ForeignKey("admin_users.id"), nullable=False)
    tenant_id: Mapped[uuid.UUID | None] = mapped_column(UUID(as_uuid=True))
    token_hash: Mapped[bytes] = mapped_column(LargeBinary, unique=True, nullable=False)
    expires_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True))
    created_at: Mapped[datetime] = _created_at()
