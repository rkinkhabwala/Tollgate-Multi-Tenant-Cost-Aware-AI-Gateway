"""Initial schema: spec §5 tables, admin users, RLS, monthly partitions, usage_daily.

Revision ID: 0001_initial
Revises:
Create Date: 2026-10-14

Notes
-----
* RLS: every tenant-scoped table gets ENABLE + FORCE RLS and a `tenant_isolation`
  policy keyed on `current_setting('app.tenant_id')` (see gateway/db/rls.py).
* Partitions: `usage_events` is range-partitioned by month. This migration creates
  last month through three months ahead, plus a DEFAULT partition as a safety net.
  The rollup worker keeps creating future months via `ensure_usage_partition()`.
* Partition creation and materialized-view refresh need table ownership, so they are
  wrapped in SECURITY DEFINER functions that only `tollgate_system` may execute.
* `usage_daily` is a materialized view. Postgres does not apply RLS to matviews, so
  only `tollgate_system` is granted SELECT and callers must filter by tenant
  explicitly.
"""

from datetime import UTC, date, datetime

from alembic import op

from gateway.db.rls import TENANT_TABLES, enable_rls_sql

revision = "0001_initial"
down_revision = None
branch_labels = None
depends_on = None

APP_ROLE = "tollgate_app"
SYSTEM_ROLE = "tollgate_system"

DDL = r"""
CREATE EXTENSION IF NOT EXISTS pgcrypto;

CREATE TABLE tenants (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  name          TEXT NOT NULL UNIQUE,
  status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE apps (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  name          TEXT NOT NULL,
  environment   TEXT NOT NULL DEFAULT 'prod' CHECK (environment IN ('dev', 'staging', 'prod')),
  baseline_model TEXT NOT NULL,
  status        TEXT NOT NULL DEFAULT 'active' CHECK (status IN ('active', 'suspended')),
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now(),
  UNIQUE (tenant_id, name, environment)
);
CREATE INDEX apps_tenant_idx ON apps (tenant_id);

CREATE TABLE api_keys (
  id            UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  app_id        UUID NOT NULL REFERENCES apps(id),
  prefix        TEXT NOT NULL UNIQUE,
  secret_hash   BYTEA NOT NULL,
  scopes        TEXT[] NOT NULL,
  ip_allowlist  CIDR[],
  expires_at    TIMESTAMPTZ,
  revoked_at    TIMESTAMPTZ,
  last_used_at  TIMESTAMPTZ,
  created_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX api_keys_app_idx ON api_keys (tenant_id, app_id);

CREATE TABLE app_policies (
  app_id        UUID PRIMARY KEY REFERENCES apps(id),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  policy        JSONB NOT NULL,
  version       INT NOT NULL DEFAULT 1,
  updated_at    TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE budgets (
  app_id        UUID PRIMARY KEY REFERENCES apps(id),
  tenant_id     UUID NOT NULL REFERENCES tenants(id),
  monthly_usd   NUMERIC(12,4) NOT NULL,
  soft_pct      INT NOT NULL DEFAULT 80,
  hard_pct      INT NOT NULL DEFAULT 100
);

CREATE TABLE models (
  id                TEXT PRIMARY KEY,
  provider          TEXT NOT NULL,
  provider_model    TEXT NOT NULL,
  tier              TEXT NOT NULL CHECK (tier IN ('local', 'small', 'medium', 'large')),
  input_usd_per_mtok  NUMERIC(10,4) NOT NULL,
  output_usd_per_mtok NUMERIC(10,4) NOT NULL,
  context_window    INT NOT NULL,
  supports_tools    BOOLEAN NOT NULL DEFAULT false,
  supports_json     BOOLEAN NOT NULL DEFAULT false,
  enabled           BOOLEAN NOT NULL DEFAULT true
);

CREATE TABLE usage_events (
  id              UUID NOT NULL DEFAULT gen_random_uuid(),
  ts              TIMESTAMPTZ NOT NULL DEFAULT now(),
  tenant_id       UUID NOT NULL,
  app_id          UUID NOT NULL,
  api_key_id      UUID NOT NULL,
  request_id      TEXT NOT NULL,
  requested_model TEXT,
  routed_model    TEXT,
  route_reason    TEXT,
  escalations     INT NOT NULL DEFAULT 0,
  cache_status    TEXT NOT NULL,
  input_tokens    INT NOT NULL DEFAULT 0,
  output_tokens   INT NOT NULL DEFAULT 0,
  actual_cost_usd   NUMERIC(12,6) NOT NULL,
  baseline_cost_usd NUMERIC(12,6) NOT NULL,
  saved_usd         NUMERIC(12,6) GENERATED ALWAYS AS (baseline_cost_usd - actual_cost_usd) STORED,
  pii_entities    JSONB,
  latency_ms      INT,
  gateway_ms      INT,
  status_code     INT NOT NULL,
  PRIMARY KEY (ts, id)
) PARTITION BY RANGE (ts);
CREATE INDEX usage_events_scope_idx ON usage_events (tenant_id, app_id, ts);
CREATE TABLE usage_events_default PARTITION OF usage_events DEFAULT;

CREATE TABLE audit_log (
  id          BIGSERIAL PRIMARY KEY,
  ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
  tenant_id   UUID,
  actor       TEXT NOT NULL,
  action      TEXT NOT NULL,
  target      TEXT NOT NULL,
  diff        JSONB
);
CREATE INDEX audit_log_tenant_idx ON audit_log (tenant_id, ts);

-- Control-plane users (spec §6.3; not listed in §5).
CREATE TABLE admin_users (
  id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  tenant_id       UUID REFERENCES tenants(id),
  email           TEXT NOT NULL UNIQUE,
  password_hash   TEXT NOT NULL,
  role            TEXT NOT NULL,
  failed_attempts INT NOT NULL DEFAULT 0,
  locked_until    TIMESTAMPTZ,
  created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
  CHECK (
    (role = 'platform_admin' AND tenant_id IS NULL)
    OR (role IN ('tenant_admin', 'tenant_viewer') AND tenant_id IS NOT NULL)
  )
);

CREATE TABLE admin_refresh_tokens (
  id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
  user_id     UUID NOT NULL REFERENCES admin_users(id),
  tenant_id   UUID,
  token_hash  BYTEA NOT NULL UNIQUE,
  expires_at  TIMESTAMPTZ NOT NULL,
  revoked_at  TIMESTAMPTZ,
  created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Creates the monthly partition containing p_month if missing.
CREATE FUNCTION ensure_usage_partition(p_month date) RETURNS text
LANGUAGE plpgsql SECURITY DEFINER SET search_path = public AS $$
DECLARE
  start_d date := date_trunc('month', p_month)::date;
  end_d   date := (date_trunc('month', p_month) + interval '1 month')::date;
  part    text := format('usage_events_%s', to_char(start_d, 'YYYYMM'));
BEGIN
  IF to_regclass(part) IS NULL THEN
    EXECUTE format('CREATE TABLE %I PARTITION OF usage_events FOR VALUES FROM (%L) TO (%L)', part, start_d, end_d);
  END IF;
  RETURN part;
END $$;

CREATE MATERIALIZED VIEW usage_daily AS
SELECT tenant_id,
       app_id,
       (ts AT TIME ZONE 'UTC')::date      AS day,
       COALESCE(routed_model, '')         AS routed_model,
       cache_status,
       count(*)                           AS requests,
       sum(input_tokens)::bigint          AS input_tokens,
       sum(output_tokens)::bigint         AS output_tokens,
       sum(actual_cost_usd)               AS actual_cost_usd,
       sum(baseline_cost_usd)             AS baseline_cost_usd,
       sum(saved_usd)                     AS saved_usd
FROM usage_events
GROUP BY 1, 2, 3, 4, 5;
CREATE UNIQUE INDEX usage_daily_pk ON usage_daily (tenant_id, app_id, day, routed_model, cache_status);

CREATE FUNCTION refresh_usage_daily() RETURNS void
LANGUAGE sql SECURITY DEFINER SET search_path = public AS $$
  REFRESH MATERIALIZED VIEW CONCURRENTLY usage_daily;
$$;
"""

GRANTS = f"""
GRANT SELECT ON tenants TO {APP_ROLE};
GRANT SELECT, INSERT, UPDATE ON apps, api_keys, app_policies, budgets TO {APP_ROLE};
GRANT SELECT ON usage_events TO {APP_ROLE};
GRANT SELECT, INSERT ON audit_log TO {APP_ROLE};
GRANT USAGE ON SEQUENCE audit_log_id_seq TO {APP_ROLE};
GRANT SELECT, INSERT, UPDATE ON admin_users TO {APP_ROLE};
GRANT SELECT ON models TO {APP_ROLE};

GRANT SELECT, INSERT, UPDATE, DELETE ON
  tenants, apps, api_keys, app_policies, budgets, models, usage_events, audit_log,
  admin_users, admin_refresh_tokens
TO {SYSTEM_ROLE};
GRANT USAGE ON SEQUENCE audit_log_id_seq TO {SYSTEM_ROLE};
GRANT SELECT ON usage_daily TO {SYSTEM_ROLE};

REVOKE ALL ON FUNCTION ensure_usage_partition(date) FROM PUBLIC;
REVOKE ALL ON FUNCTION refresh_usage_daily() FROM PUBLIC;
GRANT EXECUTE ON FUNCTION ensure_usage_partition(date), refresh_usage_daily() TO {SYSTEM_ROLE};
"""


def _split_sql(script: str) -> list[str]:
    """Split a script on top-level `;` (asyncpg can't run multi-statement strings).

    Semicolons inside `$$ ... $$` function bodies are left alone.
    """
    statements, buf, in_body = [], [], False
    for line in script.splitlines():
        if line.strip().startswith("--"):
            continue
        in_body ^= line.count("$$") % 2 == 1
        buf.append(line)
        if not in_body and line.rstrip().endswith(";"):
            statements.append("\n".join(buf).strip().rstrip(";"))
            buf = []
    if "".join(buf).strip():
        statements.append("\n".join(buf).strip())
    return statements


def _run(script: str) -> None:
    for stmt in _split_sql(script):
        op.execute(stmt)


def _add_months(d: date, n: int) -> date:
    y, m = divmod(d.month - 1 + n, 12)
    return date(d.year + y, m + 1, 1)


def upgrade() -> None:
    _run(DDL)
    for table, column in TENANT_TABLES.items():
        for stmt in enable_rls_sql(table, column):
            op.execute(stmt)
    this_month = datetime.now(UTC).date().replace(day=1)
    for offset in range(-1, 4):
        op.execute(f"SELECT ensure_usage_partition('{_add_months(this_month, offset).isoformat()}')")
    _run(GRANTS)


def downgrade() -> None:
    _run(
        """
        DROP FUNCTION IF EXISTS refresh_usage_daily();
        DROP MATERIALIZED VIEW IF EXISTS usage_daily;
        DROP FUNCTION IF EXISTS ensure_usage_partition(date);
        DROP TABLE IF EXISTS admin_refresh_tokens, admin_users, audit_log, usage_events,
          models, budgets, app_policies, api_keys, apps, tenants CASCADE;
        """
    )
