"""Row-Level Security DDL helpers, used by the Alembic migration.

Policy shape for every tenant-scoped table (spec §4):

    USING      (tenant_id = current_setting('app.tenant_id', true)::uuid)
    WITH CHECK (tenant_id = current_setting('app.tenant_id', true)::uuid)

`current_setting(..., true)` returns NULL (or '') when unset, so a session that
forgot to set the tenant sees *nothing* rather than everything. `FORCE ROW LEVEL
SECURITY` also applies policies to the table owner.
"""

# Tables protected by RLS -> the column holding the tenant id.
TENANT_TABLES: dict[str, str] = {
    "tenants": "id",
    "apps": "tenant_id",
    "api_keys": "tenant_id",
    "app_policies": "tenant_id",
    "budgets": "tenant_id",
    "usage_events": "tenant_id",
    "audit_log": "tenant_id",
    "admin_users": "tenant_id",
    "admin_refresh_tokens": "tenant_id",
}

CURRENT_TENANT = "NULLIF(current_setting('app.tenant_id', true), '')::uuid"


def enable_rls_sql(table: str, column: str) -> list[str]:
    """DDL statements enabling + forcing RLS with a tenant-isolation policy."""
    cond = f"{column} = {CURRENT_TENANT}"
    return [
        f"ALTER TABLE {table} ENABLE ROW LEVEL SECURITY",
        f"ALTER TABLE {table} FORCE ROW LEVEL SECURITY",
        f"CREATE POLICY tenant_isolation ON {table} USING ({cond}) WITH CHECK ({cond})",
    ]
