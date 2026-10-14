"""Seed demo data: 2 tenants × 2 apps, each with default policy, budget and one API key.

    uv run python -m scripts.seed_demo_tenants            # from the host
    docker compose exec gateway python -m scripts.seed_demo_tenants   # in the container
    ... --json                                            # machine-readable (demo script)

Idempotent for tenants, apps and users: existing ones are reused. A **new** API key
is minted per app on every run, because keys are only ever shown once. Older keys
stay valid until revoked.

The printed keys are the only copy, so store them; they are never written to disk.
All writes go through `gateway.services.admin`, the same code path as the admin API,
so the audit log records them with actor `system:seed`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from decimal import Decimal

from gateway.catalog import ModelCatalog
from gateway.db.session import create_database
from gateway.policy.loader import load_default_policy
from gateway.services import admin as svc
from gateway.settings import get_settings

ACTOR = "system:seed"
DEMO_TENANT_PASSWORD = "demo-tenant-password"  # local demo only

DEMO = [
    {
        "tenant": "acme",
        "apps": [
            {"name": "support-bot", "environment": "prod", "baseline_model": "anthropic/claude-sonnet-5-5", "budget": "250"},
            {"name": "doc-search", "environment": "dev", "baseline_model": "anthropic/claude-sonnet-5-5", "budget": "50"},
        ],
    },
    {
        "tenant": "globex",
        "apps": [
            {"name": "sales-assistant", "environment": "prod", "baseline_model": "anthropic/claude-opus-5-5", "budget": "500"},
            {"name": "code-review", "environment": "staging", "baseline_model": "anthropic/claude-sonnet-5-5", "budget": "100"},
        ],
    },
]


async def seed() -> list[dict[str, str]]:
    settings = get_settings()
    db = create_database()
    catalog = ModelCatalog.from_yaml(settings.models_yaml)
    await catalog.sync_to_db(db)
    default_policy = load_default_policy(settings.default_policy_json)
    out: list[dict[str, str]] = []
    try:
        for spec in DEMO:
            tenant = await svc.get_tenant_by_name(db, spec["tenant"]) or await svc.create_tenant(db, actor=ACTOR, name=spec["tenant"])
            admin_email = f"admin@{spec['tenant']}.example"
            try:
                await svc.create_admin_user(
                    db, actor=ACTOR, email=admin_email, password=DEMO_TENANT_PASSWORD, role="tenant_admin", tenant_id=tenant.id
                )
            except svc.Conflict:
                pass
            existing = {(a.name, a.environment): a for a in await svc.list_apps(db, tenant.id)}
            for a in spec["apps"]:
                app = existing.get((a["name"], a["environment"])) or await svc.create_app(
                    db,
                    actor=ACTOR,
                    tenant_id=tenant.id,
                    name=a["name"],
                    environment=a["environment"],
                    baseline_model=a["baseline_model"],
                    catalog=catalog,
                    default_policy=default_policy,
                    monthly_budget_usd=Decimal(a["budget"]),
                )
                issued = await svc.create_key(db, actor=ACTOR, tenant_id=tenant.id, app_id=app.id)
                out.append(
                    {
                        "tenant": tenant.name,
                        "tenant_id": str(tenant.id),
                        "tenant_admin": admin_email,
                        "app": app.name,
                        "environment": app.environment,
                        "app_id": str(app.id),
                        "key_id": str(issued.row.id),
                        "api_key": issued.full_key,
                    }
                )
    finally:
        await db.dispose()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--json", action="store_true", help="print machine-readable JSON")
    args = parser.parse_args()
    rows = asyncio.run(seed())
    if args.json:
        json.dump(rows, sys.stdout, indent=2)
        print()
        return
    print("Seeded demo tenants. API keys are shown ONCE — store them now.\n")
    for r in rows:
        print(f"  {r['tenant']:<8} {r['app']:<16} {r['environment']:<8} {r['api_key']}")
    print(f"\nTenant admins: admin@acme.example / admin@globex.example  (password: {DEMO_TENANT_PASSWORD})")


if __name__ == "__main__":
    main()
