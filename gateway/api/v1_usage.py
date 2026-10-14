"""`GET /v1/usage?from=YYYY-MM-DD&to=YYYY-MM-DD` — this app's usage (spec §12).

Requires the `usage:read` scope. Reads `usage_events` through a tenant-scoped RLS
session *and* filters by the key's `app_id`, so a key can only ever see its own
app's ledger. Both bounds are inclusive UTC dates; the default range is the last 30
days. Events arrive about a second after a request, once the batch writer flushes.

Response shape (Tollgate-specific; OpenAI has no equivalent):

    {"object": "usage", "app_id": "...", "from": "...", "to": "...",
     "totals": {requests, input_tokens, output_tokens, actual_cost_usd,
                baseline_cost_usd, saved_usd},
     "data": [{"date": "...", ...same fields}, ...]}
"""

from __future__ import annotations

from datetime import UTC, date, datetime, time, timedelta
from typing import Any

from fastapi import APIRouter, Query, Request
from fastapi.responses import JSONResponse
from sqlalchemy import Date, cast, func, select

from gateway.api.deps import get_services, new_context, tollgate_headers
from gateway.db.models import UsageEvent
from gateway.errors import bad_request

router = APIRouter()
MAX_RANGE_DAYS = 366


def _row(r: Any) -> dict[str, Any]:
    return {
        "requests": int(r.requests or 0),
        "input_tokens": int(r.input_tokens or 0),
        "output_tokens": int(r.output_tokens or 0),
        "actual_cost_usd": float(r.actual or 0),
        "baseline_cost_usd": float(r.baseline or 0),
        "saved_usd": float(r.saved or 0),
    }


@router.get("/v1/usage")
async def get_usage(
    request: Request,
    from_: date | None = Query(None, alias="from"),
    to: date | None = Query(None),
) -> JSONResponse:
    svc = get_services(request)
    ctx = await svc.auth_pipeline.run(new_context(request, svc.settings, "usage", "usage:read"))
    auth = ctx.require_auth()

    end = to or datetime.now(UTC).date()
    start = from_ or end - timedelta(days=29)
    if start > end or (end - start).days > MAX_RANGE_DAYS:
        raise bad_request(f"'from' must be on or before 'to', within {MAX_RANGE_DAYS} days.")
    lo = datetime.combine(start, time.min, UTC)
    hi = datetime.combine(end + timedelta(days=1), time.min, UTC)

    day = cast(func.timezone("UTC", UsageEvent.ts), Date).label("day")
    aggs = (
        func.count().label("requests"),
        func.sum(UsageEvent.input_tokens).label("input_tokens"),
        func.sum(UsageEvent.output_tokens).label("output_tokens"),
        func.sum(UsageEvent.actual_cost_usd).label("actual"),
        func.sum(UsageEvent.baseline_cost_usd).label("baseline"),
        func.sum(UsageEvent.saved_usd).label("saved"),
    )
    scope = (UsageEvent.app_id == auth.app_id, UsageEvent.ts >= lo, UsageEvent.ts < hi)
    async with svc.db.tenant_session(auth.tenant_id) as s:  # RLS: tenant; WHERE: app
        daily = (await s.execute(select(day, *aggs).where(*scope).group_by(day).order_by(day))).all()
        totals = (await s.execute(select(*aggs).where(*scope))).one()

    body = {
        "object": "usage",
        "app_id": str(auth.app_id),
        "from": start.isoformat(),
        "to": end.isoformat(),
        "totals": _row(totals),
        "data": [{"date": r.day.isoformat(), **_row(r)} for r in daily],
    }
    return JSONResponse(body, headers=tollgate_headers(ctx))
