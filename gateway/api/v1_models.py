"""`GET /v1/models` — the models and aliases this key may use (spec §12).

Lists `auto`, the `tollgate/<tier>` aliases, and every enabled catalog model in the
app policy's `allowed_models`. Needs a valid key but no particular scope.
"""

from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from gateway.api.deps import get_services, new_context, tollgate_headers
from gateway.pipeline.authz import TIER_ALIASES

router = APIRouter()

CREATED = 1_760_000_000  # static: catalog entries have no creation time


@router.get("/v1/models")
async def list_models(request: Request) -> JSONResponse:
    svc = get_services(request)
    ctx = await svc.auth_pipeline.run(new_context(request, svc.settings, "models", None))
    policy = ctx.require_policy()

    data = [{"id": "auto", "object": "model", "created": CREATED, "owned_by": "tollgate"}]
    data += [{"id": alias, "object": "model", "created": CREATED, "owned_by": "tollgate"} for alias in sorted(TIER_ALIASES)]
    for model_id in policy.routing.allowed_models:
        if (m := svc.catalog.get(model_id)) is not None:
            data.append({"id": m.id, "object": "model", "created": CREATED, "owned_by": m.provider})
    return JSONResponse({"object": "list", "data": data}, headers=tollgate_headers(ctx))
