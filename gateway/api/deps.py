"""Shared API plumbing: the `Services` container, context construction, response headers.

`Services` is built once in the app lifespan (see `gateway/main.py`) and stored on
`app.state.services`. Routers pull everything from it, which keeps them free of
globals and lets tests inject fakes.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from fastapi import Request
from redis.asyncio import Redis

from gateway.catalog import ModelCatalog
from gateway.db.session import Database
from gateway.pipeline.base import Pipeline
from gateway.pipeline.context import RequestContext
from gateway.policy.loader import PolicyStore
from gateway.policy.schema import AppPolicy
from gateway.providers.registry import ProviderRegistry
from gateway.redaction.engine import RedactionEngine
from gateway.security.admin_auth import AdminAuthProvider
from gateway.settings import Settings


@dataclass
class Services:
    settings: Settings
    db: Database
    redis: Redis
    catalog: ModelCatalog
    registry: ProviderRegistry
    policies: PolicyStore
    default_policy: AppPolicy
    chat_pipeline: Pipeline  # full pipeline for /v1/chat/completions
    auth_pipeline: Pipeline  # authn → authz → ratelimit for /v1/models, /v1/usage
    admin_auth: AdminAuthProvider
    redaction: RedactionEngine | None = None  # shared Presidio engine (None only in some unit tests)


def get_services(request: Request) -> Services:
    return request.app.state.services


def client_ip(request: Request, trust_proxy: bool) -> str | None:
    if trust_proxy and (xff := request.headers.get("x-forwarded-for")):
        return xff.split(",")[0].strip()
    return request.client.host if request.client else None


def bearer_token(request: Request) -> str | None:
    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    return token.strip() if scheme.lower() == "bearer" and token else None


def new_context(
    request: Request, settings: Settings, endpoint: Literal["chat", "models", "usage", "embeddings"], scope: str | None
) -> RequestContext:
    request_id = f"req_{uuid.uuid4().hex}"
    request.state.request_id = request_id  # so error handlers can echo it
    return RequestContext(
        request_id=request_id,
        endpoint=endpoint,
        required_scope=scope,
        client_ip=client_ip(request, settings.trust_proxy_headers),
        api_key=bearer_token(request),
    )


def tollgate_headers(ctx: RequestContext, *, cost_known: bool = True) -> dict[str, str]:
    """The `x-tollgate-*` response headers from spec §11, plus rate-limit headers."""
    headers = {
        "x-tollgate-request-id": ctx.request_id,
        "x-tollgate-cache": ctx.cache_status,
        "x-tollgate-redacted": str(sum(ctx.pii_counts.values())),
        **ctx.rate_limit_headers,
    }
    if ctx.route is not None:
        headers["x-tollgate-model"] = ctx.route.model_id
        headers["x-tollgate-route-reason"] = ctx.route.reason
    if cost_known:
        headers["x-tollgate-cost-usd"] = f"{ctx.cost_usd:.6f}"
        headers["x-tollgate-saved-usd"] = f"{ctx.saved_usd:.6f}"
    else:  # streaming: headers go out before the final usage frame
        headers["x-tollgate-cost-usd"] = "pending"
        headers["x-tollgate-saved-usd"] = "pending"
    return headers
