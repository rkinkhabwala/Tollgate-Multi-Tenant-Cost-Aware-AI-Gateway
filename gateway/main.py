"""FastAPI app factory: wires settings, stores, pipelines, routers and workers.

The pipeline order follows spec §3.1. Phase 1 stages are implemented. Later-phase
stages are pass-through placeholders, listed here so the final shape is visible:

    authn → authz → ratelimit → budget* → redact* → cache* → optimize*
          → classify* → route (static in Phase 1) → invoke
    post: meter          (validate / cache write / rehydrate join in Phases 2-4)
    (* = pass-through placeholder in Phase 1)

Cross-cutting HTTP concerns are handled here as well:
* a request body size limit (2 MB, spec §15), enforced while streaming the body in;
* OpenAI-shaped error bodies for every error path, including FastAPI's own
  validation errors, which normally echo the input back. We never do: the input
  may contain prompts.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from redis.asyncio import Redis
from sqlalchemy import text
from starlette.exceptions import HTTPException as StarletteHTTPException
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from gateway.api import v1_chat, v1_models, v1_usage
from gateway.api.admin import apps as admin_apps
from gateway.api.admin import auth as admin_auth
from gateway.api.admin import keys as admin_keys
from gateway.api.admin import tenants as admin_tenants
from gateway.api.admin.deps import admin_rate_limit
from gateway.api.deps import Services
from gateway.catalog import ModelCatalog
from gateway.db.session import Database
from gateway.errors import GatewayError
from gateway.observability.logging import configure_logging
from gateway.pipeline.authn import AuthNStage, DbKeyResolver
from gateway.pipeline.authz import AuthZStage
from gateway.pipeline.base import Pipeline
from gateway.pipeline.budget import BudgetStage
from gateway.pipeline.cache import CacheLookupStage
from gateway.pipeline.classify import ClassifyStage
from gateway.pipeline.invoke import InvokeStage
from gateway.pipeline.meter import MeterStage
from gateway.pipeline.optimize import OptimizeStage
from gateway.pipeline.ratelimit import RateLimiter, RateLimitStage
from gateway.pipeline.redact import RedactStage
from gateway.pipeline.route import StaticRouteStage
from gateway.policy.loader import PolicyStore, load_default_policy
from gateway.providers.ollama import OllamaAdapter
from gateway.providers.registry import ProviderRegistry
from gateway.security.admin_auth import LocalAdminAuth
from gateway.services.admin import ensure_bootstrap_admin
from gateway.settings import Settings, get_settings
from gateway.workers.key_touch import KeyToucher
from gateway.workers.rollups import RollupWorker
from gateway.workers.usage_writer import UsageWriter

log = logging.getLogger("gateway")


class BodySizeLimit:
    """Reject request bodies over `max_bytes` (413) without buffering them first."""

    def __init__(self, app: ASGIApp, max_bytes: int) -> None:
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        declared = dict(scope.get("headers", [])).get(b"content-length")
        if declared is not None and declared.isdigit() and int(declared) > self.max_bytes:
            await _too_large(send)
            return
        seen = 0

        async def limited_receive() -> Message:
            nonlocal seen
            message = await receive()
            if message["type"] == "http.request":
                seen += len(message.get("body", b""))
                if seen > self.max_bytes:
                    raise _BodyTooLarge
            return message

        try:
            await self.app(scope, limited_receive, send)
        except _BodyTooLarge:
            await _too_large(send)


class _BodyTooLarge(Exception):
    pass


async def _too_large(send: Send) -> None:
    body = GatewayError(413, "request_too_large", "Request body exceeds the size limit.").body()
    resp = JSONResponse(body, status_code=413)
    await resp({"type": "http"}, _noop_receive, send)


async def _noop_receive() -> Message:  # pragma: no cover
    return {"type": "http.disconnect"}


def build_services(settings: Settings, db: Database, redis: Redis, registry: ProviderRegistry, catalog: ModelCatalog) -> tuple[Services, KeyToucher]:
    policies = PolicyStore(db, redis)
    limiter = RateLimiter(redis)
    toucher = KeyToucher(db, settings.key_touch_interval_s)
    authn = AuthNStage(redis, DbKeyResolver(db), settings.auth_cache_ttl_s, on_key_used=toucher.touch)
    authz = AuthZStage(policies, catalog)
    ratelimit = RateLimitStage(limiter)

    chat = Pipeline(
        stages=[
            authn,
            authz,
            ratelimit,
            BudgetStage(),
            RedactStage(),
            CacheLookupStage(),
            OptimizeStage(),
            ClassifyStage(),
            StaticRouteStage(catalog, registry, settings.default_model),
            InvokeStage(catalog, registry),
        ],
        post_stages=[MeterStage(redis, catalog, limiter, settings.usage_stream)],
    )
    services = Services(
        settings=settings,
        db=db,
        redis=redis,
        catalog=catalog,
        registry=registry,
        policies=policies,
        default_policy=load_default_policy(settings.default_policy_json),
        chat_pipeline=chat,
        auth_pipeline=Pipeline([authn, authz, ratelimit]),
        admin_auth=LocalAdminAuth(db, settings),
    )
    return services, toucher


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    settings: Settings = app.state.settings
    configure_logging(settings.log_level)
    db = Database(settings.database_url_app, settings.database_url_system)
    redis = Redis.from_url(settings.redis_url, decode_responses=False)
    registry = ProviderRegistry({"ollama": OllamaAdapter(settings.ollama_base_url, settings.ollama_timeout_s)})
    catalog = ModelCatalog.from_yaml(settings.models_yaml)
    await catalog.sync_to_db(db)
    if settings.bootstrap_admin_email and settings.bootstrap_admin_password:
        if await ensure_bootstrap_admin(db, settings.bootstrap_admin_email, settings.bootstrap_admin_password.get_secret_value()):
            log.info("created bootstrap platform admin", extra={"email": settings.bootstrap_admin_email})

    services, toucher = build_services(settings, db, redis, registry, catalog)
    app.state.services = services

    tasks: list[asyncio.Task[None]] = []
    if settings.run_workers:
        writer = UsageWriter(
            redis,
            db,
            stream=settings.usage_stream,
            group=settings.usage_group,
            batch_size=settings.usage_batch_size,
            flush_interval_s=settings.usage_flush_interval_s,
        )
        tasks = [
            asyncio.create_task(writer.run(), name="usage-writer"),
            asyncio.create_task(toucher.run(), name="key-toucher"),
            asyncio.create_task(RollupWorker(db, settings.rollup_interval_s).run(), name="rollups"),
        ]
    log.info("tollgate ready", extra={"default_model": settings.default_model, "workers": len(tasks)})
    try:
        yield
    finally:
        for t in tasks:
            t.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        try:
            await toucher.flush()
        except Exception:
            log.exception("final last_used_at flush failed")
        await registry.aclose()
        await redis.aclose()
        await db.dispose()


def _error_response(request: Request, err: GatewayError) -> JSONResponse:
    headers = dict(err.headers)
    if rid := getattr(request.state, "request_id", None):
        headers["x-tollgate-request-id"] = rid
    return JSONResponse(err.body(), status_code=err.status, headers=headers)


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    app = FastAPI(title="Tollgate", version="0.1.0", lifespan=lifespan)
    app.state.settings = settings

    @app.exception_handler(GatewayError)
    async def _gateway_error(request: Request, exc: GatewayError) -> JSONResponse:
        return _error_response(request, exc)

    @app.exception_handler(RequestValidationError)
    async def _validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        parts = [f"{'.'.join(str(p) for p in e['loc'])}: {e['msg']}" for e in exc.errors()[:5]]  # no 'input'
        return _error_response(request, GatewayError(400, "invalid_request", "Invalid request: " + "; ".join(parts)))

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(request: Request, exc: StarletteHTTPException) -> JSONResponse:
        code = {404: "not_found", 405: "method_not_allowed"}.get(exc.status_code, "http_error")
        return _error_response(request, GatewayError(exc.status_code, code, str(exc.detail)))

    @app.exception_handler(Exception)
    async def _unhandled(request: Request, exc: Exception) -> JSONResponse:
        log.exception("unhandled error", extra={"path": request.url.path})
        return _error_response(request, GatewayError(500, "internal_error", "Internal server error.", type_="server_error"))

    @app.get("/healthz", include_in_schema=False)
    async def healthz() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/readyz", include_in_schema=False)
    async def readyz(request: Request) -> JSONResponse:
        s: Services = request.app.state.services
        checks: dict[str, str] = {}
        try:
            async with s.db.system_session() as session:
                await session.execute(text("SELECT 1"))
            checks["postgres"] = "ok"
        except Exception:
            checks["postgres"] = "down"
        try:
            await s.redis.ping()
            checks["redis"] = "ok"
        except Exception:
            checks["redis"] = "down"
        ok = all(v == "ok" for v in checks.values())
        return JSONResponse({"status": "ok" if ok else "degraded", **checks}, status_code=200 if ok else 503)

    app.include_router(v1_chat.router)
    app.include_router(v1_models.router)
    app.include_router(v1_usage.router)
    for r in (admin_auth.router, admin_tenants.router, admin_apps.router, admin_keys.router):
        app.include_router(r, dependencies=[Depends(admin_rate_limit)])

    app.add_middleware(BodySizeLimit, max_bytes=settings.max_body_bytes)
    return app


app = create_app()
