"""Admin playgrounds (spec §12): `POST /admin/redaction/test` and `POST /admin/routing/test`.

`POST /admin/redaction/test` is a dry run of text against an app's *current*
redaction policy, using the same engine, recognizers and actions as live traffic. It
returns what was found and the redacted text:

    {"blocked": false,
     "entities": [{"type": "EMAIL_ADDRESS", "start": 9, "end": 26, "score": 0.9,
                   "action": "tokenize", "placeholder": "<<EMAIL_1>>"}],
     "counts": {"EMAIL_ADDRESS": 1},
     "redacted_text": "Write to <<EMAIL_1>> ..."}

Original values are never echoed back as separate fields; spans index into the
caller's own input. Blocked spans are shown masked (`<<SECRET>>`) in `redacted_text`.
Nothing about the text is logged.
"""

from __future__ import annotations

import uuid
from collections import Counter
from typing import Any

from fastapi import APIRouter, Depends, Request
from pydantic import BaseModel, Field

from gateway.api.admin.deps import ADMIN_ROLES, require_roles, service_error, tenant_for_app
from gateway.api.deps import get_services
from gateway.errors import GatewayError
from gateway.pipeline.context import RequestContext
from gateway.pipeline.redact import RedactStage
from gateway.policy.schema import AppPolicy
from gateway.providers.base import ChatMessage, ChatRequest
from gateway.redaction.engine import apply_actions, effective_actions
from gateway.redaction.vault import Vault
from gateway.security.admin_auth import AdminPrincipal
from gateway.services import admin as svc_admin

router = APIRouter(prefix="/admin", tags=["admin:tools"])


class RedactionTestIn(BaseModel):
    app_id: uuid.UUID
    text: str = Field(min_length=1, max_length=200_000)


class EntityOut(BaseModel):
    type: str
    start: int
    end: int
    score: float
    action: str
    placeholder: str | None


class RedactionTestOut(BaseModel):
    blocked: bool
    entities: list[EntityOut]
    counts: dict[str, int]
    redacted_text: str


@router.post("/redaction/test")
async def redaction_test(
    body: RedactionTestIn, request: Request, principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES))
) -> RedactionTestOut:
    s = get_services(request)
    if s.redaction is None:
        raise GatewayError(503, "redaction_unavailable", "The redaction engine is not loaded.", type_="server_error")
    try:
        tenant_id = await tenant_for_app(request, principal, body.app_id)
        row = await svc_admin.get_policy(s.db, tenant_id=tenant_id, app_id=body.app_id)
    except Exception as exc:
        raise service_error(exc) from exc
    policy = AppPolicy.model_validate(row.policy).redaction

    (findings,) = await s.redaction.analyze_many([body.text], policy)
    actions = effective_actions(policy)
    _, applied = apply_actions(body.text, findings, actions, Vault())
    # Preview text: blocked spans are shown masked rather than left in clear.
    preview_actions: dict[str, Any] = {k: ("mask" if v == "block" else v) for k, v in actions.items()}
    redacted_text, _ = apply_actions(body.text, findings, preview_actions, Vault())

    return RedactionTestOut(
        blocked=any(a.action == "block" for a in applied),
        entities=[EntityOut(type=a.entity_type, start=a.start, end=a.end, score=round(a.score, 3), action=a.action, placeholder=a.placeholder) for a in applied],
        counts=dict(Counter(a.entity_type for a in applied)),
        redacted_text=redacted_text,
    )


# --- routing playground --------------------------------------------------------------------
class RoutingTestIn(BaseModel):
    app_id: uuid.UUID
    messages: list[ChatMessage] = Field(min_length=1)
    model: str = "auto"
    tools: list[dict[str, Any]] | None = None
    response_format: dict[str, Any] | None = None
    max_tokens: int | None = Field(None, ge=1)


class RoutingTestOut(BaseModel):
    redaction: dict[str, int]
    classification: dict[str, Any] | None
    target_tier: str
    tier: str
    model: str
    fallback_order: list[str]
    route_reason: str
    eligible_by_tier: dict[str, list[str]]
    rejected: dict[str, str]


@router.post("/routing/test")
async def routing_test(
    body: RoutingTestIn, request: Request, principal: AdminPrincipal = Depends(require_roles(*ADMIN_ROLES))
) -> RoutingTestOut:
    """Dry-run classification + routing for a sample request against an app's policy.

    The text is redacted first (exactly as live traffic), and the classifier sees only
    the redacted last user turn. The response holds labels, numbers and model ids,
    never prompt text. No model is called and nothing is metered.
    """
    s = get_services(request)
    if s.router is None or s.classifier is None or s.redaction is None:
        raise GatewayError(503, "routing_unavailable", "The router is not loaded.", type_="server_error")
    try:
        tenant_id = await tenant_for_app(request, principal, body.app_id)
        row = await svc_admin.get_policy(s.db, tenant_id=tenant_id, app_id=body.app_id)
    except Exception as exc:
        raise service_error(exc) from exc
    policy = AppPolicy.model_validate(row.policy)
    req = ChatRequest(messages=body.messages, model=body.model, tools=body.tools, response_format=body.response_format, max_tokens=body.max_tokens)

    working, counts = req, {}
    if policy.redaction.enabled:
        ctx = RequestContext(request_id="routing-test", endpoint="chat", required_scope=None, client_ip=None)
        ctx.original, ctx.policy = req, policy
        try:
            ctx = await RedactStage(s.redaction).process(ctx)
        except GatewayError as exc:
            if exc.code != "sensitive_data_blocked":
                raise
            counts = {"blocked": 1}
        else:
            working, counts = ctx.working or req, ctx.pii_counts

    classification = None
    if req.model == "auto" or not (policy.routing.allow_pinning or req.model.startswith("tollgate/")):
        classification = await s.classifier.classify(working, policy.routing.min_confidence)
    plan = s.router.plan(working, policy, classification)
    return RoutingTestOut(
        redaction=counts,
        classification=None
        if classification is None
        else {"task_type": classification.task_type, "complexity": classification.complexity, "confidence": classification.confidence, **classification.detail},
        target_tier=plan.target,
        tier=plan.tier,
        model=plan.candidates[0].id,
        fallback_order=[m.id for m in plan.candidates],
        route_reason=plan.reason,
        eligible_by_tier=plan.eligible_by_tier,
        rejected=plan.rejected,
    )
