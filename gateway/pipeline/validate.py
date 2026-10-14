"""Stage 11 — Validate: catch bad answers and escalate one tier up (spec §8.4).

It runs after invoke and before rehydrate, on the upstream (still redacted) response.
It handles **non-streaming responses only**: streamed tokens can't be taken back,
so streams only get availability fallback.

A response fails validation when:
* `empty`: there is no text and no tool call;
* `refusal`: the text is short (< 300 chars) and matches a refusal phrase, or the
  provider reports `content_filter`;
* `truncated`: `finish_reason == "length"` while the output is under the max the
  client requested. The model stopped early at its own limit, not the client's;
* `invalid_json`: JSON mode was requested and the text doesn't `json.loads`, or
  fails the client's JSON schema (checked with `jsonschema`, against the
  *original* schema);
* `invalid_tool_args`: a tool call's arguments aren't a JSON object, the tool name
  wasn't offered, or the arguments fail that tool's parameter schema.

On failure, if `escalations < policy.routing.max_escalations` and a higher tier has an
eligible model:
* the router re-plans with `min_tier = current tier + 1`, climbing further if that
  tier only offers the same model (local-only: local = small = 1.5b);
* invoke runs again with attempt kind `escalation`;
* `ctx.escalations` is incremented and `;escalate:<why>><tier>` appended to the
  route reason.

Both attempts are billed by meter. If nothing higher is available, the original
response is kept.
"""

from __future__ import annotations

import json
import re
from typing import Any

import jsonschema

from gateway.errors import GatewayError
from gateway.pipeline.context import RequestContext, RouteDecision
from gateway.pipeline.invoke import InvokeStage
from gateway.catalog import CatalogModel
from gateway.policy.schema import TIER_ORDER, AppPolicy, Tier
from gateway.providers.base import ChatRequest, ChatResponse
from gateway.routing.router import Router
from gateway.routing.table import bump

_REFUSAL = re.compile(
    r"^\s*(i'?m sorry,? but|i (cannot|can'?t|won'?t|am unable to|'m unable to) (help|assist|comply|provide|do that)|as an ai\b|i must decline)",
    re.IGNORECASE,
)


def failure_reason(req: ChatRequest, resp: ChatResponse) -> str | None:
    """Why `resp` fails validation for `req`, or None if it's acceptable."""
    if not resp.choices:
        return "empty"
    choice = resp.choices[0]
    text = choice.message.text() if choice.message.content is not None else ""
    calls = choice.message.tool_calls or []
    if not text.strip() and not calls:
        return "empty"
    if choice.finish_reason == "content_filter" or (len(text) < 300 and not calls and _REFUSAL.search(text)):
        return "refusal"
    limit = req.output_limit()
    if choice.finish_reason == "length" and limit is not None and resp.usage.completion_tokens < limit:
        return "truncated"
    rf = req.response_format or {}
    if rf.get("type") in ("json_object", "json_schema") and not calls:
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            return "invalid_json"
        schema = (rf.get("json_schema") or {}).get("schema")
        if schema is not None:
            try:
                jsonschema.validate(value, schema)
            except (jsonschema.ValidationError, jsonschema.SchemaError):
                return "invalid_json"
    if calls:
        offered = {t.get("function", {}).get("name"): t.get("function", {}).get("parameters") for t in req.tools or []}
        for call in calls:
            fn = call.get("function", {})
            if fn.get("name") not in offered:
                return "invalid_tool_args"
            try:
                args: Any = json.loads(fn.get("arguments") or "")
            except json.JSONDecodeError:
                return "invalid_tool_args"
            if not isinstance(args, dict):
                return "invalid_tool_args"
            params = offered[fn.get("name")]
            if params:
                try:
                    jsonschema.validate(args, params)
                except (jsonschema.ValidationError, jsonschema.SchemaError):
                    return "invalid_tool_args"
    return None


class ValidateStage:
    name = "validate"

    def __init__(self, invoke: InvokeStage, router: Router) -> None:
        self._invoke = invoke
        self._router = router

    def _next_plan(self, ctx: RequestContext, policy: AppPolicy) -> tuple[Tier, list[CatalogModel]] | None:
        """Climb tiers until one offers a model *different* from the one that just failed.

        Profiles can repeat a model across tiers (local-only: local = small = qwen2.5:1.5b),
        and re-asking the same model is pointless.
        """
        assert ctx.route is not None and ctx.working is not None
        tier: Tier = ctx.route.tier  # type: ignore[assignment]
        while tier != TIER_ORDER[-1]:
            tier = bump(tier)
            try:
                plan = self._router.plan(ctx.working, policy, ctx.classification, min_tier_override=tier)
            except GatewayError:
                return None
            if TIER_ORDER.index(plan.tier) <= TIER_ORDER.index(ctx.route.tier):  # type: ignore[arg-type]
                return None  # capped by max_tier
            candidates = [m for m in plan.candidates if m.id != ctx.route.model_id]
            if candidates:
                # Report the tier the new model actually comes from (≥ plan.tier).
                served_tier = next(
                    (t for t in TIER_ORDER[TIER_ORDER.index(plan.tier) :] if candidates[0].id in plan.eligible_by_tier.get(t, [])),
                    plan.tier,
                )
                return served_tier, candidates  # type: ignore[return-value]
        return None

    async def process(self, ctx: RequestContext) -> RequestContext:
        if ctx.stream is not None or ctx.response is None or ctx.working is None or ctx.route is None:
            return ctx
        policy = ctx.require_policy()
        while ctx.escalations < policy.routing.max_escalations:
            why = failure_reason(ctx.working, ctx.response)
            if why is None or ctx.route.tier == TIER_ORDER[-1]:
                return ctx
            escalation = self._next_plan(ctx, policy)
            if escalation is None:
                return ctx  # max_tier ceiling, or no different model above: keep the answer we have
            tier, candidates = escalation
            previous_response, previous_route = ctx.response, ctx.route
            ctx.candidates = candidates
            ctx.route = RouteDecision(candidates[0].id, tier, f"{ctx.route.reason};escalate:{why}>{tier}")
            ctx.escalations += 1
            try:
                await self._invoke.process(ctx, kind="escalation")
            except GatewayError:
                # Escalation target failed: return the first answer (both attempts are still billed).
                ctx.response, ctx.status_code = previous_response, 200
                ctx.route = RouteDecision(previous_route.model_id, previous_route.tier, f"{previous_route.reason};escalate-failed:{why}")
                return ctx
        return ctx
