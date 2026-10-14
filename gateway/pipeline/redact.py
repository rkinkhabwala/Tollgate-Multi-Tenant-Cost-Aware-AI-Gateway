"""Stage 5 — Redact: replace PII and secrets with placeholders before anything leaves the gateway (spec §7).

What is scanned (§7.1) is every text the provider will see:
* `system`/`developer` prompts and *all* history turns, not just the last user turn;
* text parts of multi-part content;
* tool results (`role: tool`);
* tool-call `arguments` in assistant history. These are parsed as JSON and each
  string leaf is scanned, so the result is still valid JSON.

All texts are analysed in one thread-pool hop (`RedactionEngine.analyze_many`).
Actions are then applied in message order, so placeholder numbering follows first
appearance across the whole conversation.

Outcome:
* any `block` finding → 400 `sensitive_data_blocked`; the message names entity
  *types* only;
* `ctx.working` is the redacted copy, `ctx.vault` holds the placeholder → original
  mapping (memory only), and `ctx.pii_counts` holds counts by type, which feed the
  usage event and `x-tollgate-redacted`;
* if anything was tokenized, the §7.4 placeholder instruction is prepended as a
  system message.

`redact_for_local: false` (§7.6): redaction still runs, so `block` is still enforced
and routing (Phase 3) can classify the redacted text. The unredacted copy is kept in
`ctx.unredacted`. If the request is then routed to the local tier,
`use_unredacted_for_local()` (called by invoke) sends that copy instead, since the
data never leaves the host.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Callable
from typing import Any

from gateway.errors import GatewayError
from gateway.pipeline.context import RequestContext
from gateway.providers.base import ChatMessage, ChatRequest
from gateway.redaction.engine import Applied, RedactionEngine, apply_actions, effective_actions
from gateway.redaction.vault import Vault

PLACEHOLDER_INSTRUCTION = (
    "Text like <<TYPE_N>> is a placeholder for redacted data. "
    "Keep placeholders exactly as written when you need to refer to them."
)


class _Slots:
    """Collects every scannable string in a request together with a writer for it."""

    def __init__(self) -> None:
        self.texts: list[str] = []
        self.writers: list[Callable[[str], None]] = []
        self.finalizers: list[Callable[[], None]] = []

    def add(self, text: str, writer: Callable[[str], None]) -> None:
        self.texts.append(text)
        self.writers.append(writer)

    def add_json_leaves(self, obj: Any) -> None:
        if isinstance(obj, dict):
            for k, v in obj.items():
                if isinstance(v, str):
                    self.add(v, lambda new, d=obj, key=k: d.__setitem__(key, new))
                else:
                    self.add_json_leaves(v)
        elif isinstance(obj, list):
            for i, v in enumerate(obj):
                if isinstance(v, str):
                    self.add(v, lambda new, lst=obj, idx=i: lst.__setitem__(idx, new))
                else:
                    self.add_json_leaves(v)

    @classmethod
    def of(cls, req: ChatRequest) -> _Slots:
        slots = cls()
        for msg in req.messages:
            if isinstance(msg.content, str):
                slots.add(msg.content, lambda new, m=msg: setattr(m, "content", new))
            elif isinstance(msg.content, list):
                for part in msg.content:
                    if part.get("type") == "text" and isinstance(part.get("text"), str):
                        slots.add(part["text"], lambda new, p=part: p.__setitem__("text", new))
            for tc in msg.tool_calls or []:
                fn = tc.get("function")
                if not isinstance(fn, dict) or not isinstance(fn.get("arguments"), str) or not fn["arguments"]:
                    continue
                try:
                    parsed = json.loads(fn["arguments"])
                except json.JSONDecodeError:
                    slots.add(fn["arguments"], lambda new, f=fn: f.__setitem__("arguments", new))
                    continue
                slots.add_json_leaves(parsed)
                slots.finalizers.append(lambda f=fn, obj=parsed: f.__setitem__("arguments", json.dumps(obj, ensure_ascii=False)))
        return slots


def sensitive_data_blocked(types: set[str]) -> GatewayError:
    return GatewayError(
        400,
        "sensitive_data_blocked",
        f"The request contains data this app's policy blocks: {', '.join(sorted(types))}.",
        param="messages",
    )


class RedactStage:
    name = "redact"

    def __init__(self, engine: RedactionEngine) -> None:
        self._engine = engine

    async def redact_request(self, req: ChatRequest, ctx: RequestContext) -> tuple[ChatRequest, Vault, list[Applied]]:
        policy = ctx.require_policy().redaction
        working = req.model_copy(deep=True)
        slots = _Slots.of(working)
        findings = await self._engine.analyze_many(slots.texts, policy)
        actions = effective_actions(policy)
        vault = Vault()
        applied: list[Applied] = []
        for text, writer, found in zip(slots.texts, slots.writers, findings, strict=True):
            if not found:
                continue
            new_text, done = apply_actions(text, found, actions, vault)
            applied.extend(done)
            if new_text != text:
                writer(new_text)
        for finalize in slots.finalizers:
            finalize()
        return working, vault, applied

    async def process(self, ctx: RequestContext) -> RequestContext:
        policy = ctx.require_policy()
        if ctx.original is None or not policy.redaction.enabled:
            return ctx
        working, vault, applied = await self.redact_request(ctx.original, ctx)

        blocked = {a.entity_type for a in applied if a.action == "block"}
        if blocked:
            raise sensitive_data_blocked(blocked)

        if vault:
            working.messages.insert(0, ChatMessage(role="system", content=PLACEHOLDER_INSTRUCTION))
        ctx.working = working
        ctx.vault = vault
        ctx.pii_counts = dict(Counter(a.entity_type for a in applied if a.action in ("tokenize", "mask")))
        if not policy.redaction.redact_for_local:
            ctx.unredacted = ctx.original.model_copy(deep=True)
        return ctx


def use_unredacted_for_local(ctx: RequestContext) -> None:
    """§7.6: when routed to the local tier and `redact_for_local` is off, send the original."""
    if ctx.unredacted is not None and ctx.route is not None and ctx.route.tier == "local":
        ctx.working = ctx.unredacted
        ctx.vault = Vault()
        ctx.pii_counts = {}
    ctx.unredacted = None  # drop the extra copy either way
