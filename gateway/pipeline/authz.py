"""Stage 2 — AuthZ: is this authenticated key allowed to make *this* request?

Checks in spec §6.2 order. Key revocation and expiry are handled by authn (401).
1. Tenant and app are `active` → otherwise 403.
2. Client IP is within the key's `ip_allowlist`, if one is set → otherwise 403.
3. The endpoint's scope is present (`chat:write`, `usage:read`, ...) → otherwise 403.
4. The app policy is loaded into `ctx.policy`. A pinned model (anything other than
   `auto` or a `tollgate/<tier>` alias) must be in `routing.allowed_models` → 403.
   It must also exist in the catalog → 404.
5. Request features: the estimated input tokens must not exceed
   `limits.max_input_tokens` → 400 `context_length_exceeded`. This runs before any
   expensive stage (spec §15). Tool and JSON-mode capability checks move to routing
   in Phase 3, as hard constraints.
"""

from __future__ import annotations

from ipaddress import ip_address, ip_network

from gateway.catalog import ModelCatalog
from gateway.errors import bad_request, not_found, permission_denied
from gateway.pipeline.context import RequestContext
from gateway.policy.loader import PolicyStore
from gateway.providers.base import estimate_tokens

TIER_ALIASES = frozenset({"tollgate/local", "tollgate/small", "tollgate/medium", "tollgate/large"})


def is_pinned(model: str) -> bool:
    return model != "auto" and model not in TIER_ALIASES


def ip_allowed(client_ip: str | None, allowlist: tuple[str, ...]) -> bool:
    if not allowlist:
        return True
    if client_ip is None:
        return False
    try:
        addr = ip_address(client_ip)
    except ValueError:
        return False
    return any(addr in ip_network(cidr, strict=False) for cidr in allowlist)


class AuthZStage:
    name = "authz"

    def __init__(self, policies: PolicyStore, catalog: ModelCatalog) -> None:
        self._policies = policies
        self._catalog = catalog

    async def process(self, ctx: RequestContext) -> RequestContext:
        auth = ctx.require_auth()

        if not (auth.tenant_active and auth.app_active):
            raise permission_denied("This app or tenant is suspended.")
        if not ip_allowed(ctx.client_ip, auth.ip_allowlist):
            raise permission_denied("Requests from this IP address are not allowed for this key.")
        if ctx.required_scope is not None and ctx.required_scope not in auth.scopes:
            raise permission_denied(f"This API key lacks the '{ctx.required_scope}' scope.")

        ctx.policy = await self._policies.get(auth.tenant_id, auth.app_id, auth.policy_version)

        req = ctx.original
        if req is not None:
            if is_pinned(req.model):
                if req.model not in ctx.policy.routing.allowed_models:
                    raise permission_denied(f"Model '{req.model}' is not allowed by this app's policy.")
                if self._catalog.get(req.model) is None:
                    raise not_found(f"The model '{req.model}' does not exist.", code="model_not_found")
            ctx.estimated_input_tokens = estimate_tokens(req)
            if ctx.estimated_input_tokens > ctx.policy.limits.max_input_tokens:
                raise bad_request(
                    f"Input is ~{ctx.estimated_input_tokens} tokens; this app allows at most "
                    f"{ctx.policy.limits.max_input_tokens}.",
                    code="context_length_exceeded",
                    param="messages",
                )
        return ctx
