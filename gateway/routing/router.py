"""Router: from classification + policy to an ordered list of candidate models (spec §8.2).

`Router.plan()` is a pure decision over the catalog. It makes no network calls; it
only consults the circuit breaker's view of which models are currently available.
In order:

1. **Pinning:** a pinned model (not `auto`, not a `tollgate/<tier>` alias) is honored
   when `allow_pinning` is on and it is eligible. Otherwise it is treated as `auto`.
2. **Hard constraints** decide eligibility per model before anything else:
   - the provider must be configured;
   - the circuit must not be open;
   - tools require `supports_tools`, and JSON mode `supports_json`;
   - input tokens (counted for that model) + output limit ≤ context window.
3. **Target tier:**
   - an alias's tier; or
   - the routing table for the classification, with the confidence bump, budget
     cap and min/max clamps (see `routing/table.py`).
4. **Tier selection:**
   - the lowest tier ≥ target (up to `max_tier`) that has an eligible model;
   - if none, the nearest eligible tier below (reason `downgrade:no-eligible-<tier>`).
     That covers a `hosted` app without API keys, which lands on its local tier;
   - if nothing is eligible at all → 503.
5. **Candidates:** the eligible models of the chosen tier in profile order, then those
   of each higher tier up to `max_tier`. Invoke walks this list for availability
   fallback; validate re-plans one tier up for quality escalation.

The route reason, e.g. `classifier:code/medium@0.71;bump:low-confidence;clamp:max=medium;>medium`
(ASCII only: it is sent as an HTTP header),
holds labels and numbers only, never prompt text.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from gateway.catalog import CatalogModel, ModelCatalog
from gateway.errors import upstream_error
from gateway.pipeline.authz import TIER_ALIASES, is_pinned
from gateway.pipeline.context import Classification
from gateway.policy.schema import TIER_ORDER, AppPolicy, Tier
from gateway.providers.base import ChatRequest
from gateway.providers.circuit_breaker import CircuitBreaker
from gateway.providers.registry import ProviderRegistry
from gateway.providers.tokens import TokenCounter
from gateway.routing.table import clamp, rank, target_tier


@dataclass(frozen=True)
class Needs:
    tools: bool
    json: bool
    output_tokens: int

    @classmethod
    def of(cls, req: ChatRequest, default_max_tokens: int) -> Needs:
        rf = (req.response_format or {}).get("type")
        return cls(
            tools=bool(req.tools) and req.tool_choice != "none",
            json=rf in ("json_object", "json_schema"),
            output_tokens=req.output_limit() or default_max_tokens,
        )


@dataclass
class RoutePlan:
    tier: Tier
    candidates: list[CatalogModel]
    reason: str
    target: Tier
    eligible_by_tier: dict[str, list[str]] = field(default_factory=dict)
    rejected: dict[str, str] = field(default_factory=dict)  # model → why ineligible


class Router:
    def __init__(self, catalog: ModelCatalog, registry: ProviderRegistry, breaker: CircuitBreaker, counter: TokenCounter) -> None:
        self.catalog = catalog
        self.registry = registry
        self.breaker = breaker
        self.counter = counter

    # --- eligibility -------------------------------------------------------------
    def why_ineligible(self, model: CatalogModel, req: ChatRequest, needs: Needs) -> str | None:
        if model.provider not in self.registry:
            return "provider-not-configured"
        if not self.breaker.available(model.id):
            return "circuit-open"
        if needs.tools and not model.supports_tools:
            return "no-tools"
        if needs.json and not model.supports_json:
            return "no-json"
        if self.counter.count(model, req) + needs.output_tokens > model.context_window:
            return "context-window"
        return None

    def _eligible_by_tier(self, policy: AppPolicy, req: ChatRequest, needs: Needs) -> tuple[dict[str, list[CatalogModel]], dict[str, str]]:
        profile = self.catalog.profile(policy.routing.profile)
        out: dict[str, list[CatalogModel]] = {}
        rejected: dict[str, str] = {}
        for tier in TIER_ORDER:
            models = []
            for mid in profile.get(tier, []):
                m = self.catalog.get(mid)
                if m is None:
                    rejected[mid] = "not-in-catalog"
                    continue
                why = self.why_ineligible(m, req, needs)
                if why:
                    rejected[mid] = why
                else:
                    models.append(m)
            out[tier] = models
        return out, rejected

    # --- planning ----------------------------------------------------------------
    def plan(
        self,
        req: ChatRequest,
        policy: AppPolicy,
        classification: Classification | None,
        *,
        min_tier_override: Tier | None = None,
        budget_pressure: bool = False,
    ) -> RoutePlan:
        routing = policy.routing
        needs = Needs.of(req, policy.optimize.default_max_tokens)
        by_tier, rejected = self._eligible_by_tier(policy, req, needs)
        tags: list[str] = []

        # 1. pinning
        if is_pinned(req.model):
            pinned = self.catalog.get(req.model)
            if routing.allow_pinning and pinned is not None:
                why = self.why_ineligible(pinned, req, needs)
                if why is None:
                    peers = [m for m in by_tier.get(pinned.tier, []) if m.id != pinned.id]
                    return RoutePlan(pinned.tier, [pinned, *peers], "pinned", pinned.tier, _ids(by_tier), rejected)  # type: ignore[arg-type]
                tags.append(f"pin-ignored:{why}")
            else:
                tags.append("pin-ignored:pinning-disabled")

        # 2–3. target tier
        if req.model in TIER_ALIASES:
            target: Tier = req.model.split("/", 1)[1]  # type: ignore[assignment]
            tags.insert(0, f"alias:{req.model}")
            target, clamp_tag = clamp(target, routing)
            if clamp_tag:
                tags.append(clamp_tag)
        elif classification is not None:
            tags.insert(0, f"classifier:{classification.task_type}/{classification.complexity}@{classification.confidence:.2f}")
            target, more = target_tier(classification.task_type, classification.complexity, classification.confidence, routing, budget_pressure)
            tags += more
        else:
            target, more = routing.min_tier, ["unclassified"]
            tags += more
        if min_tier_override is not None and rank(min_tier_override) > rank(target):
            target = min_tier_override  # escalation: at least this tier

        # 4. tier selection
        max_rank = rank(routing.max_tier)
        up = [t for t in TIER_ORDER if rank(target) <= rank(t) <= max(max_rank, rank(target)) and by_tier.get(t)]
        if up:
            tier: Tier = up[0]  # type: ignore[assignment]
            if tier != target:
                tags.append(f"up:no-eligible-{target}")
        else:
            down = [t for t in reversed(TIER_ORDER) if rank(t) < rank(target) and by_tier.get(t)]
            if not down:
                raise upstream_error(503, "No eligible model is available for this request.")
            tier = down[0]  # type: ignore[assignment]
            tags.append(f"downgrade:no-eligible-{target}")

        # 5. candidates: chosen tier, then higher tiers within max_tier
        candidates: list[CatalogModel] = []
        for t in TIER_ORDER:
            if rank(t) >= rank(tier) and (rank(t) <= max_rank or t == tier):
                for m in by_tier.get(t, []):
                    if m not in candidates:
                        candidates.append(m)
        tags.append(f">{tier}")
        return RoutePlan(tier, candidates, ";".join(tags), target, _ids(by_tier), rejected)


def _ids(by_tier: dict[str, list[CatalogModel]]) -> dict[str, list[str]]:
    return {t: [m.id for m in ms] for t, ms in by_tier.items()}
