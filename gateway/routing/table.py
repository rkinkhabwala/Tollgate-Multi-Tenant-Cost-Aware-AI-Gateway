"""Routing table and tier arithmetic (spec §8.2 steps 5–8).

`DEFAULT_TABLE` maps `(task_type, complexity)` to the *minimum* tier that should
handle it. Apps override cells with `policy.routing.table_overrides`. The helpers
then apply, in order:
* the confidence bump: classifier confidence below `min_confidence` → one tier up;
* budget pressure: past the soft budget limit, cap at `soft_limit_max_tier`. This is
  a hook only; budgets arrive in Phase 4;
* the app clamps: `min_tier` ≤ tier ≤ `max_tier`.
"""

from __future__ import annotations

from gateway.policy.schema import TIER_ORDER, RoutingPolicy, Tier

DEFAULT_TABLE: dict[str, dict[str, Tier]] = {
    **{t: {"low": "local", "medium": "local", "high": "small"} for t in ("chitchat", "classification", "extraction")},
    **{t: {"low": "local", "medium": "small", "high": "medium"} for t in ("summarization", "rewrite", "qa_with_context")},
    "code": {"low": "small", "medium": "medium", "high": "large"},
    **{t: {"low": "small", "medium": "medium", "high": "large"} for t in ("reasoning_math", "agentic_tool_use")},
    "creative_long": {"low": "small", "medium": "medium", "high": "medium"},
}


def rank(tier: str) -> int:
    return TIER_ORDER.index(tier)  # type: ignore[arg-type]


def bump(tier: Tier, steps: int = 1) -> Tier:
    return TIER_ORDER[min(len(TIER_ORDER) - 1, rank(tier) + steps)]


def table_tier(task_type: str, complexity: str, routing: RoutingPolicy) -> Tier:
    override = routing.table_overrides.get(task_type, {}).get(complexity)  # type: ignore[call-overload]
    return override or DEFAULT_TABLE.get(task_type, DEFAULT_TABLE["chitchat"])[complexity]


def clamp(tier: Tier, routing: RoutingPolicy) -> tuple[Tier, str | None]:
    """Clamp to [min_tier, max_tier]; returns the tier and a reason tag if clamped."""
    if rank(tier) < rank(routing.min_tier):
        return routing.min_tier, f"clamp:min={routing.min_tier}"
    if rank(tier) > rank(routing.max_tier):
        return routing.max_tier, f"clamp:max={routing.max_tier}"
    return tier, None


def target_tier(
    task_type: str, complexity: str, confidence: float, routing: RoutingPolicy, budget_pressure: bool = False
) -> tuple[Tier, list[str]]:
    """Table → confidence bump → budget cap → clamps. Returns (tier, reason tags)."""
    tags: list[str] = []
    tier = table_tier(task_type, complexity, routing)
    if confidence < routing.min_confidence:
        tier = bump(tier)
        tags.append("bump:low-confidence")
    if budget_pressure and rank(tier) > rank(routing.soft_limit_max_tier):  # Phase 4 wires the signal
        tier = routing.soft_limit_max_tier
        tags.append(f"budget:max={tier}")
    tier, clamp_tag = clamp(tier, routing)
    if clamp_tag:
        tags.append(clamp_tag)
    return tier, tags
