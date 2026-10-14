"""Pydantic v2 schema for app policies (spec §9).

Unknown keys are rejected everywhere (`extra="forbid"`), so a typo in a policy
fails loudly instead of silently falling back to a default. Every section has
defaults equal to `config/default_policy.json`, so `AppPolicy()` is a valid policy.

Phase 1 enforces `routing.allowed_models`, `routing.allow_pinning`, `limits.*` and
`optimize.default_max_tokens`. The other sections are validated and stored now and
enforced by later phases.
"""

from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

Tier = Literal["local", "small", "medium", "large"]
TIER_ORDER: tuple[Tier, ...] = ("local", "small", "medium", "large")
Complexity = Literal["low", "medium", "high"]
TaskType = Literal[
    "chitchat",
    "classification",
    "extraction",
    "summarization",
    "rewrite",
    "qa_with_context",
    "code",
    "reasoning_math",
    "creative_long",
    "agentic_tool_use",
]
RedactionAction = Literal["tokenize", "mask", "block", "allow"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class RoutingPolicy(_Strict):
    mode: Literal["auto"] = "auto"
    allow_pinning: bool = False
    allowed_models: list[str] = Field(
        default_factory=lambda: ["ollama/qwen2.5:7b", "anthropic/claude-haiku-5-5", "anthropic/claude-sonnet-5-5"]
    )
    min_tier: Tier = "local"
    max_tier: Tier = "medium"
    min_confidence: float = Field(0.6, ge=0, le=1)
    max_escalations: int = Field(1, ge=0, le=3)
    soft_limit_max_tier: Tier = "small"
    table_overrides: dict[TaskType, dict[Complexity, Tier]] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _tiers_ordered(self) -> RoutingPolicy:
        if TIER_ORDER.index(self.min_tier) > TIER_ORDER.index(self.max_tier):
            raise ValueError("routing.min_tier must not be above routing.max_tier")
        return self


class CustomPattern(_Strict):
    # ≤ 20 chars keeps every placeholder (<<NAME_123>>) within the streaming
    # re-hydrator's 32-char tail buffer (spec §7.4).
    name: str = Field(pattern=r"^[A-Z][A-Z0-9_]{0,19}$")
    regex: str
    action: RedactionAction = "tokenize"

    @field_validator("regex")
    @classmethod
    def _compiles(cls, v: str) -> str:
        try:
            re.compile(v)
        except re.error as exc:
            raise ValueError(f"invalid regex: {exc}") from exc
        return v


class RedactionPolicy(_Strict):
    enabled: bool = True
    redact_for_local: bool = True
    entities: dict[str, RedactionAction] = Field(
        default_factory=lambda: {
            "SECRET": "block",
            "CREDIT_CARD": "mask",
            "US_SSN": "mask",
            "PERSON": "tokenize",
            "EMAIL_ADDRESS": "tokenize",
            "PHONE_NUMBER": "tokenize",
        }
    )
    custom_patterns: list[CustomPattern] = Field(default_factory=list, max_length=50)
    # Literal terms (e.g. internal codenames) matched case-insensitively as entity
    # DENY_TERM. Spec §7.2 calls for tenant deny-lists; §9's schema had no field.
    deny_terms: list[str] = Field(default_factory=list, max_length=500)
    score_threshold: float = Field(0.5, ge=0, le=1)

    @field_validator("deny_terms")
    @classmethod
    def _terms_nonblank(cls, v: list[str]) -> list[str]:
        if any(not t.strip() or len(t) > 200 for t in v):
            raise ValueError("deny_terms must be non-blank strings of at most 200 chars")
        return v


class CachePolicy(_Strict):
    exact: bool = True
    semantic: bool = True
    semantic_threshold: float = Field(0.95, ge=0, le=1)
    ttl_seconds: int = Field(86400, ge=0)
    scope: Literal["app", "tenant"] = "app"
    skip_if_temperature_above: float = Field(0.7, ge=0, le=2)


class OptimizePolicy(_Strict):
    max_history_turns: int = Field(12, ge=1)
    summarize_history_over_tokens: int = Field(6000, ge=0)
    default_max_tokens: int = Field(1024, ge=1)
    dedupe_context_chunks: bool = True


class LimitsPolicy(_Strict):
    rpm: int = Field(600, ge=1)
    tpm: int = Field(200_000, ge=1)
    max_input_tokens: int = Field(32_000, ge=1)


class LoggingPolicy(_Strict):
    log_prompts: bool = False
    retention_days: int = Field(30, ge=1)


class AppPolicy(_Strict):
    routing: RoutingPolicy = Field(default_factory=RoutingPolicy)
    redaction: RedactionPolicy = Field(default_factory=RedactionPolicy)
    cache: CachePolicy = Field(default_factory=CachePolicy)
    optimize: OptimizePolicy = Field(default_factory=OptimizePolicy)
    limits: LimitsPolicy = Field(default_factory=LimitsPolicy)
    logging: LoggingPolicy = Field(default_factory=LoggingPolicy)
