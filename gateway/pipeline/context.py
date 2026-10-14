"""`RequestContext`: the single object every pipeline stage reads and writes (spec §13).

A context is created per request by the API layer and threaded through the stages.
It carries sensitive material (the raw API key, the original request with raw
prompts, and, from Phase 2, the redaction vault). Those fields are `repr=False`, so
that printing or logging a context by accident cannot leak them.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import datetime
from decimal import Decimal
from typing import Literal

from gateway.policy.schema import AppPolicy
from gateway.providers.base import ChatChunk, ChatRequest, ChatResponse

CacheStatus = Literal["miss", "exact", "semantic"]


@dataclass(frozen=True)
class AuthContext:
    """Resolved identity of the calling app (cached in Redis for 60 s, spec §6.1)."""

    tenant_id: uuid.UUID
    app_id: uuid.UUID
    key_id: uuid.UUID
    key_prefix: str
    scopes: frozenset[str]
    policy_version: int
    baseline_model: str
    tenant_active: bool = True
    app_active: bool = True
    expires_at: datetime | None = None
    ip_allowlist: tuple[str, ...] = ()


@dataclass(frozen=True)
class Classification:
    """Classifier output (Phase 3)."""

    task_type: str
    complexity: str
    confidence: float


@dataclass(frozen=True)
class RouteDecision:
    model_id: str  # catalog id, e.g. 'ollama/qwen2.5:7b'
    tier: str
    reason: str  # e.g. 'static:default', 'pinned'


@dataclass
class Attempt:
    """One upstream model call. All attempts are billed (spec §8.4)."""

    model_id: str
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    status_code: int = 200
    error: str | None = None


@dataclass
class RequestContext:
    request_id: str
    endpoint: Literal["chat", "models", "usage", "embeddings"]
    required_scope: str | None
    client_ip: str | None
    api_key: str | None = field(default=None, repr=False)  # raw bearer token: never logged

    auth: AuthContext | None = None
    policy: AppPolicy | None = None
    original: ChatRequest | None = field(default=None, repr=False)  # never logged
    working: ChatRequest | None = field(default=None, repr=False)  # copy sent upstream
    vault: dict[str, str] = field(default_factory=dict, repr=False)  # in-memory only (Phase 2)
    pii_counts: dict[str, int] = field(default_factory=dict)
    classification: Classification | None = None
    route: RouteDecision | None = None
    attempts: list[Attempt] = field(default_factory=list)
    cache_status: CacheStatus = "miss"
    response: ChatResponse | None = field(default=None, repr=False)
    stream: AsyncIterator[ChatChunk] | None = field(default=None, repr=False)

    # Filled in along the way.
    estimated_input_tokens: int = 0
    reserved_tokens: int = 0  # TPM tokens pre-charged by the rate limiter
    rate_limit_headers: dict[str, str] = field(default_factory=dict)
    status_code: int = 200
    cost_usd: Decimal = Decimal(0)
    baseline_usd: Decimal = Decimal(0)
    started_at: float = field(default_factory=time.perf_counter)
    stage_ms: dict[str, float] = field(default_factory=dict)
    metered: bool = False
    finished: bool = False  # post stages have run (guards against double-running)

    @property
    def saved_usd(self) -> Decimal:
        return self.baseline_usd - self.cost_usd

    def require_auth(self) -> AuthContext:
        if self.auth is None:  # programming error: authn must run first
            raise RuntimeError("stage requires authn to have run")
        return self.auth

    def require_policy(self) -> AppPolicy:
        if self.policy is None:
            raise RuntimeError("stage requires authz to have loaded the policy")
        return self.policy
