"""Per-message redaction findings cache (Phase 2.5).

Multi-turn chat resends the whole history every turn, so the same messages get
analysed again and again. This cache remembers, per message text, *where* the
sensitive spans are, so only new messages pay for Presidio/spaCy.

What is stored, and where:
* **Key:** `HMAC-SHA256(secret, tenant_id | app_id | policy_version | text)`.
  - The secret is 32 random bytes generated per process (or set via
    `TOLLGATE_REDACTION_CACHE_SECRET`), so keys can't be correlated across
    processes or restarts, and the text can't be recovered from them.
  - Tenant, app and policy version are part of the key. Apps never share entries,
    and a policy change (version bump) invalidates naturally.
* **Value:** a tuple of `Finding(entity_type, start, end, score)`. Offsets and types
  only, never text or values.
* **Storage:** in process memory only (LRU), never Redis or any other shared store,
  so no data derived from prompts leaves the process. It is bounded both by entry
  count and by the total number of findings held.

Re-applying cached findings goes through the same `apply_actions()` in the same
order, so placeholders and vault numbering are identical to a fresh analysis
(tested).
"""

from __future__ import annotations

import hashlib
import hmac
import os
import threading
import uuid
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover
    from gateway.redaction.engine import Finding


@dataclass(frozen=True)
class CacheScope:
    """Who a cached result belongs to. Part of every key."""

    tenant_id: uuid.UUID
    app_id: uuid.UUID
    policy_version: int


class FindingsCache:
    def __init__(self, max_entries: int = 20_000, max_findings: int = 500_000, secret: bytes | None = None) -> None:
        self._secret = secret or os.urandom(32)
        self._max_entries = max_entries
        self._max_findings = max_findings
        self._data: OrderedDict[bytes, tuple[Finding, ...]] = OrderedDict()
        self._findings = 0
        self._lock = threading.Lock()  # accessed from the event loop and the thread pool
        self.hits = 0
        self.misses = 0

    def key(self, scope: CacheScope, text: str) -> bytes:
        msg = b"|".join([scope.tenant_id.bytes, scope.app_id.bytes, str(scope.policy_version).encode(), text.encode("utf-8")])
        return hmac.new(self._secret, msg, hashlib.sha256).digest()

    def get(self, key: bytes) -> tuple[Finding, ...] | None:
        with self._lock:
            found = self._data.get(key)
            if found is None:
                self.misses += 1
                return None
            self._data.move_to_end(key)
            self.hits += 1
            return found

    def put(self, key: bytes, findings: Sequence[Finding]) -> None:
        value = tuple(findings)
        with self._lock:
            old = self._data.pop(key, None)
            if old is not None:
                self._findings -= len(old)
            self._data[key] = value
            self._findings += len(value)
            while self._data and (len(self._data) > self._max_entries or self._findings > self._max_findings):
                _, evicted = self._data.popitem(last=False)
                self._findings -= len(evicted)

    def __len__(self) -> int:
        return len(self._data)

    @property
    def findings_held(self) -> int:
        return self._findings

    def clear(self) -> None:
        with self._lock:
            self._data.clear()
            self._findings = 0
