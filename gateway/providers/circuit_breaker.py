"""Per-model circuit breaker (spec §8.4): open after 5 failures in 30 s, half-open after 60 s.

States, per model id:
* **closed:** calls flow. Failures are counted within a sliding 30 s window; the
  5th failure inside the window opens the circuit.
* **open:** the model is skipped. The router treats it as ineligible, and invoke
  falls back to the next candidate. After 60 s the next `allow()` moves it to
  half-open.
* **half-open:** exactly one trial call is allowed. Success closes the circuit and
  clears the counters; failure re-opens it for another 60 s.

Only *retryable* failures (timeouts, connection errors, 429, 5xx, overload) count.
A 400 is the request's fault, not the model's.

State lives in-process, so each gateway replica learns independently. That is
acceptable: a replica that hasn't seen the failures pays at most 5 failed attempts
before it opens too, and each of those falls back.
"""

from __future__ import annotations

import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Literal

State = Literal["closed", "open", "half_open"]


@dataclass
class _Circuit:
    state: State = "closed"
    failures: deque[float] = field(default_factory=deque)
    opened_at: float = 0.0
    trial_in_flight: bool = False


class CircuitBreaker:
    def __init__(
        self,
        failure_threshold: int = 5,
        window_s: float = 30.0,
        cooldown_s: float = 60.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.failure_threshold = failure_threshold
        self.window_s = window_s
        self.cooldown_s = cooldown_s
        self._clock = clock
        self._circuits: dict[str, _Circuit] = {}
        self._lock = threading.Lock()

    def _get(self, model: str) -> _Circuit:
        return self._circuits.setdefault(model, _Circuit())

    def state(self, model: str) -> State:
        with self._lock:
            c = self._get(model)
            if c.state == "open" and self._clock() - c.opened_at >= self.cooldown_s:
                return "half_open"
            return c.state

    def available(self, model: str) -> bool:
        """For routing: is the model worth considering? (Doesn't consume the trial.)"""
        with self._lock:
            c = self._get(model)
            if c.state == "closed":
                return True
            if c.state == "open":
                return self._clock() - c.opened_at >= self.cooldown_s
            return not c.trial_in_flight

    def allow(self, model: str) -> bool:
        """For invoking: may a call go out now? Claims the single half-open trial."""
        with self._lock:
            c = self._get(model)
            if c.state == "closed":
                return True
            if c.state == "open":
                if self._clock() - c.opened_at < self.cooldown_s:
                    return False
                c.state = "half_open"
            if c.trial_in_flight:
                return False
            c.trial_in_flight = True
            return True

    def record_success(self, model: str) -> None:
        with self._lock:
            c = self._get(model)
            c.state, c.trial_in_flight = "closed", False
            c.failures.clear()

    def record_failure(self, model: str) -> None:
        with self._lock:
            c = self._get(model)
            now = self._clock()
            if c.state == "half_open":
                c.state, c.opened_at, c.trial_in_flight = "open", now, False
                return
            c.failures.append(now)
            while c.failures and now - c.failures[0] > self.window_s:
                c.failures.popleft()
            if len(c.failures) >= self.failure_threshold:
                c.state, c.opened_at = "open", now
                c.failures.clear()

    def snapshot(self) -> dict[str, State]:
        return {m: self.state(m) for m in list(self._circuits)}
