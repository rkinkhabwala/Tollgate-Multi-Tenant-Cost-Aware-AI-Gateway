"""Structured JSON logging with a hard guarantee against leaking sensitive data.

Spec §4 and §15: raw prompts, redaction-vault contents and API keys must never
appear in logs. Two layers enforce that, so a careless `logger.info(...)` anywhere in
the codebase cannot leak:

1. `SensitiveDataFilter` (attached to every handler):
   * drops record attributes whose names imply sensitive payloads (`messages`,
     `prompt`, `content`, `vault`, `api_key`, `password`, ...), and
   * scrubs credential-shaped strings from the message and its args.
2. `JsonFormatter` runs the same scrubber over the final rendered line, including
   exception tracebacks (which can embed local variables or request bodies).

Prompts are never passed to loggers in this codebase; the filter is a safety net.
"""

from __future__ import annotations

import json
import logging
import re
import sys
from datetime import UTC, datetime
from typing import Any

# Record attributes that must never be emitted, whatever their value.
SENSITIVE_FIELDS: frozenset[str] = frozenset(
    {
        "messages",
        "prompt",
        "prompts",
        "content",
        "vault",
        "api_key",
        "authorization",
        "password",
        "secret",
        "token",
        "refresh_token",
        "access_token",
        "body",
    }
)

# (pattern, replacement) pairs applied to every rendered string.
_SCRUBBERS: list[tuple[re.Pattern[str], str]] = [
    # Tollgate API keys: keep env + public prefix (shown in UI anyway), drop secret.
    (re.compile(r"\btg_(live|test)_([a-z0-9]{8})_[A-Za-z0-9]{8,}"), r"tg_\1_\2_[REDACTED]"),
    (re.compile(r"(?i)\bbearer\s+[A-Za-z0-9._~+/=-]+"), "Bearer [REDACTED]"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{10,}"), "sk-[REDACTED]"),
    (re.compile(r"\beyJ[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]{5,}\.[A-Za-z0-9_-]+"), "[REDACTED_JWT]"),
    (re.compile(r"(?i)(password|secret|api[_-]?key)(\"?\s*[:=]\s*\"?)[^\s\",}]+"), r"\1\2[REDACTED]"),
]

# Attributes every LogRecord has; anything else came from `extra=`.
_STANDARD_ATTRS = set(vars(logging.LogRecord("", 0, "", 0, "", (), None))) | {"message", "asctime"}


def scrub(text: str) -> str:
    """Remove credential-shaped substrings from `text`."""
    for pattern, repl in _SCRUBBERS:
        text = pattern.sub(repl, text)
    return text


class SensitiveDataFilter(logging.Filter):
    """Drop sensitive `extra` fields and scrub credentials from message + args."""

    def filter(self, record: logging.LogRecord) -> bool:
        for name in list(vars(record)):
            if name not in _STANDARD_ATTRS and name.lower() in SENSITIVE_FIELDS:
                delattr(record, name)
        # Render once so args can't smuggle secrets past the scrubber.
        try:
            rendered = record.getMessage()
        except Exception:  # malformed format string: fall back to the raw template
            rendered = str(record.msg)
        record.msg = scrub(rendered)
        record.args = None
        return True


class JsonFormatter(logging.Formatter):
    """One JSON object per line; final pass through the scrubber."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, UTC).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
        }
        for name, value in vars(record).items():
            if name not in _STANDARD_ATTRS and name.lower() not in SENSITIVE_FIELDS:
                payload[name] = value
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return scrub(json.dumps(payload, default=str))


def configure_logging(level: str = "INFO") -> None:
    """Install the JSON handler + filter on the root logger and uvicorn's loggers."""
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    handler.addFilter(SensitiveDataFilter())

    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level.upper())
    for name in ("uvicorn", "uvicorn.error", "uvicorn.access"):
        lg = logging.getLogger(name)
        lg.handlers = []
        lg.propagate = True
    # httpx logs full URLs at INFO; keep it quiet.
    logging.getLogger("httpx").setLevel(logging.WARNING)
