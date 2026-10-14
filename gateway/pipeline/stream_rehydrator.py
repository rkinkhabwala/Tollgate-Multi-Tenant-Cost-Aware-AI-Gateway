"""Placeholder → value re-hydration, non-streaming and streaming (spec §7.4).

`rehydrate_text()` replaces every `<<LABEL_N>>` that exists in *this request's* vault.
Unknown placeholders are left untouched (spec §15: a model cannot conjure values
that were never in the request). Mask placeholders (`<<EMAIL>>`, no number) never
match, so masked data stays masked.

`StreamRehydrator` does the same over a stream of text fragments. A placeholder can
be split across chunks (`<<EMA` | `IL_1>>`). Each `feed()` therefore holds back the
longest trailing piece that could still *become* a placeholder (`<`, `<<`,
`<<EMAIL_1`, `<<EMAIL_1>`) and emits the rest, re-hydrated. Because every
placeholder is at most 32 chars, holding at most 32 chars is enough. `flush()`
releases the tail at end of stream.

Invariant (fuzz-tested): for any split of a text into chunks,
`"".join(feed(c) for c in chunks) + flush() == rehydrate_text(text)`.

`json_escape=True` is for tool-call arguments, where values are inserted inside JSON
strings and must be escaped (quotes, backslashes, newlines).
"""

from __future__ import annotations

import json
import re

from gateway.redaction.vault import MAX_PLACEHOLDER_LEN, PLACEHOLDER_RE, Vault

# Every string that is a prefix of some `<<LABEL_N>>` (including the complete form
# minus its final '>').
_PREFIX_RE = re.compile(r"<(?:<(?:[A-Z][A-Z0-9_]*(?:_\d+>)?)?)?")


def is_placeholder_prefix(s: str) -> bool:
    return _PREFIX_RE.fullmatch(s) is not None


def _value(v: str, json_escape: bool) -> str:
    return json.dumps(v, ensure_ascii=False)[1:-1] if json_escape else v


def rehydrate_text(text: str, vault: Vault, *, json_escape: bool = False) -> str:
    if not vault or "<<" not in text:
        return text

    def sub(m: re.Match[str]) -> str:
        original = vault.get(m.group(0))
        return m.group(0) if original is None else _value(original, json_escape)

    return PLACEHOLDER_RE.sub(sub, text)


class StreamRehydrator:
    """Incremental re-hydration with a ≤ 32-char tail buffer."""

    def __init__(self, vault: Vault, *, json_escape: bool = False, max_hold: int = MAX_PLACEHOLDER_LEN) -> None:
        self._vault = vault
        self._json_escape = json_escape
        self._max_hold = max_hold
        self._buf = ""

    def _hold_from(self, buf: str) -> int:
        """Index where the held-back tail starts (len(buf) = hold nothing)."""
        for i in range(max(0, len(buf) - self._max_hold), len(buf)):
            if buf[i] == "<" and is_placeholder_prefix(buf[i:]):
                return i
        return len(buf)

    def feed(self, fragment: str) -> str:
        self._buf += fragment
        cut = self._hold_from(self._buf)
        ready, self._buf = self._buf[:cut], self._buf[cut:]
        return rehydrate_text(ready, self._vault, json_escape=self._json_escape)

    def flush(self) -> str:
        tail, self._buf = self._buf, ""
        return rehydrate_text(tail, self._vault, json_escape=self._json_escape)

    @property
    def pending(self) -> int:
        return len(self._buf)
