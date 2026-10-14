"""Per-request placeholder vault (spec §7.4).

The vault maps placeholders to the original values (`<<EMAIL_1>>` → `jane@x.com`).
It lives **only** in the `RequestContext`, in memory, for one request. It is never
written to Redis, Postgres, logs or traces, and its `repr` shows only a count.

Numbering is per request, in order of first appearance, and stable: the same value
always gets the same placeholder within a request. Emails are matched
case-insensitively, so `Jane@X.com` and `jane@x.com` share a placeholder.

Labels are kept short, so every placeholder fits the streaming re-hydrator's
32-char tail buffer.
"""

from __future__ import annotations

import re
from collections import defaultdict

PLACEHOLDER_RE = re.compile(r"<<([A-Z][A-Z0-9_]*_\d+)>>")
MAX_PLACEHOLDER_LEN = 32

# Presidio entity type → placeholder label.
LABELS: dict[str, str] = {
    "EMAIL_ADDRESS": "EMAIL",
    "PHONE_NUMBER": "PHONE",
    "CREDIT_CARD": "CARD",
    "US_SSN": "SSN",
    "PERSON": "PERSON",
    "LOCATION": "LOCATION",
    "IP_ADDRESS": "IP",
    "IBAN_CODE": "IBAN",
    "US_BANK_NUMBER": "BANK_ACCOUNT",
    "US_DRIVER_LICENSE": "DRIVER_LICENSE",
    "US_PASSPORT": "PASSPORT",
    "MEDICAL_LICENSE": "MEDICAL_LICENSE",
    "DATE_TIME": "DATE",
    "SECRET": "SECRET",
    "DENY_TERM": "TERM",
}
_CASE_INSENSITIVE = {"EMAIL"}


def label_for(entity_type: str) -> str:
    """Short placeholder label. Custom entity names are already short (validated in policy)."""
    return LABELS.get(entity_type, entity_type)[:20]


class Vault:
    """placeholder → original value, for one request only."""

    __slots__ = ("_by_placeholder", "_by_key", "_counters")

    def __init__(self) -> None:
        self._by_placeholder: dict[str, str] = {}
        self._by_key: dict[tuple[str, str], str] = {}
        self._counters: defaultdict[str, int] = defaultdict(int)

    def tokenize(self, entity_type: str, value: str) -> str:
        """Return the stable `<<LABEL_N>>` placeholder for `value`, allocating if new."""
        label = label_for(entity_type)
        key = (label, value.casefold() if label in _CASE_INSENSITIVE else value)
        placeholder = self._by_key.get(key)
        if placeholder is None:
            self._counters[label] += 1
            placeholder = f"<<{label}_{self._counters[label]}>>"
            self._by_key[key] = placeholder
            self._by_placeholder[placeholder] = value
        return placeholder

    @staticmethod
    def mask(entity_type: str) -> str:
        """Non-reversible `<<LABEL>>` (no number, so never re-hydrated)."""
        return f"<<{label_for(entity_type)}>>"

    def get(self, placeholder: str) -> str | None:
        return self._by_placeholder.get(placeholder)

    def __contains__(self, placeholder: object) -> bool:
        return placeholder in self._by_placeholder

    def __len__(self) -> int:
        return len(self._by_placeholder)

    def __bool__(self) -> bool:
        return bool(self._by_placeholder)

    def __repr__(self) -> str:  # never show contents
        return f"Vault({len(self)} entries)"

    __str__ = __repr__

    def clear(self) -> None:
        self._by_placeholder.clear()
        self._by_key.clear()
        self._counters.clear()
