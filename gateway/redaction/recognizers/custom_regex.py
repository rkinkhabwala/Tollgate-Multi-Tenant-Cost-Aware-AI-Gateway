"""Tenant-defined recognizers built from an app's policy (spec §7.2).

* `redaction.custom_patterns`: `{name, regex, action}`. The entity type is `name`, for
  example `ACCOUNT_ID`.
* `redaction.deny_terms`: literal terms such as internal project codenames, matched
  case-insensitively on word boundaries. They are reported as entity `DENY_TERM`.

These are passed to Presidio per call as *ad-hoc* recognizers, so one app's patterns
can never apply to another app's requests. They are built per policy version and
memoised, so steady-state requests don't recompile regexes.
"""

from __future__ import annotations

import re
from functools import lru_cache

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer

DENY_TERM = "DENY_TERM"


@lru_cache(maxsize=512)
def build_custom_recognizers(
    patterns: tuple[tuple[str, str], ...], deny_terms: tuple[str, ...]
) -> tuple[EntityRecognizer, ...]:
    """`patterns` is ((name, regex), ...). Hashable arguments make this cacheable."""
    recognizers: list[EntityRecognizer] = [
        PatternRecognizer(
            supported_entity=name,
            name=f"custom:{name}",
            patterns=[Pattern(name.lower(), regex, 0.9)],
            supported_language="en",
        )
        for name, regex in patterns
    ]
    if deny_terms:
        alternation = "|".join(re.escape(t) for t in sorted(deny_terms, key=len, reverse=True))
        recognizers.append(
            PatternRecognizer(
                supported_entity=DENY_TERM,
                name="custom:deny_terms",
                patterns=[Pattern("deny_terms", rf"(?i)(?<!\w)(?:{alternation})(?!\w)", 0.95)],
                supported_language="en",
            )
        )
    return tuple(recognizers)
