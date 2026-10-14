"""Secret and credential recognizers (spec §7.2). Every match is reported as entity `SECRET`.

Two kinds:

* `SecretPatternRecognizer`: well-known token formats with distinctive prefixes or
  structure, so they're high-confidence on their own. Covers AWS access key ids,
  OpenAI/Stripe style `sk-`/`sk_live_`, GitHub `ghp_`…, Slack `xox?-`, Google
  `AIza…`, JWTs, PEM private-key blocks, connection strings that embed a password
  (`scheme://user:pass@host`), and Tollgate's own `tg_…` keys.
* `KeywordSecretRecognizer`: catches generic secrets that have no fixed format.
  - A high-entropy token (≥ 32 chars, Shannon entropy ≥ 3.5 bits/char, mixed character
    classes) appearing within ~40 chars after a word like key, secret, token,
    password, credential or auth.
  - The value in `password = ...` / `pwd: ...` style assignments, at any length.

The default policy's action for `SECRET` is `block` (spec §7.3).
"""

from __future__ import annotations

import math
import re
from collections import Counter

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts

SECRET = "SECRET"

_PATTERNS: list[tuple[str, str, float]] = [
    ("aws_access_key_id", r"\b(?:AKIA|ASIA|AGPA|AIDA|AROA|ANPA)[0-9A-Z]{16}\b", 0.95),
    ("openai_style_key", r"\bsk-(?:proj-|live-|test-|ant-)?[A-Za-z0-9_-]{20,}", 0.95),
    ("stripe_key", r"\b[rs]k_(?:live|test)_[A-Za-z0-9]{16,}\b", 0.95),
    ("github_token", r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{36,}\b", 0.95),
    ("github_pat", r"\bgithub_pat_[A-Za-z0-9_]{50,}\b", 0.95),
    ("slack_token", r"\bxox[abposr]-[A-Za-z0-9-]{10,}", 0.95),
    ("google_api_key", r"\bAIza[0-9A-Za-z_-]{35}\b", 0.9),
    ("jwt", r"\beyJ[A-Za-z0-9_-]{8,}\.eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}", 0.95),
    (
        "pem_private_key",
        r"-----BEGIN (?:[A-Z0-9]+ )*PRIVATE KEY-----[\s\S]+?-----END (?:[A-Z0-9]+ )*PRIVATE KEY-----",
        1.0,
    ),
    ("connection_string_with_password", r"\b[a-zA-Z][a-zA-Z0-9+.-]{1,20}://[^\s:/@]+:[^\s@/]+@[^\s/?#]+", 0.9),
    ("tollgate_api_key", r"\btg_(?:live|test)_[a-z0-9]{8}_[A-Za-z0-9]{32}\b", 1.0),
]


class SecretPatternRecognizer(PatternRecognizer):
    def __init__(self) -> None:
        super().__init__(
            supported_entity=SECRET,
            name="SecretPatternRecognizer",
            patterns=[Pattern(name, regex, score) for name, regex, score in _PATTERNS],
            supported_language="en",
        )


_KEYWORD = r"(?:api[_-]?key|access[_-]?key|secret|token|passw(?:or)?d|pwd|credential|auth|bearer|private[_-]?key)"
_CANDIDATE = re.compile(r"[A-Za-z0-9+/=_.\-]{32,}")
_KEYWORD_NEAR = re.compile(_KEYWORD, re.IGNORECASE)
_ASSIGNMENT = re.compile(r"(?i)\b(?:passw(?:or)?d|pwd|passphrase)\b\s*[:=]\s*[\"']?(?P<value>[^\s\"',;]{4,})")
_LOOKBEHIND_CHARS = 40


def shannon_entropy(s: str) -> float:
    counts = Counter(s)
    n = len(s)
    return -sum(c / n * math.log2(c / n) for c in counts.values())


def looks_random(token: str) -> bool:
    """High entropy and at least 3 of: lower, upper, digit, symbol."""
    classes = sum(
        [
            any(c.islower() for c in token),
            any(c.isupper() for c in token),
            any(c.isdigit() for c in token),
            any(not c.isalnum() for c in token),
        ]
    )
    return shannon_entropy(token) >= 3.5 and classes >= 3


class KeywordSecretRecognizer(EntityRecognizer):
    """High-entropy tokens near secret-ish keywords, and password assignments."""

    def __init__(self) -> None:
        super().__init__(supported_entities=[SECRET], name="KeywordSecretRecognizer", supported_language="en")

    def load(self) -> None:  # no model to load
        pass

    def analyze(self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None) -> list[RecognizerResult]:
        results: list[RecognizerResult] = []
        for m in _CANDIDATE.finditer(text):
            token = m.group(0).strip("=.-_")
            window = text[max(0, m.start() - _LOOKBEHIND_CHARS) : m.start()]
            if looks_random(token) and _KEYWORD_NEAR.search(window):
                results.append(RecognizerResult(SECRET, m.start(), m.end(), 0.8))
        for m in _ASSIGNMENT.finditer(text):
            results.append(RecognizerResult(SECRET, m.start("value"), m.end("value"), 0.85))
        return results
