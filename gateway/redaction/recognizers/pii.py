"""Pattern recognizers that run *alongside* Presidio's built-ins to raise recall.

Presidio's defaults miss two cases that matter for a gateway:
* **Email:** Presidio validates the top-level domain, so addresses on internal or
  reserved domains (`ops@corp.internal`, `jane@acme.example`) are not reported.
  This recognizer is TLD-agnostic.
* **Phone:** Presidio's `PhoneRecognizer` is replaced (the engine removes it). It
  scores a separated `415-555-0132` at only 0.4, below the default 0.5 threshold,
  yet it treats the word "number" as phone context, so "order number 4155550132"
  is boosted to 0.75. Here:
  - separated or `+country` formats score 0.6;
  - a bare 10-digit NANP run counts only when a phone keyword ("call", "phone",
    "mobile", ...) appears within 30 chars before it (`BarePhoneRecognizer`), so
    order numbers and ids are not reported. A plain keyword window is used instead
    of Presidio's lemma-based enhancer, which misses phrasings like "call me at".

`MastercardSeries2Recognizer` adds Luhn-validated Mastercard 2-series cards
(2221–2720). Presidio's card regex predates that range.

`SsnPatternRecognizer` covers dashed SSNs and excludes invalid ranges (000/666/9xx
area, 00 group, 0000 serial), which Presidio scores only weakly without context.

When a span is found by both Presidio and these, overlap resolution in the engine
keeps the higher-scoring result.
"""

from __future__ import annotations

import re

from presidio_analyzer import EntityRecognizer, Pattern, PatternRecognizer, RecognizerResult
from presidio_analyzer.nlp_engine import NlpArtifacts


class EmailPatternRecognizer(PatternRecognizer):
    def __init__(self) -> None:
        super().__init__(
            supported_entity="EMAIL_ADDRESS",
            name="TollgateEmailRecognizer",
            patterns=[Pattern("email", r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9-]+(?:\.[A-Za-z0-9-]+)*\.[A-Za-z]{2,}\b", 0.9)],
            supported_language="en",
        )


class PhonePatternRecognizer(PatternRecognizer):
    def __init__(self) -> None:
        super().__init__(
            supported_entity="PHONE_NUMBER",
            name="TollgatePhoneRecognizer",
            patterns=[
                # NANP with separators: 415-555-0132, (415) 555-0132, 415.555.0132, +1 415 555 0132
                Pattern(
                    "nanp",
                    r"(?<![\w#/-])(?:\+?1[\s.-]?)?(?:\(\d{3}\)\s?|\b\d{3}[\s.-])\d{3}[\s.-]\d{4}\b(?![-/]\d)",
                    0.6,
                ),
                # International with explicit +country code: +44 20 7946 0958, +49-30-901820
                Pattern("e164", r"(?<![\w+])\+(?:[2-9]\d{0,2})[\s.-]?\d{1,4}(?:[\s.-]?\d{2,4}){2,4}\b", 0.6),
            ],
            context=["phone", "call", "tel", "telephone", "mobile", "cell", "fax", "text", "reach", "contact", "dial", "sms", "whatsapp"],
            supported_language="en",
        )


_BARE_NANP = re.compile(r"(?<![\w#/-])[2-9]\d{2}[2-9]\d{6}\b(?![-/]\d)")
_PHONE_WORDS = re.compile(r"(?i)\b(?:phone|call|tel|telephone|mobile|cell|fax|text|reach|contact|dial|sms|whatsapp)\w*\b")


class BarePhoneRecognizer(EntityRecognizer):
    """10 unseparated digits are a phone only with a phone keyword shortly before."""

    def __init__(self) -> None:
        super().__init__(supported_entities=["PHONE_NUMBER"], name="TollgateBarePhoneRecognizer", supported_language="en")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None) -> list[RecognizerResult]:
        return [
            RecognizerResult("PHONE_NUMBER", m.start(), m.end(), 0.7)
            for m in _BARE_NANP.finditer(text)
            if _PHONE_WORDS.search(text[max(0, m.start() - 30) : m.start()])
        ]


def luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        d = int(ch)
        if i % 2 == 1:
            d = d * 2 - 9 if d > 4 else d * 2
        total += d
    return total % 10 == 0


_MC2 = re.compile(r"\b2(?:2[2-9]\d|[3-6]\d\d|7[01]\d|720)[ -]?\d{4}[ -]?\d{4}[ -]?\d{4}\b")


class MastercardSeries2Recognizer(EntityRecognizer):
    def __init__(self) -> None:
        super().__init__(supported_entities=["CREDIT_CARD"], name="TollgateMc2Recognizer", supported_language="en")

    def load(self) -> None:
        pass

    def analyze(self, text: str, entities: list[str], nlp_artifacts: NlpArtifacts | None = None) -> list[RecognizerResult]:
        out = []
        for m in _MC2.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if len(digits) == 16 and 2221 <= int(digits[:4]) <= 2720 and luhn_ok(digits):
                out.append(RecognizerResult("CREDIT_CARD", m.start(), m.end(), 1.0))
        return out


class SsnPatternRecognizer(PatternRecognizer):
    def __init__(self) -> None:
        super().__init__(
            supported_entity="US_SSN",
            name="TollgateSsnRecognizer",
            patterns=[Pattern("ssn_dashed", r"\b(?!000|666|9\d\d)\d{3}-(?!00)\d{2}-(?!0000)\d{4}\b", 0.6)],
            context=["ssn", "social", "security"],
            supported_language="en",
        )
