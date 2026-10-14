"""Redaction engine: Presidio analysis + policy actions (spec §7.2–7.3).

`RedactionEngine` wraps a single Presidio `AnalyzerEngine` backed by spaCy
`en_core_web_lg`. Loading the model takes seconds, so the engine is built **once**
at gateway startup and shared by all requests. spaCy inference is CPU-bound, so
`analyze_many()` runs it on a dedicated thread pool and never blocks the event loop.

Performance (measured with `scripts/bench_redaction.py` on an M1 Pro):
* spaCy's dependency `parser` is disabled. Presidio uses tokens, lemmas and NER,
  never the parse, and dropping it saves ~25% with identical eval results.
* the messages of a request are analysed in parallel across the pool. spaCy's
  numeric kernels release the GIL, so 4 threads give about 2.5–3.5× on multi-turn
  requests.

Recognizers:
* Presidio built-ins: PERSON, LOCATION, EMAIL, PHONE, CREDIT_CARD (Luhn), US_SSN,
  IBAN, IP, US_BANK_NUMBER, US_DRIVER_LICENSE, US_PASSPORT, MEDICAL_LICENSE,
  DATE_TIME, ...
* Tollgate additions: `recognizers/pii.py` (TLD-agnostic email, separator-aware
  phone, dashed SSN) and `recognizers/secrets.py` (entity `SECRET`).
* Per-app ad-hoc recognizers from policy: `recognizers/custom_regex.py`.

Pipeline for one text:
1. analyze, filtered by `score_threshold`;
2. drop entity types whose effective action is `allow`;
3. resolve overlaps greedily. The most protective action wins (block > mask >
   tokenize), then the higher score, then the longer span. For example, a token that
   spaCy also calls a PERSON is still treated as a SECRET;
4. `apply_actions()` rewrites the text right to left: tokenize → vault placeholder,
   mask → `<<LABEL>>`, block → reported to the caller, which rejects the request.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal

from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import NlpEngineProvider

from gateway.policy.schema import RedactionPolicy
from gateway.redaction.recognizers.custom_regex import DENY_TERM, build_custom_recognizers
from gateway.redaction.recognizers.pii import (
    BarePhoneRecognizer,
    EmailPatternRecognizer,
    MastercardSeries2Recognizer,
    PhonePatternRecognizer,
    SsnPatternRecognizer,
)
from gateway.redaction.recognizers.secrets import KeywordSecretRecognizer, SecretPatternRecognizer
from gateway.redaction.vault import Vault

Action = Literal["tokenize", "mask", "block", "allow"]

# Spec §7.3 defaults: secrets → block; government ids and card/bank numbers → mask;
# names, emails, phones, addresses → tokenize. Anything not listed → allow.
# `policy.redaction.entities` overrides per entity type.
DEFAULT_ACTIONS: dict[str, Action] = {
    "SECRET": "block",
    "CREDIT_CARD": "mask",
    "US_SSN": "mask",
    "US_PASSPORT": "mask",
    "US_DRIVER_LICENSE": "mask",
    "US_BANK_NUMBER": "mask",
    "IBAN_CODE": "mask",
    "MEDICAL_LICENSE": "mask",
    "PERSON": "tokenize",
    "EMAIL_ADDRESS": "tokenize",
    "PHONE_NUMBER": "tokenize",
    "LOCATION": "tokenize",
    "IP_ADDRESS": "tokenize",
    DENY_TERM: "tokenize",
    "DATE_TIME": "allow",
}


# spaCy sometimes tags a capitalised message word as PERSON ("Email jane@…",
# "Call 415-…"). Single-word PERSON hits on these are dropped. Real names are
# never on this list.
PERSON_STOPWORDS = frozenset(
    "email e-mail mail call phone text sms contact hi hello hey dear thanks regards cheers ssn "
    "deploy key card mobile fax tel ticket invoice order account customer user admin support".split()
)


@dataclass(frozen=True)
class Finding:
    entity_type: str
    start: int
    end: int
    score: float


@dataclass(frozen=True)
class Applied:
    """One finding after its action was applied. Carries no original value."""

    entity_type: str
    start: int
    end: int
    score: float
    action: Action
    placeholder: str | None  # None for block/allow


def effective_actions(policy: RedactionPolicy) -> dict[str, Action]:
    actions: dict[str, Action] = {**DEFAULT_ACTIONS, **policy.entities}
    for p in policy.custom_patterns:
        actions[p.name] = p.action
    return actions


_SEVERITY: dict[str, int] = {"block": 3, "mask": 2, "tokenize": 1, "allow": 0}


def resolve_overlaps(findings: Sequence[Finding], actions: dict[str, Action] | None = None) -> list[Finding]:
    """Keep a non-overlapping subset, sorted by start.

    Priority: most protective action, then score, then span length.
    """
    actions = actions or {}

    def rank(f: Finding) -> tuple[int, float, int, int]:
        return (-_SEVERITY[actions.get(f.entity_type, "allow")], -f.score, -(f.end - f.start), f.start)

    chosen: list[Finding] = []
    for f in sorted(findings, key=rank):
        if all(f.end <= c.start or f.start >= c.end for c in chosen):
            chosen.append(f)
    return sorted(chosen, key=lambda f: f.start)


def apply_actions(text: str, findings: Sequence[Finding], actions: dict[str, Action], vault: Vault) -> tuple[str, list[Applied]]:
    """Rewrite `text` according to each finding's action.

    Placeholders are allocated left to right (first appearance → lowest number), and
    the text is then rebuilt in one pass.
    """
    applied: list[Applied] = []
    pieces: list[str] = []
    cursor = 0
    for f in findings:  # sorted by start, non-overlapping
        action = actions.get(f.entity_type, "allow")
        placeholder: str | None = None
        if action == "tokenize":
            placeholder = vault.tokenize(f.entity_type, text[f.start : f.end])
        elif action == "mask":
            placeholder = vault.mask(f.entity_type)
        applied.append(Applied(f.entity_type, f.start, f.end, f.score, action, placeholder))
        if placeholder is not None:
            pieces.append(text[cursor : f.start])
            pieces.append(placeholder)
            cursor = f.end
    pieces.append(text[cursor:])
    return "".join(pieces), applied


class RedactionEngine:
    """Shared, thread-pooled Presidio analyzer. Build once per process."""

    def __init__(self, workers: int = 4, model: str = "en_core_web_lg") -> None:
        logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)
        nlp = NlpEngineProvider(
            nlp_configuration={"nlp_engine_name": "spacy", "models": [{"lang_code": "en", "model_name": model}]}
        ).create_engine()
        for pipeline in nlp.nlp.values():  # Presidio never reads the dependency parse
            if "parser" in pipeline.pipe_names:
                pipeline.disable_pipe("parser")
        self._analyzer = AnalyzerEngine(nlp_engine=nlp, supported_languages=["en"])
        self._analyzer.registry.remove_recognizer("PhoneRecognizer")  # replaced, see recognizers/pii.py
        for recognizer in (
            EmailPatternRecognizer(),
            PhonePatternRecognizer(),
            BarePhoneRecognizer(),
            MastercardSeries2Recognizer(),
            SsnPatternRecognizer(),
            SecretPatternRecognizer(),
            KeywordSecretRecognizer(),
        ):
            self._analyzer.registry.add_recognizer(recognizer)
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="redact")
        self.analyze_sync("Warm-up for Jane Doe at jane@corp.example.", RedactionPolicy())

    def analyze_sync(self, text: str, policy: RedactionPolicy) -> list[Finding]:
        """Blocking analysis of one text → non-overlapping actionable findings."""
        if not text.strip():
            return []
        actions = effective_actions(policy)
        ad_hoc = build_custom_recognizers(
            tuple((p.name, p.regex) for p in policy.custom_patterns), tuple(policy.deny_terms)
        )
        results: list[RecognizerResult] = self._analyzer.analyze(
            text=text, language="en", score_threshold=policy.score_threshold, ad_hoc_recognizers=list(ad_hoc) or None
        )
        findings = [
            Finding(r.entity_type, r.start, r.end, r.score)
            for r in results
            if actions.get(r.entity_type, "allow") != "allow"
            and r.end > r.start
            and not (r.entity_type == "PERSON" and text[r.start : r.end].strip(" .,:;").lower() in PERSON_STOPWORDS)
        ]
        return resolve_overlaps(findings, actions)

    async def analyze_many(self, texts: Sequence[str], policy: RedactionPolicy) -> list[list[Finding]]:
        """Analyze texts in parallel on the pool; results keep the input order."""
        loop = asyncio.get_running_loop()
        jobs = [loop.run_in_executor(self._pool, self.analyze_sync, t, policy) if t.strip() else None for t in texts]
        return [await j if j is not None else [] for j in jobs]

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
