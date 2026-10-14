"""Redaction engine: pattern + NER analysis, then policy actions (spec §7.2–7.3).

The engine is built **once** at gateway startup (loading spaCy takes seconds) and is
shared by all requests. Analysis runs on a dedicated thread pool and never blocks the
event loop.

Two passes per text (Phase 2.5):
1. **Pattern pass, whole text.** A Presidio `AnalyzerEngine` runs on a *blank*
   spaCy pipeline: a tokenizer plus a component that sets `lemma = lowercase`, so
   Presidio's context-word boosts still work. It runs every pattern recognizer:
   secrets, email, phone, card (Luhn), SSN, IBAN, IP, licences, passports and the
   app's custom rules. This is cheap.
2. **NER pass, natural-language segments only.** `segments.nl_segments()` skips
   code fences, JSON payloads (except prose-like string values), base64 runs and
   stack traces. The remaining segments of *all* texts in a request are batched
   through `nlp.pipe()` on an `en_core_web_lg` pipeline reduced to just its `ner`
   component, which has its own embedded tok2vec, so the entities are identical
   at ~45% of the cost. They are mapped to Presidio types at Presidio's default
   NER score (0.85).

On top of that, `analyze_many()`:
* consults the per-message findings cache (`cache.py`), so repeated history is
  never re-analysed;
* fans the pattern passes and NER buckets out across the thread pool.

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

import spacy
from presidio_analyzer import AnalyzerEngine, RecognizerResult
from presidio_analyzer.nlp_engine import SpacyNlpEngine
from spacy.language import Language

from gateway.policy.schema import RedactionPolicy
from gateway.redaction.cache import CacheScope, FindingsCache
from gateway.redaction.recognizers.custom_regex import DENY_TERM, build_custom_recognizers
from gateway.redaction.recognizers.pii import (
    BarePhoneRecognizer,
    EmailPatternRecognizer,
    MastercardSeries2Recognizer,
    PhonePatternRecognizer,
    SsnPatternRecognizer,
)
from gateway.redaction.recognizers.secrets import KeywordSecretRecognizer, SecretPatternRecognizer
from gateway.redaction.segments import nl_segments
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


# spaCy NER label → Presidio entity type (Presidio's default mapping).
NER_LABELS: dict[str, str] = {
    "PERSON": "PERSON",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "FAC": "LOCATION",
    "NORP": "NRP",
    "ORG": "ORGANIZATION",
    "DATE": "DATE_TIME",
    "TIME": "DATE_TIME",
}
NER_SCORE = 0.85  # Presidio's default score for spaCy entities
NER_MIN_BUCKET_CHARS = 1_000  # NER work is split into ≤ `workers` jobs, none smaller than this


@Language.component("tollgate_lowercase_lemma")
def _lowercase_lemma(doc):  # type: ignore[no-untyped-def]
    """Cheap stand-in for a lemmatizer: Presidio's context words match lowercase forms."""
    for token in doc:
        token.lemma_ = token.lower_
    return doc


class _TokenizerOnlyNlpEngine(SpacyNlpEngine):
    """Presidio NLP engine backed by a blank English pipeline (no NER, no tagger)."""

    def __init__(self) -> None:
        super().__init__(models=[{"lang_code": "en", "model_name": "blank:en"}])

    def load(self) -> None:
        nlp = spacy.blank("en")
        nlp.add_pipe("tollgate_lowercase_lemma")
        self.nlp = {"en": nlp}


def _filter(text: str, findings: list[Finding], actions: dict[str, Action], threshold: float) -> list[Finding]:
    kept = [
        f
        for f in findings
        if actions.get(f.entity_type, "allow") != "allow"
        and f.end > f.start
        and f.score >= threshold
        and not (f.entity_type == "PERSON" and text[f.start : f.end].strip(" .,:;").lower() in PERSON_STOPWORDS)
    ]
    return resolve_overlaps(kept, actions)


class RedactionEngine:
    """Shared, thread-pooled two-pass analyzer. Build once per process."""

    def __init__(self, workers: int = 4, model: str = "en_core_web_lg", cache: FindingsCache | None = None) -> None:
        logging.getLogger("presidio-analyzer").setLevel(logging.WARNING)
        self.cache = cache if cache is not None else FindingsCache()
        self.model = model

        # Pass 1: pattern recognizers on a tokenizer-only pipeline.
        nlp_engine = _TokenizerOnlyNlpEngine()
        nlp_engine.load()
        self._patterns = AnalyzerEngine(nlp_engine=nlp_engine, supported_languages=["en"])
        self._patterns.registry.remove_recognizer("SpacyRecognizer")  # NER is pass 2
        self._patterns.registry.remove_recognizer("PhoneRecognizer")  # replaced, see recognizers/pii.py
        for recognizer in (
            EmailPatternRecognizer(),
            PhonePatternRecognizer(),
            BarePhoneRecognizer(),
            MastercardSeries2Recognizer(),
            SsnPatternRecognizer(),
            SecretPatternRecognizer(),
            KeywordSecretRecognizer(),
        ):
            self._patterns.registry.add_recognizer(recognizer)

        # Pass 2: NER only (its own tok2vec); everything else excluded.
        self._ner = spacy.load(model, exclude=["parser", "tagger", "attribute_ruler", "lemmatizer", "tok2vec", "senter"])
        self._workers = workers
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="redact")
        self.analyze_sync("Warm-up for Jane Doe at jane@corp.example.", RedactionPolicy())

    # --- passes ----------------------------------------------------------------
    def _pattern_findings(self, text: str, policy: RedactionPolicy) -> list[Finding]:
        ad_hoc = build_custom_recognizers(tuple((p.name, p.regex) for p in policy.custom_patterns), tuple(policy.deny_terms))
        results: list[RecognizerResult] = self._patterns.analyze(
            text=text, language="en", score_threshold=policy.score_threshold, ad_hoc_recognizers=list(ad_hoc) or None
        )
        return [Finding(r.entity_type, r.start, r.end, r.score) for r in results]

    def _ner_findings(self, pieces: list[tuple[int, int, str]]) -> list[tuple[int, Finding]]:
        """NER over (text_index, offset, segment) pieces → (text_index, finding)."""
        out: list[tuple[int, Finding]] = []
        for (idx, offset, _), doc in zip(pieces, self._ner.pipe(p[2] for p in pieces), strict=True):
            for ent in doc.ents:
                etype = NER_LABELS.get(ent.label_)
                if etype is not None:
                    out.append((idx, Finding(etype, offset + ent.start_char, offset + ent.end_char, NER_SCORE)))
        return out

    def analyze_sync(self, text: str, policy: RedactionPolicy) -> list[Finding]:
        """Blocking analysis of one text → non-overlapping actionable findings."""
        if not text.strip():
            return []
        found = self._pattern_findings(text, policy)
        pieces = [(0, s, text[s:e]) for s, e in nl_segments(text)]
        found += [f for _, f in self._ner_findings(pieces)]
        return _filter(text, found, effective_actions(policy), policy.score_threshold)

    async def analyze_many(
        self, texts: Sequence[str], policy: RedactionPolicy, scope: CacheScope | None = None
    ) -> list[list[Finding]]:
        """Analyze texts in parallel on the pool; results keep the input order.

        With a `scope`, each text's findings are looked up in, and stored into, the
        per-message cache (see `redaction/cache.py`). Only cache misses are analysed.
        """
        loop = asyncio.get_running_loop()
        results: list[list[Finding] | None] = [None] * len(texts)
        keys: list[bytes | None] = [None] * len(texts)
        misses: list[int] = []
        for i, text in enumerate(texts):
            if not text.strip():
                results[i] = []
                continue
            if scope is not None:
                keys[i] = self.cache.key(scope, text)
                cached = self.cache.get(keys[i])
                if cached is not None:
                    results[i] = list(cached)
                    continue
            misses.append(i)
        if not misses:
            return [r or [] for r in results]

        # Fan out: one pattern job per text, NER segments packed into ~equal buckets
        # (one per worker), so even a single-request burst uses the whole pool.
        pattern_jobs = {i: loop.run_in_executor(self._pool, self._pattern_findings, texts[i], policy) for i in misses}
        segments = {i: nl_segments(texts[i]) for i in misses}
        total = sum(e - s for segs in segments.values() for s, e in segs)
        target = max(NER_MIN_BUCKET_CHARS, -(-total // self._workers))
        buckets: list[list[tuple[int, int, str]]] = [[]]
        size = 0
        for i in misses:
            for s, e in segments[i]:
                if size >= target and len(buckets) < self._workers:
                    buckets.append([])
                    size = 0
                buckets[-1].append((i, s, texts[i][s:e]))
                size += e - s
        ner_jobs = [loop.run_in_executor(self._pool, self._ner_findings, b) for b in buckets if b]

        raw: dict[int, list[Finding]] = {i: await job for i, job in pattern_jobs.items()}
        for job in ner_jobs:
            for i, finding in await job:
                raw[i].append(finding)
        actions = effective_actions(policy)
        for i in misses:
            results[i] = _filter(texts[i], raw[i], actions, policy.score_threshold)
            if keys[i] is not None:
                self.cache.put(keys[i], results[i])
        return [r or [] for r in results]

    def close(self) -> None:
        self._pool.shutdown(wait=False, cancel_futures=True)
