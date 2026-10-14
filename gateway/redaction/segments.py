"""Split a text into natural-language segments for NER (Phase 2.5).

spaCy NER is the expensive part of redaction, and it is pointless on code, JSON, or
base64 blobs. Pattern recognizers (secrets, email, phone, card, SSN, custom regex)
still scan the *whole* text; only the NER pass is restricted to the segments this
module returns.

Excluded from NER:
* fenced code blocks (```` ``` ```` or `~~~`, closed or running to end of text);
* JSON payloads: the whole text if it parses as JSON, or any balanced `{…}` / `[…]`
  span of at least 40 chars that parses;
* base64-like runs of at least 40 chars;
* stack-trace lines (Python `File "…", line N`, Java/JS `at foo(…)`, `Traceback`).

Two exceptions, both to protect PERSON recall. Prose inside skipped regions is
returned as small extra segments:
* string *values* in JSON payloads that read like prose (letters plus whitespace,
  ≤ 300 chars, no escapes). `{"customer": "Jane Doe"}` is still checked for names;
  keys and other values are not;
* comments inside fenced code (`# …`, `// …`, `-- …`, `/* … */`, ` * …`), e.g.
  `# owner: Jane Doe`.

Segments are `(start, end)` offsets into the original text, so NER findings map
straight back.
"""

from __future__ import annotations

import json
import re

_FENCE = re.compile(r"(```|~~~)[\s\S]*?(?:\1|\Z)")
_BASE64ISH = re.compile(r"[A-Za-z0-9+/=_-]{40,}")
_STACK_LINE = re.compile(r"(?m)^[ \t]*(?:Traceback \(most recent call last\):|File \".*\", line \d+.*|at [\w$.<>]+\(.*\)\s*)$")
_JSON_STRING = re.compile(r'"((?:[^"\\\n]|\\.){1,300})"(\s*:)?')
_PROSE = re.compile(r"[A-Za-z][^\n]*\s[^\n]*[A-Za-z]")
_COMMENT = re.compile(r"(?m)(?:#|//|--|/\*|^\s*\*)[ \t]*(?P<text>[^\n]*?)[ \t]*(?:\*/)?$")
_MIN_JSON_SPAN = 40
_DECODER = json.JSONDecoder()


def _json_spans(text: str) -> list[tuple[int, int]]:
    """Balanced JSON objects/arrays (≥ 40 chars) that actually parse."""
    stripped = text.strip()
    if stripped[:1] in "{[":
        try:
            json.loads(stripped)
            start = text.index(stripped[0])
            return [(start, start + len(stripped))]
        except ValueError:
            pass
    spans: list[tuple[int, int]] = []
    i = 0
    while True:
        i = min((p for p in (text.find("{", i), text.find("[", i)) if p != -1), default=-1)
        if i == -1:
            return spans
        try:
            _, end = _DECODER.raw_decode(text, i)
        except ValueError:
            i += 1
            continue
        if end - i >= _MIN_JSON_SPAN:
            spans.append((i, end))
        i = end


def _merge(spans: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[tuple[int, int]] = []
    for s, e in sorted(spans):
        if merged and s <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], e))
        else:
            merged.append((s, e))
    return merged


def nl_segments(text: str) -> list[tuple[int, int]]:
    """(start, end) spans of `text` that should go through NER."""
    if not text.strip():
        return []
    json_spans = _json_spans(text)
    fences = [m.span() for m in _FENCE.finditer(text)]
    excluded = list(json_spans) + fences
    excluded += [m.span() for m in _BASE64ISH.finditer(text)]
    excluded += [m.span() for m in _STACK_LINE.finditer(text)]
    excluded = _merge(excluded)

    segments: list[tuple[int, int]] = []
    cursor = 0
    for s, e in excluded:
        if text[cursor:s].strip():
            segments.append((cursor, s))
        cursor = max(cursor, e)
    if text[cursor:].strip():
        segments.append((cursor, len(text)))

    # Prose-like string values inside JSON payloads still get NER (not keys).
    for s, e in json_spans:
        for m in _JSON_STRING.finditer(text, s, e):
            if m.group(2) is None and "\\" not in m.group(1) and _PROSE.search(m.group(1)):
                segments.append((m.start(1), m.end(1)))
    # Comments inside fenced code are prose too.
    for s, e in fences:
        for m in _COMMENT.finditer(text, s, e):
            if m.group("text") and _PROSE.search(m.group("text")):
                segments.append((m.start("text"), m.end("text")))
    return sorted(set(segments))
