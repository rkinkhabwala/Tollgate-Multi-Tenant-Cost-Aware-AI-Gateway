"""NER segmenter: skip code/JSON/base64/stack traces, keep prose (incl. JSON values + code comments)."""

import json

from gateway.redaction.segments import nl_segments


def texts(text: str) -> list[str]:
    return [text[s:e] for s, e in nl_segments(text)]


def test_plain_prose_is_one_segment() -> None:
    assert texts("Hello Maria, see you Monday.") == ["Hello Maria, see you Monday."]


def test_fenced_code_skipped_but_comments_kept() -> None:
    t = "Fix this:\n```python\n# owner: Maria Gonzalez\nx = compute(a, b)\n```\nThanks Kenji Watanabe."
    segs = texts(t)
    assert "Fix this:\n" in segs and any("Thanks Kenji Watanabe" in s for s in segs)
    assert "owner: Maria Gonzalez" in segs
    assert not any("compute" in s for s in segs)


def test_json_payload_skipped_but_prose_values_kept() -> None:
    t = json.dumps({"customer": "Jane Doe", "id": "ord_12345", "notes": "call back after lunch", "n": 3})
    segs = texts(t)
    assert "Jane Doe" in segs and "call back after lunch" in segs
    assert not any("customer" in s or "ord_12345" in s for s in segs)


def test_embedded_json_span_skipped() -> None:
    t = 'Tool returned {"rows": [{"name": "Priya Raman", "score": 0.9}], "total": 1} — summarise it.'
    segs = texts(t)
    assert "Priya Raman" in segs
    assert any(s.startswith("Tool returned") for s in segs) and any("summarise it" in s for s in segs)
    assert not any('"rows"' in s for s in segs)


def test_base64_and_stack_traces_skipped() -> None:
    blob = "QUJDREVGR0hJSktMTU5PUFFSU1RVVldYWVphYmNkZWZnaGlqa2xtbm9w"
    trace = 'Traceback (most recent call last):\n  File "app.py", line 12, in run\nValueError: bad'
    segs = texts(f"payload {blob} failed.\n{trace}")
    assert not any(blob in s for s in segs)
    assert not any('File "app.py"' in s for s in segs)
    assert any("payload" in s for s in segs)


def test_offsets_map_back() -> None:
    t = 'x {"a": "Jane Doe was here", "b": 1, "c": [1, 2, 3, 4, 5]} y'
    for s, e in nl_segments(t):
        assert t[s:e] in ("x ", " y", "Jane Doe was here")
