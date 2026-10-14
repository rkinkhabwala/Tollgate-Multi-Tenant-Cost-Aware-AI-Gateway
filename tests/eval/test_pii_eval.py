"""PII eval gate (spec §16 Phase 2, §17): recall per entity on `data/pii_eval.jsonl`.

Fails if recall is below 95% for EMAIL/PHONE/CREDIT_CARD/US_SSN/SECRET, or below
85% for PERSON. The full precision/recall table is printed (run with `-s` to see
it) and attached to the assertion message on failure.
"""

from gateway.redaction.engine import RedactionEngine
from scripts.eval_pii import RECALL_TARGETS, Score, evaluate, format_table, load_rows


def test_eval_set_is_large_and_covers_every_default_type() -> None:
    rows = load_rows()
    assert len(rows) >= 200
    types = {e["type"] for r in rows for e in r["entities"]}
    assert set(RECALL_TARGETS) <= types
    assert {"near_miss", "mixed", "secret"} <= {r["category"] for r in rows}


def test_recall_meets_targets(engine: RedactionEngine) -> None:
    scores = evaluate(engine, load_rows())
    table = format_table(scores)
    print("\n" + table)
    failing = {t: f"{scores.get(t, Score()).recall:.1%} < {target:.0%}" for t, target in RECALL_TARGETS.items() if scores.get(t, Score()).recall < target}
    assert not failing, f"recall below target: {failing}\n{table}"
