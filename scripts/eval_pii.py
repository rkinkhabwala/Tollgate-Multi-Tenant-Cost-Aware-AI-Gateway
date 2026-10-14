"""Score the redaction engine against `data/pii_eval.jsonl` (precision/recall per entity).

    uv run python -m scripts.eval_pii

Matching: a labeled value counts as **recalled** if some detection of the same type
overlaps its span. A detection is a **true positive** if it overlaps a labeled span
of the same type, and a false positive otherwise. Detections of types that are never
labeled are pooled under `OTHER`, so their false positives still show up.

The default policy (default actions, threshold 0.5) is used, which is what live
traffic sees.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from gateway.policy.schema import RedactionPolicy
from gateway.redaction.engine import RedactionEngine

EVAL_FILE = Path(__file__).resolve().parent.parent / "data" / "pii_eval.jsonl"

# Thresholds from spec §16 Phase 2 (recall).
RECALL_TARGETS: dict[str, float] = {
    "EMAIL_ADDRESS": 0.95,
    "PHONE_NUMBER": 0.95,
    "CREDIT_CARD": 0.95,
    "US_SSN": 0.95,
    "SECRET": 0.95,
    "PERSON": 0.85,
}


@dataclass
class Score:
    labeled: int = 0
    recalled: int = 0
    detected: int = 0
    true_pos: int = 0
    misses: list[str] = field(default_factory=list)  # row ids (never values)

    @property
    def recall(self) -> float:
        return self.recalled / self.labeled if self.labeled else 1.0

    @property
    def precision(self) -> float:
        return self.true_pos / self.detected if self.detected else 1.0


def load_rows(path: Path = EVAL_FILE) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def evaluate(engine: RedactionEngine, rows: list[dict]) -> dict[str, Score]:
    labeled_types = {e["type"] for r in rows for e in r["entities"]}
    scores: dict[str, Score] = defaultdict(Score)
    policy = RedactionPolicy()
    for row in rows:
        text = row["text"]
        spans = [(e["type"], text.find(e["value"]), text.find(e["value"]) + len(e["value"])) for e in row["entities"]]
        found = engine.analyze_sync(text, policy)
        for etype, start, end in spans:
            s = scores[etype]
            s.labeled += 1
            if any(f.entity_type == etype and f.start < end and f.end > start for f in found):
                s.recalled += 1
            else:
                s.misses.append(row["id"])
        for f in found:
            bucket = f.entity_type if f.entity_type in labeled_types else "OTHER"
            s = scores[bucket]
            s.detected += 1
            if any(f.entity_type == t and f.start < e and f.end > st for t, st, e in spans):
                s.true_pos += 1
    return dict(scores)


def format_table(scores: dict[str, Score]) -> str:
    lines = [f"{'entity':<16}{'labeled':>8}{'recall':>9}{'precision':>11}{'target':>9}  status"]
    for etype in sorted(scores, key=lambda t: (t == "OTHER", t)):
        s = scores[etype]
        target = RECALL_TARGETS.get(etype)
        status = "" if target is None else ("PASS" if s.recall >= target else "FAIL")
        tgt = f"{target:.0%}" if target else "-"
        lines.append(f"{etype:<16}{s.labeled:>8}{s.recall:>9.1%}{s.precision:>11.1%}{tgt:>9}  {status}")
    return "\n".join(lines)


def main() -> None:
    rows = load_rows()
    scores = evaluate(RedactionEngine(), rows)
    print(f"PII eval: {len(rows)} samples from {EVAL_FILE.name}\n")
    print(format_table(scores))
    failed = [t for t, target in RECALL_TARGETS.items() if scores.get(t, Score()).recall < target]
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
